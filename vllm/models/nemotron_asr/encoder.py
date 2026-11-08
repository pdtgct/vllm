# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache-aware FastConformer encoder — P2 full-context regime.

Op-for-op translation of the pinned NeMo modules for the shipped
checkpoint's configuration (HF config.json, fetched 2026-07-11:
d_model 1024, 8 heads, FF 4096, 24 layers, conv kernel 9, causal
dw-striding 8x subsampling at 256 channels, attention/convolution bias
FALSE, activation silu, scale_input FALSE):

- dw-striding causal subsampling (subsampling.py:184-257,421-478)
- Transformer-XL relative-position MHA with per-layer pos biases
  (multi_head_attention.py:212-354; mask polarity True = masked)
- causal depthwise conv module (conformer_modules.py:236-340,
  causal_convs.py:73-151; left pad 8, right 0)
- macaron conformer layer (conformer_modules.py:160-230)
- centered rel-pos encoding (multi_head_attention.py:1056-1100)

Streaming (per-chunk cache threading over spec pages) is the P3
re-plumb; the probe/bring-up configuration runs one window over the
whole utterance with cross-chunk state dormant (PORT-REGIME-001). All
source refs @ NeMo de242add.
"""

import math

import torch
from torch import nn

from vllm.models.nemotron_asr.batch_invariance import (
    batch_invariant_enabled,
    depthwise_conv1d_fp32,
    invariant_gemm,
    invariant_linear,
    invariant_matmul,
    layer_norm_fp32,
    pointwise_conv_as_linear,
    softmax_fp32_sum,
    strided_conv2d_fp32,
)
from vllm.models.nemotron_asr.masks import (
    chunked_limited_mask,
)

_LOG_BASE = 10000.0
# Finite additive mask value of NeMo's SDPA path (INF_VAL,
# NVIDIA-NeMo/Speech@981fd4e0 multi_head_attention.py:54). Representable in
# FP16; exp(-10000 + O(scores)) underflows to exactly 0 in every dtype.
_SDPA_MASK_VALUE = 10000.0
# Memory-efficient SDPA needs attention-bias rows on a 16-element pitch or
# it pads (copies) the bias itself; the bias is computed at such a pitch.
_SDPA_BIAS_ALIGNMENT = 16


def _conv_out_len(length: int, *, pad: int, kernel: int, stride: int) -> int:
    return (length + pad - kernel) // stride + 1


# @spec PORT-PREC-019, PORT-PREC-021: branch only on the installed record.
def _conv(module: nn.Conv1d | nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    if not batch_invariant_enabled(module):
        return module(x)
    if (
        all(k == 1 for k in module.kernel_size)
        and all(s == 1 for s in module.stride)
        and all(p == 0 for p in module.padding)
        and all(d == 1 for d in module.dilation)
        and module.groups == 1
    ):
        return pointwise_conv_as_linear(x, module.weight, module.bias)
    if isinstance(module, nn.Conv1d):
        return depthwise_conv1d_fp32(
            x,
            module.weight,
            module.bias,
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
        )
    return strided_conv2d_fp32(
        x,
        module.weight,
        module.bias,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
    )


def _norm(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if batch_invariant_enabled(module) and isinstance(module, nn.LayerNorm):
        return layer_norm_fp32(
            x, module.normalized_shape, module.weight, module.bias, module.eps
        )
    return module(x)


def _softmax(owner: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return (
        softmax_fp32_sum(x, dim=-1)
        if batch_invariant_enabled(owner)
        else torch.softmax(x, dim=-1)
    )


class CausalConv2dSub(nn.Conv2d):
    """CausalConv2D: (k-1, s-1) asymmetric zero pad on BOTH axes."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, padding=0, **kwargs)
        k = self.kernel_size[0]
        s = self.stride[0]
        self._pad = (k - 1, s - 1, k - 1, s - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if batch_invariant_enabled(self):
            return _conv(self, nn.functional.pad(x, self._pad))
        return super().forward(nn.functional.pad(x, self._pad))


class SubsamplingDwStriding(nn.Module):
    """Causal dw-striding ConvSubsampling, 8x (three stride-2 stages)."""

    def __init__(
        self,
        *,
        feat_in: int,
        d_model: int,
        conv_channels: int,
        kernel: int = 3,
        stride: int = 2,
        stages: int = 3,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.stages = stages
        layers: list[nn.Module] = [
            CausalConv2dSub(1, conv_channels, kernel, stride),
            nn.ReLU(),
        ]
        for _ in range(stages - 1):
            layers.append(
                CausalConv2dSub(
                    conv_channels,
                    conv_channels,
                    kernel,
                    stride,
                    groups=conv_channels,
                )
            )
            layers.append(nn.Conv2d(conv_channels, conv_channels, 1))
            layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)
        freq = feat_in
        pad = (kernel - 1) + (stride - 1)
        for _ in range(stages):
            freq = _conv_out_len(freq, pad=pad, kernel=kernel, stride=stride)
        self.out = nn.Linear(conv_channels * freq, d_model)

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel - 1) + (self.stride - 1)
        out = lengths
        for _ in range(self.stages):
            out = (
                torch.div(out + pad - self.kernel, self.stride, rounding_mode="floor")
                + 1
            )
        return out.to(torch.int64)

    def forward(
        self,
        mel: torch.Tensor,
        lengths: torch.Tensor,
        *,
        mask_padding: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, feat_in, T)`` mel -> ``(B, T//8, d_model)``.

        ``mask_padding`` zeroes every time step at or past a row's valid
        length before each strided conv, with the length advanced by the
        same formula as :meth:`output_lengths` — the masked subsampling of
        NeMo's ``MaskedConvSequential`` (``nemo/collections/asr/parts/
        submodules/subsampling.py``). The right pad of each causal conv
        makes the last valid output of a layer read one step past an even
        valid length; masking gives that step the zero an exact-width
        input would have. Pointwise layers need no mask of their own: the
        next strided conv's mask covers their padded outputs. A row whose
        lengths reach every layer's full width selects ``x`` unchanged.
        """
        # Activations follow the policy-cast weights from this seam on;
        # the fp32 mel front-end sits upstream of it (PORT-PREC-001).
        mel = mel.to(next(self.parameters()).dtype)
        x = mel.transpose(1, 2).unsqueeze(1)  # (B, 1, T, F)
        valid = lengths.to(torch.int64)
        pad = (self.kernel - 1) + (self.stride - 1)
        invariant = batch_invariant_enabled(self)
        for module in self.conv:
            if mask_padding and isinstance(module, CausalConv2dSub):
                step = torch.arange(x.shape[2], device=x.device).view(1, 1, -1, 1)
                x = torch.where(step < valid.view(-1, 1, 1, 1), x, x.new_zeros(()))
                valid = (
                    torch.div(
                        valid + pad - self.kernel, self.stride, rounding_mode="floor"
                    )
                    + 1
                )
            if (
                invariant
                and isinstance(module, nn.Conv2d)
                and not isinstance(module, CausalConv2dSub)
            ):
                x = _conv(module, x)
            else:
                x = module(x)
        b, c, t, f = x.size()
        x = x.transpose(1, 2).reshape(b, t, c * f)
        if batch_invariant_enabled(self):
            # Keep the biased projection out of Inductor's addmm padding pass.
            x = invariant_gemm(
                x.reshape(b * t, c * f), self.out.weight.t(), self.out.bias
            ).reshape(b, t, -1)
        else:
            x = self.out(x)
        return x, self.output_lengths(lengths)


class RelPositionalEncoding(nn.Module):
    """Centered TXL relative positions L-1 .. -(L-1); no input scale."""

    pe: torch.Tensor

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.register_buffer("pe", torch.zeros(1, 0, d_model), persistent=False)

    def _extend(self, length: int, ref: torch.Tensor) -> None:
        if self.pe.size(1) >= 2 * length - 1:
            return
        positions = torch.arange(
            length - 1, -length, -1, dtype=torch.float32, device=ref.device
        ).unsqueeze(1)
        pe = torch.zeros(positions.size(0), self.d_model, device=ref.device)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=ref.device)
            * -(math.log(_LOG_BASE) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.pe = pe.unsqueeze(0).to(ref.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return pos_emb ``(1, 2T-1, d)`` for input ``(B, T, d)``."""
        self._extend(x.size(1), x)
        length = x.size(1)
        center = self.pe.size(1) // 2 + 1
        return self.pe[:, center - length : center + length - 1]


class RelPositionMHA(nn.Module):
    """Transformer-XL rel-pos attention (per-layer pos biases)."""

    def __init__(self, *, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.s_d_k = math.sqrt(self.d_k)
        self.linear_q = nn.Linear(d_model, d_model, bias=False)
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model, bias=False)
        self.linear_out = nn.Linear(d_model, d_model, bias=False)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(n_heads, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(n_heads, self.d_k))

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        b, h, qlen, pos_len = x.size()
        x = nn.functional.pad(x, pad=(1, 0))
        x = x.view(b, h, -1, qlen)
        return x[:, :, 1:].view(b, h, qlen, pos_len)

    def forward(
        self,
        x: torch.Tensor,
        *,
        pos_emb: torch.Tensor,
        masked: torch.Tensor,
    ) -> torch.Tensor:
        """Self-attention; ``masked`` is (B, T, T) True = MAY NOT attend."""
        b, t, _ = x.shape
        q = invariant_linear(self.linear_q, x).view(b, t, self.h, self.d_k)
        k = (
            invariant_linear(self.linear_k, x)
            .view(b, t, self.h, self.d_k)
            .transpose(1, 2)
        )
        v = (
            invariant_linear(self.linear_v, x)
            .view(b, t, self.h, self.d_k)
            .transpose(1, 2)
        )
        p = (
            invariant_linear(self.linear_pos, pos_emb)
            .view(pos_emb.size(0), -1, self.h, self.d_k)
            .transpose(1, 2)
        )

        q_u = (q + self.pos_bias_u).transpose(1, 2)
        q_v = (q + self.pos_bias_v).transpose(1, 2)
        matrix_bd = self._rel_shift(invariant_matmul(self, q_v, p.transpose(-2, -1)))
        matrix_ac = invariant_matmul(self, q_u, k.transpose(-2, -1))
        scores = (matrix_ac + matrix_bd[:, :, :, : matrix_ac.size(-1)]) / self.s_d_k
        mask = masked.unsqueeze(1)
        scores = scores.masked_fill(mask, -_LOG_BASE)
        attn = _softmax(self, scores).masked_fill(mask, 0.0)
        out = invariant_matmul(self, attn, v)
        out = out.transpose(1, 2).reshape(b, t, self.h * self.d_k)
        return invariant_linear(self.linear_out, out)


class ConformerConv(nn.Module):
    """Pointwise->GLU->causal depthwise->norm->silu->pointwise."""

    def __init__(self, *, d_model: int, kernel: int, norm_type: str) -> None:
        super().__init__()
        self.norm_type = norm_type
        self.left_pad = kernel - 1
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, 1, bias=False)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel, groups=d_model, bias=False
        )
        if norm_type == "layer_norm":
            self.batch_norm: nn.Module = nn.LayerNorm(d_model)
        elif norm_type == "batch_norm":
            self.batch_norm = nn.BatchNorm1d(d_model)
        else:
            raise ValueError(f"unsupported conv norm: {norm_type}")
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, 1, bias=False)

    def forward(self, x: torch.Tensor, pad_zero: torch.Tensor | None) -> torch.Tensor:
        """``(B, T, d)`` -> ``(B, T, d)``; ``pad_zero`` True = padding."""
        x = x.transpose(1, 2)
        x = nn.functional.glu(_conv(self.pointwise_conv1, x), dim=1)
        if pad_zero is not None:
            x = x.masked_fill(pad_zero.unsqueeze(1), 0.0)
        x = _conv(self.depthwise_conv, nn.functional.pad(x, (self.left_pad, 0)))
        if self.norm_type == "layer_norm":
            x = _norm(self.batch_norm, x.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.batch_norm(x)
        x = nn.functional.silu(x)
        return _conv(self.pointwise_conv2, x).transpose(1, 2)


class FeedForward(nn.Module):
    """Linear -> silu -> Linear (dropout is eval-noop, omitted).

    Bias-free: the checkpoint's encoder ``use_bias: False`` covers every
    Linear/Conv in the layer (confirmed from the restored model config).
    """

    def __init__(self, *, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return invariant_linear(
            self.linear2, nn.functional.silu(invariant_linear(self.linear1, x))
        )


class ConformerLayer(nn.Module):
    """Macaron block: 0.5FF -> MHA -> conv -> 0.5FF -> norm_out."""

    def __init__(
        self,
        *,
        d_model: int,
        d_ff: int,
        n_heads: int,
        conv_kernel: int,
        conv_norm_type: str,
    ) -> None:
        super().__init__()
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = FeedForward(d_model=d_model, d_ff=d_ff)
        self.norm_self_att = nn.LayerNorm(d_model)
        self.self_attn = RelPositionMHA(d_model=d_model, n_heads=n_heads)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConv(
            d_model=d_model, kernel=conv_kernel, norm_type=conv_norm_type
        )
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = FeedForward(d_model=d_model, d_ff=d_ff)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        *,
        pos_emb: torch.Tensor,
        masked: torch.Tensor,
        pad_zero: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x + 0.5 * self.feed_forward1(_norm(self.norm_feed_forward1, x))
        x = x + self.self_attn(
            _norm(self.norm_self_att, x), pos_emb=pos_emb, masked=masked
        )
        x = x + self.conv(_norm(self.norm_conv, x), pad_zero)
        x = x + 0.5 * self.feed_forward2(_norm(self.norm_feed_forward2, x))
        return _norm(self.norm_out, x)


class FastConformerEncoder(nn.Module):
    """FastConformer encoder (PORT-REGIME-001).

    Constructed here in the full-context probe/bring-up configuration:
    one attention window over the whole utterance at the configured
    ``att_context_size``, cross-chunk state pages dormant. The P3
    streaming path re-plumbs the same layers over paged caches.
    """

    def __init__(
        self,
        *,
        feat_in: int = 128,
        d_model: int = 1024,
        d_ff: int = 4096,
        n_layers: int = 24,
        n_heads: int = 8,
        conv_kernel: int = 9,
        conv_norm_type: str = "layer_norm",
        subsampling_channels: int = 256,
        att_context: tuple[int, int] = (56, 13),
    ) -> None:
        super().__init__()
        self.att_context = att_context
        self.pre_encode = SubsamplingDwStriding(
            feat_in=feat_in,
            d_model=d_model,
            conv_channels=subsampling_channels,
        )
        self.pos_enc = RelPositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            ConformerLayer(
                d_model=d_model,
                d_ff=d_ff,
                n_heads=n_heads,
                conv_kernel=conv_kernel,
                conv_norm_type=conv_norm_type,
            )
            for _ in range(n_layers)
        )
        # Streaming projections are derived from frozen ``linear_pos``
        # weights during startup-owned execution materialization. Their
        # lengths are transition windows (new output width + left cache),
        # not session state; never add or replace them from stream_step.
        self._stream_relative_position_lengths: tuple[int, ...] = ()

    @staticmethod
    def _stream_relative_position_name(length: int) -> str:
        return f"_stream_relative_position_{length}"

    def invalidate_stream_relative_position_projections(self) -> None:
        """Drop derived serving projections before a model mutation."""
        for layer in self.layers:
            attn = layer.self_attn
            for length in self._stream_relative_position_lengths:
                name = self._stream_relative_position_name(length)
                if name in attn._buffers:
                    delattr(attn, name)
        self._stream_relative_position_lengths = ()

    def _apply(self, fn, recurse: bool = True):
        # A device or dtype transition must rebuild from the original full
        # precision weights; casting a derived projection changes its origin.
        self.invalidate_stream_relative_position_projections()
        return super()._apply(fn, recurse=recurse)

    def train(self, mode: bool = True) -> "FastConformerEncoder":
        if mode:
            self.invalidate_stream_relative_position_projections()
        return super().train(mode)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        self.invalidate_stream_relative_position_projections()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def prepare_stream_relative_position_projections(
        self,
        *,
        out_widths: tuple[int, ...],
        cache_len: int,
        reference: torch.Tensor,
    ) -> None:
        """Precompute every declared exact relative-position projection.

        Each layer owns a nonpersistent buffer for each complete transition
        window.  The source is the original centered positional slice from
        the full positional table, passed through that layer's original
        ``linear_pos``; it is never a crop of another projected window.
        Each buffer is stored head-major and contiguous, ``(h, 2T-1, d_k)``,
        so the batch-independent projection is never copied per batch row.
        """
        if self.training or torch.is_grad_enabled():
            raise ValueError(
                "stream relative-position projections require frozen inference"
            )
        if cache_len < 0 or not out_widths or any(width <= 0 for width in out_widths):
            raise ValueError("stream relative-position projection geometry is invalid")
        lengths = tuple(sorted({int(width) + cache_len for width in out_widths}))
        if self._stream_relative_position_lengths:
            if self._stream_relative_position_lengths != lengths:
                raise ValueError(
                    "stream relative-position projection geometry changed after "
                    "materialization"
                )
            return
        self.pos_enc._extend(max(lengths), reference)
        center = self.pos_enc.pe.size(1) // 2 + 1
        prepared: list[tuple[RelPositionMHA, str, torch.Tensor]] = []
        for layer in self.layers:
            attn = layer.self_attn
            weight = attn.linear_pos.weight
            if weight.device != reference.device or weight.dtype != reference.dtype:
                raise ValueError(
                    "stream relative-position projection weight metadata differs from "
                    "execution"
                )
            for length in lengths:
                pos_emb = self.pos_enc.pe[:, center - length : center + length - 1]
                projection = _head_major_position_projection(attn, pos_emb)
                prepared.append(
                    (attn, self._stream_relative_position_name(length), projection)
                )
        for attn, name, projection in prepared:
            attn.register_buffer(name, projection, persistent=False)
        self._stream_relative_position_lengths = lengths

    def stream_relative_position_projections(
        self,
        *,
        out_width: int,
        cache_len: int,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, ...] | None:
        """Return immutable startup projections or select the eager oracle."""
        if self.training or torch.is_grad_enabled():
            return None
        length = int(out_width) + cache_len
        if length not in self._stream_relative_position_lengths:
            return None
        name = self._stream_relative_position_name(length)
        projections = tuple(
            getattr(layer.self_attn, name, None) for layer in self.layers
        )
        if any(
            not isinstance(projection, torch.Tensor)
            or projection.device != reference.device
            or projection.dtype != reference.dtype
            for projection in projections
        ):
            return None
        return projections

    def forward(
        self, mel: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, feat_in, T_mel)`` -> ``(B, T, d_model)`` encoder_raw.

        Output is (B, T, D) — NeMo transposes to (B, D, T) at its
        encoder boundary; the port keeps time-major throughout (layout,
        not math; the capture-hook comparison transposes accordingly).
        """
        x, out_lens = self.pre_encode(mel, lengths)
        pos_emb = self.pos_enc(x)
        t = x.size(1)
        may_attend = chunked_limited_mask(t, self.att_context, device=x.device)
        valid = torch.arange(t, device=x.device).unsqueeze(0) < out_lens.unsqueeze(1)
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
        masked = ~(may_attend.unsqueeze(0) & pair_valid)
        pad_zero = ~valid
        for layer in self.layers:
            x = layer(x, pos_emb=pos_emb, masked=masked, pad_zero=pad_zero)
        return x, out_lens


class StreamingCaches:
    """Per-session encoder caches (the spec pages' in-module form).

    ``channel``: per-layer normed attention inputs, ``(L, B, 56, d)``
    (NeMo ``cache_last_channel``). ``time``: per-layer conv tails,
    ``(L, B, d, kernel-1)`` (``cache_last_time``). ``valid``: filled
    cache rows per batch element. The 56-slot capacity IS the attention
    window (8 chunks x 7 frames at every published config), so the
    streaming mask reduces to 'all valid cached rows + full intra-chunk'.
    """

    def __init__(
        self,
        *,
        n_layers: int,
        batch: int,
        d_model: int,
        left_context: int,
        conv_kernel: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.left_context = left_context
        self.channel = torch.zeros(
            n_layers, batch, left_context, d_model, device=device, dtype=dtype
        )
        self.time = torch.zeros(
            n_layers,
            batch,
            d_model,
            conv_kernel - 1,
            device=device,
            dtype=dtype,
        )
        self.valid = torch.zeros(batch, dtype=torch.long, device=device)


def _stream_attention_mask(
    valid: torch.Tensor,
    new_valid: torch.Tensor,
    capacity: int,
) -> torch.Tensor:
    """Mask dead cached keys, padded new keys, and padded queries."""
    batch, new_frames = new_valid.shape
    row = torch.arange(capacity, device=new_valid.device).unsqueeze(0)
    dead = row < (capacity - valid.unsqueeze(1))
    mask = torch.zeros(
        batch,
        1,
        new_frames,
        capacity + new_frames,
        dtype=torch.bool,
        device=new_valid.device,
    )
    mask[:, :, :, :capacity] = dead.unsqueeze(1).unsqueeze(2)
    mask[:, :, :, capacity:] = (~new_valid).unsqueeze(1).unsqueeze(2)
    return mask | (~new_valid).unsqueeze(1).unsqueeze(-1)


def _gather_into(
    source: torch.Tensor, dim: int, index: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """Gather ``source`` along ``dim`` into the storage of ``out``.

    Eager writes directly with ``out=`` (no intermediate). While tracing,
    the ``out=`` meta check cannot run on symbolic batch sizes, so the
    functional gather is copied into ``out`` and Inductor fuses the write.
    """
    if torch.compiler.is_compiling():
        return out.copy_(source.gather(dim, index))
    return torch.gather(source, dim, index, out=out)


def _sdpa_bias_width(keys: int) -> int:
    """Bias row width: ``keys`` rounded up to the SDPA alignment when the
    shifted position-score view can supply it (``<= 2T-2`` columns)."""
    aligned = -(-keys // _SDPA_BIAS_ALIGNMENT) * _SDPA_BIAS_ALIGNMENT
    return aligned if keys > 1 and aligned <= 2 * keys - 2 else keys


class StreamSDPAMask:
    """Per-transition read-only inputs of the SDPA streaming attention.

    Built once per encoder transition from the boolean ``(B, 1, F, T)``
    mask and shared by every layer: ``additive`` is ``(B, 1, F, W)`` with
    0 for attendable and ``-_SDPA_MASK_VALUE`` for masked keys, where
    ``W = _sdpa_bias_width(T)``; columns ``>= T`` are alignment padding
    that SDPA never reads. ``dead_queries`` is ``(B, 1, F, 1)`` True where
    a query row is fully masked.
    """

    __slots__ = ("additive", "dead_queries")

    def __init__(self, mask: torch.Tensor, *, dtype: torch.dtype) -> None:
        batch, _, queries, keys = mask.shape
        width = _sdpa_bias_width(keys)
        additive = torch.zeros(
            batch, 1, queries, width, dtype=dtype, device=mask.device
        )
        additive[..., :keys].masked_fill_(mask, -_SDPA_MASK_VALUE)
        self.additive = additive
        self.dead_queries = mask.all(dim=-1, keepdim=True)


def _stream_cache_indices(new_lengths: torch.Tensor, capacity: int) -> torch.Tensor:
    """Select each row's retained history after its logical append."""
    return new_lengths.view(-1, 1) + torch.arange(
        capacity, device=new_lengths.device
    ).unsqueeze(0)


def _head_major_position_projection(
    attn: RelPositionMHA, pos_emb: torch.Tensor
) -> torch.Tensor:
    """Project ``(1, 2T-1, d)`` positions to contiguous ``(h, 2T-1, d_k)``."""
    if pos_emb.size(0) != 1:
        raise ValueError("stream positional embeddings are batch-independent")
    return (
        invariant_linear(attn.linear_pos, pos_emb)
        .view(-1, attn.h, attn.d_k)
        .transpose(0, 1)
        .contiguous()
    )


def _stream_rel_shift(
    raw: torch.Tensor,
    *,
    batch: int,
    queries: int,
    keys: int,
    width: int | None = None,
) -> torch.Tensor:
    """Relative shift of head-major position scores, as a strided view.

    ``raw`` is the contiguous ``(h, B*F, 2T-1)`` product. For query ``i``
    and key ``j < T`` the Transformer-XL shift (``RelPositionMHA._rel_shift``
    followed by the ``[..., :T]`` crop) selects ``raw[..., i, F-1-i+j]``:
    a pure index map, so the result is a ``(B, h, F, T)`` view with no
    padding copy and values identical to the shift-and-crop. Within each
    (head, row) block that element sits at flat offset
    ``F-1 + i*(2T-2) + j``, so the view drops ``F-1`` leading elements and
    re-reads the block with row pitch ``2T-2``.

    ``width`` (``keys <= width <= 2T-2``) widens the view past the crop;
    columns ``>= keys`` are in-bounds but are not position scores, so a
    caller must exclude them (the SDPA bias uses them only as alignment
    padding).
    """
    heads, rows, pos_len = raw.shape
    width = keys if width is None else width
    if rows != batch * queries or pos_len != 2 * keys - 1 or queries > keys:
        raise ValueError("stream position scores have an unexpected geometry")
    if width != keys and not keys < width <= pos_len - 1:
        raise ValueError("stream position score width is out of range")
    blocks = raw.view(heads, batch, queries * pos_len)
    if keys == 1:
        shifted = blocks.view(heads, batch, queries, 1)
    else:
        pitch = pos_len - 1
        shifted = blocks[..., queries - 1 : queries - 1 + queries * pitch].view(
            heads, batch, queries, pitch
        )
        shifted = shifted[..., :width]
    return shifted.transpose(0, 1)


def _stream_attention(
    layer: ConformerLayer,
    x: torch.Tensor,
    *,
    cache: torch.Tensor,
    valid: torch.Tensor,
    pos_emb: torch.Tensor | None,
    new_valid: torch.Tensor,
    new_lengths: torch.Tensor,
    projected_pos: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    cache_indices: torch.Tensor | None = None,
    cache_out: torch.Tensor | None = None,
    sdpa: StreamSDPAMask | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One layer's attention over [cache | new] keys (NeMo update_cache).

    ``x`` is the normed attention input for the NEW frames (B, F, d);
    ``cache`` holds the previous normed inputs (B, C, d). ``new_valid``
    is the (B, F) per-row frame-validity mask and ``new_lengths`` the
    (B,) logical frame counts: padded new frames are masked as keys AND
    queries (a fully masked query row softmaxes uniform then zeroes,
    so its output is exactly 0 — never NaN), and the cache advances by
    each row's LOGICAL length via a per-row gather over [cache | new].
    Returns the attention output for the new frames and the advanced
    cache. With ``cache_out`` (which may be ``cache`` itself) the advance
    is gathered straight into that storage and it is returned.

    With ``sdpa`` the score/softmax/value chain is one fused
    ``scaled_dot_product_attention`` over ``q_u, k, v`` with the scaled,
    additively masked position scores as its float bias, then fully
    masked query rows are zeroed (NeMo's public formulation,
    NVIDIA-NeMo/Speech@981fd4e0 multi_head_attention.py:320-345). It is
    the same math with a different reduction order, so it is selected
    only by the encoder FP16 precision policy, never for FP32.
    """
    attn = layer.self_attn
    batch, new_frames, _ = x.shape
    capacity = cache.shape[1]
    # The cache keeps its own policy axis (attention_cache); compute
    # runs in the activations dtype, so read-cast here, write-cast on
    # advance (PORT-PREC-001/005 — state dtype never follows compute).
    keys = torch.cat([cache.to(x.dtype), x], dim=1)  # (B, C+F, d)

    b, t2 = batch, keys.shape[1]
    q = invariant_linear(attn.linear_q, x).view(b, new_frames, attn.h, attn.d_k)
    k = (
        invariant_linear(attn.linear_k, keys)
        .view(b, t2, attn.h, attn.d_k)
        .transpose(1, 2)
    )
    v = (
        invariant_linear(attn.linear_v, keys)
        .view(b, t2, attn.h, attn.d_k)
        .transpose(1, 2)
    )
    if projected_pos is None:
        if pos_emb is None:
            raise ValueError(
                "stream attention requires positional embeddings or a prepared "
                "projection"
            )
        p = _head_major_position_projection(attn, pos_emb)
    else:
        p = projected_pos
    # Content scores: the bias add writes q_u head-major directly, so the
    # batched matmul reads it without a separate layout copy.
    q_u = q.new_empty((b, attn.h, new_frames, attn.d_k))
    torch.add(q.transpose(1, 2), attn.pos_bias_u.unsqueeze(1), out=q_u)
    # Position scores: p is shared by every row, so fold the batch into
    # the GEMM rows -- one head-batched (B*F, d_k) @ (d_k, 2T-1) product.
    # Batch-invariant mode keeps this shape (PORT-PREC-027): its kernel
    # reduces each output row in an order independent of the row count.
    q_v = (q + attn.pos_bias_v).view(b * new_frames, attn.h, attn.d_k).transpose(0, 1)
    matrix_bd = _stream_rel_shift(
        invariant_matmul(attn, q_v, p.transpose(-2, -1)),
        batch=b,
        queries=new_frames,
        keys=t2,
        width=None if sdpa is None else sdpa.additive.shape[-1],
    )
    # Mask: cache rows beyond each element's valid count are dead;
    # padded new frames are dead keys; padded queries mask fully.
    if mask is None and sdpa is None:
        mask = _stream_attention_mask(valid, new_valid, capacity)
    if sdpa is not None:
        # bias = mask + bd / sqrt(d_k) in one kernel (SDPA applies the same
        # 1/sqrt(d_k) to q_u.k^T). It is computed over the aligned width and
        # cropped, so its rows keep a 16-element pitch and SDPA needs no
        # padding copy; the padding columns are never read. Batch-invariant
        # mode does not toggle this lane (PORT-PREC-027); with the mode on
        # it stays unqualified until PORT-PREC-023 passes with it selected.
        bias = torch.add(sdpa.additive, matrix_bd, alpha=1.0 / attn.s_d_k)[..., :t2]
        out = torch.nn.functional.scaled_dot_product_attention(
            q_u, k, v, attn_mask=bias
        )
        # A fully masked query row is exactly zero (never NaN/uniform mix).
        out = out.masked_fill(sdpa.dead_queries, 0.0)
    else:
        assert mask is not None
        matrix_ac = invariant_matmul(attn, q_u, k.transpose(-2, -1))
        scores = (matrix_ac + matrix_bd) / attn.s_d_k
        scores = scores.masked_fill(mask, -_LOG_BASE)
        weights = _softmax(attn, scores).masked_fill(mask, 0.0)
        out = invariant_matmul(attn, weights, v)
    out = out.transpose(1, 2).reshape(batch, new_frames, attn.h * attn.d_k)
    # Advance cache by each row's logical length: slot j of the new
    # cache is [cache | x][j + F_b] — F_b = 0 leaves the row's cache
    # bit-identical; gather indices never touch padded frames.
    if cache_indices is None:
        cache_indices = (
            _stream_cache_indices(new_lengths, capacity)
            .unsqueeze(-1)
            .expand(b, capacity, keys.shape[2])
        )
    # The gather source is a fresh concatenation, so writing the advance
    # into ``cache_out`` (even when it is ``cache``) cannot alias a read:
    # every read of ``cache`` (``keys`` and this source) precedes it.
    source = torch.cat([cache, x.to(cache.dtype)], dim=1)
    if cache_out is None:
        new_cache = source.gather(1, cache_indices)
    else:
        new_cache = _gather_into(source, 1, cache_indices, cache_out)
    return invariant_linear(attn.linear_out, out), new_cache


def _stream_conv(
    layer: ConformerLayer,
    x: torch.Tensor,
    cache: torch.Tensor,
    *,
    new_lengths: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    preserve_cache_precision: bool = False,
    cache_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Conv module over [time_cache | new] (CausalConv1D.update_cache).

    The causal depthwise conv makes every valid output independent of
    padded columns (and the norm is per-position), so only the cache
    tail needs length awareness: slot j of the new tail gathers
    [cache | glu][j + F_b], each row's own logical append. ``cache_out``
    (which may be ``cache`` itself) receives the advance in place.
    """
    conv = layer.conv
    batch, frames, width = x.shape
    y = x.permute(2, 0, 1).contiguous().view(1, width, batch * frames)
    y = _conv(conv.pointwise_conv1, y)
    y = (
        y.view(conv.pointwise_conv1.out_channels, batch, frames)
        .permute(1, 0, 2)
        .contiguous()
    )
    y = torch.nn.functional.glu(y, dim=1)
    # conv_state axis: read-cast to compute dtype, write-cast back.
    padded = torch.cat([cache.to(y.dtype), y], dim=-1)
    cw = cache.shape[-1]
    if cache_indices is None:
        cache_indices = (
            _stream_cache_indices(new_lengths, cw)
            .unsqueeze(1)
            .expand(y.shape[0], y.shape[1], cw)
        )
    # Both gather sources are fresh concatenations (no alias with
    # ``cache_out``), and every read of ``cache`` precedes the write.
    if preserve_cache_precision:
        # Keep retained FP32 history exact in the encoder-only experiment,
        # including zero-length rows. Only newly computed values are widened.
        source = torch.cat([cache, y.to(cache.dtype)], dim=-1)
        if cache_out is None:
            new_cache = source.gather(2, cache_indices)
        else:
            new_cache = _gather_into(source, 2, cache_indices, cache_out)
    elif cache_out is None:
        new_cache = padded.gather(2, cache_indices).to(cache.dtype)
    elif padded.dtype == cache_out.dtype:
        new_cache = _gather_into(padded, 2, cache_indices, cache_out)
    else:
        # The write-cast happens in the copy (same rounding as ``.to``).
        new_cache = cache_out.copy_(padded.gather(2, cache_indices))
    y = _conv(conv.depthwise_conv, padded)
    y = _norm(conv.batch_norm, y.transpose(1, 2)).transpose(1, 2)
    y = torch.nn.functional.silu(y)
    return _conv(conv.pointwise_conv2, y).transpose(1, 2), new_cache


def stream_step(
    encoder: FastConformerEncoder,
    chunk_mel: torch.Tensor,
    caches: StreamingCaches,
    *,
    drop_extra: int | None = None,
    out_offsets: torch.Tensor | None = None,
    out_lengths: torch.Tensor | None = None,
    out_width: int | None = None,
    preserve_conv_cache_precision: bool = False,
    sdpa_attention: bool = False,
    mel_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """One cached streaming encoder step (batch of sessions).

    ``chunk_mel``: (B, feat, mel [+9-mel pre-encode context for
    non-first chunks]). Advances the caches in place — numerically the
    cached form of the prefix computation the P3 probes proved
    golden-exact.

    Length-aware form (PORT-ADV-004): ``out_offsets`` (B,) is each
    row's pre-encode drop, ``out_lengths`` (B,) its logical encoder
    frame count, and ``out_width`` the fixed host-derived output
    width. ``mel_lengths`` (B,) is each row's valid input mel width
    (pre-encode prefix included); the subsampling masks every step past
    it (see :meth:`SubsamplingDwStriding.forward`), so a partial final
    chunk encodes exactly like an exact-width one. ``None`` means every
    row spans the full input width and skips the mask. Rows are
    realigned per-row after subsampling, attention masks padded keys
    AND queries, caches append by logical
    lengths, ``window_valid`` accumulates logical lengths, and output
    frames at or past a row's length are exactly zero. A zero-length
    row leaves its caches bit-identical. No tensor value determines a
    shape; the call issues no host/device synchronization.

    ``drop_extra`` is the uniform adapter used by the P3/P4 parity
    probes: equivalent to offsets = ``drop_extra``, lengths = full
    width, over the same single algorithm.

    ``preserve_conv_cache_precision`` is enabled by the encoder-only
    FP16 experiment so retained FP32 history avoids a compute-dtype round trip.

    ``sdpa_attention`` selects the fused SDPA attention core (see
    ``_stream_attention``). It is resolved from the immutable encoder FP16
    precision policy; the FP32 path keeps the bitwise eager oracle.
    """
    b = chunk_mel.shape[0]
    device = chunk_mel.device
    if mel_lengths is None:
        lengths = torch.full((b,), chunk_mel.shape[2], device=device)
        x, _ = encoder.pre_encode(chunk_mel, lengths)
    else:
        x, _ = encoder.pre_encode(chunk_mel, mel_lengths, mask_padding=True)
    if drop_extra is not None:
        if out_offsets is not None or out_lengths is not None:
            raise ValueError(
                "pass either drop_extra (uniform legacy) or the per-row "
                "out_offsets/out_lengths/out_width form"
            )
        out_width = x.shape[1] - drop_extra
        out_offsets = torch.full((b,), drop_extra, dtype=torch.long, device=device)
        out_lengths = torch.full((b,), out_width, dtype=torch.long, device=device)
    assert out_offsets is not None and out_lengths is not None and out_width is not None
    # Per-row realignment: row b's encoder frames start at its own
    # pre-encode drop. Clamp keeps padded columns in-bounds; their
    # values are masked everywhere below.
    gidx = (
        out_offsets.view(-1, 1) + torch.arange(out_width, device=device).unsqueeze(0)
    ).clamp(max=max(x.shape[1] - 1, 0))
    x = x.gather(1, gidx.unsqueeze(-1).expand(b, out_width, x.shape[2]))
    new_valid = torch.arange(out_width, device=device).unsqueeze(0) < out_lengths.view(
        -1, 1
    )
    cache_len = caches.channel.shape[2]
    # All layers share the same logical lengths and cache geometry. Build
    # their read-only masks and gather indices once for this transition.
    mask = _stream_attention_mask(caches.valid, new_valid, cache_len)
    sdpa = StreamSDPAMask(mask, dtype=x.dtype) if sdpa_attention else None
    attn_indices = (
        _stream_cache_indices(out_lengths, cache_len)
        .unsqueeze(-1)
        .expand(b, cache_len, x.shape[2])
    )
    conv_len = caches.time.shape[-1]
    conv_indices = (
        _stream_cache_indices(out_lengths, conv_len)
        .unsqueeze(1)
        .expand(b, x.shape[2], conv_len)
    )
    projections = encoder.stream_relative_position_projections(
        out_width=out_width,
        cache_len=cache_len,
        reference=x,
    )
    pos_emb = None
    if projections is None:
        pos_emb = encoder.pos_enc(
            torch.zeros(
                1,
                out_width + cache_len,
                x.shape[2],
                device=device,
                dtype=x.dtype,
            )
        )
    for idx, layer in enumerate(encoder.layers):
        residual = x
        y = _norm(layer.norm_feed_forward1, x)
        residual = residual + 0.5 * layer.feed_forward1(y)
        y = _norm(layer.norm_self_att, residual)
        # Each advance gathers straight into its layer's cache storage
        # (static addresses under graph capture; no copy-back).
        channel = caches.channel[idx]
        attn_out, _ = _stream_attention(
            layer,
            y,
            cache=channel,
            valid=caches.valid,
            pos_emb=pos_emb,
            new_valid=new_valid,
            new_lengths=out_lengths,
            projected_pos=None if projections is None else projections[idx],
            mask=mask,
            cache_indices=attn_indices,
            cache_out=channel,
            sdpa=sdpa,
        )
        residual = residual + attn_out
        y = _norm(layer.norm_conv, residual)
        time = caches.time[idx]
        conv_out, _ = _stream_conv(
            layer,
            y,
            time,
            new_lengths=out_lengths,
            cache_indices=conv_indices,
            preserve_cache_precision=preserve_conv_cache_precision,
            cache_out=time,
        )
        residual = residual + conv_out
        y = _norm(layer.norm_feed_forward2, residual)
        residual = residual + 0.5 * layer.feed_forward2(y)
        x = _norm(layer.norm_out, residual)
    caches.valid = torch.clamp(
        caches.valid + out_lengths.to(caches.valid.dtype),
        max=caches.left_context,
    )
    # Padded-output zeroing (PORT-ADV-004): frames at or past each
    # row's logical length are exactly zero.
    return torch.where(new_valid.unsqueeze(-1), x, x.new_zeros(()))
