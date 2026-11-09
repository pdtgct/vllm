# Nemotron ASR Whole-Transaction CUDA Graphs

**Status:** opt-in; default-on is gated below. **Acceptance criteria:** the
numbered `TXG-*` criteria below.

## Scope and Provenance

This is the governing design for the `transaction_graph_populations` option of
the Nemotron streaming ASR model (`vllm/models/nemotron_asr/`). It lives beside
the model and moves with it when the model becomes an out-of-tree plugin.
The model's tests (`tests/models/nemotron_asr/`,
`tests/v1/engine/test_nemotron_asr.py`) move with it; the tests of core
behaviour this design depends on (`tests/v1/worker/test_mixed_warmup_gate.py`,
`tests/v1/core/test_kv_cache_planning.py` and the runner tests in
`tests/v1/worker/test_gpu_model_runner_v2.py`) stay in core. Numbered
acceptance criteria take the place of EARS specifications. For project
context only (not a governing parent), see the section "vLLM core
effort and the streaming plugin" of `docs/high-level-design.md` in
`pdtgct/vllm-omni-notes`.

Geometries are named by encoder lookahead: **g0** is the 80 ms cadence
(lookahead 0, `CADENCES` index 0 in `manifests.py`) and **g1** is the 160 ms
cadence (lookahead 1), which is the model's default.

"Endpoint", `endpoint_enabled` and "forced endpoint" refer to utterance
endpointing, a model capability, not to endpoint plugins (HTTP routes). The
model's endpointing is independent of whether any endpoint plugin is loaded.

## Context and Design Philosophy

A Nemotron streaming step is one model-defined transaction over resident state
slots. `advance_model_rows` (`advance.py:2743`) validates the row plan on the
host, gathers the state of continuing rows, runs the frontend, encoder and
RNN-T decode for each CHUNK bucket, projects emissions, and commits through
predicated page scatters followed by one commit-sink stage. The encoder,
decoder and CHUNK-bucket regions already replay their own graphs. The step
still pays for per-op ATen dispatch, kernel launches and caching-allocator work
between those regions. This option replays the whole device body and its
scatters as one CUDA graph per declared *cell*. A covered step then costs one
host-to-device upload, one graph launch and the commit stage.

### Tenets, in priority order

1. **Exact semantics before replay coverage.** A step either replays a graph
   whose cell and signature match it, or runs the eager transaction, which is
   the reference. Coverage gaps are counted. They are closed by declaring or
   validating more cells, never by loosening the match.
2. **Resident capacity before per-step latency.** Graph memory is admitted into
   the device budget before resident capacity is fixed, so the trade of slots
   for latency is visible. Resident slots alone do not establish sustainable
   concurrency: step latency at the served population must also fit the
   cadence.

Bitwise equality of outputs and resident state with the eager transaction is
an acceptance requirement for every covered cell. It is not a tenet.

### Exploratory measurements

Pre-registered exploratory runs on an A40 at g0 (core `50a3aa3a36`, harness
`fc883ca`), in image
`docker.io/ptanski/nemotron-omni-port@sha256:c63fb9f47c48c7d64a0c5fdda51d4a198f9235c3f3e5ecd91cb39a6be0ebfc63`
(`/opt/venv`), FP32 with TF32 off, `eager-graphed` encoder, `dense-graphed`
decoder, native burst off. Evidence: `pdtgct/nemotron-omni-port` draft PR #182,
`experiments/2026-11-05-h1-populations`. The B1 `U_core` bracket result is in
`experiments/2026-11-05-h1-ucore-brackets`.

| Population | Eager step | Replayed step | Saving |
|-----------:|-----------:|--------------:|-------:|
| 1  | 22.6 ms | 16.2 ms | 6.4 ms |
| 31 | 46.0 ms | 40.0 ms | 6.0 ms |
| 63 | 70.8 ms | 65.6 ms | 5.2 ms |

Outputs and pages were bitwise equal to eager, with zero fallbacks on
continuing CHUNK steps. Nine cells (`[1, 31, 63]` × three kinds) retained
1.96 GB of graph memory, the sum of the per-cell growth; capture took 2.2 s.
g1 is unmeasured.

## Transaction Structure

`advance_model_rows` has three phases with one launch boundary:

1. `prepare_transaction` (`advance.py:2913`) is the host prologue: structural
   preflight, decode resolution, CHUNK bucket formation, per-step batch stats
   (`advance.py:3151`) and every host-only check. It produces
   `TransactionInputs` and touches no resident page.
2. `run_transaction_body` (`advance.py:3254`) is the device body: gathers,
   frontend, encoder, decode, projection, row status, progress and the
   complete scatter plan. The whole plan is validated before any scatter.
3. `commit_transaction_body` (`advance.py:4109`) makes one composite
   reservation, launches the prevalidated predicated scatters, then performs
   the no-fail stage.

The eager path runs all three phases. The graph path runs phase 1 eagerly.
`TransactionGraphBinding.execute` then classifies the call, prepares the
replay, reserves the commit, replays phase 2 together with the scatters, and
stages status.

### Execute order

`execute` runs in a fixed order:

1. the failed latch (Failure Containment);
2. the process constants (Process constants and excluded paths);
3. classification, including the mode reasons;
4. the signature;
5. the wrapper entry;
6. arena upload and static-input copies;
7. escaped-output allocation;
8. the commit reservation;
9. `graph.replay()`, then status staging and the replay counter.

A call declined at step 3 or 4 (no cell, an undeclared population, a mode
reason or a signature mismatch) increments its fallback counter and returns for
eager execution; the wrapper-entry check at step 5 is reached only by a call
that passed both. An exception raised at any step increments no counter: at
step 1, 2 or 5 it is worker-fatal; at steps 3, 4 and 6 to 8 it follows the
pre-boundary contract (Failure Containment).

### Cells and classification

A `TransactionCell` (`transaction_graph.py:62`) is `(kind, geometry,
population)`, where population is the exact real-row count. Only rows in the
step count: admitted, parked or never-fed slots without a row in the step
never contribute to its population.

| Kind | Rows |
|------|------|
| `chunk-fresh` | one CHUNK bucket, every row a session's first CHUNK |
| `chunk-continuing` | one CHUNK bucket, no fresh row |
| `replay` | no CHUNK row and no fresh row: REPLAY, park-echo, FLUSH, forced-endpoint |

`classify_transaction` (`transaction_graph.py:81`) returns the cell, or the
reason the call has none. A declined call runs eager and is counted under
`(cell or "uncovered", reason)`:

| Reason | Meaning |
|--------|---------|
| `empty` | no real rows |
| `fresh-non-chunk` | a fresh row without a CHUNK |
| `multi-geometry` | more than one CHUNK geometry (unreachable within the envelope) |
| `mixed-roles` | CHUNK and non-CHUNK rows in one step |
| `mixed-fresh` | fresh and continuing rows in one CHUNK bucket |
| `undeclared` | population not declared |
| `uncaptured` | declared cell with no published graph (unreachable once startup succeeds) |
| `signature` | structural inputs, including the arena layout, differ from the captured ones |
| `capture`, `memory-profile`, `native-burst` | modes never covered; `native-burst` is unreachable while the option is on, because configuration requires native burst off |

### Flat capture

The body is captured with the nested encoder, decoder and CHUNK-bucket
bindings running eagerly inside the outer capture. The single graph therefore
records their kernels directly rather than nested graph launches.

`capture_scope.py` holds a `ContextVar`: while `outer_transaction_capture` is
active, `nested_graph_mode` returns the eager mode (`capture_scope.py:37`), and
every nested launch site consults it (`encoder_execution.py:1560`, `:1627`,
`decode_graph.py:442`, `chunk_bucket_graph.py:249`). Host-to-device staging
inside the scope raises (`capture_scope.py:44`, `advance.py:1601`), because a
captured copy would replay a frozen pinned source.

### Inputs, outputs and signature

- **Arena.** Each cell owns a `TransactionArena` (`transaction_graph.py:328`):
  one device buffer holding every int64/bool vector the body reads, plus two
  pinned host buffers that alternate per call. Each host buffer is fenced by
  an event, so it is rewritten only after its previous upload has completed.
  There is one host-to-device copy per call.
- **Static inputs and owned outputs.** Each cell's `static_ids` and
  `static_embeds` receive the call's inputs before replay. Inside the graph,
  projection rows, status and progress are copied into per-cell buffers
  allocated outside the graph pool, so no pool tensor outlives a replay.
- **Signature.** `transaction_signature` (`transaction_graph.py:141`) covers
  every structural input the body bakes in: population, control ids,
  `endpoint_enabled`, `n_fresh`, decode and transition callable identities,
  resident pool layouts (address, shape, stride), lazily bound encoder
  buffers, input shapes, and the arena's name/dtype/shape layout. A layout
  mismatch is a `signature` fallback detected before any upload.

### Host control flow inside the body

The body is not branch-free. Its host control flow depends only on fields
fixed by the cell or covered by the signature: `endpoint_enabled` (signature);
the fresh-mask class (cell kind, plus `n_fresh`); the geometry's lookahead and
hop (cell geometry; the host reads a lookahead only for CHUNK buckets,
`advance.py:3484-3487`, so a `replay` cell has no host geometry branch);
decode output widths and `k_max` (geometry plus the decode
callable's identity); and the capture flag (forced off; `capture` is a
fallback reason). Any further host-dependent field the body branches on must
join the signature or the cell key before its cell is published. Within one
cell and signature the graph is data-independent. Per-row protocol state is
device data, not host control flow: whether a CHUNK row is a session's first
chunk (`chunk_sequence` 0) or a final tail (`final_tail`), which sets its
frontend target and its lookahead handling (`advance.py:1235-1240`), and, for
non-CHUNK rows, the per-row cadence the prologue stages from the plan's
geometry (`advance.py:3196-3199`) and the body reads from the arena
(`advance.py:3398`), and the book's geometry check (`advance.py:3440`).

Nested host-side checks and counters run only while a cell is captured and
are frozen under replay: `_validate_result_structure` (`advance.py:3541`), the
guarded-transition invocation counters (`encoder_execution.py:830`, `:2174`)
and the CHUNK binding's replay and fallback counts (`chunk_bucket_graph.py:275`,
`:479`). For replayed steps, encoder invocation counts and CHUNK-binding replay
and fallback counts do not advance; per-step batch stats still do, because the
prologue stages them. Those nested counts advance only when their host code
runs: on eager steps and during startup, including each cell's eager
reference, warm-ups and capture in both inventories (Capture and the
replay-cell oracle) and the eager profile transactions. Serving measurements
of nested counts are deltas from a baseline read after startup. The
transaction counters are the per-replay record.

### Capture and the replay-cell oracle

Capturing a cell (`transaction_graph.py:1233`) runs an eager reference on
capture rows the transaction rejects (so no page is stored), three eager
warm-ups through the wrapper, the capture, and a replay that must reproduce the
reference outputs and pages bitwise before the cell is published. Capture rows
carry the served geometry in their plan, for `replay` cells as for CHUNK cells
(`transaction_graph.py:1202`). The eager reference launches nested graphs,
while the warm-ups and the capture run them eagerly inside the capture scope
(Flat capture), so the self-check also compares the two nested paths, on
rejected rows. That self-check covers rejected rows only. Correctness for
accepted rows rests on tests that drive accepted rows of every kind through
replay at each servable geometry and compare with eager bitwise. At g1 they
cover every case in Lookahead rows at g1, with the CHUNK-bucket binding on and
off, and with the bucket on at a population outside its configured set. They
include `replay` cells carrying accepted FLUSH and park-echo
rows with endpointing on and off, and accepted forced-endpoint rows with
endpointing on. With endpointing off the forced-endpoint control is not
recognized (`advance.py:3423-3431`), so such a row is rejected, under replay
exactly as under eager.

### Lookahead rows at g1

At g1 the lookahead `L` is 1 and a regular chunk is `C = 16` mel frames of
2,560 samples. The frontend sets each row's target to `(8L + 1) + seq·C` for
chunk sequence `seq`; a final-tail row ignores its target, and a final tail of
`C·hop` samples or more is rejected with `ROW_STATUS_FINAL_OVERSIZE`
(`advance.py:1235-1240`). A session's first chunk takes no mel-tail prefix and
drops nothing; a continuing row takes the nine-frame prefix and drops the
configured pre-encode outputs; every row's encoder length is its own
`output_lengths(prefix + new frames) - drop`, clamped at zero, and zero when
the row produced no new frames (`advance.py:1247-1296`). Decode reads only the first `L + 1` encoder frames,
bounded per row by that length (`advance.py:1338-1347`). These are per-row
device values within one cell and signature. For every case below, replay is
bitwise equal to eager in outputs, statuses, progress and pages:

- **Fresh and final in one chunk.** A one-chunk session (`seq` 0, final tail
  under `C·hop` samples) is a `chunk-fresh` row: no prefix, no drop, and an
  encoder length that may be below `L + 1`. It may share its cell
  with regular first chunks.
- **Final tail among regular continuing rows.** Freshness and finality are
  independent (`advance.py:1235-1240`), and a final tail is not a separate
  kind. A continuing final-tail row therefore shares a `chunk-continuing` cell
  with regular rows; only its frame count and encoder length differ, and the
  regular rows' outputs are unaffected by it.
- **Final CHUNK, then REPLAY, then FLUSH.** After the final tail, the
  session's REPLAY rows drain its queued emissions: each validated echo emits
  the next queued label and advances the queue head, and a drained REPLAY
  parks and clears the pending echo (`advance.py:897-930`). Its FLUSH then
  emits park and leaves the queue and book unchanged (`advance.py:994-995`).
  These are non-CHUNK rows in `replay` cells, which read their cadence from
  the arena, and replay matches eager at every step of the sequence.
- **Oversized final tail.** The row is rejected with
  `ROW_STATUS_FINAL_OVERSIZE`, a terminal status: its pages stay unchanged, the
  other rows commit, and collection fails its request (Failure Containment).

### Callable identity

The signature compares decode and CHUNK callables by `id`. This is sound
because the dense decode binding caches one callable per graph key for the
binding's lifetime (`decode_graph.py:365-370`), and the CHUNK binding accepts
only its sealed callables (`chunk_bucket_graph.py:464-469`). The comparison
uses a normalized identity, `(id(__self__), id(__func__))` for a bound method
and the object's own `id` otherwise (`transaction_graph.py:104-106`,
`:166-167`): a sealed callable is the same object on every resolution, while
the CHUNK binding's eager path is a bound method that is a new object on each
attribute read but has a constant normalized identity. Ids are not reused
while those objects are alive. A resolver that rebuilds closures per call would
require a different identity mechanism in the signature.

## Supported Envelope

The option is `hf_overrides["transaction_graph_populations"]`: a list of
distinct, exact populations. An empty list means off. For example:

```python
hf_overrides = {
    "encoder_execution_arm": "eager-graphed",
    "decode_dispatch_arm": "dense-graphed",
    "experimental_native_burst": False,
    "supported_num_lookahead_tokens": [0],
    "transaction_graph_populations": [1, 31, 63],
}
```

This serves g0. Serving g1 sets `supported_num_lookahead_tokens` to `[1]` and
may also set `chunk_bucket_graph_populations` (g1 and the CHUNK-bucket
binding).

The engine also needs `cudagraph_mode` `NONE` and `max_num_seqs` of at least
the largest population. The model validates at
construction, each failure a configuration error naming the option:

- native burst is disabled, and the decode arm is `dense-graphed` or
  `dense-eager`;
- the runner's `cudagraph_mode` is `NONE` with no runner-managed encoder
  graphs, so `needs_cudagraph_capture()` is false;
- the served geometry set is exactly one geometry, g0 or g1:
  `supported_num_lookahead_tokens` is a one-element list or tuple whose
  element is the `int` 0 or 1, with no coercion, so an absent value (every
  published cadence), `[0, 1]`, `[3]`, `[1.0]` and `[True]` are rejected;
- every population is an integer in `1..min(max_num_seqs,
  max_num_batched_tokens)` (each row is one token), and populations are
  distinct;
- under `num_gpu_blocks_override`, every population is at most
  `override - 1`, the non-null slot count.

When memory, not the `max_num_seqs + 1` cap, binds capacity, the usable slot
count is known only after profiling. In that case the serving-capture slot
check (`transaction_graph.py:1129-1133`) fails startup.

### One served geometry: configuration and row admission

An engine with the option on serves exactly one geometry, g0 or g1. Every
other served set, `{g0, g1}` included, is rejected at configuration. Both the
model's configuration check and the binding's guard accept exactly the served
sets `(0,)` and `(1,)`; an unsupported singleton such as `(3,)` and any set of
another size raise a `ValueError` that names `transaction_graph_populations`,
the permitted values (`[0]` or `[1]`) and the rejected declaration. The
binding receives the served set from the model as `served_geometry_ids`
(`manifests.py:160-175`): `CADENCES` indices, the same function the processor
and `bind_rows` use. An index equals its lookahead only for g0 and g1 (index 2
is lookahead 3), so the configuration check reads lookahead values and the
binding guard reads indices. The binding declares its CHUNK cells at
that geometry; it captures
each `replay` cell with rows of the same geometry. The cell count per
population and the inventory's one dedicated graph pool are the same at either
geometry; at g1 only the sizes that scale with the chunk width (samples, mel
and encoder frames, decode output widths) grow. `replay` cells carry no
geometry in their key and cover non-CHUNK rows of any geometry, so leaving a
geometry uncaptured cannot keep its sessions' echo and FLUSH rows out of
replay. The served set, not the captured cells, therefore bounds the
geometries a replay can reach, and every row a replayed step carries is a row
of the geometry its cell was captured and validated at.

The served geometry set is enforced twice:

- **Per request, in the processor.** `NemotronASRMultiModalProcessor`
  (`processor.py`) rejects an envelope whose geometry is outside the served set
  before the offending input reaches engine execution, so one bad request
  fails alone and other sessions continue. An unadmitted request is never
  admitted. An admitted stream already holds a slot, because RFC 1 admits an
  empty stream before any input; a frontend preprocessing failure aborts it,
  and its abort and fenced retirement are core-owned.
- **Per step, in the worker.** The plan context receives the configured set
  instead of every `CADENCES` index. A row whose envelope header names any
  other geometry fails host structural validation in `bind_rows`, which
  validates the complete row set before its delta commit. The failure happens
  before the registry mutates and before any eager path could run the row. This
  is the worker invariant for inputs that bypass the processor or internal
  defects, and it keeps the model's existing structural-failure contract.

Both checks apply whenever the model serves, independent of
`transaction_graph_populations`.

Serving g0 and g1 in one engine is outside the envelope (Decisions &
Alternatives, "Served geometries"; Open Question 6).

### g1 and the CHUNK-bucket binding

At g1 the option runs with or without the CHUNK-bucket binding
(`chunk_bucket_graph_populations`). That binding serves only g1 populations 1,
31 and 63 and has its own construction requirements: the `eager-graphed`
encoder arm, the `dense-graphed` decoder, native burst off, synchronous,
non-speculative single-device execution, `supported_num_lookahead_tokens`
`[1]`, expert parallelism off and no logits processors, and, per declared
bucket population, decoder tier 1 for population 1, 32 for 31 and 64 for 63
(`chunk_bucket_graph.py:113`, `:299-317`, `encoder_execution.py:765-787`,
`native_burst.py:221-249`). Decoder tiers are the powers of two below
`max_num_seqs` plus `max_num_seqs` itself (`decode_graph.py:23-33`), and the
encoder's maximum population is `max_num_seqs` (`nemotron_asr.py:552`).
Population 1 therefore holds at any `max_num_seqs`, population 31 needs
`max_num_seqs >= 32` and population 63 needs `max_num_seqs >= 64`. Below the
population, tier resolution raises "decode population N exceeds captured
maximum M" (`decode_graph.py:186-192`); at exactly 31 (for 31) or 63 (for 63),
the tier equals the population and the bucket binding raises "CHUNK pilot
requires unchanged physical decoder tiers 1/32/64" (`chunk_bucket_graph.py:310-317`).
The bucket binding's construction owns those errors, including the ones its
reused native-burst guard raises, and they do not name
`transaction_graph_populations`; with `[0]` and the bucket on, that guard
rejects the configuration ("native burst requires the declared 160-ms
lookahead profile", `native_burst.py:243-249`), so the
two never meet at g0. The option's own validation runs first
(`nemotron_asr.py:482`, before the bucket's construction at `:584-608`), and
that precedence is kept, so a served set the option rejects always fails with
the option's error. When both are on:

- **Startup order.** `prepare_execution_memory` warms up and publishes the
  bucket inventory before admission (`nemotron_asr.py:925-927`, `:935`), so
  both transaction inventories capture against the published bucket
  inventory. The eager CHUNK-cell and `max_num_seqs` profiles then run with
  the retained nested inventory (encoder, decode and CHUNK-bucket graphs) and
  the admission inventory alive (Ordering, step 3); the serving inventory does
  not yet exist.
- **Call binding.** The bucket binding is the transaction's
  `bucket_transition` call binding, so its identity is in the startup key, and
  the transition it resolves for a CHUNK bucket is in the signature. At a
  bucket population it resolves the sealed transition; at any other
  population, its bound eager path (`chunk_bucket_graph.py:461-462`,
  `:474-480`), a bound method read afresh on each resolution, whose normalized
  identity is stable for the binding's lifetime (Callable identity). Transaction populations need not equal bucket populations.
- **Replayed and eager steps.** Flat capture runs the resolved transition
  eagerly inside the outer capture (`chunk_bucket_graph.py:249`), so a
  replayed step executes the bucket's eager body (`chunk_bucket_graph.py:150-163`)
  as recorded kernels and launches no bucket graph. An eager step (a fallback,
  or a capture's eager reference) launches the bucket graph for a bucket at
  one of its populations. Replay and eager therefore run different nested
  paths for the same accepted rows, and neither self-check covers accepted
  rows: the bucket capture checks profile rows, and the transaction capture
  rejected rows. Accepted-row bitwise equality is a test obligation (Capture
  and the replay-cell oracle). Both paths write the bucket's scratch (staged
  inputs, state copy and owned outputs), which is nested-binding scratch
  (Invariants): each call stages all of it before reading it.
- **Positional projections.** The bucket capture requires unprepared
  positional projections (`chunk_bucket_graph.py:127`), and the
  `eager-graphed` arm rejects prepared ones (`encoder_execution.py:1012-1022`).
  The signature's `_stream_relative_position_lengths` entry is therefore empty
  at capture and stays empty; the bucket binding adds no source of `signature`
  fallback.
- **Counters.** The bucket's replay and fallback counts
  (`chunk_bucket_graph.py:275`, `:479`) count invocations of its sealed
  transitions and of its eager path, whichever a CHUNK bucket resolves. A
  replayed transaction step increments neither (Host control flow inside the
  body). They advance on eager steps with a CHUNK bucket and during startup:
  each CHUNK cell's eager reference, warm-ups and
  capture in both transaction inventories, and the eager profile transactions,
  which pass the bucket binding (`nemotron_asr.py:945-962`,
  `profile_execution.py:311`). `replay` cells invoke no bucket. Serving
  measurements of these counts are deltas from a baseline read after startup.
  An eager step whose bucket resolves a sealed transition runs the bucket
  graph, whose body calls the native encoder directly
  (`chunk_bucket_graph.py:151-161`, `advance.py:1310-1314`), so the guarded
  encoder transition's invocation counters do not advance; an eager step whose
  bucket resolves the eager path passes the guarded transition, and they do.

### Echo eligibility, mixed steps and signature mismatch

A `replay` cell covers a step whose real rows are all non-CHUNK and non-fresh,
with a declared population. Steps that mix CHUNK and echo rows (`mixed-roles`)
or fresh and continuing CHUNK rows (`mixed-fresh`) run eager by design and are
counted. Their serving rate is part of the default-on gate. A signature
mismatch runs the call eager and counts `signature`. Within the envelope the
signature is constant after startup, so a nonzero count is a defect to
investigate, not a tolerated rate.

### Lifetime: one serving capture per worker

The serving inventory is captured once per worker lifetime, before any session
is admitted, because capture reads free blocks `1..P` of the resident pools.
That read, and the rejected-row reference over the same blocks, is startup
access to unowned slots, outside the per-step access to scheduled handles that
models otherwise get. The model reaches the pools for it only through
`StateSlotBindings.startup_pool_access(layer).component_pools`, which core
makes valid only inside the runner's `initialize_state_slots` call
(`vllm/v1/worker/gpu/model_runner.py:896-898`). The access grants no slot
ownership and authorizes no store to any resident page, the null page
included. Capture, its warm-ups and its self-check therefore run only rows the
transaction rejects, on non-null free blocks, and store no page (Capture and
the replay-cell oracle). The commit-scatter warm-up that precedes the first
serving capture follows the same rule: for each resident pool it launches one
row with a rejected status that names the first non-null free block, and the
kernel's status predicate returns before it reads the block id or any
resident address. The model keeps no access object after the call returns.
Publication records three plain-value references for post-warmup
verification: the `pool_layout(layer)` snapshot, the projected resident
layout and the device of the resident pools.

When CUDA graph memory is profiled, core calls `initialize_state_slots` twice:
first with bindings whose `is_profiling` is true, over temporary pools, then
with serving bindings over new pools. On profiling bindings the model captures
nothing, requests no startup access and retains nothing; the admission holder
stays alive for serving capture. The option requires `cudagraph_mode` `NONE`,
under which the runner does not profile CUDA graph memory, so this path is
unreachable while the option is on; the guard keeps the rule independent of
that configuration.

A repeated `initialize_state_slots` with serving bindings whose pool layout and
call bindings are identical is a verified no-op; any difference is
worker-fatal. Profiling bindings take no part in that comparison. Nothing is
recaptured outside startup.

Runner shutdown discards the binding's graphs, admission inventory, arenas and
owned buffers. After a device synchronize, `GPUModelRunner.shutdown` calls
`ModelState.shutdown()` before it releases the state bindings and the KV cache
(`model_runner.py:2531-2542`). The model state's override calls
`TransactionGraphBinding.discard()` and tolerates a model state whose
initialization did not complete and a binding that never captured. Core logs
an exception from the hook and continues its cleanup, and never calls the
hook twice.

### Process constants and excluded paths

The batch-invariant mode, the per-operation FP32 precision settings
(`torch.backends.cuda.matmul.fp32_precision`,
`torch.backends.cudnn.conv.fp32_precision` and
`torch.backends.cudnn.rnn.fp32_precision`) and `CUBLAS_WORKSPACE_CONFIG` are
process constants for the binding's lifetime. The per-operation settings also
reflect the legacy `torch.set_float32_matmul_precision` setter, while the
legacy getters raise once a per-operation value has been set, so the check
reads only the per-operation settings. `execute` compares their effective
values (not cached construction values) with their capture-time values using
host-only reads, before every return, including uncovered and mixed-row
fallbacks. A change is worker-fatal, because
an eager fallback would also run against nested graphs captured under the old
settings. Sleep mode and the CuMem allocator are rejected for
all persistent state (`vllm/v1/core/state_slot_manager.py:52-53`), as is weight
reload after execution materialization (`nemotron_asr.py:649-653`).

### Invariants

- Arena upload, static-input copies, replay and every resident write run on
  the runner stream (`model_state_v2.py:123-124`). Replays are serialized.
- No slot is freed or reassigned between the commit reservation and the
  commit-sink collection in `postprocess_state`. Collection precedes the next
  replay because persistent state requires synchronous scheduling
  (`vllm/v1/core/state_slot_manager.py:47`).
- A request abort is not commit-ticket cancellation. A launched transaction
  completes its normal stage, collection and fence path whatever happens to
  its requests; core suppresses an aborted request's output and controls the
  slot's retirement through the release fence.
- Slot ids are arena data, so a graph is slot-agnostic. It reads and writes
  resident pages only through uploaded block indices and predicated per-row
  status.
- Cells share no arena, static input or owned output. Intermediates may share
  graph-pool memory, because replays are serialized and no intermediate is
  live across a replay.
- Nested-binding scratch (encoder, decode and, at g1, CHUNK-bucket) is shared
  by the outer graph and the eager-path nested launches. It is fully written
  before it is read within a call, and its addresses and sizes are checked
  after the warmups (Post-Warmup Verification, check 6).

## Memory Admission

### Budget mechanics

- `determine_available_memory` wraps `profile_run` in `memory_profiling`
  (`vllm/v1/worker/gpu_worker.py:720-735`). `non_kv_cache_memory` is the
  free-memory drop since worker creation plus the transient peak headroom
  (`vllm/utils/mem_utils.py:327-336`). Anything still alive when the window
  closes is charged at its driver-visible size.
- `profile_run` calls the model's `prepare_execution_memory` inside that window
  (`vllm/v1/worker/gpu/model_runner.py:1117-1119`). That hook retains the
  nested inventory and runs eager profile transactions alongside it
  (`nemotron_asr.py:902-969`).
- The budget is the memory left after `non_kv_cache_memory`, or
  `kv_cache_memory_bytes` when that is set. Resident capacity for persistent
  state is `count = min(budget // page_size_bytes, max_num_seqs + 1)` pages.
  A set `num_gpu_blocks_override` replaces the count, not the budget: core
  budget validation must still accept `count * page_size_bytes <= budget`,
  and startup fails unless it does and `count >= 2`. Usable slots are `count -
  1`, excluding the null block (`vllm/v1/core/kv_cache_planning.py:1645-1655`
  rejects an oversized override, `:1681-1691` a budget too small for the null
  page and one slot, and `:1676-1679` a failed automatic fit of
  `max_model_len`, all before allocation). Admission therefore reduces
  usable slots only when memory, not the `max_num_seqs` cap, binds.
- Resident pools exist only inside `initialize_kv_cache`. The KV allocation
  precedes `initialize_state_slots`, and the capture hook runs from
  `initialize_state_slots` after the pools are allocated
  (`model_runner.py:869`, `:896-898`; `model_state_v2.py:109-122`).

Graphs bake in resident pool addresses, and those addresses do not exist until
capacity is fixed. The inventory that serves therefore cannot be the inventory
that is measured.

### Normalized measurement

`N()` denotes a normalized free-memory reading. It is taken after a device
synchronize, `gc.collect()` and an allocator `empty_cache()`, with no other
allocation in flight, using `mem_get_info`. Every admission figure is a
difference of two `N()` readings. This includes memory the torch allocator does
not track: instantiated graphs and fragmentation. Only the admission release
check reads the allocator itself (Ordering): free memory does not rise when a
release leaves its block in a segment the allocator keeps, so the check
compares the allocator's allocated bytes, which are exact for storage of any
size.

### Capture-lifetime rule

Captures are performed through a per-cell capture context. That context holds
the capture `TransactionInputs` (and through them the pools), page snapshots,
expected outputs, and the capture and verify callbacks. The wrapper's body
reads only through it.

After a cell is verified, the binding clears the context. The published cell
retains only what replay needs: the wrapper's graph entry, the arena, static
inputs, owned outputs, the signature and the descriptor. During admission the
stand-in storage the capture inputs reached is retained only by the admission
holder's explicit reference, until it is released after `Na` (Ordering).

Admission graphs go into a separate admission holder that is never published,
has no replay path, is not part of `_graphs` or `_pool_signature`, and never
makes the binding ready.

### Ordering

1. **Construction.** The binding is built and the envelope is validated.
2. **Admission (inside the profile window).** In `prepare_execution_memory`,
   after the nested inventory is ready and before the eager profile
   transactions: read `N0`; allocate stand-in storage of `max(populations) +
   1` slots with the real layout (`build_nemotron_persistent_state_spec`,
   `allocate_state_scratch`, one raw allocation, `state_profile.py:240-263`);
   capture the complete declared inventory over it into a dedicated graph pool
   through the same capture path, cell order, capture stream and verification
   as serving capture; clear every capture-only temporary (page snapshots,
   expected outputs, capture and verify callbacks, capture inputs) while the
   admission holder retains the stand-in storage explicitly; read `Na`, then
   the allocator's allocated bytes on the device; release the stand-in
   storage; read `N1`, then the allocated bytes again. `Na` is the settled
   point at which the allocated-bytes reading for the release check is taken;
   no admitted figure uses `Na`'s free memory. Release is verified
   when the allocated bytes drop by at least the stand-in bytes between the
   two allocated-bytes readings and no owning reference to the stand-in
   storage remains (checked separately through a weak reference), else
   startup fails. Admitted bytes are `A = N0 - N1`, a free-memory difference:
   stand-in storage released into a segment the allocator keeps stays in `A`,
   which can only over-charge. Stand-in
   storage provides capture indices and matching component strides, not
   proven-identical allocator or graph-exec bytes; the final check is the
   authority.
3. **Eager profiles.** The existing CHUNK-cell and `max_num_seqs` profile
   transactions run with both inventories alive, the retained nested inventory
   (encoder, decode and, when on, CHUNK-bucket graphs) and the admission
   inventory, so their transient peak is measured on top of retained graph
   memory.
4. **Window closes.** `non_kv_cache_memory` includes the admission inventory
   (graph-pool segments, instantiated graphs, arenas, owned outputs). The peak
   headroom includes the preparation and capture peak.
5. **Capacity fixed** from the remaining budget, by the formula above.
6. **Serving capture.** In `capture_transaction_graphs`: release the admission
   holder and its pool; read `Ns0`; capture over the resident pools into a
   fresh dedicated pool; clear the capture contexts; read `Ns1`. Serving
   retained bytes are `S = Ns0 - Ns1`.
7. **Admission check.** Startup succeeds only if every cell is published and
   `S <= A`.

**Escaped outputs.** The per-call escaped row buffer is at most `population ×
hidden_size` float32. It is allocated from the headroom reserved by the
`max_num_seqs` eager profile transaction, whose own projection rows are a
superset.

### Capacity overrides

The two overrides fix different quantities. `kv_cache_memory_bytes` fixes
the budget: it skips `memory_profiling` but still runs `profile_run`
(`vllm/v1/worker/gpu_worker.py:691-716`), so `prepare_execution_memory`,
admission and the measurement of `A` still run, and the count follows from
that budget by the formula above. `num_gpu_blocks_override` fixes the count,
but only if core budget validation accepts it against the budget in force:
the profiled remainder, which includes `A`, or `kv_cache_memory_bytes` when
both are set. An override that exceeds that budget is rejected before
allocation (Failure modes and diagnostics).

Budget-derived capacity from the profiled remainder (Ordering step 5)
applies only without either override. Under either override, admission is
still measured and reported and `S <= A` still checked, but neither changes
the count the overrides determine. If serving capture then raises or runs out
of memory, the model reports an admission failure naming the option. A
failure of the KV allocation itself happens before `initialize_state_slots`,
outside any model hook, so core names the option there from the state spec's
memory contributors (Core Obligations, item 5).

### Failure modes and diagnostics

Every case below is a startup failure. None degrades to eager serving. Each
names the option. Core names it in the planner rejection and the KV-allocation
failure, which no model call surrounds, because the model lists
`transaction_graph_populations` in its state spec's `memory_contributors`
while the option is on (Core Obligations, items 4 and 5).

- Admission does not fit (capture failure or OOM in the window):
  `prepare_execution_memory` fails and latches `_execution_memory_failed`
  (`nemotron_asr.py:964-968`).
- Release not verified, or too few usable slots.
- No capacity: core budget validation rejects the persistent-state budget,
  because fewer than two pages fit, a `num_gpu_blocks_override` exceeds the
  remaining budget, or automatically fitting `max_model_len` fails
  (`vllm/v1/core/kv_cache_planning.py:1645-1655`, `:1681-1691`, `:1676-1679`).
  It runs in the engine-core process from `EngineCore._initialize_kv_caches`
  (`vllm/v1/engine/core.py:357`), after `determine_available_memory` returns
  and before any worker `initialize_kv_cache`, so no model call surrounds it
  and the model cannot wrap it. The `ValueError` names the option.
- KV allocation fails or runs out of memory in `initialize_kv_cache`, before
  `initialize_state_slots` and therefore outside the model's capture hook.
  Core re-raises the same exception with a note naming the option
  (`model_runner.py:207-220`, `:865-867`). The note is in the worker's logged
  traceback, not in the exception's message; the multiprocess executor sends
  the engine only the message.
- Serving capture raises, or `S > A`: every graph is discarded and
  `initialize_kv_cache` raises.

A capture OOM reports the normalized bytes admitted so far, the cells
captured, the failing cell, and the device's free and total memory. The full
required size of an unfinished inventory is not known and is not reported.

**Cost.** Capture runs twice per startup (about twice the single-inventory figure, plus
eager warm-ups). Admitted bytes come out of resident capacity when memory
binds. The peak headroom may count part of the retained inventory twice, which
errs toward fewer slots.

## Post-Warmup Verification

Worker warmups run after serving capture and can resize workspaces:
`kernel_warmup`, `warmup_kernels` and the compile dummy runs. Core calls
`ModelState.post_warmup()` once, after every warmup stage and before the
worker serves (`vllm/v1/worker/gpu_worker.py:1207`); an exception from it fails
engine startup. The model state's override runs the post-warmup verification
there, over its current serving bindings.

Verification holds no resident tensor. Its inputs are the live
`StateSlotBindings.pool_layout(layer)` snapshot (address, shape, element
strides and dtype of each component pool, as plain values), the live call
bindings, the core and the nested bindings. It rebuilds the resident pools as
layout-only pools: one meta-device tensor per component pool, placed in a
shared meta storage at the live layout's address with its shape, strides and
dtype, and projected as the resident pools are. It checks, in order:

1. the live pool layout against the snapshot recorded at publication, address
   included;
2. the device the resident pools were captured on against the device of the
   core's encoder parameters;
3. the projected resident layout against the one recorded at publication. For
   each transaction role (each layer's channel, time and valid-length pools,
   `h`, `c`, `queue`, `book`, and the frontend and endpoint pools) it holds the
   component pool the role starts in, the role's address and byte offset in
   that component, and its shape, element strides and dtype. The rebuilt roles
   sit at the live addresses and differ from the resident tensors only in
   their meta device, which the record omits. A projection that swaps two
   same-shaped roles or changes a role's strides fails here, although every
   component pool is unchanged;
4. every published cell's signature, rebuilt as at capture over the
   layout-only pools. Its resident-pool and input-layout entries name the meta
   device and are not compared: checks 1 to 3 cover the pools, and the cell
   and the call bindings fix the inputs;
5. the call bindings against those recorded at publication; and
6. the address and size of every binding-external buffer the graphs read. These
   are the core's parameters and buffers and the entry scratch of the nested
   encoder and decode graph bindings, which each reports through its
   `external_tensors()` inventory. When the CHUNK-bucket binding is on, they
   also include, for each of its sealed transitions, the staged inputs, the
   state copy and the owned outputs, which the bucket binding reports through
   its own `external_tensors()` inventory and which the model includes among
   its nested execution bindings. Those buffers are held by the sealed
   transitions (`chunk_bucket_graph.py:135-180`), outside both
   `EncoderExecution.external_tensors` (`encoder_execution.py:1636-1665`) and
   the model's nested-binding list (`nemotron_asr.py:1065-1071`), so the bucket
   binding must provide that inventory before the option serves with it on
   (Decisions & Alternatives, "Bucket scratch inventory"). A changed address or
   size is worker-fatal like every other mismatch here. The comparison covers
   only recorded entries (`transaction_graph.py:1541-1547`), so verification
   also checks completeness against a schema derived from the binding, not from
   what it reports. The bucket is on when the call bindings'
   `bucket_transition` is not `None`; the expected entries are then
   `encoder.bucket.<g>.<n>.<field>` for each sealed cell `(g, n)` of the binding, with fields `env`, `prompt` and `status` (staged inputs); `state.raw_tail`, `state.mel_tail`, `state.frontend_counters`, `state.channel.<i>`, `state.window_valid.<i>` and `state.time.<i>` for each encoder layer `i`, `state.h`, `state.c` and `state.last_label` (state copy); and `out.token_ids`, `out.token_lengths`, `out.row_status`, `out.frame_emission_counts`, `out.frame_final_labels`, `out.frame_valid_lengths`, `out.valid_samples`, `out.geometry_id`, `out.final_tail`, `out.prompt_index`, `out.chunk_sequence`, `out.counter_invariant_bad` and `out.counter_delta_bad` (owned outputs). Verification fails if the recorded or the live inventory lacks any
   expected entry; an empty or incomplete bucket inventory while the bucket is
   on is therefore a mismatch, never a pass. Publication records the inventory
   it finds and does not check it; verification is the one failure point, and
   its failure latches the binding as worker-fatal.

A mismatch is worker-fatal: it latches the binding and fails startup.
Verification does not run again at serving time. Every non-dummy
`prepare_step` checks the binding's failed latch before its registry pruning
(`model_state_v2.py:190`), and therefore before `state_slots.begin_step`,
request activation or any transaction, and raises if it is set.

## Failure Containment

The launch boundary is `graph.replay()` in `TransactionGraphBinding.execute`;
the checks before it run in the Execute order.

**Per-row rejections are not step failures.** A row the transaction rejects
has a nonzero status, its pages stay unchanged and the step's other rows
commit. Under replay exactly as under eager, its status flows through
commit-sink collection, which classifies it as eager does: a terminal
rejection goes to `state_slots.fail_requests` (`model_state_v2.py:324-327`)
and core finishes that request with an error, while a recoverable status
(`ROW_STATUS_GEOMETRY`, `commit_sink.py:74`) does not fail the request. Such a
step neither raises nor latches the binding.

**Unchanged-state baseline.** "Unchanged" means the resident state after core
slot initialization in `begin_step` (`model_runner.py:1891`,
`vllm/v1/worker/gpu/state_slots.py:77-86`), which zeroes and initializes newly
bound slots before the model runs.

**Before the boundary**, work touches only binding-owned buffers and host
bookkeeping: classification and signature, process-constant and wrapper-entry
checks, arena upload and static-input copies, escaped-output allocation, and
the commit reservation. A non-fatal failure here (classification, signature,
arena upload, static-input copy, escaped-output allocation or reservation)
follows the eager structural-failure contract: the call raises, resident state
equals the baseline, and a reserved ticket is cancelled. Ticket cancellation
is error cleanup only: this pre-boundary path, and a replay that raises
(below); a request abort never cancels a ticket (Invariants). Process-constant,
wrapper-entry and post-warmup failures are worker-fatal and latch the binding,
although they occur before the boundary.

**Wrapper entry.** Before replay, `execute` checks on the host that the cell's
wrapper entry still holds its graph. A missing entry is worker-fatal and never
leads to a capture. It can be emptied by `CUDAGraphWrapper.clear_all_graphs()`
(`cudagraph_utils.py:1081`, `model_runner.py:2546`). The configuration
requirement `cudagraph_mode == NONE` keeps the runner's graph estimator
(`cudagraph_utils.py:1006-1010`) and its minimal state-slot bootstrap
(`cudagraph_utils.py:1110-1147`) from ever invoking the capture hook.

**From the boundary on**, any failure is worker-fatal:

- Every allocation the step needs is made before the launch; afterwards only
  stream-ordered copies into preallocated memory, `CommitTicket.stage` and
  host counter updates run.
- Any exception from the replay onward, including `CommitTicket.stage` failing
  after it has set `_done` (`commit_sink.py:95-105`), latches the binding as
  failed and propagates. Cancellation of the reservation is attempted on
  every such raise; its outcome, including the ticket's state after a partial
  stage, is irrelevant because the worker is never reused; later `execute` calls raise
  "a fresh worker is required".
- In the engine, the exception leaves `execute_model`, is re-raised from
  `step` (`vllm/v1/engine/core.py:756`, `:773`), and the busy loop signals the
  engine dead (`core.py:1794-1800`).
- Asynchronous device faults surface at the next synchronizing call, normally
  the commit-sink collection (`commit_sink.py:270`, via
  `model_state_v2.py:324`), which fails before any progress is published and
  propagates to the same engine-death path. Collection runs outside
  `execute`, so it does not latch the binding; the engine, not the model,
  stops later steps. Faults are not attributed to cells and not resumed.
- In-process callers of `advance_model_rows` see pre-boundary failures as
  structural failures; engine callers see any raise as engine death. Recovery
  is a fresh worker process.
- Neither cancellation nor abort implies rollback. A reservation cancelled
  after a replay raised, or a request aborted after a replay wrote its pages,
  leaves those pages as the replay left them until core rebinds the slot for
  a new generation, which zeroes and initializes it (`vllm/v1/worker/gpu/state_slots.py:66-88`),
  or a later authorized write changes them. Slot retirement and fencing stay
  with the core state-slot manager.

This extends the model's existing rule that a catastrophic failure during the
prevalidated commit is worker-fatal (`advance.py:2815-2821`) to the whole
replayed body, because the body and the scatters are one launch.

**Output lifetime.** The escaped row tensor is caller-owned and survives any
later replay. The owned status and progress buffers are valid until the same
cell's next replay. The commit sink consumes them earlier in the step.

## Observability

- The counters and the receipt are an in-process harness facility. Evidence
  for this design comes from in-process `EngineCore` runs; a reader reachable
  under `EngineCoreProc` is a Default-On Gate prerequisite.
- `vllm.models.nemotron_asr.transaction_graph:transaction_graph_counters` (its
  path on this branch) takes a model, runner, worker, executor or in-process
  EngineCore. It reaches the model by walking runner and worker attributes
  (`_resolve_model`, `transaction_graph.py:1707-1723`), a harness convenience,
  not a core contract. It returns
  host-only counts that never synchronize: captured cells, replays per cell,
  fallbacks per `(cell, reason)`, and prologue time. With the option off it
  returns an empty mapping; if the model is unreachable in process (for
  example under a multiprocess executor) it raises, so that case is never
  mistaken for off.
- The execution profile receipt includes readiness and, per cell, arena bytes,
  reserved-memory growth, warm-up and capture seconds, and capture order. It
  also includes `A`, `S`, the release check, and the admission and serving
  capture seconds, and the served geometry. Cell labels stay as they are
  (`transaction_graph.py:76-77`): a `replay` label names no geometry, so the
  receipt's served geometry is what distinguishes evidence from g0 and g1
  engines.

## Core Dependency: Persistent-State Warmup

This is a core fix owned by the core persistent-state design (its criterion
M1), not by the model. `run_mixed_prefill_decode_warmup` builds synthetic
token-prefill and decode requests, which are invalid for any
`PersistentStateSpec` group: a persistent-state request holds one state record
for its lifetime and starts with a model-defined transaction. The warmup
returns early when any KV cache group is persistent state
(`vllm/v1/worker/gpu/warmup.py:157-162`). The other gates (pooling model,
`max_num_reqs < 2`, `num_tokens < 3`, too few blocks) stay; multi-session
populations pass the `max_num_reqs` gate and reach the persistent-state gate.
Non-state models are unchanged.

## Core Obligations

This design places the obligations below on the core persistent-state design
(`docs/design/persistent_state.md`, "Model hooks", "Profiling and serving
bindings", "Startup pool access" and "Memory contributors", criteria M1 to
M8). Core delivers all six obligations. Core-side criteria name a model-supplied
contributor label rather than this model's option, so core tests stay
independent of the out-of-tree model.

1. **Persistent-state mixed-warmup skip.** Delivered:
   `run_mixed_prefill_decode_warmup` skips persistent-state groups
   (`vllm/v1/worker/gpu/warmup.py:157-162`; M1). TXG-CORE-001, TXG-CORE-002.
2. **Post-warmup hook.** Delivered: `ModelState.post_warmup()`, called once
   through `GPUModelRunner.post_warmup` after every worker warmup and before
   the worker serves (`gpu_worker.py:1207`, `model_runner.py:2527-2529`; M2).
   The model verifies there (Post-Warmup Verification). TXG-VER-004.
3. **Shutdown hook.** Delivered: `ModelState.shutdown()`, called first in
   `GPUModelRunner.shutdown`, while the state bindings and the KV cache exist
   (`model_runner.py:2531-2542`; M3). The model discards there (Lifetime: one
   serving capture per worker). TXG-LIFE-007.
4. **Option-attributed capacity rejection.** Delivered:
   `PersistentStateSpec.memory_contributors` names the model options that
   also consume device memory, and when core budget validation rejects the
   persistent-state budget, including a `num_gpu_blocks_override` that exceeds
   the remaining budget or a failed automatic fit of `max_model_len`, the
   `ValueError` names them (`vllm/v1/kv_cache_interface.py:1586-1588`,
   `vllm/v1/core/kv_cache_planning.py:84-104`; M7). The model lists
   `transaction_graph_populations` there while the option is on. TXG-MEM-011.
5. **Option-attributed KV-allocation failure.** Delivered: a KV allocation
   that raises before `initialize_state_slots` is re-raised with a note naming
   the same contributors (`model_runner.py:207-220`, `:865-867`; M8).
   TXG-MEM-012.
6. **Startup pool access.** Delivered: `StateSlotBindings.startup_pool_access`
   (every page of each component pool, valid only inside the runner's
   `initialize_state_slots` call, authorizing no store),
   `StateSlotBindings.pool_layout` (plain-value layout, usable at any time) and
   `StateSlotBindings.is_profiling` (`vllm/v1/worker/gpu/state_slots.py:147-256`,
   `model_runner.py:892-898`; M4 to M6). Capture uses the first, post-warmup
   verification the second, and the profiling guard the third. TXG-LIFE-008,
   TXG-LIFE-009.

## Default-On Gate (future intent)

Turning the option on by default requires, against a declared baseline
configuration: correctness with the model installed as an out-of-tree plugin;
serving latency and sustained-concurrency evidence at the served cadence from
unprofiled serving measurements; realistic mixed-step fallback rates under
admitted, staggered sessions; the complete startup cost, including both
captures; and resident capacity preserved, or its reduction explicitly
accepted. It also requires:

- a counter and receipt reader reachable under `EngineCoreProc`, for example
  through `collective_rpc` and the plugin's `vllm.general_plugins` worker
  entry point, so serving fallback rates are observed outside in-process runs.

The default cadence is g1, so this evidence is taken at g1, with the
CHUNK-bucket binding on or off as the baseline configuration declares.
Serving g0 and g1 in one engine is not a prerequisite.

## Decisions & Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
|----------|--------|------------------------|-----------|
| Placement | Design doc beside the model, moving with it to the plugin; numbered acceptance criteria | Core design doc; EARS with `@spec` tags | The model owns the option; the plugin move must not orphan intent |
| Posture | Opt-in, with admission, one served geometry and counted fallbacks | Default-on; opt-in without admission | Startup and capacity costs and mixed-step rates are unmeasured; hidden graph memory breaks capacity planning |
| Coverage key | Exact population per kind and geometry | Padded population buckets | Padding changes row counts through decode and commit and needs its own exactness proof |
| Capture shape | Flat: nested bindings eager inside one outer capture | Nested graph launches | One launch; nested launches keep per-region cost and separate pools |
| Host inputs | Persistent double-buffered pinned arena, one copy | Per-call `HostStaging`; captured host-to-device copies | Captured copies replay frozen sources; per-call staging adds launches |
| Served geometries | Exactly one per engine, g0 or g1; any other set rejected at configuration and its rows at row admission; CHUNK cells declared, and `replay` cells captured, at the served geometry | (a) g0 and g1 together in one engine; (b) a general geometry set over every published cadence; (c) other geometries captured but unvalidated, or left uncaptured and counted | g1 is the default cadence, and one served geometry keeps the inventory at three cells per population; (a) has no mixed-geometry use yet, raises the inventory to five cells per population, makes `multi-geometry` reachable, may need a geometry key on `replay` cells (Open Question 6), and cannot run with the CHUNK-bucket binding, which accepts only `[1]`; (b) adds the costs of (a) for cadences with no accepted-row evidence; (c) `replay` cells key no geometry, so an unvalidated or uncaptured geometry still reaches replay through its echo rows |
| CHUNK-bucket binding at g1 | Composes with the option: flat capture records the bucket's eager body, while eager steps launch its graph; accepted-row bitwise equality between the two is a test obligation, with the bucket on and off and at a population outside its set; its scratch joins post-warmup verification | Require it off; require it on; nest its graph launch inside the outer graph | Both arms are served g1 configurations, and the bucket graph is the eager reference when it is on; flat capture already covers its kernels, and a nested launch keeps its per-region cost and separate pool (Capture shape); because replay and eager run different nested paths and neither self-check covers accepted rows, equivalence must be tested rather than inferred |
| Admission method | Model-local: retain the full admission inventory over stand-in storage through the profile window, then check `S <= A` | (a) Reserved-memory delta of serving capture; (b) placeholder tensor of the measured size; (c) extrapolate from one cell; (d) core graph estimator; (e) measure the admission capture's normalized delta and peak, release it, and charge both through a core accounting hook before capacity selection | (a) and (b) miss graph-exec memory, fragmentation and peak; (c) is unsound without reuse; (d) does not see model graphs; (e) is viable but needs a new core hook and an explicit composition of the measured peak with the profile peak, while retention reuses the existing window unchanged |
| Graph pool | One dedicated pool per inventory | Platform global pool shared with nested graphs | Releasing admission must return exactly its segments, whatever stream serving capture uses |
| Failure boundary | All step allocations before replay; any failure from the launch on is worker-fatal with a latch | Roll back pages; fall back to eager after a replay error | Body and scatters are one launch, so partial mutation is unknowable; rollback needs hot-path snapshots |
| Signature mismatch | Count and run eager; no serving-time recapture | Recapture on demand; raise | Recapture needs unadmitted memory and stalls; eager is the exact reference |
| Process constants | Assert at `execute`, worker-fatal on change | Include them in the signature | The nested inventories and every captured cell were built under the same settings, so a change invalidates the whole process, not one cell; a signature fallback would silently run eager against stale nested graphs |
| Bucket scratch inventory | The CHUNK-bucket binding owns its sealed transitions' scratch (staged inputs, state copy, owned outputs) and reports it through its own `external_tensors()`; the model adds the bucket binding to its nested execution bindings; post-warmup check 6 compares it and, at verification under the worker-fatal latch, requires every entry of the schema derived from the binding's sealed cells | The transaction binding reads the bucket's closures; the encoder execution reports the bucket entries; exclude the bucket scratch on the ground that no warmup rebinds it | The buffers are closure locals of the bucket's capture (`chunk_bucket_graph.py:135-180`), so only the bucket binding can name them; the encoder execution publishes and discards the sealed callables (`encoder_execution.py:789-807`, `:950`) but treats them as opaque callables; an exclusion would leave addresses the transaction graph replays unverified; check 6 compares only recorded entries, so completeness is checked against the derived schema, and an incomplete inventory fails rather than passes |
| Post-warmup check | Core `ModelState.post_warmup()` hook, over `pool_layout`, the recorded projected layout and device, and layout-only pools | Re-verify on every call; verify on the first real step; compare only the component pool layout; skip | Per-call signature checks miss binding-external buffers; a first-step check lets the engine report ready before a mismatch is found; a projection that swaps same-shaped roles or changes their strides leaves every component unchanged; startup access ends with `initialize_state_slots`, so verification reads plain-value layouts |
| Resident pool access | Core startup pool access inside `initialize_state_slots`; capture rows rejected, so no page is stored | Private resident views; a model-owned copy of the pools | Startup access is the authorized source of every page; a copy would need unadmitted memory and its addresses would not serve |
| Scatter warm-up row | A rejected row on the first non-null free block | A rejected row on the null page | The kernel's predicate stores nothing either way; a non-null free block keeps every startup launch inside the rule that capture rows use non-null free blocks |
| Profiling bindings | Capture and retain nothing; compare only serving bindings | Capture against them and replace on serving bindings | Graphs over temporary pools never serve, and a third inventory would need unadmitted memory |
| Warmup fix | Core persistent-state fix | Model-side workaround | It is a core persistent-state defect, independent of graphs |

## Open Questions

1. **No cross-cell graph memory reuse.** Retained memory is the sum of
   per-cell growth. Likely cause, not yet confirmed:
   `GraphRuntime.capture_context` calls `graph_capture(device=device)` without
   a stream (`decode_graph.py:94`), which creates a new stream per call
   (`vllm/distributed/parallel_state.py:1718`); the caching allocator reuses a
   freed block only on the stream that allocated it, and `graph_capture`
   documents that reusing one stream shares free graph-pool blocks
   (`parallel_state.py:1711`). Secondary suspects: tensors kept alive until the
   deferred `consume_batch_stats()` (`transaction_graph.py:1181`), and lazily
   created module buffers.
   **Probe:** capture the nine g0 cells on one persistent stream into a
   dedicated pool, taking `memory_reserved` and `torch.cuda.memory._snapshot()`
   after each capture; repeat with capture contexts cleared and
   `consume_batch_stats()` after each cell. Reuse predicts a total near the
   largest cell's growth plus small per-cell residue; otherwise attribute the
   live pool blocks by allocation trace.
2. **Exactness of `S <= A`.** Allocator rounding against different pool
   addresses may make `S` differ slightly from `A`. **Probe:** across 10
   startups per configuration, record `A - S` and the per-cell growth. Zero
   tolerance stays unless a nonzero, bounded spread is observed.
3. **Double-counted peak.** **Probe:** compare admitted non-KV memory with
   `N()` after startup and after a sustained B63 run, to measure how much
   retained inventory the peak headroom counts twice.
4. **Module globals across same-process engines.** `_batch_stats_slot`
   (`advance.py:2514`) and the class-level platform graph pool
   (`vllm/platforms/interface.py:1330-1335`) persist into a second engine in the
   same process. **Probe:** start, shut down and restart an `EngineCore` in one
   process with the option on; check counters, receipts and pool use counts,
   and that no stale graph is replayed.
5. **Declared-population count against startup budget.** Capture time and
   memory grow linearly with populations (three cells each, captured twice).
   Per-cell eager reference and warm-ups are not part of the 2.2 s figure, and
   the engine-ready timeout is `VLLM_ENGINE_READY_TIMEOUT_S` (default 600 s).
   **Probe:** measure total startup and `A` for 1, 3, 6 and 12 populations at
   g0. Proposal: no fixed cap; configuration rejects a declaration whose
   projected startup, extrapolated from the probe's per-cell cost, exceeds
   half of the ready timeout.
6. **More than one served geometry.** Outside the envelope; each engine
   captures `replay` cells at its one served geometry (One served geometry).
   Open if a mixed-geometry use appears: whether a `replay` cell captured with
   rows of one geometry replays rows of another bitwise. Replay rows read their
   cadence from staged arena data (`advance.py:3196-3199`) and check the book's
   geometry as data (`advance.py:3440`), which suggests no geometry key is
   needed; no test establishes it. Also open: how `multi-geometry` steps are
   counted, and how the CHUNK-bucket binding, which accepts only `[1]`,
   composes. **Probe:** a `replay` cell captured with g0 rows, replaying
   accepted g1 park-echo, FLUSH and forced-endpoint rows, compared with eager
   bitwise.
7. **Lazily bound encoder buffers.** Whether any served stream rebinds them
   after warm-up, which would cause a permanent `signature` fallback.
   **Probe:** a long-stream run with the `signature` counter and the
   lazy-buffer layout logged per step, at each servable geometry. g1 windows
   are wider than g0's; materialization extends `encoder.pos_enc.pe` to its
   cap (`encoder_execution.py:980`), so a g1 stream is expected to leave its
   layout unchanged; the probe must check this on long g1 streams.
8. **Rejection reason of g1 capture rows.** Capture inputs are zero
   embeddings, whose envelope header names geometry 0, while the capture plan
   names the served geometry (`transaction_graph.py:626-666`). Rejection rests
   on terminal bits that hold at every geometry: a CHUNK capture row's zero
   header has version 0, not `ENVELOPE_VERSION`, so it carries
   `ROW_STATUS_ENVELOPE` (`advance.py:2282`, `:2295`), and a `replay` capture
   row is a non-CHUNK row with the placeholder id, so it carries
   `ROW_STATUS_ROLE_MISMATCH` (`advance.py:3401-3403`). Capture headers stay
   zero. At g1 CHUNK capture rows also carry the recoverable
   `ROW_STATUS_GEOMETRY`. The self-check requires only that every capture row is
   rejected and no page is stored. Open: whether capture rows should carry a
   header naming the served geometry, so the rejected-row path is the same at
   g0 and g1.
9. **`multi-geometry` under one served geometry.** Unreachable: the served set
   is a singleton, and `bind_rows` rejects any other geometry before the
   registry mutates. Mixed geometry stays under Open Question 6.
10. **Admission per CHUNK-bucket arm.** With the bucket on, the transaction
    captures record the bucket's eager-body intermediates, so `A` and `S`
    differ between the two arms. Admission is per configuration, so `S <= A`
    is unaffected. **Probe:** report `A` and `S` for each arm.

## Acceptance Criteria

Each criterion names its governing section and its test. Test levels: `CPU`
(CPU re-execution runtime or stubs), `CUDA` (real `torch.cuda.CUDAGraph`
replay), `engine-CUDA` (a spawned `EngineCore` on a GPU). "(new)" marks a test
that does not exist yet. "Eager" means the same call through
`advance_model_rows` with `transaction_graph=None`. "Bitwise equal" means
`torch.equal` on projection rows, row statuses, progress and every resident
page tensor. A "configuration error" is a `ValueError` whose message contains
`transaction_graph_populations`. "`endpoint_enabled` true" and "false" name the
two pool configurations: endpoint history and book pools and `eou_token_id`
present, or all absent (`advance.py:3032-3045`); neither is a step policy.
"Endpoint" always means utterance endpointing, never endpoint plugins.
Arena-only data flow into a replayed cell is a CUDA-level property, observed by
TXG-REP-010, TXG-REP-011, TXG-REP-017 and TXG-REP-019; the CPU re-executing
runtime re-runs the body on the live call's inputs and cannot observe it.
**g0** is the 80 ms cadence (`supported_num_lookahead_tokens` `[0]`) and
**g1** the 160 ms cadence (`[1]`); the "served geometry" is the one geometry an
engine with the option on serves. "CHUNK-bucket binding on" means
`chunk_bucket_graph_populations` is non-empty, which is possible only at g1. A
criterion that names no geometry holds at g0 and at g1, and one that names no
CHUNK-bucket state holds with the binding off and, at g1, with it on; a
criterion whose behaviour depends on either names the configurations it
covers. **M** is the set of three test configurations: g0; g1 with the
CHUNK-bucket binding off; and g1 with `chunk_bucket_graph_populations`
`[1, 31, 63]`. Unless a criterion's test explicitly overrides them, the bucket-on
configurations of M and the other bucket-on test configurations use the following
defaults. They use `max_num_seqs` 64, a test choice that meets every
bucket population's tier condition (TXG-ENV-009): decoder tiers
`execution_tiers(64)`, which are 1, 2, 4, 8, 16, 32 and 64, and an encoder
maximum population of 64, at least the largest bucket population 63. Engine
tests with fewer sessions than 64 keep `max_num_seqs` 64 and size
`num_gpu_blocks_override` to the session count plus one. At the binding-test
level, "CHUNK-bucket binding on" means the call bindings' `bucket_transition`
is not `None`: it is a warmed `ExactChunkGraphBinding`, built over a sealed
`eager-graphed` encoder execution of maximum population at least 63 with the
bucket cells reserved (`reserve_chunk_graph_cells`) and a `DenseGraphBinding`
with tiers `execution_tiers(64)`, with `graph_covers_decode` true and the
model's decode resolver (`build_decode_resolver`) over that decode binding.
`[CPU]` tests of routing, identity, validation and counters may use stand-in
encoder executions and stand-in sealed callables, as the existing signature
tests do (`test_transaction_graph.py:2484-2496`); `[CUDA]` and `[engine-CUDA]`
tests use the real stack. Every criterion's "Configurations" entry names the configurations its
tests run in. A criterion whose behaviour can be exercised under a g1 engine,
with or without the CHUNK-bucket binding, is tested over M; an entry narrower
than M states why the behaviour does not apply in the omitted
configurations.

### Configuration and envelope (ENV)

- **TXG-ENV-001**: Where `transaction_graph_populations` is absent or an empty list, the model shall construct no `TransactionGraphBinding`, capture no transaction graph, and call `advance_model_rows` with `transaction_graph=None` on every step.
  - Design: Supported Envelope
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_option_off_constructs_no_binding` [CPU]
  - Configurations: M
- **TXG-ENV-002**: When `transaction_graph_populations` is non-empty and native burst is enabled or the decode arm is neither `dense-graphed` nor `dense-eager`, the model shall raise a configuration error at construction.
  - Design: Supported Envelope
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_option_requires_native_burst_off_and_dense_decode` [CPU]
  - Configurations: M
- **TXG-ENV-003**: When `transaction_graph_populations` is non-empty and the configured `cudagraph_mode` is not `NONE` (including `PIECEWISE` and `FULL`) or `compilation_config.cudagraph_mm_encoder` is true, the model shall raise a configuration error at construction; the option's checks run before the model's other construction-time mode rejections, so every such rejection names the option. The runner's `needs_cudagraph_capture()` is not observable at model construction; under `NONE` with `cudagraph_mm_encoder` false the runner captures nothing for this model, because its graph manager builds no capture descriptors without a graph mode (`cudagraph_utils.py:291-292`) and no encoder graph manager exists without `cudagraph_mm_encoder` and `SupportsEncoderCudaGraph`, which this model does not implement (`model_states/interface.py:68-82`), so the estimator returns early (`cudagraph_utils.py:1006-1010`).
  - Design: Supported Envelope
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_option_rejects_every_outer_cudagraph_mode_but_none` [CPU]
  - Configurations: M
- **TXG-ENV-004**: When `transaction_graph_populations` is non-empty, the model shall accept `supported_num_lookahead_tokens` exactly when it is one of `[0]`, `[1]`, `(0,)` or `(1,)`, whose element is the `int` 0 (g0, the 80 ms cadence) or 1 (g1, the 160 ms cadence), with no coercion; if the value is absent (every published cadence), is not a list or tuple (for example `0` or `"[1]"`), is empty, has more than one element (for example `[0, 1]` or `[1, 1]`), names another lookahead (for example `[3]`) or holds a non-`int` element (for example `[0.5]`, `[1.0]`, `[True]` or `["0"]`), then the model shall raise a configuration error at construction whose message names the permitted values `[0]` and `[1]` and the rejected declaration.
  - Design: One served geometry: configuration and row admission; Supported Envelope
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_option_accepts_exactly_one_served_geometry` (new; each accepted and each rejected form above parametrized; the message names `[0]`, `[1]` and the declaration) [CPU]
  - Configurations: each declaration form the criterion enumerates, with the CHUNK-bucket binding off; `[0]` is g0 and `[1]` is g1. With the bucket on, the accepted `[1]` is TXG-ENV-009, a rejected form is TXG-ENV-010 and `[0]` is TXG-ENV-011, so those cases are not repeated here
- **TXG-ENV-005**: When `transaction_graph_populations` is non-empty, if it is not a sequence of distinct integers each in `1..min(max_num_seqs, max_num_batched_tokens)`, then the model shall raise a configuration error at construction.
  - Design: Supported Envelope
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_population_declaration_is_validated` (including the `max_num_batched_tokens` bound) [CPU]
  - Configurations: M
- **TXG-ENV-006**: When `transaction_graph_populations` is non-empty and `num_gpu_blocks_override` is set, if any declared population exceeds `num_gpu_blocks_override - 1` (the non-null slot count), then the model shall raise a configuration error at construction.
  - Design: Supported Envelope
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_population_declaration_respects_block_override` [CPU]
  - Configurations: M
- **TXG-ENV-007**: When a request envelope names a geometry outside the configured served geometry set (`supported_num_lookahead_tokens`), with or without `transaction_graph_populations`, `NemotronASRMultiModalProcessor` shall reject that input with an error before the offending input reaches engine execution; for an unadmitted request nothing is admitted, and for an admitted stream (whose empty stream RFC 1 admits before any input) the stream's abort and fenced retirement are the core-owned frontend-preprocessing-failure path; because the rejection happens in the frontend processor, the offending input never reaches `bind_rows` (TXG-ENV-008), and other sessions' isolation follows from that core-owned abort path.
  - Design: One served geometry: configuration and row admission
  - Test: `tests/models/nemotron_asr/test_processor.py::test_envelope_outside_served_geometries_is_rejected` (drives `NemotronASRMultiModalProcessor` directly, the path the in-process engine tests bypass; parametrized as its Configurations entry states) [CPU]
  - Configurations: option off and on × served `[0]` (an envelope naming geometry 1 rejected), served `[1]` (an envelope naming geometry 0 rejected), and served `[1]` with `chunk_bucket_graph_populations` `[1, 31, 63]` (geometry 0 rejected)
- **TXG-ENV-008**: If a row's envelope header names a geometry outside the configured served geometry set, with or without `transaction_graph_populations`, then `bind_rows` shall raise a host structural failure before its delta commit, leaving the session registry unchanged and running no transaction, eager or replayed, for that step.
  - Design: One served geometry: configuration and row admission
  - Test: `tests/models/nemotron_asr/test_plan_local.py::test_bind_rows_rejects_geometry_outside_served_set` (parametrized as its Configurations entry states) [CPU]
  - Configurations: option off and on × served `[0]` (a header naming geometry 1 rejected), served `[1]` (a header naming geometry 0 rejected), and served `[1]` with `chunk_bucket_graph_populations` `[1, 31, 63]` (geometry 0 rejected)
- **TXG-ENV-009**: When `transaction_graph_populations` is non-empty, `supported_num_lookahead_tokens` is `[1]` (g1), `chunk_bucket_graph_populations` is a non-empty list of distinct populations drawn from 1, 31 and 63, the encoder arm is `eager-graphed`, the decoder arm is `dense-graphed`, native burst is disabled, execution is synchronous, non-speculative and single-device with expert parallelism off and no logits processors (`model_config.logits_processors` empty), `max_num_seqs` meets each declared bucket population's tier condition (any value for population 1, at least 32 for population 31, at least 64 for population 63), and the configuration passes every check of TXG-ENV-002 to TXG-ENV-006, the model shall construct both the CHUNK-bucket binding and `TransactionGraphBinding`, whether or not the transaction populations equal the bucket populations, and shall set the `bucket_transition` of its transaction call bindings to that CHUNK-bucket binding, so that every transaction call, both captures and post-warmup verification resolve CHUNK buckets through it; and `_nested_execution_bindings()` shall include the CHUNK-bucket binding while it is on and omit it while it is off.
  - Design: g1 and the CHUNK-bucket binding; Supported Envelope
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_option_composes_with_the_chunk_bucket_binding_at_g1` (new; bucket `[1, 31, 63]` with transaction `[1, 2, 31]`, and bucket `[1, 63]` with transaction `[31]`; bucket `[1]` with `max_num_seqs` 1, and bucket `[1, 31]` with `max_num_seqs` 32; asserts `TransactionCallBindings.bucket_transition` is the model's CHUNK-bucket binding and that `_nested_execution_bindings()` includes it, and with the bucket off that it omits it) [CPU]
  - Configurations: g1 with `chunk_bucket_graph_populations` `[1, 31, 63]` and `[1, 63]` at the default `max_num_seqs` 64, `[1]` with `max_num_seqs` 1 and `[1, 31]` with `max_num_seqs` 32 (each overriding the shared default), and g1 with the bucket off as the control for the nested-binding omission; not g0, because the composition exists only at g1
- **TXG-ENV-010**: When `transaction_graph_populations` and `chunk_bucket_graph_populations` are both non-empty, if the configuration fails any check of TXG-ENV-002 to TXG-ENV-006 (for example `supported_num_lookahead_tokens` `[0, 1]`, or native burst enabled), then the model shall raise the option's configuration error before it constructs the CHUNK-bucket binding.
  - Design: g1 and the CHUNK-bucket binding
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_option_checks_precede_chunk_bucket_construction` (new; `[0, 1]` and native burst enabled, each with the bucket on; the bucket binding is never constructed) [CPU]
  - Configurations: the CHUNK-bucket binding on, with each failing check stated; not bucket off, because precedence over the bucket's construction requires the bucket to be configured
- **TXG-ENV-011**: When `transaction_graph_populations` is non-empty and passes TXG-ENV-002 to TXG-ENV-006, and `chunk_bucket_graph_populations` is non-empty, if `supported_num_lookahead_tokens` is `[0]` (raising "native burst requires the declared 160-ms lookahead profile"); scheduling is asynchronous, a speculative configuration is set, any parallel size is other than 1, expert parallelism is on, or `model_config.logits_processors` is non-empty (each raising "native burst requires synchronous non-speculative single-device execution"); the decoder arm is `dense-eager` (raising "CHUNK serving pilot requires distinct populations 1/31/63 and a dense graph decoder"); `max_num_seqs` is below a declared bucket population (31 or 63; raising "decode population N exceeds captured maximum M"); `max_num_seqs` is exactly 31 with population 31 declared or exactly 63 with population 63 declared (raising "CHUNK pilot requires unchanged physical decoder tiers 1/32/64"); or the encoder arm is not `eager-graphed` (raising "CHUNK ownership requires an untouched native encoder domain"), then model construction shall fail with the CHUNK-bucket binding's own `ValueError`, which need not name `transaction_graph_populations`, and no `TransactionGraphBinding` shall serve.
  - Design: g1 and the CHUNK-bucket binding
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_chunk_bucket_requirements_raise_the_bucket_error` (new; each failing requirement parametrized with its message, including expert parallelism on, a non-empty `logits_processors`, `max_num_seqs` 16 with bucket `[31]`, 48 with bucket `[63]`, 31 with bucket `[31]` and 63 with bucket `[63]`; the error is the bucket binding's, and the `[0]` case asserts the native-burst message, not a CHUNK message) [CPU]
  - Configurations: the CHUNK-bucket binding on, with each failing requirement stated (including `[0]`, the only g0 case); not bucket off, because the errors belong to the bucket binding
- **TXG-ENV-012**: When `TransactionGraphBinding` is constructed with a served geometry set, which the model passes as `served_geometry_ids(hf_config)`, the `CADENCES` indices that the processor and `bind_rows` also use (an index equals its lookahead only for g0 and g1; TXG-ENV-004 validates lookahead values), it shall accept exactly `(0,)` and `(1,)`, and if the set is any other, for example `()`, `(2,)` or `(0, 1)`, then it shall raise a configuration error whose message names the permitted sets `(0,)` and `(1,)` and the rejected set.
  - Design: One served geometry: configuration and row admission
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_binding_rejects_served_sets_other_than_g0_or_g1` (new; `(0,)` and `(1,)` accepted; `()`, `(2,)`, `(0, 1)` rejected with the message contents) [CPU]
  - Configurations: the served sets `(0,)` (g0) and `(1,)` (g1) accepted and each enumerated set rejected; the bucket binding is not an input to this constructor (it reaches the transaction only as a call binding, TXG-ENV-009), so bucket state does not apply

### Memory admission (MEM)

- **TXG-MEM-001**: Where `transaction_graph_populations` is absent or empty, `prepare_execution_memory` shall perform no transaction-graph admission capture, allocate no stand-in storage and create no admission holder.
  - Design: Ordering
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_option_off_performs_no_admission` [CPU]
  - Configurations: M
- **TXG-MEM-002**: When `transaction_graph_populations` is non-empty, `prepare_execution_memory` shall, after the nested inventory is ready and before the eager profile transactions, capture every declared cell (never a subset or an extrapolation) over stand-in storage of `max(populations) + 1` slots with the real persistent-state layout into a dedicated graph pool, using the same capture path, cell order, capture stream and replay self-check as serving capture.
  - Design: Ordering
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_admission_captures_full_inventory_before_eager_profiles` (call order and captured-cell inventory) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_admission_and_serving_capture_share_cell_order` (capture stream identical across both inventories and the admission graph pool handle distinct from the platform and nested pools) [CUDA]
  - Configurations: M; with the bucket on, the call-order test also asserts that bucket publication precedes admission
- **TXG-MEM-003**: When `transaction_graph_populations` is non-empty, every admission reading `N()` (`N0`, `Na`, `N1`, `Ns0`, `Ns1`) shall be a `mem_get_info` free-memory value taken after a device synchronize, `gc.collect()` and an allocator `empty_cache()`; `Na` shall be read after every capture-only temporary (page snapshots, expected outputs, capture and verify callbacks, capture inputs) is cleared and while the admission holder still retains the stand-in storage, and `N1` after that storage is released; the model shall read the allocator's allocated bytes on the device (`torch.cuda.memory_allocated`) immediately after the `Na` and the `N1` readings, `Na` being the settled point at which the allocated-bytes reading for the release check is taken, and no admitted figure shall use `Na`'s free memory; and the model shall compute `A = N0 - N1` and `S = Ns0 - Ns1` from free memory alone.
  - Design: Normalized measurement; Ordering
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_normalized_readings_synchronize_collect_and_empty_first` (also the order: temporaries cleared, `Na` and its allocated-bytes reading, stand-in release, `N1` and its allocated-bytes reading; release verified with no free-memory rise at release) [CPU, patched device calls]
  - Configurations: M
- **TXG-MEM-004**: When `transaction_graph_populations` is non-empty, if the allocator's allocated bytes drop by less than the stand-in storage bytes between the readings taken with `Na` and `N1`, or any owning reference to the stand-in storage survives its release, then startup shall fail with an error naming `transaction_graph_populations` and reporting both figures (the allocated-bytes drop and the stand-in bytes).
  - Design: Ordering
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_unverified_admission_release_fails_startup` (injected allocated-bytes readings one byte short while free memory rises past the stand-in bytes, and a weak reference to the stand-in storage kept alive by an injected owner) [CPU]
  - Configurations: M
- **TXG-MEM-005**: When `transaction_graph_populations` is non-empty, `capture_transaction_graphs` shall release the admission holder and its graph pool before reading `Ns0`, and shall capture the serving inventory into a fresh dedicated graph pool distinct from the admission pool.
  - Design: Ordering
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_serving_capture_releases_admission_before_measuring` (release before the `Ns0` read, call order) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_serving_pool_is_distinct_from_released_admission_pool` (pool handles differ, `Ns0` read after the admission pool is released) [CUDA]
  - Configurations: M
- **TXG-MEM-006**: When `transaction_graph_populations` is non-empty, startup shall succeed only if every declared cell is published and `S <= A` with zero tolerance; if `S > A` by any amount, then the binding shall discard every graph and `initialize_kv_cache` shall raise an error naming `transaction_graph_populations` and reporting `S` and `A`.
  - Design: Ordering
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_serving_bytes_above_admission_fail_startup` [CPU, injected readings including `S = A + 1`]
  - Configurations: M
- **TXG-MEM-007**: When `transaction_graph_populations` is non-empty and neither `num_gpu_blocks_override` nor `kv_cache_memory_bytes` is set, `non_kv_cache_memory` from the memory-profiling window shall include the retained admission inventory and the admission preparation and capture peak, with the peak headroom left as the profiler's own measurement (the model resets no peak memory statistics during `prepare_execution_memory`) and no deduction for retained memory the peak headroom also counts, so resident capacity is fixed from the remaining budget; and the admission holder shall be alive when `prepare_execution_memory` returns, stay alive through the eager profile transactions and the window close, and be released only in `capture_transaction_graphs`.
  - Design: Budget mechanics; Ordering
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_admission_holder_lives_until_serving_capture` (event order of holder release against `prepare_execution_memory` return, eager profiles, window close and `capture_transaction_graphs`) [CPU]; `tests/v1/engine/test_nemotron_asr.py::test_nemotron_engine_transaction_graph_admission_is_charged` (option-on minus option-off, where the option-off arm runs twice before the option-on arm, takes admission's normalized readings at the admission point, and is represented by its second run because the first engine in a process also creates process-wide device memory that later engines start with; the spread between the two option-off runs is reported, with no tolerance below `A`: `total_consumed >= A` for retention; in every run, no peak-statistics reset inside `prepare_execution_memory` and a KV budget equal to requested memory less `total_consumed` and the profiler's `transient_peak_headroom`; the `non_kv_cache_memory` and `transient_peak_headroom` differences reported, not bounded, because the profiler may observe a smaller transient peak above the retained floor; and a memory-bound run whose block count equals the count core derives from the reduced budget) [engine-CUDA]
  - Configurations: M (the option-off arms of each configuration run the same configuration)
- **TXG-MEM-008**: When `transaction_graph_populations` is non-empty and `num_gpu_blocks_override` or `kv_cache_memory_bytes` is set, the model shall still measure and report `A` and check `S <= A` under either override, and shall not itself change the resident block count, which core derives from the override once its budget validation accepts the configuration; and if serving capture then raises or runs out of memory, then startup shall fail with an admission-failure error naming `transaction_graph_populations`. Planner rejection of an oversized override is TXG-MEM-011 and KV-allocation failure is TXG-MEM-012; core reports both.
  - Design: Capacity overrides
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_override_capacity_reports_admission_and_wraps_failures` (`A` and the `S <= A` check under each override, the model requesting no block-count change, and serving-capture failure) [CPU]
  - Configurations: M
- **TXG-MEM-009**: When `transaction_graph_populations` is non-empty, if admission capture raises or runs out of memory inside the profile window, then `prepare_execution_memory` shall raise an error naming `transaction_graph_populations` and latch `_execution_memory_failed`, and for an out-of-memory error the message shall report the normalized bytes admitted so far, the cells captured, the failing cell label and the device's free and total memory, and no projected total for the unfinished inventory.
  - Design: Failure modes and diagnostics
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_admission_oom_latches_and_reports_progress` [CPU, injected OOM]
  - Configurations: M
- **TXG-MEM-010**: When `transaction_graph_populations` is non-empty, if serving capture raises, or the usable slot count after capacity is fixed is less than `max(populations)`, then the binding shall discard every graph and `initialize_kv_cache` shall raise an error naming `transaction_graph_populations`, and no step shall be served eager in its place.
  - Design: Lifetime: one serving capture per worker; Failure modes and diagnostics
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_failed_capture_publishes_nothing_and_fails_startup` (serving capture raises naming `transaction_graph_populations`, publishes nothing and latches the binding, so a later call raises instead of running eager) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_serving_capture_rejects_too_few_usable_slots` [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_serving_capture_requires_admission_on_cuda` (binding level: without admission, the CUDA serving-capture guard raises naming `transaction_graph_populations` and latches the binding) [CPU]
  - Configurations: M
- **TXG-MEM-011**: When `transaction_graph_populations` is non-empty, if core budget validation rejects the persistent-state budget after admission, because fewer than two pages fit, a set `num_gpu_blocks_override` exceeds the remaining budget, or automatically fitting `max_model_len` fails, then startup shall fail with a `ValueError` naming the model-supplied contributor label (for this model, `transaction_graph_populations`), and no step shall be served eager in its place. The model supplies the label as its state spec's `memory_contributors` while the option is on; when the option is off, the spec names no contributor.
  - Design: Failure modes and diagnostics; Core Obligations, item 4 (delivered)
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_state_spec_names_the_option_as_a_memory_contributor` (option on and off; layout unchanged) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_planner_rejection_names_the_option` (the served state spec through core's planner: a one-page budget and an oversized override, option on and off) [CPU]; `tests/v1/core/test_kv_cache_utils.py::test_persistent_state_rejection_names_memory_contributors` (generic label; capacity, override and automatic-fit rejections, parametrized) [CPU]
  - Configurations: M for the model-level tests; core's generic-label tests (`tests/v1/core/test_kv_cache_utils.py`) run no Nemotron model, so geometry and bucket state do not apply to them
- **TXG-MEM-012**: When `transaction_graph_populations` is non-empty, if the KV allocation in `initialize_kv_cache` raises or runs out of memory before `initialize_state_slots`, then startup shall fail with the same exception, noted with the model-supplied contributor label (for this model, `transaction_graph_populations`) so that the worker's logged traceback names it, and no step shall be served eager in its place.
  - Design: Capacity overrides; Failure modes and diagnostics; Core Obligations, item 5 (delivered)
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_state_spec_names_the_option_as_a_memory_contributor` (the label the served spec carries) [CPU]; `tests/v1/worker/test_gpu_model_runner_v2.py::test_state_kv_allocation_failure_names_memory_contributors` (generic label; injected allocation error and out-of-memory, profiling and serving pools, parametrized) [CPU]; `tests/v1/core/test_kv_cache_utils.py::test_persistent_state_memory_contributors_are_diagnostic` (the label survives planning and the copy that rebuilds the spec as pickle does) [CPU]
  - Configurations: M for the model-level test; core's generic-label tests (`tests/v1/worker/test_gpu_model_runner_v2.py`, `tests/v1/core/test_kv_cache_utils.py`) run no Nemotron model, so geometry and bucket state do not apply to them

### Capture lifetime (LIFE)

- **TXG-LIFE-001**: When `transaction_graph_populations` is non-empty, the model shall run serving capture exactly once per worker, from `initialize_state_slots` with serving bindings (`is_profiling` false) after the resident pools are allocated and before any session is admitted, and `binding.ready` shall become true only after every declared cell is published.
  - Design: Lifetime: one serving capture per worker
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_capture_input_stores_no_page_and_publishes_every_cell` [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_serving_capture_runs_once_from_initialize_state_slots` [CPU]
  - Configurations: M
- **TXG-LIFE-002**: When `transaction_graph_populations` is non-empty and `initialize_state_slots` repeats with serving bindings whose pool layout and call bindings are identical to the published ones, the binding shall perform no capture, and its graph objects, `captured_cells` count and receipt capture order shall be unchanged.
  - Design: Lifetime: one serving capture per worker
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_identical_reinitialize_is_a_verified_no_op` [CPU]
  - Configurations: M
- **TXG-LIFE-003**: When `transaction_graph_populations` is non-empty, if `initialize_state_slots` repeats with serving bindings that differ from the published ones in pool layout or call bindings, then the model shall raise a worker-fatal `RuntimeError` without capturing, and after startup the binding shall never capture a cell, so `captured_cells` stays constant across signature fallbacks.
  - Design: Lifetime: one serving capture per worker
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_changed_pool_layout_on_reinitialize_is_fatal` [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_relocated_pools_fall_back_instead_of_replaying` (`captured_cells` unchanged across the fallback) [CPU]
  - Configurations: M
- **TXG-LIFE-004**: When `transaction_graph_populations` is non-empty and a captured cell (admission or serving) passes verification, the binding shall clear its capture context so that the cell holds no reference to the capture `TransactionInputs`, the pools reached through them, page snapshots, expected outputs, or capture and verify callbacks; for admission cells this clearing shall complete before `Na` is read, and the stand-in storage shall then be retained only by the admission holder's explicit reference until its release after `Na`.
  - Design: Capture-lifetime rule; Ordering
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_published_cell_retains_no_capture_context` (weak references dead after `gc.collect()`, and for admission the only live referrer of the stand-in storage at `Na` is the holder) [CPU]
  - Configurations: M
- **TXG-LIFE-005**: When `transaction_graph_populations` is non-empty, the model shall hold admission graphs only in an admission holder that is absent from `_graphs` and `_pool_signature`, has no replay path, and leaves `binding.ready` false and the replay counters at zero.
  - Design: Capture-lifetime rule
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_admission_holder_is_never_published` [CPU]
  - Configurations: M
- **TXG-LIFE-006**: When `transaction_graph_populations` is non-empty and `TransactionGraphBinding.discard()` runs, whether the binding published, only admitted, or already discarded, the binding shall drop its graphs, admission inventory, arenas and owned buffers, leaving no published cell, `binding.ready` false and no live reference to any arena.
  - Design: Lifetime: one serving capture per worker
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_discard_leaves_no_published_cell_or_arena` (also a second `discard()`; weak references dead after `gc.collect()`) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_model_state_shutdown_discards_the_binding` (the admitted-only binding) [CPU]
  - Configurations: M
- **TXG-LIFE-007**: When `transaction_graph_populations` is non-empty and core calls `ModelState.shutdown()` from `GPUModelRunner.shutdown`, the model state shall call the binding's `discard()` (TXG-LIFE-006), and the hook shall return without raising for a model state whose initialization did not complete, a model without a binding, and a binding that never captured.
  - Design: Lifetime: one serving capture per worker; Core Obligations, item 3 (delivered)
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_model_state_shutdown_discards_the_binding` (served, admitted-only, option-off and uninitialized model states) [CPU]; `tests/v1/worker/test_gpu_model_runner_v2.py::test_shutdown_calls_model_state_hook_before_releasing_state` and `::test_shutdown_hook_tolerates_incomplete_initialization` (core, generic model state) [CPU]; `tests/v1/engine/test_nemotron_asr.py::test_nemotron_engine_transaction_graph_matches_eager` (`EngineCore.shutdown` discards the binding once) [engine-CUDA]
  - Configurations: M, through the engine test that TXG-REP-013 parametrizes, which keeps the shutdown assertions in every configuration; core's generic-model-state tests run no Nemotron model
- **TXG-LIFE-008**: When `transaction_graph_populations` is non-empty, the model shall reach the unowned resident pools for serving capture and the rejected-row reference only through `StateSlotBindings.startup_pool_access(layer)`, request it only inside `initialize_state_slots` and keep no access object after that call returns; every launch over those pools, the commit-scatter warm-up included, shall run only rows the transaction rejects, on non-null free blocks, and store no resident page, the null page included; post-warmup verification shall read only `StateSlotBindings.pool_layout(layer)` (address, shape, strides and dtype as plain values), never a resident tensor.
  - Design: Lifetime: one serving capture per worker; Post-Warmup Verification; Core Obligations, item 6 (delivered)
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_capture_reaches_pools_only_through_the_startup_accessor` (no `views._views` read from the model during capture or `post_warmup`; every page, the null page included, seeded with distinct finite nonzero sentinels and bitwise unchanged afterwards; no dispatched op writes resident storage and no `Tensor.numpy` or `Tensor.__dlpack__` export of it is taken, even one that would store the values a page holds; no access object alive afterwards; capture outside a startup access window fails) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_scatter_warmup_names_a_non_null_free_block` (the admission and serving scatter warm-ups each name a non-null block inside the pools) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_startup_access_stores_no_resident_page` (sentinel-seeded pages bitwise unchanged after the scatter warm-up, serving capture and `post_warmup`, and every warm-up descriptor a rejected row on a non-null block) [CUDA]; `tests/v1/worker/test_gpu_model_runner_v2.py::test_startup_pool_access_expires_after_initialize_state_slots` (core, generic model state) [CPU]
  - Configurations: M; with the bucket on, each CHUNK cell's eager reference at a configured bucket population launches a bucket graph inside the access window, and the no-store assertions cover it; `replay` cells and CHUNK cells at other populations invoke no bucket graph; core's generic test runs no Nemotron model
- **TXG-LIFE-009**: When `transaction_graph_populations` is non-empty and `initialize_state_slots` receives bindings whose `is_profiling` is true, the model shall capture nothing, request no startup pool access, leave `binding.ready` false and the admission holder alive, and keep no reference to the profiling pools once serving bindings replace them; the first serving bindings shall then capture as the first serving capture (TXG-LIFE-001), without comparison with the profiling bindings (TXG-LIFE-002, TXG-LIFE-003).
  - Design: Lifetime: one serving capture per worker; Core Obligations, item 6 (delivered)
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_profiling_bindings_capture_and_retain_nothing` [CPU]
  - Configurations: M

### Replay exactness (REP)

- **TXG-REP-001**: When `transaction_graph_populations` is non-empty and the served geometry is `G` (0 for g0, 1 for g1), the binding shall declare exactly the cells `chunk-fresh:g<G>:b<P>`, `chunk-continuing:g<G>:b<P>` and `replay:b<P>` for each declared population `P`, and no other cell.
  - Design: Cells and classification; One served geometry: configuration and row admission
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_capture_input_stores_no_page_and_publishes_every_cell` (served-geometry labels, parametrized over g0 and g1) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_receipt_captures_largest_population_first_and_reports_cost` [CPU]
  - Configurations: M; TXG-REP-009 shares the capture test
- **TXG-REP-002**: When `transaction_graph_populations` is non-empty, `classify_transaction` shall return the `chunk-fresh` cell for one CHUNK bucket whose rows are all first CHUNKs, the `chunk-continuing` cell for one CHUNK bucket with no fresh row, and the `replay` cell for steps with no CHUNK row and no fresh row, each with population equal to the number of rows in the step (admitted, parked or never-fed slots without a row in the step never counted), and otherwise `(None, reason)` with reason `empty`, `fresh-non-chunk`, `multi-geometry`, `mixed-roles` or `mixed-fresh` as defined in the reason table.
  - Design: Cells and classification
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_classification_covers_declared_shapes_only` (every kind and reason, in a step taken while other admitted, parked and never-fed slots are bound) [CPU]
  - Configurations: M
- **TXG-REP-003**: When `transaction_graph_populations` is non-empty and a call has no cell, has an undeclared population, matches a declared cell with no published graph (`uncaptured`), or runs in capture or memory-profile mode, `execute` shall run the eager transaction with results bitwise equal to eager, increment `fallbacks:<cell label or uncovered>:<reason>` by exactly one, and leave `replays:<cell label>` unchanged; once startup has succeeded, no call shall be counted `uncaptured`; and because native burst is off whenever the option is on (TXG-ENV-002), no call shall be counted `native-burst`.
  - Design: Cells and classification; Execute order
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_uncovered_calls_run_eager_and_are_counted` (parametrized over every reachable reason) [CPU]
  - Configurations: M
- **TXG-REP-004**: When `transaction_graph_populations` is non-empty, `transaction_signature` shall differ whenever exactly one of population, control ids, `endpoint_enabled`, `n_fresh`, decode or transition callable identity, resident pool address, shape or stride, lazily bound encoder buffer layout, input shapes, or arena name/dtype/shape layout differs, and shall be equal for calls that differ only in arena data (slot ids, block indices, row values).
  - Design: Inputs, outputs and signature; Host control flow inside the body
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_signature_covers_every_structural_input` [CPU]
  - Configurations: M
- **TXG-REP-005**: When `transaction_graph_populations` is non-empty and a call matches a declared cell whose captured signature, including arena layout, differs from the call's, `execute` shall detect the mismatch before any arena upload (both pinned host buffers and the device arena byte-identical), run eager over the live pools with results, including committed pages, bitwise equal to eager (and where eager raises for that input, for example the CHUNK-bucket binding's sealed transition rejecting an invocation that differs from its captured cell, `chunk_bucket_graph.py:232-242`, propagate that exception after the counter increments, without latching the binding, so a later call is classified normally), leave captured storage that is distinct from the live pools untouched, and increment `fallbacks:<cell label>:signature` by one.
  - Design: Inputs, outputs and signature; Echo eligibility, mixed steps and signature mismatch; Execute order
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_relocated_pools_fall_back_instead_of_replaying` [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_arena_layout_mismatch_is_a_signature_fallback` (execution-level, live-pool commit bitwise equal to eager; with the bucket on at population 31, a call whose `inputs_embeds` is one column wider than captured, which changes the signature's input-embeddings entry and the bucket's captured envelope signature, raises the bucket's `ValueError` after `fallbacks:<cell label>:signature` increments, the binding unlatched and a following valid call replayed) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_arena_rejects_vectors_outside_its_layout` (arena-level guard) [CPU]
  - Configurations: M
- **TXG-REP-006**: When `transaction_graph_populations` is non-empty, the dense decode binding and the CHUNK-bucket binding's sealed transitions shall resolve to the identical callable object (same `id`) for the same graph key across calls for the binding's lifetime; the CHUNK-bucket binding's eager path, resolved for a population outside its set, shall have the same normalized identity `(id(__self__), id(__func__))` on every resolution although each resolution returns a new bound-method object; and the CHUNK-bucket binding shall reject a callable it did not seal, so the signature of two structurally identical calls is equal.
  - Design: Callable identity
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_signature_callable_identities_are_stable_across_calls` (also repeated resolution at population 2 with bucket `[1, 31, 63]`: equal normalized identity and equal signatures) [CPU]
  - Configurations: M for the dense decode binding's callable identity; the CHUNK-bucket binding's identity at g1 with `chunk_bucket_graph_populations` `[1, 31, 63]` (its test resolves geometry 1), not at g0 or bucket off, where no bucket binding exists
- **TXG-REP-007**: While `outer_transaction_capture` is active, `nested_graph_mode` shall return the eager mode at every nested launch site (encoder, decode and CHUNK bucket) and any host-to-device staging shall raise; while it is inactive, `nested_graph_mode` shall return the configured mode.
  - Design: Flat capture
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_nested_bindings_run_eagerly_only_inside_the_capture_scope` (drives each binding's launch entry point, encoder, decode and CHUNK bucket, and asserts no nested graph launch inside the scope) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_host_staging_is_rejected_inside_the_capture_scope` [CPU]
  - Configurations: M for the encoder and decode launch sites; the CHUNK-bucket launch site at g1 with `chunk_bucket_graph_populations` `[1, 31, 63]`, not at g0 or bucket off, where no bucket binding exists
- **TXG-REP-008**: When `transaction_graph_populations` is non-empty and a cell replays, `execute` shall perform exactly one host-to-device copy into the cell's arena, alternate its two pinned host buffers across calls, and rewrite a host buffer only after the event fencing that buffer's previous upload has completed, with uploaded bytes equal to live staging.
  - Design: Inputs, outputs and signature
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_arena_alternates_host_buffers_across_calls` (explicit event stubs, since CPU arenas hold no events, count copies and show each buffer's wait preceding its rewrite) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_arena_stager_matches_live_staging_bit_identical` [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_arena_reuses_pinned_buffers_after_async_upload` (pinned host buffers, asynchronous uploads, uploaded bytes equal to live staging across reuse) [CUDA]
  - Configurations: M
- **TXG-REP-009**: When `transaction_graph_populations` is non-empty and a cell is captured, the binding shall run the eager reference on capture rows the transaction rejects (leaving every resident page byte-identical), then three eager warm-ups, the capture and one replay, and shall publish the cell only if that replay reproduces the reference outputs and pages bitwise; if it does not, then the cell shall not be published and startup shall fail with an error naming the cell label.
  - Design: Capture and the replay-cell oracle
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_capture_input_stores_no_page_and_publishes_every_cell` [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_capture_self_check_mismatch_publishes_nothing` [CPU]
  - Configurations: M, through TXG-REP-001's capture test and the self-check mismatch test
- **TXG-REP-010**: When `transaction_graph_populations` is non-empty and accepted CHUNK rows replay a `chunk-fresh` or `chunk-continuing` cell at the served geometry at populations 1, 31 and 63, with `endpoint_enabled` true and false and with block ids, row orders and prompts that differ from the capture inputs, results shall be bitwise equal to eager in each of these configurations: g0; g1 with the CHUNK-bucket binding off; g1 with `chunk_bucket_graph_populations` `[1, 31, 63]`; and g1 with `chunk_bucket_graph_populations` `[1, 63]`, so that population 31 resolves the bucket binding's eager path while eager steps at 1 and 63 launch the bucket graph. In a homogeneous lockstep workload (every session sends its CHUNK in the same step, so each CHUNK step is one all-fresh or all-continuing bucket at a declared population), every CHUNK step shall replay its cell with zero CHUNK-step fallbacks in each of those configurations, and echo steps at undeclared populations are counted `undeclared`.
  - Design: Context and Design Philosophy; Invariants; Capture and the replay-cell oracle; g1 and the CHUNK-bucket binding
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_population_lockstep_replays_every_chunk` (endpointing enabled and disabled; parametrized over the four configurations) [CUDA]; `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_graph_replay_matches_eager_across_varied_plans` (parametrized over the four configurations) [CUDA]
  - Configurations: M, and g1 with `chunk_bucket_graph_populations` `[1, 63]`
- **TXG-REP-011**: When `transaction_graph_populations` is non-empty and a `replay` cell replays control rows, at g0, g1 with the CHUNK-bucket binding off, and g1 with `chunk_bucket_graph_populations` `[1, 31, 63]`, results shall be bitwise equal to eager with zero fallbacks for accepted park-echo and FLUSH rows with `endpoint_enabled` true and false, and for accepted forced-endpoint rows with `endpoint_enabled` true; with `endpoint_enabled` false, a forced-endpoint row shall be rejected with the same row status, outputs and unchanged pages as eager. Utterance endpointing is a model capability, independent of whether any endpoint plugin is loaded.
  - Design: Capture and the replay-cell oracle; Host control flow inside the body
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_replay_cell_accepted_control_rows_match_eager` (accepted park-echo and FLUSH × `endpoint_enabled` true and false, accepted forced-endpoint with `endpoint_enabled` true, rejected forced-endpoint with `endpoint_enabled` false; parametrized over M) [CUDA]
  - Configurations: M
- **TXG-REP-012**: When `transaction_graph_populations` is non-empty and a covered step at g0, g1 with the CHUNK-bucket binding off, or g1 with `chunk_bucket_graph_populations` `[1, 31, 63]`, contains rows the transaction rejects (out-of-sequence CHUNK, wrong echo token, corrupt book, and at g1 a final-tail row with at least `C·hop` valid samples (2,560 at hop 160), which carries `ROW_STATUS_FINAL_OVERSIZE`), replay shall produce the same nonzero row statuses, outputs and pages as eager, leaving the rejected rows' pages unchanged and committing the other rows; commit-sink collection shall report the same failed request ids as eager, which `postprocess_state` passes to `state_slots.fail_requests` (the step's `failed_req_ids`); the step shall not raise, and the binding shall not latch.
  - Design: Capture and the replay-cell oracle; Lookahead rows at g1; Failure Containment
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_graph_replay_rejects_corrupt_rows_like_eager` (`failed_req_ids` equal between arms; no latch; parametrized over M, with the oversized final tail at g1) [CUDA]; `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_one_corrupt_row_in_a_population_is_rejected_alone` (likewise) [CUDA]
  - Configurations: M
- **TXG-REP-013**: When `transaction_graph_populations` is `[1]` or `[31]` through an `EngineCore`, at g0, at g1 with the CHUNK-bucket binding off, and at g1 with `chunk_bucket_graph_populations` `[1, 31, 63]`, the emitted tokens, echo counts and resident storage after every park shall be bitwise equal to the same configuration run with the option off; in the homogeneous lockstep workload (exactly the declared population of sessions, all sending each CHUNK in the same step), every CHUNK step shall replay its declared cell with zero CHUNK-step fallbacks, and echo steps at undeclared populations are counted `undeclared`; mixed or staggered steps keep eager equivalence and are counted under their fallback reasons.
  - Design: Context and Design Philosophy; g1 and the CHUNK-bucket binding
  - Test: `tests/v1/engine/test_nemotron_asr.py::test_nemotron_engine_transaction_graph_matches_eager` (also the per-chunk echo counts of both arms; parametrized over the three configurations, with `max_num_seqs` 64 and `num_gpu_blocks_override` equal to the session count plus one in the bucket-on configuration rather than `max_num_seqs` equal to the session count; that configuration captures `eager-graphed` encoder cells for populations 1 to 64 and profiles a 64-row transaction, so the test sets `gpu_memory_utilization` explicitly to hold them with the admission inventory, records non-KV memory, `A` and `S`, and treats a capacity failure at startup as a failed test configuration, not a pass) [engine-CUDA]; `tests/v1/engine/test_nemotron_asr.py::test_nemotron_engine_transaction_graph_b31_lockstep_matches_eager` (parametrized likewise) [engine-CUDA]
  - Configurations: M
- **TXG-REP-014**: When `transaction_graph_populations` is non-empty, before `execute` classifies the call, and therefore before it replays or returns from any fallback (uncovered, undeclared, mixed-row or signature), it shall compare the live batch-invariant mode, the per-operation `fp32_precision` settings of `torch.backends.cuda.matmul`, `torch.backends.cudnn.conv` and `torch.backends.cudnn.rnn` (which also reflect `torch.set_float32_matmul_precision()`), and `CUBLAS_WORKSPACE_CONFIG` with their capture-time values using host-only reads that never call the legacy getters (`torch.get_float32_matmul_precision()` and the `allow_tf32` flags), and if any differs, then it shall raise a worker-fatal `RuntimeError` naming the changed constant without replaying, running eager or incrementing any counter.
  - Design: Process constants and excluded paths; Execute order
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_process_constant_change_is_fatal_on_every_path` (parametrized over constant × path) [CPU]
  - Configurations: M
- **TXG-REP-015**: When `transaction_graph_populations` is non-empty and a cell replays, the arena upload, static-input copies, the replay and every resident write shall be issued on the runner's current stream.
  - Design: Invariants
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_replay_runs_on_the_current_stream` (non-default stream, pages bitwise equal after synchronizing only that stream, plus stream tracing of every copy, replay and write, for example `torch.profiler` events or an interception of the copy and replay entry points, that fails on work issued to any other stream) [CUDA]
  - Configurations: M
- **TXG-REP-016**: When `transaction_graph_populations` is non-empty, no two published cells shall share an arena buffer, `static_ids`, `static_embeds` or owned output buffer (disjoint storage), and owned outputs shall be allocated outside the graph pool.
  - Design: Inputs, outputs and signature; Invariants
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_cells_share_no_arena_static_input_or_owned_output` (storage disjointness) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_owned_outputs_are_allocated_outside_the_graph_pool` (allocator provenance, for example `torch.cuda.memory._snapshot()` segment pool ids, for rows, status and progress) [CUDA]
  - Configurations: M
- **TXG-REP-017**: When `transaction_graph_populations` is non-empty and eager fallback steps interleave with replayed steps, at g0, at g1 with the CHUNK-bucket binding off, and at g1 with `chunk_bucket_graph_populations` `[1, 31, 63]`, results shall be bitwise equal to the same sequence run entirely eager. The eager steps launch the nested encoder and decode bindings and, with the bucket binding on, the bucket graph for each CHUNK bucket at one of its populations, while the replayed steps run the bucket's eager body recorded in the transaction graph, over the same nested-binding scratch. In the bucket-on configuration the sequence uses transaction populations `[2, 31]` and interleaves: replayed 31-row CHUNK steps (the bucket's eager body for population 31); eager `mixed-roles` steps of 31 CHUNK rows and one echo row, whose 31-row bucket launches the bucket graph at 31 over the same scratch; eager `undeclared` one-row CHUNK steps, which launch the bucket graph at 1; and replayed two-row CHUNK steps, whose bucket resolves the bucket's eager path.
  - Design: Invariants; g1 and the CHUNK-bucket binding
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_interleaved_fallbacks_and_replays_match_eager` (parametrized over the three configurations; the bucket-on sequence as stated) [CUDA]
  - Configurations: M
- **TXG-REP-018**: When `transaction_graph_populations` is non-empty, the binding shall build the capture rows of every cell, `replay` cells included, with plan geometry equal to the served geometry, in the admission inventory, the serving inventory and the cell inputs that post-warmup verification rebuilds; and a `replay` cell's key and label shall carry no geometry.
  - Design: One served geometry: configuration and row admission; Capture and the replay-cell oracle
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_capture_rows_carry_the_served_geometry` (new; every cell kind, both inventories and the verification rebuild, parametrized over M) [CPU]
  - Configurations: M
- **TXG-REP-019**: When `transaction_graph_populations` is `[1, 2, 31]`, the served geometry is g1, and the CHUNK-bucket binding is off, or `chunk_bucket_graph_populations` is `[1, 31, 63]` (bucket populations 1 and 31; population 2 outside its set), or `chunk_bucket_graph_populations` is `[1, 63]` (bucket population 1; populations 2 and 31 outside its set), these accepted rows shall replay with results bitwise equal to eager in each of the three configurations: (a) a one-chunk session's row (chunk sequence 0, final tail with fewer than `C·hop` valid samples (2,560 at hop 160)) in a `chunk-fresh` cell, alone at population 1 and beside regular first chunks at populations 2 and 31; (b) a continuing final-tail row with fewer than `C·hop` valid samples (2,560 at hop 160) beside regular continuing rows in one `chunk-continuing` cell at populations 2 and 31; and (c) at populations 1 and 31, after each session's final CHUNK, each of its REPLAY rows in a `replay` cell, the emitting ones advancing the queue head and the drained one parking with its pending echo cleared, followed by its FLUSH in a `replay` cell, which emits park and leaves the queue and book unchanged.
  - Design: Lookahead rows at g1; Host control flow inside the body; Capture and the replay-cell oracle
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_g1_lookahead_rows_match_eager` (new; cases (a) to (c) at the stated populations × the three configurations) [CUDA]
  - Configurations: g1 with the CHUNK-bucket binding off, with `[1, 31, 63]` and with `[1, 63]`; not g0, because the lookahead frame and the 2,560-sample bound exist only at g1

### Post-warmup verification (VER)

- **TXG-VER-001**: When `transaction_graph_populations` is non-empty and the model's post-warmup verification runs, it shall compare with their values recorded at publication the live `pool_layout`, the device of the core's encoder parameters (against the device the resident pools were captured on), the projected resident layout rebuilt from the live `pool_layout` (for each transaction role, the component pool it starts in, its address, byte offset in that component, shape, strides and dtype), every published cell's signature rebuilt over layout-only pools (all entries but the resident-pool and input-layout entries), the live call bindings, and the address and size of every binding-external buffer the graphs read (the core's parameters and buffers, the entry scratch of the nested encoder and decode graph bindings and, when the CHUNK-bucket binding is on at g1, each sealed bucket transition's staged inputs, state copy and owned outputs, which the bucket binding reports through its `external_tensors()` inventory), holding no resident tensor; while the call bindings' `bucket_transition` is not `None`, it shall also require both the recorded and the live inventory to hold every entry of the schema derived from that binding's sealed cells, `encoder.bucket.<g>.<n>.<field>` for each sealed cell `(g, n)` of the binding, with fields `env`, `prompt` and `status` (staged inputs); `state.raw_tail`, `state.mel_tail`, `state.frontend_counters`, `state.channel.<i>`, `state.window_valid.<i>` and `state.time.<i>` for each encoder layer `i`, `state.h`, `state.c` and `state.last_label` (state copy); and `out.token_ids`, `out.token_lengths`, `out.row_status`, `out.frame_emission_counts`, `out.frame_final_labels`, `out.frame_valid_lengths`, `out.valid_samples`, `out.geometry_id`, `out.final_tail`, `out.prompt_index`, `out.chunk_sequence`, `out.counter_invariant_bad` and `out.counter_delta_bad` (owned outputs); and if any compared value differs, or any expected bucket entry is missing, then it shall raise an error naming `transaction_graph_populations`, the cell and the changed pool, device, projected role, signature entry, call binding or buffer, or the missing bucket entry. Serving capture shall require the pool layout it records.
  - Design: Post-Warmup Verification
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_post_warmup_verification_rejects_resized_external_buffer` (calls the verification with the live `pool_layout`; parametrized over every mismatch kind, including a core moved to another device, a projection that swaps the `h` and `c` roles and a projection that changes a role's strides, each with every component pool unchanged, and at g1 with the CHUNK-bucket binding on a relocated or resized bucket staged input, state copy and owned output, plus, for each expected bucket field and for a whole sealed cell, an inventory missing it, and an empty inventory, each of which fails) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_verification_keeps_no_resident_view` (after `post_warmup`, with the binding published and verified, no resident view outlives the test's own references and the model state's bindings) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_serving_capture_requires_the_pool_layout` (capture without a pool layout is a `TypeError` naming `pool_layout`) [CPU]
  - Configurations: M; the bucket-buffer mismatch kinds only with the bucket on, because those buffers exist only then
- **TXG-VER-002**: When `transaction_graph_populations` is non-empty and core calls `ModelState.post_warmup()`, the model state shall run the post-warmup verification once over its current bindings' `pool_layout`, and no `prepare_step` shall run it; every non-dummy `prepare_step` shall check the binding's failed latch before registry pruning, `state_slots.begin_step`, request activation and any transaction, and raise the TXG-FAIL-004 error if it is set.
  - Design: Post-Warmup Verification
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_post_warmup_verifies_once_and_real_steps_check_the_latch` (also a latched binding failing a later `prepare_step` before pruning) [CPU]
  - Configurations: M
- **TXG-VER-003**: When `transaction_graph_populations` is non-empty, if post-warmup verification finds a changed pool layout, device, projected resident role, signature entry, call binding or binding-external buffer address or size (including, at g1 with the CHUNK-bucket binding on, a sealed bucket transition's staged inputs, state copy or owned outputs), then `post_warmup` shall raise a worker-fatal `RuntimeError` naming the cell and the changed item and latch the binding, with the session registry unchanged and no resident page written, so every later non-dummy `prepare_step` fails at its latch check (TXG-VER-002).
  - Design: Post-Warmup Verification
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_post_warmup_mismatch_is_fatal_before_any_state_change` (parametrized over the TXG-VER-001 mismatch kinds, the bucket buffers and each missing bucket entry included; each case latches the binding and a later `prepare_step` fails) [CPU]
  - Configurations: M, inheriting TXG-VER-001's mismatch kinds, the bucket buffers only with the bucket on
- **TXG-VER-004**: When `transaction_graph_populations` is non-empty, startup shall run the model's post-warmup verification from core's `ModelState.post_warmup()` hook, after every worker warmup and before the worker serves, and a mismatch shall fail startup.
  - Design: Post-Warmup Verification; Core Obligations, item 2 (delivered)
  - Test: `tests/v1/worker/test_gpu_model_runner_v2.py::test_post_warmup_hook_runs_once_after_every_warmup` and `::test_post_warmup_hook_failure_fails_startup` (core, generic model state) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_post_warmup_mismatch_is_fatal_before_any_state_change` (the model state's hook raises) [CPU]; `tests/v1/engine/test_nemotron_asr.py::test_nemotron_engine_transaction_graph_matches_eager` (verification runs once during `EngineCore` startup, before the first step) [engine-CUDA]
  - Configurations: M, through the engine test that TXG-REP-013 parametrizes, which keeps the startup-order assertions in every configuration; core's generic-model-state tests run no Nemotron model

### Failure containment (FAIL)

- **TXG-FAIL-001**: When `transaction_graph_populations` is non-empty, if a covered call fails before `graph.replay()` (classification, signature, arena upload, static-input copy, escaped-output allocation or commit reservation), then `execute` shall raise the exception type and message the eager transaction raises for the same input where eager has one, leave every resident page bitwise equal to the post-`begin_step` baseline, cancel a reserved commit ticket exactly once, and increment no `replays:<cell label>` or `fallbacks:` counter.
  - Design: Failure Containment
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_stale_lease_fails_before_replay_writes_a_page` [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_pre_boundary_failure_keeps_baseline_and_cancels_ticket_once` (fault injected at each of classification, signature, arena upload, static-input copy, escaped-output allocation and commit reservation, parametrized; asserts the post-`begin_step` baseline, cancellation exactly once only where a ticket was reserved, and unchanged replay counts) [CPU]
  - Configurations: M
- **TXG-FAIL-002**: When `transaction_graph_populations` is non-empty and a cell replays, `execute` shall reserve the commit before any resident page write and stage status only after the replay, so the commit sink records exactly `reserve` then `stage`, and no request abort shall cancel that reservation: a launched transaction completes stage and collection, and ticket cancellation occurs only as error cleanup (TXG-FAIL-001, TXG-FAIL-004).
  - Design: Transaction Structure; Invariants
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_replay_reserves_before_any_page_write_and_stages_after` (also a request aborted after the replay: `reserve`, `stage` and collection, with no cancel) [CPU]
  - Configurations: M
- **TXG-FAIL-003**: When `transaction_graph_populations` is non-empty and a cell replays, `execute` shall allocate every buffer the step needs, including the escaped row tensor of shape `(population, hidden_size)` float32, before `graph.replay()`, the device allocator shall record no allocation from the replay to the return, and the returned row tensor shall own storage distinct from every cell buffer and stay unchanged across any later replay, and the cell's owned status and progress buffers shall keep their contents across other cells' replays until the same cell's next replay.
  - Design: Failure Containment; Escaped outputs (Ordering)
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_replay_allocates_nothing_after_launch` [CUDA]; `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_replayed_rows_survive_another_cells_replay` [CUDA]; `tests/models/nemotron_asr/test_transaction_graph.py::test_replayed_rows_are_owned_by_the_caller` [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_status_and_progress_survive_another_cells_replay` (cell A, then cell B, then A's status and progress compared with A's eager values before A replays again) [CUDA]
  - Configurations: M
- **TXG-FAIL-004**: When `transaction_graph_populations` is non-empty, if any exception arises in `execute` from `graph.replay()` onward, including `CommitTicket.stage` failing after it set `_done`, or from a worker-fatal check (process constants, wrapper entry, post-warmup, changed reinitialization per TXG-LIFE-003), then the binding shall latch as failed and propagate the exception, and every later `execute` call, covered or fallback, shall raise `RuntimeError` containing "a fresh worker is required" before any classification, upload or replay, incrementing no counter; every later non-dummy `prepare_step` shall raise the same error (TXG-VER-002).
  - Design: Failure Containment; Execute order; Post-Warmup Verification
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_replay_failure_cancels_the_reservation_once` (also the latch and a later call) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_stage_failure_after_done_latches_the_binding` [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_every_fatal_source_latches_before_classification` (parametrized over replay, stage, process constant, wrapper entry, post-warmup and changed reinitialization × a later covered and a later fallback call) [CPU]
  - Configurations: M
- **TXG-FAIL-005**: When `transaction_graph_populations` is non-empty and `execute` raises inside an `EngineCore` step, the exception shall propagate out of `execute_model` and `EngineCore.step` with no progress from that step published. Stopping later steps is the core engine-death path (`EngineCoreProc`), not a model obligation; the model adds no in-process recovery guard.
  - Design: Failure Containment
  - Test: `tests/v1/engine/test_nemotron_asr.py::test_nemotron_engine_transaction_graph_replay_failure_propagates_from_step` (injected replay failure, in-process `EngineCore`; asserts propagation and no published progress) [engine-CUDA]
  - Configurations: M
- **TXG-FAIL-006**: When `transaction_graph_populations` is non-empty, if the commit-sink collection in `postprocess_state` raises after a replayed step (the surfacing point of an asynchronous device fault), then no progress from that step shall be published and the exception shall propagate out of `EngineCore.step`. Collection runs outside `execute`, so this failure does not set the TXG-FAIL-004 latch; stopping later steps is the core engine-death path, as for TXG-FAIL-005.
  - Design: Failure Containment
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_collection_failure_after_replay_publishes_no_progress` [CPU]; `tests/v1/engine/test_nemotron_asr.py::test_nemotron_engine_transaction_graph_collection_failure_propagates_from_step` (injected collection failure, in-process `EngineCore`; asserts propagation and no published progress) [engine-CUDA]
  - Configurations: M
- **TXG-FAIL-007**: When `transaction_graph_populations` is non-empty and a commit reservation is cancelled after a replay raised, or a request is aborted after a replay wrote its resident pages, the model shall leave those pages byte-identical to their post-replay contents until core rebinds the slot for a new generation or a later authorized write changes them.
  - Design: Failure Containment; Invariants
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_abort_after_replay_does_not_roll_back_pages` (pages compared after the abort and its collection, before any rebind) [CPU]
  - Configurations: M
- **TXG-FAIL-008**: When `transaction_graph_populations` is non-empty, if a call that passed classification and the signature check (Execute order) finds that its cell's wrapper entry no longer holds its graph (for example after `CUDAGraphWrapper.clear_all_graphs()`), then `execute` shall raise a worker-fatal `RuntimeError` before arena upload, capture nothing (`captured_cells` unchanged) and leave every resident page unchanged.
  - Design: Failure Containment
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_cleared_wrapper_entry_is_fatal_and_never_captures` (also a cleared entry with a signature mismatch, which is a counted `signature` fallback and not fatal) [CPU]
  - Configurations: M

### Observability (OBS)

- **TXG-OBS-001**: Where `transaction_graph_populations` is absent or empty, `transaction_graph_counters` shall return an empty mapping for a reachable model, runner, worker, executor or in-process `EngineCore`, and the execution profile receipt shall contain no transaction-graph entry.
  - Design: Observability
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_counters_reach_the_model_from_an_engine_core` (off branch) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_option_off_receipt_has_no_transaction_graph_entry` [CPU]; `tests/v1/engine/test_nemotron_asr.py::test_nemotron_engine_transaction_graph_matches_eager` (eager arm `== {}`) [engine-CUDA]
  - Configurations: M, through the engine test that TXG-REP-013 parametrizes (the option-off arm's empty mapping in every configuration), and the CPU tests over M
- **TXG-OBS-002**: When `transaction_graph_populations` is non-empty, `transaction_graph_counters` shall accept a model, runner, worker, executor or in-process `EngineCore` and return `captured_cells`, replays per cell label, fallbacks per `(cell, reason)` and prologue time from host state, without any device synchronization.
  - Design: Observability
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_counters_reach_the_model_from_an_engine_core` (every target, with synchronization patched to fail, a proxy that cannot see implicit synchronization) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_receipt_reports_admission_and_serving_bytes` (reads the counters under `torch.cuda.set_sync_debug_mode("error")`) [CUDA]
  - Configurations: M, including the CUDA receipt test that TXG-OBS-004 shares
- **TXG-OBS-003**: If the model is unreachable in process (for example under a multiprocess executor), with or without `transaction_graph_populations`, then `transaction_graph_counters` shall raise rather than return an empty mapping.
  - Design: Observability
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_counters_raise_when_the_model_is_unreachable` [CPU]
  - Configurations: M
- **TXG-OBS-004**: When `transaction_graph_populations` is non-empty, the execution profile receipt shall include readiness; per cell, arena bytes, reserved-memory growth, warm-up seconds, capture seconds and capture index; `A`, `S`, the release-check result, and the admission and serving capture seconds; and the served geometry (0 for g0, 1 for g1), with cell labels unchanged, so that `replay` labels carry no geometry.
  - Design: Observability
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_receipt_captures_largest_population_first_and_reports_cost` (including the admission fields and the served geometry, parametrized over M) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_receipt_reports_admission_and_serving_bytes` (also the TXG-OBS-002 implicit-synchronization check) [CUDA]
  - Configurations: M
- **TXG-OBS-005**: When `transaction_graph_populations` is non-empty and a step replays a cell, at g0, g1 with the CHUNK-bucket binding off, or g1 with `chunk_bucket_graph_populations` `[1, 31, 63]`, the encoder invocation counters and the CHUNK-bucket binding's replay and fallback counts shall not advance, per-step batch stats shall advance, and `replays:<cell label>` shall increase by exactly one.
  - Design: Host control flow inside the body
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_replay_freezes_nested_counters_and_counts_once` (parametrized over M) [CUDA]
  - Configurations: M
- **TXG-OBS-006**: When `transaction_graph_populations` is `[2]` at g1 with `chunk_bucket_graph_populations` `[1, 31, 63]`, an eager step whose CHUNK bucket holds a bucket population of rows (for example a one-row CHUNK step, counted `undeclared`) shall advance that bucket cell's replay count by one, no bucket fallback count and no encoder invocation counter, because the sealed bucket body runs the native encoder rather than the guarded encoder transition (`chunk_bucket_graph.py:151-161`, `advance.py:1310-1314`); an eager step whose CHUNK bucket holds another number of rows (for example a three-row CHUNK step, counted `undeclared`) shall advance the bucket's fallback count for that population by one, no bucket replay count, and the guarded encoder transition's invocation counter by one (`chunk_bucket_graph.py:474-480`); an eager step with no CHUNK row (for example one REPLAY row) shall advance neither bucket count; and a replayed two-row CHUNK step shall advance none of them; the serving baseline of these counts is their reading after startup (after `post_warmup` and before the first served step), and serving counts are differences from that baseline.
  - Design: g1 and the CHUNK-bucket binding; Host control flow inside the body
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_bucket_counts_record_eager_invocations` (new; transaction populations `[2]`; baseline read after startup; bucket and encoder-invocation deltas across one eager one-row CHUNK step, one eager three-row CHUNK step, one eager one-row REPLAY step and one replayed two-row CHUNK step) [CUDA]
  - Configurations: g1 with `chunk_bucket_graph_populations` `[1, 31, 63]`; not g0 or bucket off, because the bucket's counts exist only with the bucket on

### Core warmup fix (CORE; owned by the core persistent-state design)

- **TXG-CORE-001**: When any KV cache group's spec is a `PersistentStateSpec`, `run_mixed_prefill_decode_warmup` shall return `False` without invoking `worker_execute_model` or `worker_sample_tokens`.
  - Design: Core Dependency: Persistent-State Warmup; Core Obligations, item 1 (delivered)
  - Test: `tests/v1/worker/test_mixed_warmup_gate.py::test_mixed_warmup_skipped_for_persistent_state_groups` [CPU]
  - Configurations: core tests with generic state specs; no Nemotron model runs, so geometry and bucket state do not apply
- **TXG-CORE-002**: When no KV cache group is persistent state, `run_mixed_prefill_decode_warmup` shall keep every existing gate: it shall skip the warmup without invoking the worker callbacks for a pooling model, when `max_num_reqs < 2`, when `num_tokens < 3`, or when the KV cache has too few blocks for the warmup requests, and shall otherwise run it.
  - Design: Core Dependency: Persistent-State Warmup; Core Obligations, item 1 (delivered)
  - Test: `tests/v1/worker/test_mixed_warmup_gate.py::test_mixed_warmup_disables_watermarking` (run path) [CPU]; `tests/v1/worker/test_mixed_warmup_gate.py::test_mixed_warmup_skipped_for_single_seq` (`max_num_reqs < 2`) [CPU]; `tests/v1/worker/test_mixed_warmup_gate.py::test_mixed_warmup_skipped_for_pooling_short_and_block_starved_runs` (pooling model, `num_tokens < 3`, insufficient blocks, parametrized) [CPU]
  - Configurations: core tests with generic state specs; no Nemotron model runs, so geometry and bucket state do not apply

### Consistency report

- **Coverage.** Every target behaviour has a criterion except: the slot-reuse invariant between commit reservation and collection and replay serialization (Invariants), which rest on core synchronous scheduling for persistent state and are tested there; nested-scratch write-before-read, a model property whose behavioural coverage is TXG-REP-017, including the CHUNK-bucket scratch at g1; the bucket binding's freedom from `signature` fallback (unprepared positional projections under the `eager-graphed` arm), observed through the zero-fallback lockstep of TXG-REP-010 and TXG-REP-013 with the bucket on; the bucket-before-admission startup order, which TXG-MEM-002's "after the nested inventory is ready" covers once the nested inventory includes the bucket graphs; the sleep-mode, CuMem and weight-reload rejections, which are existing persistent-state contracts owned elsewhere; "a nonzero `signature` count is a defect", an operating judgement observed through TXG-REP-005's counter; and the client-level `EngineDeadError` after a fatal step, which is the core-owned `EngineCoreProc` busy-loop path (`core.py:1833-1841`, `:1794-1800`), which stops later steps and is covered generically by `tests/v1/shutdown/test_forward_error.py`, with no existing harness driving a Nemotron streaming session through the multiprocess client (TXG-FAIL-005/006 observe propagation out of `EngineCore.step` instead). The Default-On Gate (including the `EngineCoreProc` counter reader; the counters are an in-process harness facility here), Open Questions probes, the startup-count proposal and the exploratory measurements carry no criteria by decision.
- **Observable proxies.** TXG-ENV-003 names the construction-observable options, because the runner's `needs_cudagraph_capture()` is not observable at model construction, and orders its checks before the model's other construction-time mode rejections; TXG-REP-014 reads the per-operation `fp32_precision` settings, because the legacy matmul-precision getter raises once those settings are used; TXG-ENV-010 observes validation precedence through which error a doubly invalid configuration raises, and TXG-ENV-011 error ownership through the bucket binding's own error; TXG-ENV-009 observes the call routing as the identity of `TransactionCallBindings.bucket_transition`; TXG-OBS-006 observes the post-startup baseline as a reading after `post_warmup` and before the first served step, and its deltas across eager steps at a bucket population, at another population and with no CHUNK row.
- **Core-owned.** Core Obligations lists every obligation on the core persistent-state design with its criteria. Core delivers all six obligations: TXG-CORE-001 and TXG-CORE-002 are tested in core; TXG-VER-004, TXG-LIFE-007 and TXG-LIFE-008 pair the core tests with model-level tests of the model state's `post_warmup`, `shutdown` and startup access, and TXG-LIFE-009 tests the model's `is_profiling` guard. TXG-MEM-011 and TXG-MEM-012 pair core's generic-label tests with a model-level test that the served state spec names the option only while it is on, and TXG-MEM-011 also routes the served spec through core's planner.
- **Conservative defaults.** TXG-MEM-006 fixes zero tolerance for `S <= A`, and TXG-MEM-008 keeps it under both capacity overrides; TXG-MEM-002 requires the complete inventory, so no cross-cell reuse is assumed; TXG-MEM-007 forbids deducting double-counted peak.
- **Contradictions.** None between the criteria and the tests they name. `test_arena_rejects_vectors_outside_its_layout` is an arena-level guard beside the execution-level `signature` fallback (TXG-REP-005). `test_cuda_graph_replay_matches_eager_across_varied_plans` excludes park tokens and is not park-echo coverage (TXG-REP-011).
- **Implicit scoping resolved.** Processor and `bind_rows` geometry enforcement (TXG-ENV-007, TXG-ENV-008) and the unreachable-model reader (TXG-OBS-003) are option-independent, because the design passes the configured set to the plan context unconditionally. "Worker-fatal" is read as the latch of TXG-FAIL-004 for every fatal check, including pre-boundary process-constant, wrapper-entry and post-warmup failures; only non-fatal pre-boundary failures follow the structural-failure contract (TXG-FAIL-001). Signature-fallback pool preservation applies only to captured storage distinct from the live pools (TXG-REP-005). Budget-derived capacity applies only without an override, and an override's count only once core budget validation accepts it (TXG-MEM-007, TXG-MEM-008). "Every CHUNK step replays" means zero CHUNK-step fallbacks in the homogeneous lockstep workload, with echo steps at undeclared populations counted `undeclared`; mixed and staggered steps keep eager equivalence and their fallback counts (TXG-REP-010, TXG-REP-013). Request abort is not ticket cancellation, and post-replay page preservation lasts until a rebind or authorized write (TXG-FAIL-002, TXG-FAIL-007). Per-row rejections are classified as eager classifies them, terminal ones as request failures, never step failures (TXG-REP-012). Processor geometry rejection is tested at the processor; the in-process engine tests bypass it (TXG-ENV-007). Geometry and CHUNK-bucket state follow the preamble rule: criteria whose behaviour depends on the served geometry or the bucket binding (TXG-ENV-004, TXG-ENV-009 to TXG-ENV-012, TXG-REP-001, TXG-REP-010 to TXG-REP-013, TXG-REP-017 to TXG-REP-019, TXG-OBS-004 to TXG-OBS-006, TXG-VER-001, TXG-VER-003) name their configurations; every other criterion holds at g0 and at g1, with the bucket off and, at g1, on. The rule states where a criterion applies; a criterion's tests establish it only in the configurations they run in, listed below.
- **Test configurations.** Every criterion carries a Configurations entry, and every criterion whose behaviour can be exercised under a g1 engine is tested over M (g0; g1 with the CHUNK-bucket binding off; g1 with `[1, 31, 63]`). Entries narrower than M, each with its reason in the entry: TXG-ENV-004 (bucket-on cases belong to TXG-ENV-009 to TXG-ENV-011); TXG-ENV-009, TXG-ENV-010, TXG-OBS-006 and the bucket parts of TXG-REP-006, TXG-REP-007, TXG-VER-001 and TXG-VER-003 (bucket-only behaviour, inapplicable at g0 or bucket off); TXG-ENV-011 (bucket configured, including its negative case at g0, `[0]`); TXG-ENV-012 (the bucket binding is not a constructor input); TXG-REP-019 (g1-only lookahead behaviour); the core generic tests of TXG-CORE-001, TXG-CORE-002, TXG-MEM-011, TXG-MEM-012, TXG-LIFE-007, TXG-LIFE-008 and TXG-VER-004 (no Nemotron model runs). Wider than M: TXG-REP-010 adds g1 with `[1, 63]`; TXG-ENV-007 and TXG-ENV-008 run option off and on in each served set. Shared tests: TXG-LIFE-007, TXG-VER-004 and TXG-OBS-001 run in TXG-REP-013's parametrized engine test, which keeps each one's assertions in every configuration; TXG-REP-009 shares TXG-REP-001's capture test; TXG-OBS-002 shares TXG-OBS-004's CUDA receipt test. Phase 5 replaces `test_option_rejects_served_geometries_other_than_g0`, whose `g1` case contradicts TXG-ENV-004.
- **Context-free reading.** Every criterion names `transaction_graph_populations` and its state in its own clause; cross-file concepts (g0, g1, served geometry, CHUNK-bucket binding on, eager, bitwise equal, configuration error, `endpoint_enabled`, `N()`) carry a parenthetical or the preamble definition.

## References

Tests: `tests/models/nemotron_asr/test_transaction_graph.py`,
`tests/v1/engine/test_nemotron_asr.py`. Related designs:
[CUDA Graphs](cuda_graphs.md), [Model Runner V2](model_runner_v2.md).
