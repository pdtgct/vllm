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

"Current Implementation Divergences" lists where the code does not yet meet
this design; all other sections describe target behavior.

## Context and Design Philosophy

A Nemotron streaming step is one model-defined transaction over resident state
slots. `advance_model_rows` (`advance.py:2741`) validates the row plan on the
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

1. `prepare_transaction` (`advance.py:2910`) is the host prologue: structural
   preflight, decode resolution, CHUNK bucket formation, per-step batch stats
   (`advance.py:3148`) and every host-only check. It produces
   `TransactionInputs` and touches no resident page.
2. `run_transaction_body` (`advance.py:3251`) is the device body: gathers,
   frontend, encoder, decode, projection, row status, progress and the
   complete scatter plan. The whole plan is validated before any scatter.
3. `commit_transaction_body` (`advance.py:4105`) makes one composite
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

A `TransactionCell` (`transaction_graph.py:47`) is `(kind, geometry,
population)`, where population is the exact real-row count. Only rows in the
step count: admitted, parked or never-fed slots without a row in the step
never contribute to its population.

| Kind | Rows |
|------|------|
| `chunk-fresh` | one CHUNK bucket, every row a session's first CHUNK |
| `chunk-continuing` | one CHUNK bucket, no fresh row |
| `replay` | no CHUNK row and no fresh row: REPLAY, park-echo, FLUSH, forced-endpoint |

`classify_transaction` (`transaction_graph.py:67`) returns the cell, or the
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
every nested launch site consults it (`encoder_execution.py:1566`, `:1633`,
`decode_graph.py:423`, `chunk_bucket_graph.py:253`). Host-to-device staging
inside the scope raises (`capture_scope.py:44`, `advance.py:1601`), because a
captured copy would replay a frozen pinned source.

### Inputs, outputs and signature

- **Arena.** Each cell owns a `TransactionArena` (`transaction_graph.py:163`):
  one device buffer holding every int64/bool vector the body reads, plus two
  pinned host buffers that alternate per call. Each host buffer is fenced by
  an event, so it is rewritten only after its previous upload has completed.
  There is one host-to-device copy per call.
- **Static inputs and owned outputs.** Each cell's `static_ids` and
  `static_embeds` receive the call's inputs before replay. Inside the graph,
  projection rows, status and progress are copied into per-cell buffers
  allocated outside the graph pool, so no pool tensor outlives a replay.
- **Signature.** `transaction_signature` (`transaction_graph.py:115`) covers
  every structural input the body bakes in: population, control ids,
  `endpoint_enabled`, `n_fresh`, decode and transition callable identities,
  resident pool layouts (address, shape, stride), lazily bound encoder
  buffers, input shapes, and the arena's name/dtype/shape layout. A layout
  mismatch is a `signature` fallback detected before any upload.

### Host control flow inside the body

The body is not branch-free. Its host control flow depends only on fields
fixed by the cell or covered by the signature: `endpoint_enabled` (signature);
the fresh-mask class (cell kind, plus `n_fresh`); the geometry's lookahead and
hop (cell geometry); decode output widths and `k_max` (geometry plus the decode
callable's identity); and the capture flag (forced off; `capture` is a
fallback reason). Any further host-dependent field the body branches on must
join the signature or the cell key before its cell is published. Within one
cell and signature the graph is data-independent.

Nested host-side checks and counters run only while a cell is captured and
are frozen under replay: `_validate_result_structure` (`advance.py:3537`), the
guarded-transition invocation counters (`encoder_execution.py:830`, `:2149`)
and the CHUNK binding's fallback counts (`chunk_bucket_graph.py:483`). For
replayed steps, encoder invocation counts and CHUNK-binding fallback counts
do not advance; per-step batch stats still do, because the prologue stages
them. The transaction counters are the per-replay record.

### Capture and the replay-cell oracle

Capturing a cell (`transaction_graph.py:479`) runs an eager reference on
capture rows the transaction rejects (so no page is stored), three eager
warm-ups through the wrapper, the capture, and a replay that must reproduce the
reference outputs and pages bitwise before the cell is published. That
self-check covers rejected rows only. Correctness for accepted rows rests on
tests that drive accepted rows of every kind through replay and compare with
eager bitwise, including `replay` cells carrying accepted FLUSH and park-echo
rows with endpointing on and off, and accepted forced-endpoint rows with
endpointing on. With endpointing off the forced-endpoint control is not
recognized (`advance.py:3420-3428`), so such a row is rejected, under replay
exactly as under eager.

### Callable identity

The signature compares decode and CHUNK callables by `id`. This is sound
because the dense decode binding caches one callable per graph key for the
binding's lifetime (`decode_graph.py:463-470`), and the CHUNK binding accepts
only its sealed callables (`chunk_bucket_graph.py:474-476`). Ids are not reused
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

The engine also needs `cudagraph_mode` `NONE` and `max_num_seqs` of at least
the largest population. The model validates at
construction, each failure a configuration error naming the option:

- native burst is disabled, and the decode arm is `dense-graphed` or
  `dense-eager`;
- the runner's `cudagraph_mode` is `NONE` with no runner-managed encoder
  graphs, so `needs_cudagraph_capture()` is false;
- the served geometry set is exactly g0 (`supported_num_lookahead_tokens ==
  [0]`);
- every population is an integer in `1..min(max_num_seqs,
  max_num_batched_tokens)` (each row is one token), and populations are
  distinct;
- under `num_gpu_blocks_override`, every population is at most
  `override - 1`, the non-null slot count.

When memory, not the `max_num_seqs + 1` cap, binds capacity, the usable slot
count is known only after profiling. In that case the serving-capture slot
check (`transaction_graph.py:430`) fails startup.

### g0 only: configuration and row admission

Non-g0 geometries are rejected at configuration. `replay` cells carry no
geometry and cover non-CHUNK rows of any geometry. Leaving g1 uncaptured
therefore cannot keep g1 sessions' echo and FLUSH rows out of replay.

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

The restriction lifts only when g1 meets the same acceptance criteria.

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
models otherwise get. It is valid only during `initialize_state_slots` and
commits no page; its authorization is a core obligation (Core Obligations).
A repeated `initialize_state_slots` with identical pool layout and call
bindings is a verified no-op; any difference is worker-fatal. Nothing is
recaptured outside startup.

Runner shutdown discards the binding's graphs, arenas and owned buffers.
`GPUModelRunner.shutdown` calls no model or `ModelState` discard hook
(`model_runner.py:2483-2511`). Discard on shutdown therefore needs a core
`ModelState` shutdown hook (Core Obligations). The model provides
`TransactionGraphBinding.discard()` (`transaction_graph.py:396`). Until the
hook exists the model takes no shutdown action: `shutdown` drops
`model_state` (`model_runner.py:2510-2511`), and graphs, arenas and owned
buffers are released with the model object.

### Process constants and excluded paths

The batch-invariant mode, TF32 enablement, `float32_matmul_precision` and
`CUBLAS_WORKSPACE_CONFIG` are process constants for the binding's lifetime.
`execute` compares their effective values (not cached construction values)
with their capture-time values using host-only reads, before every return,
including uncovered and mixed-row fallbacks. A change is worker-fatal, because
an eager fallback would also run against nested graphs captured under the old
settings. Sleep mode and the CuMem allocator are rejected for
all persistent state (`vllm/v1/core/state_slot_manager.py:52-53`), as is weight
reload after execution materialization (`nemotron_asr.py:552-556`).

### Invariants

- Arena upload, static-input copies, replay and every resident write run on
  the runner stream (`model_state_v2.py:122`). Replays are serialized.
- No slot is freed or reassigned between the commit reservation and the
  commit-sink collection in `postprocess_state`. Collection precedes the next
  replay because persistent state requires synchronous scheduling
  (`vllm/v1/core/state_slot_manager.py:35`).
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
- Nested-binding scratch is shared by the outer graph and the eager-path nested
  launches. It is fully written before it is read within a call.

## Memory Admission

### Budget mechanics

- `determine_available_memory` wraps `profile_run` in `memory_profiling`
  (`vllm/v1/worker/gpu_worker.py:698-714`). `non_kv_cache_memory` is the
  free-memory drop since worker creation plus the transient peak headroom
  (`vllm/utils/mem_utils.py:309-320`). Anything still alive when the window
  closes is charged at its driver-visible size.
- `profile_run` calls the model's `prepare_execution_memory` inside that window
  (`vllm/v1/worker/gpu/model_runner.py:1071-1073`). That hook retains the
  nested inventory and runs eager profile transactions alongside it
  (`nemotron_asr.py:805-859`).
- The budget is the memory left after `non_kv_cache_memory`, or
  `kv_cache_memory_bytes` when that is set. Resident capacity for persistent
  state is `count = min(budget // page_size_bytes, max_num_seqs + 1)` pages.
  A set `num_gpu_blocks_override` replaces the count, not the budget: core
  budget validation must still accept `count * page_size_bytes <= budget`,
  and startup fails unless it does and `count >= 2`. Usable slots are `count -
  1`, excluding the null block (on this branch
  `vllm/v1/core/kv_cache_planning.py:1496-1515`; the RFC 1 PR performs the
  same validation in its generic planner, rejecting an oversized override
  before allocation). Admission therefore reduces usable slots only when
  memory, not the `max_num_seqs` cap, binds.
- Resident pools exist only inside `initialize_kv_cache`. The KV allocation
  precedes `initialize_state_slots`, and the capture hook runs from
  `initialize_state_slots` after the pools are allocated (on this branch
  `model_runner.py:830`, `:854`; `model_state_v2.py:110-121`).

Graphs bake in resident pool addresses, and those addresses do not exist until
capacity is fixed. The inventory that serves therefore cannot be the inventory
that is measured.

### Normalized measurement

`N()` denotes a normalized free-memory reading. It is taken after a device
synchronize, `gc.collect()` and an allocator `empty_cache()`, with no other
allocation in flight, using `mem_get_info`. Every admission figure is a
difference of two `N()` readings. This includes memory the torch allocator does
not track: instantiated graphs and fragmentation.

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
   `allocate_state_scratch`, one raw allocation, `state_profile.py:233-255`);
   capture the complete declared inventory over it into a dedicated graph pool
   through the same capture path, cell order, capture stream and verification
   as serving capture; clear every capture-only temporary (page snapshots,
   expected outputs, capture and verify callbacks, capture inputs) while the
   admission holder retains the stand-in storage explicitly; read `Na`;
   release the stand-in storage; read `N1`. Release is verified when `N1 - Na
   >= stand-in bytes` and no owning reference to the stand-in storage remains
   (checked separately through a weak reference), else startup fails. Admitted
   bytes are `A = N0 - N1`. Stand-in
   storage provides capture indices and matching component strides, not
   proven-identical allocator or graph-exec bytes; the final check is the
   authority.
3. **Eager profiles.** The existing CHUNK-cell and `max_num_seqs` profile
   transactions run with both inventories alive, so their transient peak is
   measured on top of retained graph memory.
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
(`vllm/v1/worker/gpu_worker.py:670-695`), so `prepare_execution_memory`,
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
outside any model hook, so naming the option there is a core obligation
(Core Obligations).

### Failure modes and diagnostics

Every case below is a startup failure. None degrades to eager serving. Each
names the option, except the core-owned planner rejection and KV-allocation
failure until their obligations land (Core Obligations).

- Admission does not fit (capture failure or OOM in the window):
  `prepare_execution_memory` fails and latches `_execution_memory_failed`
  (`nemotron_asr.py:812-816`).
- Release not verified, or too few usable slots.
- No capacity: core budget validation rejects the persistent-state budget,
  because fewer than two pages fit or a `num_gpu_blocks_override` exceeds the
  remaining budget (on this branch `vllm/v1/core/kv_cache_planning.py:1496-1515`).
  It runs in the engine-core process from `EngineCore._initialize_kv_caches`
  (`vllm/v1/engine/core.py:336`), after `determine_available_memory` returns
  and before any worker `initialize_kv_cache`, so no model call surrounds it
  and the model cannot wrap it.
- KV allocation fails or runs out of memory in `initialize_kv_cache`, before
  `initialize_state_slots` and therefore outside the model's capture hook.
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
`kernel_warmup`, `warmup_kernels` and the compile dummy runs
(`gpu_worker.py:980-1004`). Every published cell's signature, together with
the address and size of every binding-external buffer its graph reads, is
verified after those warmups and before the worker serves.

This branch has no runner or `ModelState` hook after
`_compile_or_warm_up_model` (with `cudagraph_mode` `NONE`, `capture_model`
returns before `capture_inner_cudagraphs`). The verification therefore needs a
core `ModelState` post-warmup hook called at the end of
`_compile_or_warm_up_model`, where a mismatch is a startup failure (Core
Obligations). Until it exists, the model runs the same
verification once on the first non-dummy `prepare_step`, before its registry
pruning (`model_state_v2.py:155-158`) and therefore before
`state_slots.begin_step` (`model_runner.py:1837`, `:1845`), request activation
or any transaction; a mismatch there is worker-fatal. This interim is a safety
gate, not startup qualification: the engine may report ready before the first
step discovers a mismatch. Independently of that one-time verification, every
non-dummy `prepare_step` checks the binding's failed latch at the same point
and raises if it is set.

## Failure Containment

The launch boundary is `graph.replay()` in `TransactionGraphBinding.execute`;
the checks before it run in the Execute order.

**Per-row rejections are not step failures.** A row the transaction rejects
has a nonzero status, its pages stay unchanged and the step's other rows
commit. Under replay exactly as under eager, its status flows through
commit-sink collection, which classifies it as eager does: a terminal
rejection goes to `state_slots.fail_requests` (`model_state_v2.py:292-295`)
and core finishes that request with an error, while a recoverable status
(`ROW_STATUS_GEOMETRY`, `commit_sink.py:74`) does not fail the request. Such a
step neither raises nor latches the binding.

**Unchanged-state baseline.** "Unchanged" means the resident state after core
slot initialization in `begin_step` (`model_runner.py:1845`,
`vllm/v1/worker/gpu/state_slots.py:76-85`), which zeroes and initializes newly
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
(`cudagraph_utils.py:1020`, `model_runner.py:2490`). The configuration
requirement `cudagraph_mode == NONE` keeps the runner's graph estimator
(`cudagraph_utils.py:945`) and its minimal state-slot bootstrap
(`cudagraph_utils.py:1057-1077`) from ever invoking the capture hook.

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
  `step` (`vllm/v1/engine/core.py:728`, `:623`), and the busy loop signals the
  engine dead (`core.py:1668-1674`).
- Asynchronous device faults surface at the next synchronizing call, normally
  the commit-sink collection (`commit_sink.py:270`, via
  `model_state_v2.py:292`), which fails before any progress is published and
  propagates to the same engine-death path. Collection runs outside
  `execute`, so it does not latch the binding; the engine, not the model,
  stops later steps. Faults are not attributed to cells and not resumed.
- In-process callers of `advance_model_rows` see pre-boundary failures as
  structural failures; engine callers see any raise as engine death. Recovery
  is a fresh worker process.
- Neither cancellation nor abort implies rollback. A reservation cancelled
  after a replay raised, or a request aborted after a replay wrote its pages,
  leaves those pages as the replay left them until core rebinds the slot for
  a new generation, which zeroes and initializes it (`vllm/v1/worker/gpu/state_slots.py:65-87`),
  or a later authorized write changes them. Slot retirement and fencing stay
  with the core state-slot manager.

This extends the model's existing rule that a catastrophic failure during the
prevalidated commit is worker-fatal (`advance.py:2813-2819`) to the whole
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
  (`_resolve_model`, `transaction_graph.py:793-810`), a harness convenience,
  not a core contract. It returns
  host-only counts that never synchronize: captured cells, replays per cell,
  fallbacks per `(cell, reason)`, and prologue time. With the option off it
  returns an empty mapping; if the model is unreachable in process (for
  example under a multiprocess executor) it raises, so that case is never
  mistaken for off.
- The execution profile receipt includes readiness and, per cell, arena bytes,
  reserved-memory growth, warm-up and capture seconds, and capture order. It
  also includes `A`, `S`, the release check, and the admission and serving
  capture seconds.

## Core Dependency: Persistent-State Warmup (carried)

This is a core bugfix owned by the core persistent-state design, not by the
model; this branch carries it until that design lands it.
`run_mixed_prefill_decode_warmup` builds synthetic token-prefill and decode
requests, which are invalid for any `PersistentStateSpec` group: a
persistent-state request holds one state record for its lifetime and starts
with a model-defined transaction. The fix returns early when any KV cache group
is persistent state (`vllm/v1/worker/gpu/warmup.py:101-106`). The existing
gates (pooling model, `max_num_reqs < 2`, `num_tokens < 3`, too few blocks;
`warmup.py:98`, `:131`) stay; multi-session populations pass the
`max_num_reqs` gate and reach the warmup first. Non-state models are unchanged.

## Core Obligations

This design places the obligations below on the core persistent-state design.
At the RFC 1 PR revision (`574efbd8f6`), none is yet recorded in that design;
each is tracked by the criteria named. TXG-CORE-001 and TXG-CORE-002 track the
carried warmup fix and are testable now on this branch. TXG-VER-004,
TXG-LIFE-007, TXG-LIFE-008, TXG-MEM-011 and TXG-MEM-012 are deferred until
core lands their obligation. Core-side criteria name a model-supplied
contributor label rather than this model's option, so core tests stay
independent of the out-of-tree model.

1. **Persistent-state mixed-warmup skip.** `run_mixed_prefill_decode_warmup`
   skips persistent-state groups. This branch carries the fix (Core
   Dependency); the RFC 1 PR's `warmup.py` lacks the gate. TXG-CORE-001,
   TXG-CORE-002.
2. **Post-warmup hook.** A `ModelState` hook called after every worker warmup
   and before the worker serves (Post-Warmup Verification). TXG-VER-004.
3. **Shutdown hook.** A `ModelState` hook, called from runner shutdown, that
   calls the model's `discard()` (Lifetime: one serving capture per worker).
   TXG-LIFE-007.
4. **Option-attributed capacity rejection.** When core budget validation
   rejects the persistent-state budget, including a `num_gpu_blocks_override`
   that exceeds the remaining budget, the error names the option that
   contributed admitted memory. How core learns the option is core's choice.
   TXG-MEM-011.
5. **Option-attributed KV-allocation failure.** A KV allocation that fails
   before `initialize_state_slots` reports the same attribution. TXG-MEM-012.
6. **Startup pool access.** An authorized way for a model to reach the
   unowned resident pools for serving capture and the rejected-row reference,
   valid only during `initialize_state_slots` and committing no page. Today
   capture reads the private `views._views` (`model_state_v2.py:121`;
   divergence 13). TXG-LIFE-008.

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

Covering the default cadence first requires validating g1 as an envelope
extension.

## Current Implementation Divergences

Each item gives the current behavior on this branch, then the target.

1. **Geometry at configuration.** `nemotron_asr.py:503-526` accepts any served
   geometry set, and `transaction_graph.py:365-370` builds cells for every
   served geometry. Target: g0 only.
2. **Geometry at admission.** `_stage_plan` passes `len(CADENCES)`
   (`nemotron_asr.py:709`), and `plan.py:425` checks only that range. Target:
   the configured set.
3. **Outer graph mode.** Only FULL is rejected (`plan.py:230-249`); PIECEWISE
   is accepted. Under PIECEWISE the runner's estimator bootstraps minimal
   state slots, which calls `initialize_state_slots`, and so the capture hook
   runs over throwaway pools (`cudagraph_utils.py:1057-1077`), whose graphs
   `clear_all_graphs` later empties. Target: `cudagraph_mode == NONE`, plus the
   wrapper-entry check.
4. **No admission.** `prepare_execution_memory` performs no transaction-graph
   admission. Capture runs only after capacity is fixed
   (`model_state_v2.py:115-121`). Target: the full ordering above, with
   normalized `A`/`S` and the `S <= A` check.
5. **Capture references retained.** `_TransactionGraph` keeps its
   `capture_graph` and `verify` closures (`transaction_graph.py:267-268`,
   `:664-665`). Those closures retain the capture `inputs` (pools), page
   snapshots and expected outputs. The wrapper keeps `body`, which closes over
   `inputs` (`transaction_graph.py:566`, `cuda_graph.py:185`). Target: the
   capture-lifetime rule.
6. **Replay boundary.** `execute` cancels only on replay failure and has no
   failed latch. It stages outside that handler and clones `rows_out` after
   replay (`transaction_graph.py:729-738`). Target: preallocated escaped
   output, latch, stage failure latched.
7. **Arena mismatch.** `TransactionArena.upload` raises on a layout mismatch
   (`transaction_graph.py:216-232`), and the layout is not in the signature.
   Target: a `signature` fallback.
8. **Recapture.** `capture` discards and recaptures on any change of pool
   layout (`transaction_graph.py:420-423`, `nemotron_asr.py:861-866`). Target:
   a single serving capture; a change is fatal.
9. **Missing checks.** No latch, process-constant, wrapper-entry or
   post-warmup checks exist: `execute` classifies first and counts a fallback
   (`transaction_graph.py:686-709`) with nothing fatal before it. Population
   validation (`transaction_graph.py:350-362`) omits `max_num_batched_tokens`
   and the override slot bound. Target: the Execute order.
10. **Reader and shutdown.** `_resolve_model` returns `None` for an unreachable
    model, so the reader reports off (`transaction_graph.py:793-823`);
    `GPUModelRunner.shutdown` (`model_runner.py:2483-2511`) does not discard
    the binding, and no `ModelState` shutdown hook exists; `discard()`
    (`transaction_graph.py:396-399`) clears graphs only, not arenas or owned
    buffers. Target: `discard()` drops all three, and the core shutdown hook
    calls it.
11. **Inaccurate comments.** `advance.py:3535-3536` says structural rejection
    runs on every invocation including graph replay; under outer replay it is
    frozen at capture. `transaction_graph.py:451-455` and the receipt
    docstring (`:759-761`) claim largest-first segment reuse, which
    measurement contradicts.

12. **Processor geometry check.** `NemotronASRMultiModalProcessor`
    (`processor.py:106-122`) passes envelopes through without checking their
    geometry, so today an off-set geometry is caught only in the worker.
13. **Private pool access at capture.** `initialize_state_slots` hands capture
    the private `views._views` mapping (`model_state_v2.py:121`), outside any
    authorized core access. Target: the core startup pool access (Core
    Obligations, item 6).

## Decisions & Alternatives

| Decision | Chosen | Alternatives Considered | Rationale |
|----------|--------|------------------------|-----------|
| Placement | Design doc beside the model, moving with it to the plugin; numbered acceptance criteria | Core design doc; EARS with `@spec` tags | The model owns the option; the plugin move must not orphan intent |
| Posture | Opt-in, with admission, the g0 restriction and counted fallbacks | Default-on; opt-in without admission | Startup and capacity costs and mixed-step rates are unmeasured; hidden graph memory breaks capacity planning |
| Coverage key | Exact population per kind and geometry | Padded population buckets | Padding changes row counts through decode and commit and needs its own exactness proof |
| Capture shape | Flat: nested bindings eager inside one outer capture | Nested graph launches | One launch; nested launches keep per-region cost and separate pools |
| Host inputs | Persistent double-buffered pinned arena, one copy | Per-call `HostStaging`; captured host-to-device copies | Captured copies replay frozen sources; per-call staging adds launches |
| Non-g0 geometries | Rejected at configuration and at row admission | Captured but unvalidated; left uncaptured and counted | Replay cells are geometry-agnostic, so uncaptured g1 still reaches replay through its echo rows |
| Admission method | Model-local: retain the full admission inventory over stand-in storage through the profile window, then check `S <= A` | (a) Reserved-memory delta of serving capture; (b) placeholder tensor of the measured size; (c) extrapolate from one cell; (d) core graph estimator; (e) measure the admission capture's normalized delta and peak, release it, and charge both through a core accounting hook before capacity selection | (a) and (b) miss graph-exec memory, fragmentation and peak; (c) is unsound without reuse; (d) does not see model graphs; (e) is viable but needs a new core hook and an explicit composition of the measured peak with the profile peak, while retention reuses the existing window unchanged |
| Graph pool | One dedicated pool per inventory | Platform global pool shared with nested graphs | Releasing admission must return exactly its segments, whatever stream serving capture uses |
| Failure boundary | All step allocations before replay; any failure from the launch on is worker-fatal with a latch | Roll back pages; fall back to eager after a replay error | Body and scatters are one launch, so partial mutation is unknowable; rollback needs hot-path snapshots |
| Signature mismatch | Count and run eager; no serving-time recapture | Recapture on demand; raise | Recapture needs unadmitted memory and stalls; eager is the exact reference |
| Process constants | Assert at `execute`, worker-fatal on change | Include them in the signature | The nested inventories and every captured cell were built under the same settings, so a change invalidates the whole process, not one cell; a signature fallback would silently run eager against stale nested graphs |
| Post-warmup check | Core `ModelState` post-warmup hook; first-real-step check until it exists | Re-verify on every call; skip | Per-call signature checks miss binding-external buffers; startup is the right failure point |
| Warmup fix | Separate core bugfix carried by this branch | Model-side workaround | It is a core persistent-state defect, independent of graphs |

## Open Questions

1. **No cross-cell graph memory reuse.** Retained memory is the sum of
   per-cell growth. Likely cause, not yet confirmed:
   `GraphRuntime.capture_context` calls `graph_capture(device=device)` without
   a stream (`decode_graph.py:96`), which creates a new stream per call
   (`vllm/distributed/parallel_state.py:1718`); the caching allocator reuses a
   freed block only on the stream that allocated it, and `graph_capture`
   documents that reusing one stream shares free graph-pool blocks
   (`parallel_state.py:1711`). Secondary suspects: tensors kept alive until the
   deferred `consume_batch_stats()` (`transaction_graph.py:473`), the retained
   capture closures (divergence 5), and lazily created module buffers.
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
   (`advance.py:2512`), the capture-enabled flag
   (`vllm/compilation/monitor.py:113`) and the class-level platform graph pool
   (`vllm/platforms/interface.py:1329`) persist into a second engine in the
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
6. **g1 extension.** Whether `replay` cells need a geometry key once more than
   one geometry is served, and what g1 evidence lifts the restriction.
   **Probe:** a g1-only bitwise lockstep run with accepted rows of every kind,
   and a mixed g0/g1 row-level replay comparison.
7. **Lazily bound encoder buffers.** Whether any served stream rebinds them
   after warm-up, which would cause a permanent `signature` fallback.
   **Probe:** a long-stream run with the `signature` counter and the
   lazy-buffer layout logged per step.

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
present, or all absent (`advance.py:3029-3035`); neither is a step policy.
"Endpoint" always means utterance endpointing, never endpoint plugins.

### Configuration and envelope (ENV)

- **TXG-ENV-001**: Where `transaction_graph_populations` is absent or an empty list, the model shall construct no `TransactionGraphBinding`, capture no transaction graph, and call `advance_model_rows` with `transaction_graph=None` on every step.
  - Design: Supported Envelope
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_option_off_constructs_no_binding` (new) [CPU]
- **TXG-ENV-002**: When `transaction_graph_populations` is non-empty and native burst is enabled or the decode arm is neither `dense-graphed` nor `dense-eager`, the model shall raise a configuration error at construction.
  - Design: Supported Envelope
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_option_requires_native_burst_off_and_dense_decode` (new) [CPU]
- **TXG-ENV-003**: When `transaction_graph_populations` is non-empty and the runner's `cudagraph_mode` is not `NONE` (including `PIECEWISE` and `FULL`) or `needs_cudagraph_capture()` is true, the model shall raise a configuration error at construction.
  - Design: Supported Envelope
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_option_rejects_every_outer_cudagraph_mode_but_none` (new) [CPU]
- **TXG-ENV-004**: When `transaction_graph_populations` is non-empty and `supported_num_lookahead_tokens` is not exactly `[0]` (g0, the 80 ms cadence), the model shall raise a configuration error at construction.
  - Design: g0 only: configuration and row admission
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_option_rejects_served_geometries_other_than_g0` (new) [CPU]
- **TXG-ENV-005**: When `transaction_graph_populations` is non-empty, if it is not a sequence of distinct integers each in `1..min(max_num_seqs, max_num_batched_tokens)`, then the model shall raise a configuration error at construction.
  - Design: Supported Envelope
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_population_declaration_is_validated` (existing; extend with the `max_num_batched_tokens` bound) [CPU]
- **TXG-ENV-006**: When `transaction_graph_populations` is non-empty and `num_gpu_blocks_override` is set, if any declared population exceeds `num_gpu_blocks_override - 1` (the non-null slot count), then the model shall raise a configuration error at construction.
  - Design: Supported Envelope
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_population_declaration_respects_block_override` (new) [CPU]
- **TXG-ENV-007**: When a request envelope names a geometry outside the configured served geometry set (`supported_num_lookahead_tokens`), with or without `transaction_graph_populations`, `NemotronASRMultiModalProcessor` shall reject that input with an error before the offending input reaches engine execution; for an unadmitted request nothing is admitted, and for an admitted stream (whose empty stream RFC 1 admits before any input) the stream's abort and fenced retirement are the core-owned frontend-preprocessing-failure path; because the rejection happens in the frontend processor, the offending input never reaches `bind_rows` (TXG-ENV-008), and other sessions' isolation follows from that core-owned abort path.
  - Design: g0 only: configuration and row admission
  - Test: `tests/models/nemotron_asr/test_processor.py::test_envelope_outside_served_geometries_is_rejected` (new; drives `NemotronASRMultiModalProcessor` directly, the path the in-process engine tests bypass) [CPU]
- **TXG-ENV-008**: If a row's envelope header names a geometry outside the configured served geometry set, with or without `transaction_graph_populations`, then `bind_rows` shall raise a host structural failure before its delta commit, leaving the session registry unchanged and running no transaction, eager or replayed, for that step.
  - Design: g0 only: configuration and row admission
  - Test: `tests/models/nemotron_asr/test_plan_local.py::test_bind_rows_rejects_geometry_outside_served_set` (new) [CPU]

### Memory admission (MEM)

- **TXG-MEM-001**: Where `transaction_graph_populations` is absent or empty, `prepare_execution_memory` shall perform no transaction-graph admission capture, allocate no stand-in storage and create no admission holder.
  - Design: Ordering
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_option_off_performs_no_admission` (new) [CPU]
- **TXG-MEM-002**: When `transaction_graph_populations` is non-empty, `prepare_execution_memory` shall, after the nested inventory is ready and before the eager profile transactions, capture every declared cell (never a subset or an extrapolation) over stand-in storage of `max(populations) + 1` slots with the real persistent-state layout into a dedicated graph pool, using the same capture path, cell order, capture stream and replay self-check as serving capture.
  - Design: Ordering
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_admission_captures_full_inventory_before_eager_profiles` (new; call order and captured-cell inventory) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_admission_and_serving_capture_share_cell_order` (new; capture stream identical across both inventories and the admission graph pool handle distinct from the platform and nested pools) [CUDA]
- **TXG-MEM-003**: When `transaction_graph_populations` is non-empty, every admission reading `N()` (`N0`, `Na`, `N1`, `Ns0`, `Ns1`) shall be a `mem_get_info` free-memory value taken after a device synchronize, `gc.collect()` and an allocator `empty_cache()`; `Na` shall be read after every capture-only temporary (page snapshots, expected outputs, capture and verify callbacks, capture inputs) is cleared and while the admission holder still retains the stand-in storage, and `N1` after that storage is released; and the model shall compute `A = N0 - N1` and `S = Ns0 - Ns1`.
  - Design: Normalized measurement; Ordering
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_normalized_readings_synchronize_collect_and_empty_first` (new; also the order: temporaries cleared, `Na`, stand-in release, `N1`) [CPU, patched device calls]
- **TXG-MEM-004**: When `transaction_graph_populations` is non-empty, if `N1 - Na` is less than the stand-in storage bytes after admission release, or any owning reference to the stand-in storage survives its release, then startup shall fail with an error naming `transaction_graph_populations` and reporting both figures.
  - Design: Ordering
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_unverified_admission_release_fails_startup` (new; injected readings, and a weak reference to the stand-in storage kept alive by an injected owner) [CPU]
- **TXG-MEM-005**: When `transaction_graph_populations` is non-empty, `capture_transaction_graphs` shall release the admission holder and its graph pool before reading `Ns0`, and shall capture the serving inventory into a fresh dedicated graph pool distinct from the admission pool.
  - Design: Ordering
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_serving_capture_releases_admission_before_measuring` (new; release before the `Ns0` read, call order) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_serving_pool_is_distinct_from_released_admission_pool` (new; pool handles differ, `Ns0` read after the admission pool is released) [CUDA]
- **TXG-MEM-006**: When `transaction_graph_populations` is non-empty, startup shall succeed only if every declared cell is published and `S <= A` with zero tolerance; if `S > A` by any amount, then the binding shall discard every graph and `initialize_kv_cache` shall raise an error naming `transaction_graph_populations` and reporting `S` and `A`.
  - Design: Ordering
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_serving_bytes_above_admission_fail_startup` (new) [CPU, injected readings including `S = A + 1`]
- **TXG-MEM-007**: When `transaction_graph_populations` is non-empty and neither `num_gpu_blocks_override` nor `kv_cache_memory_bytes` is set, `non_kv_cache_memory` from the memory-profiling window shall include the retained admission inventory and the admission preparation and capture peak, with no deduction for retained memory the peak headroom also counts, so resident capacity is fixed from the remaining budget; and the admission holder shall be alive when `prepare_execution_memory` returns, stay alive through the eager profile transactions and the window close, and be released only in `capture_transaction_graphs`.
  - Design: Budget mechanics; Ordering
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_admission_holder_lives_until_serving_capture` (new; event order of holder release against `prepare_execution_memory` return, eager profiles, window close and `capture_transaction_graphs`) [CPU]; `tests/v1/engine/test_nemotron_asr.py::test_nemotron_engine_transaction_graph_admission_is_charged` (new; option-on minus option-off: `non_kv_cache_memory >= A` as the integration check, `total_consumed >= A` for retention, `transient_peak_headroom` not smaller, and a memory-bound run whose block count equals the count core derives from the reduced budget) [engine-CUDA]
- **TXG-MEM-008**: When `transaction_graph_populations` is non-empty and `num_gpu_blocks_override` or `kv_cache_memory_bytes` is set, the model shall still measure and report `A` and check `S <= A` under either override, and shall not itself change the resident block count, which core derives from the override once its budget validation accepts the configuration; and if serving capture then raises or runs out of memory, then startup shall fail with an admission-failure error naming `transaction_graph_populations`. Planner rejection of an oversized override is TXG-MEM-011 and KV-allocation failure is TXG-MEM-012, both core-owned.
  - Design: Capacity overrides
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_override_capacity_reports_admission_and_wraps_failures` (new; `A` and the `S <= A` check under each override, the model requesting no block-count change, and serving-capture failure) [CPU]
- **TXG-MEM-009**: When `transaction_graph_populations` is non-empty, if admission capture raises or runs out of memory inside the profile window, then `prepare_execution_memory` shall raise an error naming `transaction_graph_populations` and latch `_execution_memory_failed`, and for an out-of-memory error the message shall report the normalized bytes admitted so far, the cells captured, the failing cell label and the device's free and total memory, and no projected total for the unfinished inventory.
  - Design: Failure modes and diagnostics
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_admission_oom_latches_and_reports_progress` (new) [CPU, injected OOM]
- **TXG-MEM-010**: When `transaction_graph_populations` is non-empty, if serving capture raises, or the usable slot count after capacity is fixed is less than `max(populations)`, then the binding shall discard every graph and `initialize_kv_cache` shall raise an error naming `transaction_graph_populations`, and no step shall be served eager in its place.
  - Design: Lifetime: one serving capture per worker; Failure modes and diagnostics
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_failed_capture_publishes_nothing_and_every_call_runs_eager` (existing; rewrite to assert startup failure) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_serving_capture_rejects_too_few_usable_slots` (new) [CPU]
- **TXG-MEM-011**: When `transaction_graph_populations` is non-empty, if core budget validation rejects the persistent-state budget after admission, because fewer than two pages fit or a set `num_gpu_blocks_override` exceeds the remaining budget, then startup shall fail with an error naming the model-supplied contributor label (for this model, `transaction_graph_populations`), and no step shall be served eager in its place. How the model supplies that label is the core persistent-state design's choice.
  - Design: Failure modes and diagnostics; Core Obligations, item 4 (deferred until core lands it)
  - Test: `tests/v1/core/test_kv_cache_planning.py::test_state_planner_rejection_names_the_contributor_label` (new; requires the core change; generic label; too-small budget and oversized override, parametrized) [CPU]
- **TXG-MEM-012**: When `transaction_graph_populations` is non-empty, if the KV allocation in `initialize_kv_cache` raises or runs out of memory before `initialize_state_slots`, then startup shall fail with an error naming the model-supplied contributor label (for this model, `transaction_graph_populations`), and no step shall be served eager in its place.
  - Design: Capacity overrides; Failure modes and diagnostics; Core Obligations, item 5 (deferred until core lands it)
  - Test: `tests/v1/worker/test_gpu_model_runner_v2.py::test_state_kv_allocation_failure_names_the_contributor_label` (new; requires the core change; generic label; injected allocation failure) [CPU]

### Capture lifetime (LIFE)

- **TXG-LIFE-001**: When `transaction_graph_populations` is non-empty, the model shall run serving capture exactly once per worker, from `initialize_state_slots` after the resident pools are allocated and before any session is admitted, and `binding.ready` shall become true only after every declared cell is published.
  - Design: Lifetime: one serving capture per worker
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_capture_input_stores_no_page_and_publishes_every_cell` (existing) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_serving_capture_runs_once_from_initialize_state_slots` (new) [CPU]
- **TXG-LIFE-002**: When `transaction_graph_populations` is non-empty and `initialize_state_slots` repeats with identical pool layout and call bindings, the binding shall perform no capture, and its graph objects, `captured_cells` count and receipt capture order shall be unchanged.
  - Design: Lifetime: one serving capture per worker
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_identical_reinitialize_is_a_verified_no_op` (new) [CPU]
- **TXG-LIFE-003**: When `transaction_graph_populations` is non-empty, if `initialize_state_slots` repeats with any difference in pool layout or call bindings, then the model shall raise a worker-fatal `RuntimeError` without capturing, and after startup the binding shall never capture a cell, so `captured_cells` stays constant across signature fallbacks.
  - Design: Lifetime: one serving capture per worker
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_changed_pool_layout_on_reinitialize_is_fatal` (new) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_relocated_pools_fall_back_instead_of_replaying` (existing; extend to assert `captured_cells` unchanged) [CPU]
- **TXG-LIFE-004**: When `transaction_graph_populations` is non-empty and a captured cell (admission or serving) passes verification, the binding shall clear its capture context so that the cell holds no reference to the capture `TransactionInputs`, the pools reached through them, page snapshots, expected outputs, or capture and verify callbacks; for admission cells this clearing shall complete before `Na` is read, and the stand-in storage shall then be retained only by the admission holder's explicit reference until its release after `Na`.
  - Design: Capture-lifetime rule; Ordering
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_published_cell_retains_no_capture_context` (new; weak references dead after `gc.collect()`, and for admission the only live referrer of the stand-in storage at `Na` is the holder) [CPU]
- **TXG-LIFE-005**: When `transaction_graph_populations` is non-empty, the model shall hold admission graphs only in an admission holder that is absent from `_graphs` and `_pool_signature`, has no replay path, and leaves `binding.ready` false and the replay counters at zero.
  - Design: Capture-lifetime rule
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_admission_holder_is_never_published` (new) [CPU]
- **TXG-LIFE-006**: When `transaction_graph_populations` is non-empty and `TransactionGraphBinding.discard()` runs, the binding shall drop its graphs, arenas and owned buffers, leaving no published cell, `binding.ready` false and no live reference to any arena.
  - Design: Lifetime: one serving capture per worker
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_discard_leaves_no_published_cell_or_arena` (new; weak references dead after `gc.collect()`) [CPU]
- **TXG-LIFE-007**: When `transaction_graph_populations` is non-empty and `GPUModelRunner.shutdown` runs, the core `ModelState` shutdown hook shall call the binding's `discard()` before the model state is released.
  - Design: Lifetime: one serving capture per worker; Core Obligations, item 3 (deferred until core lands it)
  - Test: `tests/v1/worker/test_gpu_model_runner_v2.py::test_shutdown_calls_model_state_shutdown_hook` (new; requires the core hook) [CPU]
- **TXG-LIFE-008**: When `transaction_graph_populations` is non-empty, the model shall reach the unowned resident pools for serving capture and the rejected-row reference only through the core-authorized startup pool access, use it only during `initialize_state_slots`, and commit no page through it.
  - Design: Lifetime: one serving capture per worker; Core Obligations, item 6 (deferred until core lands it)
  - Test: `tests/v1/worker/test_gpu_model_runner_v2.py::test_startup_pool_access_expires_after_initialize_state_slots` (new; requires the core change; generic model state) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_capture_reaches_pools_only_through_the_startup_accessor` (new; requires the core change; never reads `views._views`) [CPU]

### Replay exactness (REP)

- **TXG-REP-001**: When `transaction_graph_populations` is non-empty, the binding shall declare exactly the cells `chunk-fresh:g0:b<P>`, `chunk-continuing:g0:b<P>` and `replay:b<P>` for each declared population `P`, and no other cell.
  - Design: Cells and classification; g0 only: configuration and row admission
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_capture_input_stores_no_page_and_publishes_every_cell` (existing; update to g0-only labels) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_receipt_captures_largest_population_first_and_reports_cost` (existing) [CPU]
- **TXG-REP-002**: When `transaction_graph_populations` is non-empty, `classify_transaction` shall return the `chunk-fresh` cell for one CHUNK bucket whose rows are all first CHUNKs, the `chunk-continuing` cell for one CHUNK bucket with no fresh row, and the `replay` cell for steps with no CHUNK row and no fresh row, each with population equal to the number of rows in the step (admitted, parked or never-fed slots without a row in the step never counted), and otherwise `(None, reason)` with reason `empty`, `fresh-non-chunk`, `multi-geometry`, `mixed-roles` or `mixed-fresh` as defined in the reason table.
  - Design: Cells and classification
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_classification_covers_declared_shapes_only` (existing; extend to every kind and reason, and to a step taken while other admitted, parked and never-fed slots are bound) [CPU]
- **TXG-REP-003**: When `transaction_graph_populations` is non-empty and a call has no cell, has an undeclared population, matches a declared cell with no published graph (`uncaptured`), or runs in capture or memory-profile mode, `execute` shall run the eager transaction with results bitwise equal to eager, increment `fallbacks:<cell label or uncovered>:<reason>` by exactly one, and leave every replay counter unchanged; once startup has succeeded, no call shall be counted `uncaptured`; and because native burst is off whenever the option is on (TXG-ENV-002), no call shall be counted `native-burst`.
  - Design: Cells and classification; Execute order
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_uncovered_calls_run_eager_and_are_counted` (existing; extend parameters to every reachable reason) [CPU]
- **TXG-REP-004**: When `transaction_graph_populations` is non-empty, `transaction_signature` shall differ whenever exactly one of population, control ids, `endpoint_enabled`, `n_fresh`, decode or transition callable identity, resident pool address, shape or stride, lazily bound encoder buffer layout, input shapes, or arena name/dtype/shape layout differs, and shall be equal for calls that differ only in arena data (slot ids, block indices, row values).
  - Design: Inputs, outputs and signature; Host control flow inside the body
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_signature_covers_every_structural_input` (new) [CPU]
- **TXG-REP-005**: When `transaction_graph_populations` is non-empty and a call matches a declared cell whose captured signature, including arena layout, differs from the call's, `execute` shall detect the mismatch before any arena upload (both pinned host buffers and the device arena byte-identical), run eager over the live pools with results, including committed pages, bitwise equal to eager, leave captured storage that is distinct from the live pools untouched, and increment `fallbacks:<cell label>:signature` by one.
  - Design: Inputs, outputs and signature; Echo eligibility, mixed steps and signature mismatch; Execute order
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_relocated_pools_fall_back_instead_of_replaying` (existing) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_arena_layout_mismatch_is_a_signature_fallback` (new; execution-level, live-pool commit bitwise equal to eager) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_arena_rejects_vectors_outside_its_layout` (existing; arena-level guard retained) [CPU]
- **TXG-REP-006**: When `transaction_graph_populations` is non-empty, the dense decode binding and the CHUNK-bucket binding shall return the identical callable object (same `id`) for the same graph key across calls for the binding's lifetime, and the CHUNK-bucket binding shall reject a callable it did not seal, so the signature of two structurally identical calls is equal.
  - Design: Callable identity
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_signature_callable_identities_are_stable_across_calls` (new) [CPU]
- **TXG-REP-007**: While `outer_transaction_capture` is active, `nested_graph_mode` shall return the eager mode at every nested launch site (encoder, decode and CHUNK bucket) and any host-to-device staging shall raise; while it is inactive, `nested_graph_mode` shall return the configured mode.
  - Design: Flat capture
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_nested_bindings_run_eagerly_only_inside_the_capture_scope` (existing; extend to drive each binding's launch entry point, encoder, decode and CHUNK bucket, and assert no nested graph launch inside the scope) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_host_staging_is_rejected_inside_the_capture_scope` (existing) [CPU]
- **TXG-REP-008**: When `transaction_graph_populations` is non-empty and a cell replays, `execute` shall perform exactly one host-to-device copy into the cell's arena, alternate its two pinned host buffers across calls, and rewrite a host buffer only after the event fencing that buffer's previous upload has completed, with uploaded bytes equal to live staging.
  - Design: Inputs, outputs and signature
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_arena_alternates_host_buffers_across_calls` (existing; extend with explicit event stubs, since CPU arenas hold no events, to count copies and assert each buffer's wait precedes its rewrite) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_arena_stager_matches_live_staging_bit_identical` (existing) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_arena_reuses_pinned_buffers_after_async_upload` (new; pinned host buffers, asynchronous uploads, uploaded bytes equal to live staging across reuse) [CUDA]
- **TXG-REP-009**: When `transaction_graph_populations` is non-empty and a cell is captured, the binding shall run the eager reference on capture rows the transaction rejects (leaving every resident page byte-identical), then three eager warm-ups, the capture and one replay, and shall publish the cell only if that replay reproduces the reference outputs and pages bitwise; if it does not, then the cell shall not be published and startup shall fail with an error naming the cell label.
  - Design: Capture and the replay-cell oracle
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_capture_input_stores_no_page_and_publishes_every_cell` (existing) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_capture_self_check_mismatch_publishes_nothing` (new) [CPU]
- **TXG-REP-010**: When `transaction_graph_populations` is non-empty and accepted CHUNK rows replay a `chunk-fresh` or `chunk-continuing` cell at populations 1, 31 and 63, with `endpoint_enabled` true and false and with block ids, row orders and prompts that differ from the capture inputs, results shall be bitwise equal to eager; in a homogeneous lockstep workload (every session sends its CHUNK in the same step, so each CHUNK step is one all-fresh or all-continuing bucket at a declared population), every CHUNK step shall replay its cell with zero CHUNK-step fallbacks, and echo steps at undeclared populations are counted `undeclared`.
  - Design: Context and Design Philosophy; Invariants
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_population_lockstep_replays_every_chunk` (existing; extend with endpointing disabled) [CUDA]; `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_graph_replay_matches_eager_across_varied_plans` (existing) [CUDA]
- **TXG-REP-011**: When `transaction_graph_populations` is non-empty and a `replay` cell replays control rows, results shall be bitwise equal to eager with zero fallbacks for accepted park-echo and FLUSH rows with `endpoint_enabled` true and false, and for accepted forced-endpoint rows with `endpoint_enabled` true; with `endpoint_enabled` false, a forced-endpoint row shall be rejected with the same row status, outputs and unchanged pages as eager. Utterance endpointing is a model capability, independent of whether any endpoint plugin is loaded.
  - Design: Capture and the replay-cell oracle
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_replay_cell_accepted_control_rows_match_eager` (new; accepted park-echo and FLUSH × `endpoint_enabled` true and false, accepted forced-endpoint with `endpoint_enabled` true, rejected forced-endpoint with `endpoint_enabled` false) [CUDA]
- **TXG-REP-012**: When `transaction_graph_populations` is non-empty and a covered step contains rows the transaction rejects (out-of-sequence CHUNK, wrong echo token, corrupt book), replay shall produce the same nonzero row statuses, outputs and pages as eager, leaving the rejected rows' pages unchanged and committing the other rows; commit-sink collection shall report the same failed request ids as eager, which `postprocess_state` passes to `state_slots.fail_requests` (the step's `failed_req_ids`); the step shall not raise, and the binding shall not latch.
  - Design: Capture and the replay-cell oracle; Failure Containment
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_graph_replay_rejects_corrupt_rows_like_eager` (existing; extend to compare `failed_req_ids` between arms and assert no latch) [CUDA]; `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_one_corrupt_row_in_a_population_is_rejected_alone` (existing; extend likewise) [CUDA]
- **TXG-REP-013**: When `transaction_graph_populations` is `[1]` or `[31]` at g0 through an `EngineCore`, the emitted tokens, echo counts and resident storage after every park shall be bitwise equal to the same run with the option off; in the homogeneous lockstep workload (exactly the declared population of sessions, all sending each CHUNK in the same step), every CHUNK step shall replay its declared cell with zero CHUNK-step fallbacks, and echo steps at undeclared populations are counted `undeclared`; mixed or staggered steps keep eager equivalence and are counted under their fallback reasons.
  - Design: Context and Design Philosophy
  - Test: `tests/v1/engine/test_nemotron_asr.py::test_nemotron_engine_transaction_graph_matches_eager` (existing; extend to compare the per-chunk echo counts both arms return, which it now discards) [engine-CUDA]; `tests/v1/engine/test_nemotron_asr.py::test_nemotron_engine_transaction_graph_b31_lockstep_matches_eager` (existing) [engine-CUDA]
- **TXG-REP-014**: When `transaction_graph_populations` is non-empty, before `execute` classifies the call, and therefore before it replays or returns from any fallback (uncovered, undeclared, mixed-row or signature), it shall compare the live batch-invariant mode, TF32 enablement, `torch.get_float32_matmul_precision()` and `CUBLAS_WORKSPACE_CONFIG` with their capture-time values using host-only reads, and if any differs, then it shall raise a worker-fatal `RuntimeError` naming the changed constant without replaying, running eager or incrementing any counter.
  - Design: Process constants and excluded paths; Execute order
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_process_constant_change_is_fatal_on_every_path` (new; parametrized over constant × path) [CPU]
- **TXG-REP-015**: When `transaction_graph_populations` is non-empty and a cell replays, the arena upload, static-input copies, the replay and every resident write shall be issued on the runner's current stream.
  - Design: Invariants
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_replay_runs_on_the_current_stream` (new; non-default stream, pages bitwise equal after synchronizing only that stream, plus stream tracing of every copy, replay and write, for example `torch.profiler` events or an interception of the copy and replay entry points, that fails on work issued to any other stream) [CUDA]
- **TXG-REP-016**: When `transaction_graph_populations` is non-empty, no two published cells shall share an arena buffer, `static_ids`, `static_embeds` or owned output buffer (disjoint storage), and owned outputs shall be allocated outside the graph pool.
  - Design: Inputs, outputs and signature; Invariants
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_cells_share_no_arena_static_input_or_owned_output` (new; storage disjointness) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_owned_outputs_are_allocated_outside_the_graph_pool` (new; allocator provenance, for example `torch.cuda.memory._snapshot()` segment pool ids, for rows, status and progress) [CUDA]
- **TXG-REP-017**: When `transaction_graph_populations` is non-empty and eager fallback steps, which launch the nested encoder, decode and CHUNK-bucket bindings, interleave with replayed steps, results shall be bitwise equal to the same sequence run entirely eager.
  - Design: Invariants
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_interleaved_fallbacks_and_replays_match_eager` (new) [CUDA]

### Post-warmup verification (VER)

- **TXG-VER-001**: When `transaction_graph_populations` is non-empty and the model's post-warmup verification function runs, it shall verify every published cell's signature and the address and size of every binding-external buffer its graph reads, and if any differs from its capture-time value, then it shall raise an error naming `transaction_graph_populations`, the cell and the buffer.
  - Design: Post-Warmup Verification
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_post_warmup_verification_rejects_resized_external_buffer` (new; calls the verification function directly with a resized stub buffer) [CPU]
- **TXG-VER-002**: When `transaction_graph_populations` is non-empty and no core post-warmup hook exists, the model shall run the same verification exactly once, on the first non-dummy `prepare_step`, before registry pruning, `state_slots.begin_step`, request activation and any transaction, and dummy steps shall not consume it; every non-dummy `prepare_step`, the first included, shall also check the binding's failed latch at that point and raise the TXG-FAIL-004 error if it is set.
  - Design: Post-Warmup Verification
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_first_real_step_verifies_once_before_pruning` (new; also a latched binding failing a later `prepare_step` before pruning, with verification not rerun) [CPU]
- **TXG-VER-003**: When `transaction_graph_populations` is non-empty and no core post-warmup hook exists, if the first-real-step verification finds a changed signature or binding-external buffer address or size, then the model shall raise a worker-fatal `RuntimeError` naming the cell and buffer and latch the binding, with the session registry unchanged and no resident page written, so every later non-dummy `prepare_step` fails at its latch check (TXG-VER-002).
  - Design: Post-Warmup Verification
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_first_real_step_mismatch_is_fatal_before_any_state_change` (new) [CPU]
- **TXG-VER-004**: When `transaction_graph_populations` is non-empty and the core `ModelState` post-warmup hook exists, `_compile_or_warm_up_model` shall call it after every worker warmup and before the worker serves, the hook shall run the model's post-warmup verification, and a mismatch shall fail startup.
  - Design: Post-Warmup Verification; Core Obligations, item 2 (deferred until core lands it)
  - Test: `tests/v1/worker/test_gpu_model_runner_v2.py::test_post_warmup_hook_runs_after_warmups` (new; requires the core hook) [CPU]

### Failure containment (FAIL)

- **TXG-FAIL-001**: When `transaction_graph_populations` is non-empty, if a covered call fails before `graph.replay()` (classification, signature, arena upload, static-input copy, escaped-output allocation or commit reservation), then `execute` shall raise the exception type and message the eager transaction raises for the same input where eager has one, leave every resident page bitwise equal to the post-`begin_step` baseline, cancel a reserved commit ticket exactly once, and increment no replay or fallback counter.
  - Design: Failure Containment
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_stale_lease_fails_before_replay_writes_a_page` (existing) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_pre_boundary_failure_keeps_baseline_and_cancels_ticket_once` (new; fault injected at each of classification, signature, arena upload, static-input copy, escaped-output allocation and commit reservation, parametrized; asserts the post-`begin_step` baseline, cancellation exactly once only where a ticket was reserved, and unchanged replay counts) [CPU]
- **TXG-FAIL-002**: When `transaction_graph_populations` is non-empty and a cell replays, `execute` shall reserve the commit before any resident page write and stage status only after the replay, so the commit sink records exactly `reserve` then `stage`, and no request abort shall cancel that reservation: a launched transaction completes stage and collection, and ticket cancellation occurs only as error cleanup (TXG-FAIL-001, TXG-FAIL-004).
  - Design: Transaction Structure; Invariants
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_replay_reserves_before_any_page_write_and_stages_after` (existing; extend with a request aborted after the replay, asserting `reserve`, `stage` and collection with no cancel) [CPU]
- **TXG-FAIL-003**: When `transaction_graph_populations` is non-empty and a cell replays, `execute` shall allocate every buffer the step needs, including the escaped row tensor of shape `(population, hidden_size)` float32, before `graph.replay()`, the device allocator shall record no allocation from the replay to the return, and the returned row tensor shall own storage distinct from every cell buffer and stay unchanged across any later replay, and the cell's owned status and progress buffers shall keep their contents across other cells' replays until the same cell's next replay.
  - Design: Failure Containment; Escaped outputs (Ordering)
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_replay_allocates_nothing_after_launch` (new) [CUDA]; `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_replayed_rows_survive_another_cells_replay` (existing) [CUDA]; `tests/models/nemotron_asr/test_transaction_graph.py::test_replayed_rows_are_owned_by_the_caller` (existing) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_status_and_progress_survive_another_cells_replay` (new; cell A, then cell B, then A's status and progress compared with A's eager values before A replays again) [CUDA]
- **TXG-FAIL-004**: When `transaction_graph_populations` is non-empty, if any exception arises in `execute` from `graph.replay()` onward, including `CommitTicket.stage` failing after it set `_done`, or from a worker-fatal check (process constants, wrapper entry, post-warmup, changed reinitialization per TXG-LIFE-003), then the binding shall latch as failed and propagate the exception, and every later `execute` call, covered or fallback, shall raise `RuntimeError` containing "a fresh worker is required" before any classification, upload or replay, incrementing no counter; every later non-dummy `prepare_step` shall raise the same error (TXG-VER-002).
  - Design: Failure Containment; Execute order; Post-Warmup Verification
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_replay_failure_cancels_the_reservation_once` (existing; extend with the latch and a later call) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_stage_failure_after_done_latches_the_binding` (new) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_every_fatal_source_latches_before_classification` (new; parametrized over replay, stage, process constant, wrapper entry, post-warmup and changed reinitialization × a later covered and a later fallback call) [CPU]
- **TXG-FAIL-005**: When `transaction_graph_populations` is non-empty and `execute` raises inside an `EngineCore` step, the exception shall propagate out of `execute_model` and `EngineCore.step` with no progress from that step published. Stopping later steps is the core engine-death path (`EngineCoreProc`), not a model obligation; the model adds no in-process recovery guard.
  - Design: Failure Containment
  - Test: `tests/v1/engine/test_nemotron_asr.py::test_nemotron_engine_transaction_graph_replay_failure_propagates_from_step` (new; injected replay failure, in-process `EngineCore`; asserts propagation and no published progress) [engine-CUDA]
- **TXG-FAIL-006**: When `transaction_graph_populations` is non-empty, if the commit-sink collection in `postprocess_state` raises after a replayed step (the surfacing point of an asynchronous device fault), then no progress from that step shall be published and the exception shall propagate out of `EngineCore.step`. Collection runs outside `execute`, so this failure does not set the TXG-FAIL-004 latch; stopping later steps is the core engine-death path, as for TXG-FAIL-005.
  - Design: Failure Containment
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_collection_failure_after_replay_publishes_no_progress` (new) [CPU]; `tests/v1/engine/test_nemotron_asr.py::test_nemotron_engine_transaction_graph_collection_failure_propagates_from_step` (new; injected collection failure, in-process `EngineCore`; asserts propagation and no published progress) [engine-CUDA]
- **TXG-FAIL-007**: When `transaction_graph_populations` is non-empty and a commit reservation is cancelled after a replay raised, or a request is aborted after a replay wrote its resident pages, the model shall leave those pages byte-identical to their post-replay contents until core rebinds the slot for a new generation or a later authorized write changes them.
  - Design: Failure Containment; Invariants
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_abort_after_replay_does_not_roll_back_pages` (new; pages compared after the abort and its collection, before any rebind) [CPU]
- **TXG-FAIL-008**: When `transaction_graph_populations` is non-empty, if a call that passed classification and the signature check (Execute order) finds that its cell's wrapper entry no longer holds its graph (for example after `CUDAGraphWrapper.clear_all_graphs()`), then `execute` shall raise a worker-fatal `RuntimeError` before arena upload, capture nothing (`captured_cells` unchanged) and leave every resident page unchanged.
  - Design: Failure Containment
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_cleared_wrapper_entry_is_fatal_and_never_captures` (new; also a cleared entry with a signature mismatch, which is a counted `signature` fallback and not fatal) [CPU]

### Observability (OBS)

- **TXG-OBS-001**: Where `transaction_graph_populations` is absent or empty, `transaction_graph_counters` shall return an empty mapping for a reachable model, runner, worker, executor or in-process `EngineCore`, and the execution profile receipt shall contain no transaction-graph entry.
  - Design: Observability
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_counters_reach_the_model_from_an_engine_core` (existing; off branch) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_option_off_receipt_has_no_transaction_graph_entry` (new) [CPU]; `tests/v1/engine/test_nemotron_asr.py::test_nemotron_engine_transaction_graph_matches_eager` (existing; eager arm `== {}`) [engine-CUDA]
- **TXG-OBS-002**: When `transaction_graph_populations` is non-empty, `transaction_graph_counters` shall accept a model, runner, worker, executor or in-process `EngineCore` and return `captured_cells`, replays per cell label, fallbacks per `(cell, reason)` and prologue time from host state, without any device synchronization.
  - Design: Observability
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_counters_reach_the_model_from_an_engine_core` (existing; extend to every target with synchronization patched to fail, a proxy that cannot see implicit synchronization) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_receipt_reports_admission_and_serving_bytes` (new; reads the counters under `torch.cuda.set_sync_debug_mode("error")`) [CUDA]
- **TXG-OBS-003**: If the model is unreachable in process (for example under a multiprocess executor), with or without `transaction_graph_populations`, then `transaction_graph_counters` shall raise rather than return an empty mapping.
  - Design: Observability
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_counters_raise_when_the_model_is_unreachable` (new) [CPU]
- **TXG-OBS-004**: When `transaction_graph_populations` is non-empty, the execution profile receipt shall include readiness; per cell, arena bytes, reserved-memory growth, warm-up seconds, capture seconds and capture index; and `A`, `S`, the release-check result, and the admission and serving capture seconds.
  - Design: Observability
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_receipt_captures_largest_population_first_and_reports_cost` (existing; extend with the admission fields) [CPU]; `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_receipt_reports_admission_and_serving_bytes` (new; also the TXG-OBS-002 implicit-synchronization check) [CUDA]
- **TXG-OBS-005**: When `transaction_graph_populations` is non-empty and a step replays a cell, the encoder invocation counters and CHUNK-binding fallback counts shall not advance, per-step batch stats shall advance, and `replays:<cell label>` shall increase by exactly one.
  - Design: Host control flow inside the body
  - Test: `tests/models/nemotron_asr/test_transaction_graph.py::test_cuda_replay_freezes_nested_counters_and_counts_once` (new) [CUDA]

### Carried core warmup fix (CORE; owned by the core persistent-state design)

- **TXG-CORE-001**: When any KV cache group's spec is a `PersistentStateSpec`, `run_mixed_prefill_decode_warmup` shall return `False` without invoking `worker_execute_model` or `worker_sample_tokens`.
  - Design: Core Dependency: Persistent-State Warmup (carried); Core Obligations, item 1
  - Test: `tests/v1/worker/test_mixed_warmup_gate.py::test_mixed_warmup_skipped_for_persistent_state_groups` (existing) [CPU]
- **TXG-CORE-002**: When no KV cache group is persistent state, `run_mixed_prefill_decode_warmup` shall keep every existing gate: it shall skip the warmup without invoking the worker callbacks for a pooling model, when `max_num_reqs < 2`, when `num_tokens < 3`, or when the KV cache has too few blocks for the warmup requests, and shall otherwise run it.
  - Design: Core Dependency: Persistent-State Warmup (carried); Core Obligations, item 1
  - Test: `tests/v1/worker/test_mixed_warmup_gate.py::test_mixed_warmup_disables_watermarking` (existing; run path) [CPU]; `tests/v1/worker/test_mixed_warmup_gate.py::test_mixed_warmup_skipped_for_single_seq` (existing; `max_num_reqs < 2`) [CPU]; `tests/v1/worker/test_mixed_warmup_gate.py::test_mixed_warmup_skipped_for_pooling_short_and_block_starved_runs` (new; pooling model, `num_tokens < 3`, insufficient blocks, parametrized) [CPU]

### Consistency report

- **Coverage.** Every target behaviour has a criterion except: the slot-reuse invariant between commit reservation and collection and replay serialization (Invariants), which rest on core synchronous scheduling for persistent state and are tested there; nested-scratch write-before-read, a model property whose behavioural coverage is TXG-REP-017; the sleep-mode, CuMem and weight-reload rejections, which are existing persistent-state contracts owned elsewhere; "a nonzero `signature` count is a defect", an operating judgement observed through TXG-REP-005's counter; and the client-level `EngineDeadError` after a fatal step, which is the core-owned `EngineCoreProc` busy-loop path (`core.py:1668-1674`), which stops later steps and is covered generically by `tests/v1/shutdown/test_forward_error.py`, with no existing harness driving a Nemotron streaming session through the multiprocess client (TXG-FAIL-005/006 observe propagation out of `EngineCore.step` instead). The Default-On Gate (including the `EngineCoreProc` counter reader; the counters are an in-process harness facility here), Open Questions probes, the startup-count proposal and the exploratory measurements carry no criteria by decision.
- **Core-owned.** Core Obligations lists every obligation on the core persistent-state design with its criteria. TXG-CORE-001 and TXG-CORE-002 are the carried warmup fix, testable now on this branch. TXG-VER-004, TXG-LIFE-007, TXG-LIFE-008, TXG-MEM-011 and TXG-MEM-012 are deferred until core lands them; the model-side halves (TXG-VER-001, TXG-LIFE-006, TXG-LIFE-001, the serving-capture case of TXG-MEM-008) are tested directly.
- **Conservative defaults.** TXG-MEM-006 fixes zero tolerance for `S <= A`, and TXG-MEM-008 keeps it under both capacity overrides; TXG-MEM-002 requires the complete inventory, so no cross-cell reuse is assumed; TXG-MEM-007 forbids deducting double-counted peak.
- **Contradictions.** The header status said acceptance criteria were "to be added"; it now points here. Existing tests that assert superseded behaviour are marked for rewrite: `test_failed_capture_publishes_nothing_and_every_call_runs_eager` (target: startup failure, TXG-MEM-010) and the `g2` cells in `test_capture_input_stores_no_page_and_publishes_every_cell` and `_replay_parity` (target: g0 only, TXG-REP-001). `test_arena_rejects_vectors_outside_its_layout` stays as an arena-level guard beside the execution-level `signature` fallback (TXG-REP-005). `test_cuda_graph_replay_matches_eager_across_varied_plans` excludes park tokens and is not park-echo coverage (TXG-REP-011).
- **Implicit scoping resolved.** Processor and `bind_rows` geometry enforcement (TXG-ENV-007, TXG-ENV-008) and the unreachable-model reader (TXG-OBS-003) are option-independent, because the design passes the configured set to the plan context unconditionally. "Worker-fatal" is read as the latch of TXG-FAIL-004 for every fatal check, including pre-boundary process-constant, wrapper-entry and post-warmup failures; only non-fatal pre-boundary failures follow the structural-failure contract (TXG-FAIL-001). Signature-fallback pool preservation applies only to captured storage distinct from the live pools (TXG-REP-005). Budget-derived capacity applies only without an override, and an override's count only once core budget validation accepts it (TXG-MEM-007, TXG-MEM-008). "Every CHUNK step replays" means zero CHUNK-step fallbacks in the homogeneous lockstep workload, with echo steps at undeclared populations counted `undeclared`; mixed and staggered steps keep eager equivalence and their fallback counts (TXG-REP-010, TXG-REP-013). Request abort is not ticket cancellation, and post-replay page preservation lasts until a rebind or authorized write (TXG-FAIL-002, TXG-FAIL-007). Per-row rejections are classified as eager classifies them, terminal ones as request failures, never step failures (TXG-REP-012). Processor geometry rejection is tested at the processor; the in-process engine tests bypass it (TXG-ENV-007).
- **Context-free reading.** Every criterion names `transaction_graph_populations` and its state in its own clause; cross-file concepts (g0, eager, bitwise equal, configuration error, `endpoint_enabled`, `N()`) carry a parenthetical or the preamble definition.

## References

Tests: `tests/models/nemotron_asr/test_transaction_graph.py`,
`tests/v1/engine/test_nemotron_asr.py`. Related designs:
[CUDA Graphs](cuda_graphs.md), [Model Runner V2](model_runner_v2.md).
