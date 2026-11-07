# Nemotron ASR Whole-Transaction CUDA Graphs

**Status:** opt-in; default-on is gated below. **Acceptance criteria:** to be
added.

## Scope and Provenance

This is the governing design for the `transaction_graph_populations` option of
the Nemotron streaming ASR model (`vllm/models/nemotron_asr/`). It lives beside
the model and moves with it when the model becomes an out-of-tree plugin.
Numbered acceptance criteria take the place of EARS specifications. For
project context only (not a governing parent), see the section "vLLM core
effort and the streaming plugin" of `docs/high-level-design.md` in
`pdtgct/vllm-omni-notes`.

Geometries are named by encoder lookahead: **g0** is the 80 ms cadence
(lookahead 0, `CADENCES` index 0 in `manifests.py`) and **g1** is the 160 ms
cadence (lookahead 1), which is the model's default.

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

### Cells and classification

A `TransactionCell` (`transaction_graph.py:47`) is `(kind, geometry,
population)`, where population is the exact real-row count.

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
| `signature` | structural inputs, including the arena layout, differ from the captured ones |
| `capture`, `memory-profile`, `native-burst` | modes never covered |

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
eager bitwise, including `replay` cells carrying accepted FLUSH,
forced-endpoint and park-echo rows, with endpointing on and off.

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
  before the request reaches the engine, so one bad request fails alone and
  other sessions continue.
- **Per step, in the worker.** The plan context receives the configured set
  instead of every `CADENCES` index. A row whose envelope header names any
  other geometry fails host structural validation in `bind_rows`, which
  validates the complete row set before its delta commit. The failure happens
  before the registry mutates and before any eager path could run the row. This
  is the worker invariant for inputs that bypass the processor or internal
  defects, and it keeps the model's existing structural-failure contract.

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
A repeated `initialize_state_slots` with identical pool layout and call
bindings is a verified no-op; any difference is worker-fatal. Nothing is
recaptured outside startup.

Runner shutdown discards the binding's graphs, arenas and owned buffers.

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
- Resident capacity for persistent state is `count = min(budget //
  page_size_bytes, max_num_seqs + 1)`, replaced by a set
  `num_gpu_blocks_override`; startup fails unless `count >= 2` and `count *
  page_size_bytes <= budget`; usable slots are `count - 1`, excluding the null
  block (`vllm/v1/core/kv_cache_planning.py:1496-1515`). Admission therefore
  reduces usable slots only when memory, not the `max_num_seqs` cap, binds.
- Resident pools exist only inside `initialize_kv_cache`. The capture hook runs
  from `initialize_state_slots` after the pools are allocated
  (`model_runner.py:852-854`, `model_state_v2.py:110-121`).

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
inputs, owned outputs, the signature and the descriptor.

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
   as serving capture; read `Na`; clear every capture context and drop the
   stand-in storage; read `N1`. Release is verified when `N1 - Na >= stand-in
   bytes`, else startup fails. Admitted bytes are `A = N0 - N1`. Stand-in
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

Under `num_gpu_blocks_override` or `kv_cache_memory_bytes` the operator fixes
capacity. Admission is still measured and reported but cannot change that
capacity. If the KV allocation or serving capture then fails or runs out of
memory, the error is caught and reported as an admission failure naming the
option.

### Failure modes and diagnostics

Every case below is a startup failure that names the option. None degrades to
eager serving.

- Admission does not fit (capture failure or OOM in the window):
  `prepare_execution_memory` fails and latches `_execution_memory_failed`
  (`nemotron_asr.py:812-816`).
- Release not verified, no capacity, or too few slots.
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
cross-segment core obligation, owned by the core persistent-state design: a
`ModelState` post-warmup hook called at the end of `_compile_or_warm_up_model`,
where a mismatch is a startup failure. Until it exists, the model runs the same
verification once on the first non-dummy `prepare_step`, before its registry
pruning (`model_state_v2.py:155-158`) and therefore before
`state_slots.begin_step` (`model_runner.py:1837`, `:1845`), request activation
or any transaction; a mismatch there is worker-fatal. This interim is a safety
gate, not startup qualification: the engine may report ready before the first
step discovers a mismatch.

## Failure Containment

The launch boundary is `graph.replay()` in `TransactionGraphBinding.execute`.

**Unchanged-state baseline.** "Unchanged" means the resident state after core
slot initialization in `begin_step` (`model_runner.py:1845`,
`vllm/v1/worker/gpu/state_slots.py:76-85`), which zeroes and initializes newly
bound slots before the model runs.

**Before the boundary**, work touches only binding-owned buffers and host
bookkeeping: classification and signature, process-constant and wrapper-entry
checks, arena upload and static-input copies, escaped-output allocation, and
the commit reservation. A failure here follows the eager structural-failure
contract: the call raises, resident state equals the baseline, and a reserved
ticket is cancelled.

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
  failed and propagates. The ticket's release state after a partial stage is
  irrelevant because the worker is never reused; later `execute` calls raise
  "a fresh worker is required".
- In the engine, the exception leaves `execute_model`, is re-raised from
  `step` (`vllm/v1/engine/core.py:728`, `:623`), and the busy loop signals the
  engine dead (`core.py:1668-1674`).
- Asynchronous device faults surface at the next synchronizing call, normally
  the commit-sink collection (`commit_sink.py:270`, via
  `model_state_v2.py:292`), which fails before any progress is published and
  takes the same path. Faults are not attributed to cells and not resumed.
- In-process callers of `advance_model_rows` see pre-boundary failures as
  structural failures; engine callers see any raise as engine death. Recovery
  is a fresh worker process.
- Cancellation never implies rollback: cancelling a reservation or aborting a
  request leaves resident pages as the replay left them. Slot retirement and
  fencing stay with the core state-slot manager.

This extends the model's existing rule that a catastrophic failure during the
prevalidated commit is worker-fatal (`advance.py:2813-2819`) to the whole
replayed body, because the body and the scatters are one launch.

**Output lifetime.** The escaped row tensor is caller-owned and survives any
later replay. The owned status and progress buffers are valid until the same
cell's next replay. The commit sink consumes them earlier in the step.

## Observability

- `vllm.models.nemotron_asr.transaction_graph:transaction_graph_counters` takes
  a model, runner, worker, executor or in-process EngineCore. It returns
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
is persistent state (`vllm/v1/worker/gpu/warmup.py:101-106`). The warmup runs
only when `max_num_reqs >= 2`, so multi-session populations reach it first.
Non-state models are unchanged.

## Default-On Gate (future intent)

Turning the option on by default requires, against a declared baseline
configuration: correctness with the model installed as an out-of-tree plugin;
serving latency and sustained-concurrency evidence at the served cadence from
unprofiled serving measurements; realistic mixed-step fallback rates under
admitted, staggered sessions; the complete startup cost, including both
captures; and resident capacity preserved, or its reduction explicitly
accepted.

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
9. **Missing checks.** No process-constant, wrapper-entry or post-warmup
   checks exist; population validation (`transaction_graph.py:350-362`) omits
   `max_num_batched_tokens` and the override slot bound.
10. **Reader and shutdown.** `_resolve_model` returns `None` for an unreachable
    model, so the reader reports off (`transaction_graph.py:793-823`);
    `GPUModelRunner.shutdown` (`model_runner.py:2483-2500`) does not discard
    the binding.
11. **Inaccurate comments.** `advance.py:3535-3536` says structural rejection
    runs on every invocation including graph replay; under outer replay it is
    frozen at capture. `transaction_graph.py:451-455` and the receipt
    docstring (`:759-761`) claim largest-first segment reuse, which
    measurement contradicts.

12. **Processor geometry check.** `NemotronASRMultiModalProcessor`
    (`processor.py:106-122`) passes envelopes through without checking their
    geometry, so today an off-set geometry is caught only in the worker.

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

To be added.

## References

Tests: `tests/models/nemotron_asr/test_transaction_graph.py`,
`tests/v1/engine/test_nemotron_asr.py`. Related designs:
[CUDA Graphs](cuda_graphs.md), [Model Runner V2](model_runner_v2.md).
