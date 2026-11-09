# ALFWorld v2 audit and repair

## Confirmed before editing the objective

1. The v1 instruction contains `look;` as an example, but its parser rejects
   trailing semicolons. The supplied 274-task trace distribution has 7020 such
   responses. This explains nearly all reported parser failures; it does not
   establish that executing `look` would solve the tasks.
2. The simulator returns admissible commands, but v1 never renders them into
   policy context. Parser feedback also replaces the last real observation.
3. Invalid generated tokens enter the actor mask BEFORE parsing. Teacher and
   student score those exact prefixes, including subsequent parser feedback.
   All-invalid episodes can therefore consume optimizer steps. Failure itself
   does not change the token normalization; no invalid-only filtering existed.
4. A separate, confirmed Top-K PPO normalization bug: the frozen `old_logp` is
   full-vocabulary, but `build_topk_opd_reference` materializes an all-ones
   `support_mask` even for `support_mask=None`. The backward gather interprets
   that mask as an instruction to conditionalize on Top-K. For unchanged p:

       ratio = [p(a) / p(S)] / p(a) = 1 / p(S), instead of 1.

   Minimal counterexample: 8 equiprobable tokens, K=2, p(S)=1/4. Current ratio
   is 4 although no parameter has changed. The regression test was run before
   the repair and failed with error 3 in both candidates. In the toy all-invalid
   OPD/TA tests the old path produced finite loss but zero gradient via clipping.

   The v2 repair preserves full-vocabulary probability normalization in BOTH
   old and current log-probabilities on the same Student-Top-K candidate IDs.
   Student-p candidate weighting, frozen advantages, PPO clipping, CMT allocation,
   TA selection and IW sampled-action formulation are unchanged. This fixes an
   estimator inconsistency; it does not introduce reward-weighted RL.

   Scope is explicit: `full_vocab_ppo=True` is selected only by ALFWorld protocol
   v2. Legacy single-turn math/Gemma and protocol-v1 execution retain their old
   implementation for reproducibility. No existing run is silently upgraded.
   Any later correction of those legacy experiments needs its own versioned run.

## Hypotheses requiring model experiments

The observed look-only behavior may also reflect poor initial task competence,
student Base instruction-following, teacher action preferences, or learning from
mostly non-progressing prefixes. Parser repair alone cannot establish which of
these dominates. Compare untouched student and frozen teacher on the same train
subset using `alfworld_cli probe` before spending a full training budget. A Base
model that cannot follow the protocol may need a separately specified supervised
warm-start, or an instruction-tuned initialization with compatible token IDs.
No warm-start dataset or expert actions are injected by this repair.

## Reference inspection

Local data were re-read from the actual awawa-agi raw parquet export; its grammar
includes `move`, `put ... into/in`, and `slice ... with ...`. Real admissible
commands were checked against the repaired parser. Native `state.won`, not reward
or string matching, determines success.

AgentFly's [ALFWorld launcher](https://github.com/Agent-One-Lab/AgentFly/blob/main/examples/train_scripts/alfworld/train_alfworld_chain.sh)
and [action agent](https://github.com/Agent-One-Lab/AgentFly/blob/main/src/agentfly/agents/specialized/action_agent.py)
were inspected for per-turn task/action observations and preservation of full
generated histories. Its first-action extraction/GRPO recipe is not copied here:
BellmanOPD rejects multiple commands rather than silently taking the first one.
Source snapshots used during the audit are in a temporary reference checkout,
outside the project. Upstream BellmanOPD was consulted as context; edits apply to
the current local analysis checkout.

## Token audit

`input_ids` retain full context and exact sampled IDs; only the executed command
is punctuation-normalized. Actor positions are compact across turns; physical
token p uses causal logit p-1. Observation/chat tokens are conditioning context,
not loss targets. Context representations can still receive gradients through
attention, as they should. No observation-length discount is added to CMT and
no IW prefix reset is added between assistant turns. Both reset at episode rows.
Student/teacher forwards use the same full history and position IDs.

The optional vLLM check explicitly requests `processed_logprobs`, compares against
HF with the actual temperature and top-p, records raw-policy and sampled-distribution
means separately, and checks tokens from later turns. No vocabulary-logit retention
or extra diagnostic forward is enabled by default.

## Pipeline trace

`load_task_splits` validates all splits and supplies only train rows to the
trainer. `generate_training_episodes` enters one vLLM weight-sync session and
collects whole current-student episodes. `EnvironmentClient` materializes the
embedded game and calls actual TextWorld reset/step. `format_observation` renders
visible task/observation/action menu identically for train and eval.
`AppendOnlyChat` preserves native envelopes and sampled IDs.
`parse_action_details` returns a normalized executable command or an explicit
parser failure; neither branch rewrites actor IDs. `collate_episodes` makes full
context tensors plus a compact actor axis. The student and frozen teacher use
`rollout_forward_inputs` to gather causal actor logits from the same history.
Existing OPD/TA/PGT+CMT/IW scoring, allocation and `_opd_train_step` are reused.
The v2 Top-K reference opts into consistent full-vocabulary PPO normalization;
IW already uses full-vocabulary sampled log-probabilities. Checkpoint/optimizer
serialization is unchanged. Standalone `alfworld_evaluation` loads a saved model,
executes fresh episodes via the same collector, validates their fingerprint,
and aggregates each seed/split/task category independently.

## Changed files

Runtime and configuration:

- `b200_experiment/alfworld_protocol.py` — versioned prompt, strict parser,
  observation formatting, executable-code fingerprint and explicit no-filter policy.
- `b200_experiment/alfworld_diagnostics.py` — counts/rates/histograms, distributed
  reduction and warn/error/off collapse guard; no token reweighting.
- `b200_experiment/alfworld_logprob_debug.py` — equivalent-distribution comparison
  of sampled token LPs from multiple turns; disabled by default.
- `b200_experiment/alfworld_rollout.py` — protocol use, state-preserving parser
  feedback, exact IDs, optional LP collation, collective zero-actor handling.
- `b200_experiment/alfworld_env.py` — accept/validate versioned settings;
  actual TextWorld dynamics and state.won success semantics stay intact.
- `b200_experiment/alfworld_cli.py` — scripted parser/env/mask smoke and separate
  untouched-student/frozen-teacher probes with shared student tokenizer.
- `b200_experiment/alfworld_evaluation.py` — detailed per-seed/per-type diagnostics,
  mean ± sample std, fingerprint validation and explicit legacy evaluation override.
- `b200_experiment/config.py`, `b200_experiment/resume.py` — persist fingerprint
  and reject incompatible training resumes.
- `b200_experiment/scoring.py` — optional rollout log-prob distribution metadata;
  causal indexing implementation stays shared by scoring and backward.
- `b200_experiment/opd_core.py` — opt-in consistent ALFWorld-v2 PPO normalizer;
  candidate IDs, rewards and legacy defaults unchanged.
- `b200_experiment/trainer.py` — reduced interaction diagnostics before update,
  persisted failure diagnostics, guard, v2 reference and multi-turn LP check.
- `b200_experiment/vllm_rollout.py` — processed-LP debug output and actual full
  episode context reservation for ALFWorld; one sync per episode batch unchanged.
- `b200_experiment/tensorboard_logging.py` — ALFWorld rollout/task-type and
  LP-debug metrics, flushed even before a collapse guard stops the update.
- `configs/alfworld_multiturn.yaml` — protocol v2, admissible mode, collapse guard.
- `scripts/train_alfworld_multiturn.sh`, `scripts/train_alfworld_multiturn_14b.sh`
  — versioned fresh run names and protocol/guard/debug environment overrides.

Tests and documentation:

- `tests/test_alfworld_protocol_v2.py` — parser, state/history, masks, invalid
  batches, protocol resume, native success, later-turn LPs and all four optimizers.
- `tests/test_alfworld_ppo_normalizer.py` — minimal identity-ratio regression and
  unchanged candidate/reward/legacy-reference checks.
- `tests/test_alfworld_v2_gpu.py` — optional real ID/OOD scripted multiseed
  evaluation and real-environment GPU toy OPD/CMT optimizer mechanics.
- `tests/test_alfworld_evaluation.py` — protocol-aware fixtures and explicit
  rejection/labeling of legacy or changed-sampling evaluation.
- `ALFWORLD_MULTITURN.md`, `ALFWORLD_PROTOCOL_AUDIT.md`, `README.md` — commands, audit and entry links.

## Validation and remaining limits

Actual TextWorld/PDDL checks used the downloaded raw files (3553/140/134 tasks).
286 admissible commands from the six task types in both evaluation splits were
accepted unchanged by the strict parser. A scripted success episode ran through
the repaired collector with official Qwen3 tokenization: 4 real actions, 32 actor
tokens preserved, 893 total context tokens. Three fresh evaluation seeds on one
ID and one OOD diagnostic task succeeded with parser failure=0 and produced the
expected mean/std; these are scripted installation checks, not LLM scores.

On the available RTX 3090, random tiny Qwen models with the official tokenizer
passed GPU OPD/CMT mechanics checks: loss 0.33616/0.33295, gradient norm
0.28242/0.29367, identity ratio=1 and actual parameter changes. These checks use
scripted trajectories and cannot measure learning or task capability. CPU tests
also verify all-invalid gradients for OPD/TA/CMT/IW with guard explicitly off;
the default error guard instead stops such a batch before the optimizer.

The pre-repair identity-ratio failure and full test outputs are retained in local
temporary audit artifacts. Seven legacy test failures were re-run against clean
HEAD `7f38b46` and reproduced: avg@16 CSV merge, RAC config fairness, old math
prompt-version expectation, three GRPO reward tests, and IW padding expectation.

The untouched real student/teacher probes, their before/after command distributions,
production vLLM/HF equivalence, and full B200 OPD/CMT training smoke remain
unverified locally: only tokenizer/config assets are present, not the requested
1.7B/4B/14B model weights; vLLM is not installed; the available GPU is a single
24-GiB RTX 3090 rather than 2/4 B200s. Use the documented probe/smoke commands
on the target machine. Unit loss/gradient or scripted success is not evidence of
improved trained-model success, and no existing checkpoint's capability is repaired
retroactively by changing the parser.

Final complete-suite run with `ALFWORLD_REAL_DATA` enabled: **404 passed,
7 failed, 1 skipped, 46 subtests passed** (113.65 seconds). All seven failures
are the clean-HEAD legacy failures listed above. The skipped test requires
multiple CUDA GPUs for FSDP. The separately targeted ALFWorld/runtime/TensorBoard/
vLLM/distributed regression group passed 110 tests before the last three added
checks; those checks also passed in the final complete-suite run. Ruff,
`git diff --check`, and shell syntax checks passed for all changed source/launchers.
