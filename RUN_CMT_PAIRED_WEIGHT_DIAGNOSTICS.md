# Paired PROW-vs-G-only diagnostics trong random-weight control

Các thống kê trong run này được đo trên **on-policy trajectories của student
đang train bằng random-permuted PROW weights**. Chúng không phải kết quả của
original PROW run. JSON/CSV ghi rõ
`trajectory_source=random_weight_trained_student_on_policy_trajectories`.

## Chạy trên 1 GPU B200

Trong venv đang chạy được training của repo:

```bash
export STORAGE_ROOT=/workspace/storage-shared
cd "$STORAGE_ROOT/nlp/minhpn19/BellmanOPD_analysis"
export RUN_PREFIX="cmt_paired_1gpu_s42_$(date +%Y%m%d_%H%M%S_%N)"

CUDA_VISIBLE_DEVICES=0 TRAIN_NPROC_PER_NODE=1 \
TEACHER_MODEL=models/Qwen3-4B STUDENT_MODEL=models/Qwen3-1.7B \
SEED=42 ROLLOUT_SEED=42 \
MICRO_BATCH_SIZE_PER_GPU=2 SCORE_MICRO_BATCH_SIZE=1 \
CMT_PAIRED_WEIGHT_DIAGNOSTICS=true \
EVAL_TAG=final_s42_43_44 \
bash scripts/train_cmt_random_control_compe_math_4b_17b_1epoch.sh random
```

Launcher giữ Competition-MATH **1 epoch**, global prompt batch 64, 4 responses,
global PPO batch 64, cùng allocation/clipping/optimizer/rollout recipe hiện có.
Lệnh trên giảm train/score microbatch để chạy một GPU, không đổi global batch.
Không eval trong training. Sau khi train thành công, launcher tự eval checkpoint
cuối trên sáu benchmark hiện có với **seed 42/43/44**, 8 responses, rồi ghi
summary **avg@8/pass@8 mean ± sample standard deviation**.

Có thể sửa ngay vùng USER CONFIG của launcher:

```bash
export CMT_PAIRED_WEIGHT_DIAGNOSTICS="${CMT_PAIRED_WEIGHT_DIAGNOSTICS:-true}"
```

Flag mặc định của code vẫn `false`; khi tắt, không có extra G-only allocation,
gather, phép đo hay output paired diagnostics. CLI tương đương:

```bash
--set selector.cmt_random_weight_control=true \
--set selector.cmt_paired_weight_diagnostics=true
```

Chỉ bật diagnostics với mode `random`. Mode `original` hoặc `both` với flag này
sẽ bị từ chối vì diagnostics này được thiết kế cho random-trained student.

## Cách tính và bảo toàn training

Trước optimizer update, với **cùng valid token positions của mỗi global PPO
optimizer group**:

1. Giữ bản detached của final PROW weights do `cmt_allocation()` hiện có trả về.
2. Dùng nguyên permutation từ `cmt_random_control.py` cho training weights.
3. Gather local gain `G_t`, `sequential_gain_raw` và `coverage_correction` với
   cùng row order, valid mask, group/rank boundaries như PROW allocation.
4. Gọi cùng `cmt_allocation(G_t, ...)` cho G-only, cùng clipping bounds,
   allocation mode và KL budget. Đây là allocation toàn group, không theo GPU
   hoặc microbatch. Budget giống nhau; realized KL của G-only có thể khác nếu
   score suy biến hoặc solver không thể sử dụng hết budget.
5. Tính paired comparisons giữa **original PROW và G-only**, rồi kiểm tra cả
   ba weight vectors. Chỉ **random weights** được scatter và đi vào loss.

PROW/G-only và diagnostic samples đều detached; không thêm model forward,
rollout hay RNG operation. Không sửa score, token IDs, local gain, teacher/
student logits, permutation, objective hoặc evaluator.

Các kiểm tra mean-one/clipping dùng tolerance `1e-6`. Multiset PROW/random
được so sánh chính xác. KL/ESS dùng các bản copy detached đã sort để cùng
multiset cho cùng reduction order; không sort weights dùng cho training.
Legacy allocation mode `gibbs` không có clipping box nên `clipping_passed=null`;
launcher dùng `direct_bounded_gibbs` với bounds `[0.5,2.0]`.

## Output

Với lệnh trên, dùng cùng shell để xác định đúng folder:

```bash
export RUN_OUTPUT="$PWD/outputs/${RUN_PREFIX}_random/cmt_opd"
```

Các file nằm cạnh checkpoint và eval history:

```text
$RUN_OUTPUT/rebuttal_weight_diagnostics.json
$RUN_OUTPUT/rebuttal_weight_diagnostics.csv
$RUN_OUTPUT/rebuttal_weight_diagnostics_steps.jsonl
$RUN_OUTPUT/rebuttal_weight_diagnostics_samples/step-<optimizer-step>-<hash>.npz
$RUN_OUTPUT/metrics.jsonl
$RUN_OUTPUT/train_metrics.csv
$RUN_OUTPUT/tensorboard/
$RUN_OUTPUT/eval_summary_final_s42_43_44.json
$RUN_OUTPUT/eval_summary_final_s42_43_44.csv
$RUN_OUTPUT/eval_summary_final_s42_43_44.md
```

`metrics.jsonl` và `train_metrics.csv` có các field `paired_*` cho **mỗi optimizer
step**. TensorBoard dùng tags `cmt/paired_weights/*`. JSON có `overall` và
`per_step`; CSV có dòng `scope=overall` và từng dòng `scope=step`.

JSON/CSV được cập nhật sau mỗi group. Khi training kết thúc bình thường,
`status=complete`, `quantiles_finalized=true` và median/p95 overall được tính
chính xác trên pooled samples. Trong khi train, summary có `status=partial`;
median/p95 overall để `null`, còn từng step đã có quantiles. Các file NPZ lưu
lossless absolute differences để tính pooled quantiles, không lưu model tensors
hay computational graph. Giữ chúng nếu cần resume/tính lại pooled quantiles.

## Lấy số để trả lời reviewer

Ví dụ đọc và in số trực tiếp, không dùng số minh họa giả:

```bash
python3 - <<'PY'
import json, os
from pathlib import Path

path = Path(os.environ["RUN_OUTPUT"]) / "rebuttal_weight_diagnostics.json"
report = json.loads(path.read_text())
assert report["status"] == "complete" and report["quantiles_finalized"]
print("Trajectory source:", report["trajectory_source"])
print("Observed global optimizer steps:", report["observed_optimizer_steps"])
overall = report["overall"]
print("Zero/support statistics:")
print(json.dumps(overall["zero_score_statistics"], indent=2))
for subset in ("zero_prospective", "out_of_support"):
    print("Paired PROW-vs-G-only:", subset)
    print(json.dumps(overall["paired_comparison"][subset], indent=2))
print("Allocation checks:")
print(json.dumps(overall["allocations"], indent=2))
print(json.dumps(overall["checks"], indent=2))
PY
```

Các fraction là tỷ lệ `[0,1]`; nhân 100 khi viết phần trăm:

| Field trong `overall.zero_score_statistics` | Ý nghĩa / mẫu số |
|---|---|
| `valid_token_count` | Tổng valid token positions được tối ưu |
| `zero_prospective_count`, `zero_prospective_fraction` | `sequential_gain_raw == 0` chính xác / valid tokens |
| `out_of_support_count`, `out_of_support_fraction` | `coverage_correction == 0`, tức `y_t` ngoài support / valid tokens |
| `out_of_support_zero_count`, `out_of_support_zero_fraction` | Giao của hai điều kiện / valid tokens |
| `zero_attributable_to_out_of_support_fraction` | Giao của hai điều kiện / tất cả exact-zero tokens |
| `in_support_zero_count`, `in_support_zero_fraction` | Exact-zero nhưng trong support / valid tokens |
| `in_support_zero_conditional_fraction` | Exact-zero trong support / tất cả in-support tokens |
| `out_of_support_nonzero_count` | Kiểm tra không mặc định đồng nhất hai điều kiện |

Hai subsets `overall.paired_comparison.zero_prospective` và `.out_of_support`
đều có mean PROW/G-only weights, signed difference `PROW - G_only`, mean/median/
p95 absolute difference, `different_fraction` với exact inequality. Field
`different_gt_1e6_fraction` báo thêm tỷ lệ `abs(diff) > 1e-6` để phân biệt sai
khác nhỏ do floating point. Subset rỗng có count 0, các mean/quantile/fraction
`null`, không tạo NaN.

`overall.allocations.prow`, `.g_only`, `.random` có mean/min/max, ESS và realized
KL, cộng kiểm tra mean-one/clipping. `overall.checks` xác nhận các bất biến của
**mọi group**, gồm multiset và KL giống nhau giữa PROW và random.

Các counts/sums được cộng theo token, không lấy trung bình các group means.
Mỗi global optimizer step có đúng một record, không nhân lại số token của toàn
rollout cho mỗi PPO group. Median/p95 overall lấy từ toàn bộ absolute differences
trong subset, không lấy trung bình median/p95 từng step. Khi true-resume, journal
chỉ giữ records tới checkpoint step rồi deduplicate các step được replay. Nếu
bật diagnostics từ giữa run cũ, summary chỉ bao phủ các step **đã được đo**;
`observed_optimizer_steps`/`per_step` thể hiện coverage, không suy ra số liệu
cho các step chưa ghi diagnostics.

Nếu bật/tắt flag khi resume, `train_metrics.csv` giữ nguyên dữ liệu và thứ tự
các cột cũ, thêm các cột paired còn thiếu; những step không đo được để trống.

Khi viết rebuttal, ghi rõ đây là paired diagnostic **trên trajectories của
random-weight-trained student**. Không suy diễn đây là đo trực tiếp trên run
PROW gốc hay bịa kết quả trước khi chạy.

## Files và kiểm tra

- `b200_experiment/paired_weight_diagnostics.py`: phép đo detached, checks,
  aggregation, resume/dedup, JSON/CSV và lossless quantile samples.
- `b200_experiment/trainer.py`: gather đúng global group, G-only diagnostic
  allocation, log mỗi optimizer step, finalize summary.
- `b200_experiment/tensorboard_logging.py`: tags `cmt/paired_weights/*`.
- `configs/qwen3_b200_cmt.yaml`: flag mặc định false.
- `scripts/common_b200.sh`: truyền flag vào CLI.
- `scripts/train_cmt_random_control_compe_math_4b_17b_1epoch.sh`: thêm cấu hình
  tùy chọn, giữ nguyên training/final evaluation protocol.
- `tests/test_paired_weight_diagnostics.py`: unit, distributed, launcher tests.
- `RUN_CMT_PAIRED_WEIGHT_DIAGNOSTICS.md`, `RUN_CMT_RANDOM_CONTROL_B200.md`,
  `README.md`: hướng dẫn/link.

`cmt_random_control.py` và `selectors/cmt_selector.py` không thay đổi.
Test bao gồm token alignment với padding/length sorting, exact-zero counts,
paired stats, weighted pooling, true pooled quantiles, empty subsets, resume,
constraint checks, CPU/CUDA RNG/gradient isolation, cùng loss/weights/parameter
updates và số forward khi bật/tắt, Gloo 2/4 ranks có uneven groups/dummy rows,
và capture lệnh launcher một GPU cùng final 3-seed evaluation.

Kết quả kiểm tra: **127 tests passed, 24 subtests passed** (46.99s); Ruff,
`bash -n` và `git diff --check` đều pass. Test CUDA chạy trên GPU máy code,
distributed tests dùng CPU/Gloo 2 và 4 ranks. Chưa chạy full training/evaluation
Qwen3 trên B200, nên không có kết quả thực nghiệm để báo cáo.

Lệnh test (dùng Python/venv đã cài test dependencies của repo):

```bash
python3 -m pytest -q \
  tests/test_paired_weight_diagnostics.py tests/test_cmt_random_control.py \
  tests/test_cmt_selector.py tests/test_cmt_grouped_distributed.py \
  tests/test_tensorboard_logging.py tests/test_resume.py \
  tests/test_training_launchers.py tests/test_multiseed_evaluation.py \
  tests/test_gemma_evaluation_context.py tests/test_training_cost.py \
  --disable-warnings --tb=short
```
