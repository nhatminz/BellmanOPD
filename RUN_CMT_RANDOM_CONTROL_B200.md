# CMT/PROW vs random-permuted-weight control

Training ablation đáp lại yêu cầu "random-weight control under the same clipping,
KL budget, and normalization". Bật bằng CLI hiện có:

```bash
--set selector.cmt_random_weight_control=true
```

Mặc định `false`; tên method vẫn là `cmt`. Không thay theoretical CMT scoring.

## Control được áp dụng ở đâu

Trong `trainer._opd_train_step`, mỗi global PPO optimizer group thực hiện:

1. Dùng nguyên mask/group/ordering hiện có để gather valid-token scores toàn cục.
2. Gọi nguyên `cmt_allocation(...)` hiện có, nhận `w_raw`, **final `w`** và solver
   diagnostics. Không sửa local gains, prospective scores hay bounded solver.
3. Nếu flag bật, tạo permutation `pi` trên **N valid token positions của group**,
   rồi dùng `w[pi]` và `w_raw[pi]`. Không re-clip, re-scale hay normalize thêm.
4. Mỗi rank lấy slice `group_start:group_end` như trước, đưa về đúng các token
   hợp lệ của rank. Padding/dummy rows không thuộc permutation.

Vì đây là phép hoán vị thuần, multiset giá trị final và raw giữ nguyên bitwise.
Với mean-one weights, `p_i = w_i/N`,
`KL(p || uniform) = (1/N) sum_i w_i log(w_i)`. Tổng/mọi moment đối xứng đều
bất biến: mean, **realized KL**, min/max, clipping fractions, ESS và tổng weight
mass. Solver KL/status được giữ nguyên. Floating-point reductions theo thứ tự
khác có thể khác ở chữ số cuối; không có re-normalization để che khác biệt đó.

Phân phối final weights được giữ đúng so với **CMT allocation vừa tính trong
chính optimizer group đó**. Sau các update, policy hai run có thể khác nhau;
không tuyên bố multiset weights của hai run độc lập sẽ tiếp tục giống nhau.

Không shuffle token IDs, prompt/example order, logits, local gain hay score.
PPO clipping, teacher/student scoring, optimizer, weighted-token normalization,
rollout/checkpoint và evaluator đều dùng code hiện có.

## Seed và resume

Mỗi group dùng `torch.Generator(device="cpu")` riêng. Seed là 63 bit thấp của
8 byte đầu SHA-256 (little endian) của chuỗi:

```text
cmt-final-weight-permutation-v1:<experiment_seed>:<global_optimizer_step>
```

`torch.randperm(N, generator=generator, device="cpu")` không tiêu thụ global RNG
CPU/CUDA. Mọi rank có cùng N, experiment seed và global step, nên dùng cùng
global permutation trước khi slice. Không dùng rank ID hay rollout index làm
seed; global optimizer step vẫn đúng sau resume giữa một rollout.

Flag được ghi trong `resolved_config.yaml` và kiểm tra khi true-resume. Run cũ
thiếu field tương đương `false`; đổi original ↔ random khi resume mặc định bị
từ chối. Không thêm state mới vào optimizer/checkpoint format.

Uniform weights vẫn uniform sau permutation. `moved_fraction` có thể lớn dù
`changed_weight_fraction=0`; nhiều token có cùng giá trị weight là hợp lệ.

## Chạy Competition-MATH, 1 epoch, Qwen3-4B → Qwen3-1.7B

Trong venv B200 đã chạy được repo, cập nhật code rồi:

```bash
export STORAGE_ROOT=/workspace/storage-shared
cd "$STORAGE_ROOT/nlp/minhpn19/BellmanOPD_analysis"

CUDA_VISIBLE_DEVICES=0,1 TRAIN_NPROC_PER_NODE=2 \
TEACHER_MODEL=models/Qwen3-4B STUDENT_MODEL=models/Qwen3-1.7B \
SEED=42 ROLLOUT_SEED=42 \
bash scripts/train_cmt_random_control_compe_math_4b_17b_1epoch.sh both
```

Thứ tự: **train original → eval final original → train random → eval final
random**. Hai arm khởi tạo từ cùng `models/Qwen3-1.7B`, cùng teacher
`models/Qwen3-4B`, seed 42 và các cấu hình khác; random không load checkpoint
của original. Tên run tự có timestamp và suffix `_original` / `_random`.

Hoặc chạy riêng từng arm với cùng prefix:

```bash
export RUN_PREFIX="cmt_compe_math_4b_17b_1epoch_s42_$(date +%Y%m%d_%H%M%S_%N)"
export TEACHER_MODEL=models/Qwen3-4B STUDENT_MODEL=models/Qwen3-1.7B
export SEED=42 ROLLOUT_SEED=42

CUDA_VISIBLE_DEVICES=0,1 TRAIN_NPROC_PER_NODE=2 \
bash scripts/train_cmt_random_control_compe_math_4b_17b_1epoch.sh original

CUDA_VISIBLE_DEVICES=0,1 TRAIN_NPROC_PER_NODE=2 \
bash scripts/train_cmt_random_control_compe_math_4b_17b_1epoch.sh random
```

4 GPU: dùng `CUDA_VISIBLE_DEVICES=0,1,2,3 TRAIN_NPROC_PER_NODE=4` cho **cả hai arm**. Giữ cùng topology
để so sánh. Vùng USER CONFIG trong launcher cho phép sửa trực tiếp đường dẫn
model và các hyperparameter chung. Model/data path tương đối được resolve theo
`STORAGE_ROOT`; không có đường dẫn model từ máy code RTX 3090.

Cấu hình mặc định chung:

| Tham số | Giá trị |
|---|---|
| Train data | `nlp/minhpn19/data/competition_math/data/train-00000-of-00001.parquet` |
| Teacher / student | `models/Qwen3-4B` / `models/Qwen3-1.7B` |
| Epoch / max steps | `1` / `null` (không áp hard cap) |
| Global prompts / responses per prompt | `64` / `4` |
| Global PPO mini-batch | `64` trajectory/update |
| Train micro-batch / score micro-batch per GPU | `16` / `8` |
| LR / experiment seed / rollout seed | `5e-6` / `42` / `42` |
| Prompt / response token caps | `1024` / `4096` |
| Overlong prompt policy | `filter`, giữ behavior hiện tại |
| Allocation / bounds | `direct_bounded_gibbs` / `[0.5,2.0]` |
| Final KL budget / correction | `0.02` / `tanh_q99`, quantile `0.99` |
| Save interval | `150`, và checkpoint `final` |
| Eval trong training | Tắt cả periodic, eval-at-start, eval-at-end |

Số optimizer step được suy ra từ **một epoch trên các training record còn lại
sau prompt filtering** và PPO mini-batching hiện có; không giả định một số step
cố định, không đổi valid-token budget để bù cho permutation.

Sau khi training process thoát thành công, launcher gọi evaluator hiện có:

```bash
bash scripts/eval_multiseed_b200.sh \
  --run-output /ABS/PATH/TO/RUN/cmt_opd \
  --checkpoints final --seeds 42 43 44 --fresh-seeds \
  --tag final_s42_43_44
```

Eval cả sáu benchmark hiện có, toàn bộ câu hỏi, 8 response/câu/seed;
temperature `0.7`, top-p `0.95`, response cap `7168`, context `9216`.
Đây là nguyên protocol evaluator hiện tại, không có grader/sampling riêng cho
control. `--fresh-seeds` không yêu cầu baseline history khi train-time eval tắt.
Nếu training lỗi, launcher không chạy eval checkpoint chưa hoàn tất.

## Kết quả và log

```text
outputs/<prefix>_original/cmt_opd/
outputs/<prefix>_random/cmt_opd/
  resolved_config.yaml
  run_metadata.json
  metrics.jsonl
  train_metrics.csv
  tensorboard/
  final/
  eval_multiseed/final_s42_43_44/final/seed_42/
  eval_multiseed/final_s42_43_44/final/seed_43/
  eval_multiseed/final_s42_43_44/final/seed_44/
  eval_summary_final_s42_43_44.json
  eval_summary_final_s42_43_44.csv
  eval_summary_final_s42_43_44.md
```

Eval summary có `avg@8` và `pass@8` theo từng dataset, điểm ba seed và **mean ±
sample standard deviation (ddof=1)**. Mean ở phần trăm và ± ở percentage points.
Checkpoint cố định, đây là evaluation-seed variation, không phải training-seed
variation. File training `summary.json` vẫn giữ training summary hiện tại.

Mỗi optimizer event ghi flag/assignment, global optimizer step/group/N,
`allocation_kl_target`, `allocation_kl_final`, `cmt_weight_mean`, clipping bounds,
`normalized_ess`, `cmt_effective_sample_size` và permutation diagnostics:

- `cmt_permutation_seed`, SHA-256 của vector permutation.
- Token count, fixed points, moved fraction, mean absolute index displacement.
- Tỷ lệ final/raw weight đổi giá trị tại vị trí và thời gian permutation.

Raw diagnostic vector được hoán vị cùng permutation với final vector; gain/score
diagnostics vẫn ở nguyên token position. JSON/CSV giữ đầy đủ integer seed/hash;
TensorBoard ghi các scalar số tại `cmt/random_control/*` và các KL/ESS tags có
sẵn. Các diagnostics gắn đúng từng optimizer group, không copy permutation của
group cuối sang các step trước trong cùng rollout.

## Test và phạm vi xác minh

```bash
python3 -m pytest -q \
  tests/test_cmt_random_control.py tests/test_cmt_selector.py \
  tests/test_cmt_grouped_distributed.py tests/test_resume.py \
  tests/test_tensorboard_logging.py tests/test_training_launchers.py \
  tests/test_multiseed_evaluation.py tests/test_gemma_evaluation_context.py \
  tests/test_training_cost.py
```

Test mới bao phủ multiset/raw alignment, mean/KL/ESS/clipping bất biến, RNG,
deterministic repeat, step/seed variation, uniform/tied weights, bitwise disabled
behavior, objective matching manual final-weight assignment, CLI/launcher,
resume, TensorBoard; DDP/Gloo thật trên 2/4 rank gồm ragged token slices,
uneven trajectory counts và dummy padding. Test CUDA nhẹ kiểm tra GPU weights
dùng cùng CPU permutation và không đổi CUDA RNG.

Có thêm đối chiếu one-off với `_opd_train_step` từ HEAD trước patch trên tiny CPU
model: flag false cho weights/raw diagnostics, loss và optimizer parameters
giống **bitwise**, không đổi global RNG. Các test này không phải full B200
training hoặc kết quả accuracy; full Qwen train/eval sẽ chạy bằng lệnh trên.

Kết quả đã chạy trên máy code: **113 passed, 24 subtests passed** cho command
test ở trên; CUDA test chạy trên RTX 3090, distributed tests dùng Gloo CPU
2/4 rank. `ruff check`, `bash -n` ba launcher liên quan và `git diff --check`
đều pass. Chưa chạy full Qwen training hoặc multiseed generation trên B200.

## Tất cả file thay đổi

- `b200_experiment/cmt_random_control.py` (mới): CPU permutation và diagnostics.
- `b200_experiment/trainer.py`: hook sau allocation/trước scatter, log từng group.
- `b200_experiment/tensorboard_logging.py`: scalar tags của control.
- `b200_experiment/resume.py`: compatibility flag, legacy default false.
- `configs/qwen3_b200_cmt.yaml`: flag mặc định false.
- `scripts/common_b200.sh`: map env `CMT_RANDOM_WEIGHT_CONTROL` vào CLI config.
- `scripts/train_cmt_b200.sh`: expose knob trong launcher CMT hiện có.
- `scripts/train_cmt_random_control_compe_math_4b_17b_1epoch.sh` (mới): hai arm
  dùng cùng recipe; train một epoch rồi eval final ba seed.
- `tests/test_cmt_random_control.py` (mới): các test control.
- `RUN_CMT_RANDOM_CONTROL_B200.md` (mới): hướng dẫn và audit này.
- `README.md`: liên kết launcher/hướng dẫn.
