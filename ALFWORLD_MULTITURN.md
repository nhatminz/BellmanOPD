# ALFWorld multi-turn protocol v2 trên B200

Protocol v2 sửa giao thức gây `look;`/parser failure và lỗi chuẩn hóa PPO Top-K.
Đọc phân tích, phản ví dụ và phạm vi sửa ở [ALFWORLD_PROTOCOL_AUDIT.md](ALFWORLD_PROTOCOL_AUDIT.md).
Code không cam kết checkpoint sau train sẽ có success rate cao: protocol đúng và
năng lực giải task của student/teacher là hai điều phải kiểm tra riêng.

Nếu CMT dừng ở `stage=start-rollout-vllm` với timeout, xem phần
[timeout khởi động vLLM và lệnh chạy lại hai GPU](ALFWORLD_FIX_CMT_RUNTIME.md#timeout-vllm-khi-khởi-động-cmt-ngày-09112026).
Launcher ALFWorld hiện chờ health tối đa 1800 giây và ghi log INFO cùng
`vllm_rollout_server.rank-XXXXX.startup.json` để theo dõi startup.

## 1. Các thay đổi hành vi

- System prompt liệt kê từng ví dụ trên một dòng, không còn ví dụ `look;`.
- Parser nhận đúng một command, cho phép một dấu `;` hoặc `.` cuối lệnh. Không
  lấy lệnh đầu từ response nhiều lệnh. JSON `{"action":"look;"}` và XML
  `<action>look;</action>` được hỗ trợ. `slice ... with ...` và `put ... into ...`
  được kiểm tra theo grammar thật. Không tự chiếu tới action gần nhất.
- `alfworld.action_prompt_mode=admissible` là mặc định: mỗi lượt có goal quan sát
  được ở reset, observation thật và các command hiện hợp lệ, đúng tên/ID object.
  Đây là danh sách action hợp lệ, không phải expert plan. `legacy` giữ kiểu không
  có action menu để ablation. Protocol v1 cũ vẫn được nhận diện riêng.
- Parser failure không gọi simulator và không thay observation thật. Feedback
  parser có nhãn riêng, goal vẫn trong history. Raw response/token IDs được giữ.
- Không truncate goal/history/action list. Dừng episode khi không đủ context cho
  observation đầy đủ. Observation ban đầu vượt budget báo lỗi có hướng xử lý.
- Các token invalid vẫn thuộc objective on-policy; không tự lọc episode thất bại.
  `invalid_trajectory_filter=none` được kiểm tra rõ. Thử nghiệm filtering cần một
  estimator ablation riêng, không được bật ngầm.
- Collapse guard mặc định `error`: all-parser-invalid batch, hoặc batch >=32 lượt
  với parser failure >=0.95, look rate >=0.98, hoặc repeat rate >=0.98, sẽ dừng
  trước optimizer update, sau khi ghi trace. `warn` cho phép tiếp tục có cảnh báo;
  `off` chỉ dùng khi chủ động nghiên cứu estimator trên dữ liệu lỗi.
- Zero-actor episode làm mọi rank dừng đồng bộ trước scoring/backward; không tạo
  token/loss giả. Không âm thầm update batch không có target.
- ALFWorld v2 dùng **cùng full-vocabulary normalizer cho old/current PPO log-prob**
  trên đúng Student-Top-K IDs. Bản cũ có ratio khác 1 ngay khi policy chưa đổi.
  Không đổi Student-p reward, Top-K support, clipping, CMT allocation, TA budget
  hoặc IW sampled-action objective. Single-turn math/Gemma cũ giữ nguyên đường code.
- Student sinh tất cả các lượt. Teacher freeze, chỉ chấm đúng prefix student đã
  đi qua. Environment/chat tokens là context, không phải actor loss targets.
  CMT/IW dùng trục actor compact xuyên lượt, reset giữa episode; không tính
  discount/policy transition cho token môi trường.

## 2. Dữ liệu, dependencies, simulator

Dùng venv B200 hiện tại đã có dependencies của repo/vLLM CUDA IPC:

```bash
export STORAGE_ROOT=/workspace/storage-shared
cd "$STORAGE_ROOT/nlp/minhpn19/BellmanOPD_analysis"
export PYTHON_BIN="$(command -v python3)"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
"$PYTHON_BIN" -m pip install -r requirements-alfworld.txt
```

Ba file raw parquet phải tồn tại:

```text
$STORAGE_ROOT/nlp/minhpn19/data/ALFworld/train-00000-of-00001.parquet
$STORAGE_ROOT/nlp/minhpn19/data/ALFworld/eval_in_distribution-00000-of-00001.parquet
$STORAGE_ROOT/nlp/minhpn19/data/ALFworld/eval_out_of_distribution-00000-of-00001.parquet
```

Đã đọc schema thực `id/task_type/game_file_path/game_content`; `game_content`
nhúng domain/problem/grammar. Bản kiểm tra có 3553 train, 140 ID, 134 OOD.
Loader kiểm tra disjoint theo ID/path/game hash và loại walkthrough/solution khỏi
runtime game/task. Không cần THOR/Unity hay replay observation từ parquet.

Nếu chưa tải:

```bash
"$PYTHON_BIN" - <<'PYDATA'
import os, urllib.request
from pathlib import Path
root = Path(os.environ["STORAGE_ROOT"]) / "nlp/minhpn19/data/ALFworld"
root.mkdir(parents=True, exist_ok=True)
for split in ("train", "eval_in_distribution", "eval_out_of_distribution"):
    name = f"{split}-00000-of-00001.parquet"
    path = root / name
    if not path.exists():
        temporary = path.with_suffix(".download")
        urllib.request.urlretrieve(
            f"https://huggingface.co/datasets/awawa-agi/alfworld-raw/resolve/main/data/{name}", temporary)
        temporary.replace(path)
    print(path)
PYDATA
```

Kiểm tra reset/step và toàn bộ parser → environment → collector → actor mask:

```bash
python3 -m b200_experiment.alfworld_cli ready
python3 -m b200_experiment.alfworld_cli smoke
```

`smoke` dùng **scripted walkthrough chỉ trong diagnostic**, cố ý thêm `;`, rồi
chạy simulator thật qua collector. Report `parser_environment_smoke` phải success,
parser failure=0 và có punctuation recovery. Đây không phải policy/model score.
Có thể chọn `--split eval_out_of_distribution --task-index 0`, hoặc
`--output results/alfworld_v2_smoke.json` (file mới, không ghi đè).

Nếu gặp `GLIBCXX_3.4.32`, build native runtime trên chính node đang dùng:

```bash
bash scripts/rebuild_alfworld_native.sh
python3 -m b200_experiment.alfworld_cli smoke
```

Build cần gcc/g++/make/cmake; nó cài native package riêng theo host/Python, không
thay torch/vLLM hay venv chung. Chi tiết: [ALFWORLD_FIX_CMT_RUNTIME.md](ALFWORLD_FIX_CMT_RUNTIME.md).

## 3. Chọn model và kiểm tra trước train

Ví dụ Qwen3-4B teacher → Qwen3-1.7B instruction student, theo đường dẫn local B200:

```bash
export TEACHER_MODEL=models/Qwen3-4B
export STUDENT_MODEL=models/Qwen3-1.7B

python3 -m b200_experiment.alfworld_cli ready --check-models \
  --set "models.teacher_path=$TEACHER_MODEL" \
  --set "models.student_path=$STUDENT_MODEL"
```

Nếu dùng asset Base cũ, đặt `STUDENT_MODEL` tới đúng thư mục đó. Token-ID mapping
teacher/student vẫn phải tương thích tuyệt đối. Student tokenizer là tokenizer
chung; `enable_thinking=false`; native end-of-turn/EOS được dừng đúng. Gemma2 không
có system role nên instruction được đặt vào user turn đầu bằng template của nó.
Launcher không thay initialization giữa OPD/TA/CMT/IW.

Nên so sánh untouched student và teacher trên **cùng 8 train tasks** trước:

```bash
PROBE_ID="$(date +%Y%m%d_%H%M%S)"
for ROLE in student teacher; do
  python3 -m b200_experiment.alfworld_cli probe \
    --role "$ROLE" --split train --limit 8 --seed 42 --gpus 0,1 \
    --output "results/alfworld_v2_probe_${ROLE}_${PROBE_ID}" \
    --set "models.teacher_path=$TEACHER_MODEL" \
    --set "models.student_path=$STUDENT_MODEL"
done
```

Mỗi probe dùng một vLLM replica/GPU, tương tác môi trường mới, cùng student
shared tokenizer cho cả hai role. Xem `summary.json`, `episodes.jsonl`, `rank-*.log`:
SR, parser/simulator-invalid, tỷ lệ look, số action khác nhau, top response/action.
Nếu Base không theo được action protocol hoặc teacher cũng chỉ `look`, chưa có
bằng chứng OPD sẽ học được giải task. Có thể cần instruction initialization hoặc
supervised warm-start được thiết kế riêng; code không tự thêm expert data.

## 4. Train trên 2 hoặc 4 GPU

Chọn **một** topology, giữ mọi biến workload giống nhau giữa bốn phương pháp:

```bash
export CUDA_VISIBLE_DEVICES=0,1          # 2 GPU
# Hoặc: export CUDA_VISIBLE_DEVICES=0,1,2,3   # 4 GPU

export GLOBAL_BATCH_SIZE=8
export BATCH_SIZE=8
export NUM_RESPONSES=4
export PPO_MINI_BATCH_SIZE=8
export MICRO_BATCH_SIZE_PER_GPU=1
export SCORE_MICRO_BATCH_SIZE=1
export MAX_TURNS=30
export MAX_TURN_TOKENS=128
export MAX_RESPONSE_LEN=3840
export MAX_EPISODE_TOKENS=8192
export LR=5.0e-6
export MAX_STEPS=1000
export SAVE_INTERVAL=50
export SEED=42
export ROLLOUT_SEED=42
```

Đây là 8 task × 4 episode =32 episode/rollout, PPO=8 episode/update, tối đa
4 updates/rollout. Chia trên 2 hay 4 GPU không đổi global budget. Microbatch chỉ
thay bộ nhớ/accumulation. `MAX_STEPS` là tổng optimizer step, không phải số rollout.
B200 vLLM sync weights một lần trước batch episode, không có collective giữa lượt.

Chạy **smoke training ngắn** OPD và CMT, có log-prob sanity check xuyên lượt:

```bash
for METHOD in opd cmt; do
  RUN_NAME="alfworld_v2_${METHOD}_smoke_$(date +%Y%m%d_%H%M%S_%N)" \
  MAX_STEPS=2 SAVE_INTERVAL=1 \
  VLLM_LOGPROB_SANITY_ENABLED=true VLLM_LOGPROB_SANITY_FAIL=true \
  bash scripts/train_alfworld_multiturn.sh "$METHOD"
done
```

Debug mode yêu cầu vLLM `processed_logprobs` và so với HF với đúng temperature/top-p,
bao gồm lượt sau. Mean/max discrepancy và raw-policy/sampling LP means được ghi
trong `vllm_logprob_sanity` và TensorBoard `alfworld/logprob_debug/*`.
Mặc định tolerance=0.05, tối đa 32 token/rank (lấy token đầu mỗi lượt). Muốn sửa:
`--set rollout.vllm.logprob_sanity.max_tokens_per_turn=2` hoặc `.tolerance=...`.
Debug tắt mặc định ở train dài, không giữ full-vocabulary logits diagnostic.

Kiểm tra finite loss, gradient khác 0, PPO ratio ban đầu gần 1, trace có action
đa dạng và chuyển trạng thái thật. Loss giảm không chứng minh success tăng.
Guard mặc định sẽ dừng protocol collapse trước update; không tắt guard chỉ để
che lỗi. Nếu chủ động làm ablation, `COLLAPSE_GUARD_MODE=warn` vẫn giữ tất cả token.

Train fresh bốn method, lần lượt, mỗi run dùng mọi GPU trong mask đã chọn:

```bash
RUN_NAME=alfworld_v2_opd_s42 bash scripts/train_alfworld_multiturn.sh opd
RUN_NAME=alfworld_v2_ta_s42  bash scripts/train_alfworld_multiturn.sh ta
RUN_NAME=alfworld_v2_cmt_s42 bash scripts/train_alfworld_multiturn.sh cmt
RUN_NAME=alfworld_v2_iw_s42  bash scripts/train_alfworld_multiturn.sh iw
```

Nếu muốn dừng chuỗi khi một run lỗi, nối bằng `&&` hoặc dùng shell `set -e`.
Output của launcher này là `outputs/<RUN_NAME>/<method>_opd/`, kể cả `opd_opd`
và `iw_opd`. Có thể đặt rõ `OUTPUT_DIR=...`.

Cặp 14B → 4B cũng dùng protocol v2:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
TEACHER_MODEL=models/Qwen3-14B STUDENT_MODEL=models/Qwen3-4B \
RUN_NAME=alfworld_v2_cmt_14b_s42 \
bash scripts/train_alfworld_multiturn_14b.sh cmt
```

Nếu shell chưa export workload ở trên, launcher 14B giữ defaults riêng: global
batch=16, PPO=8, micro=4/GPU, save=100. Dùng cùng workload đó cho cả bốn method;
đổi đối số thành opd/ta/iw để train các baseline tương ứng.

Resume **chỉ cùng protocol/fingerprint**:

```bash
OUTPUT_DIR="$PWD/outputs/alfworld_v2_cmt_s42/cmt_opd" \
RESUME=auto MAX_STEPS=1000 bash scripts/train_alfworld_multiturn.sh cmt
```

Giữ model/data/seed/sampling/batch/LR/token limits như run gốc. Checkpoint cần
optimizer.pt. Không resume optimizer/checkpoint protocol v1 vào v2. Tên run cũ,
metrics, checkpoint và summary cũ không bị ghi đè.

## 5. Eval nhiều checkpoint: avg@8/pass@8, 3 seed và summary ±

Mở `scripts/eval_alfworld_multiseed.sh`, thêm các đường dẫn checkpoint vào
`CHECKPOINTS=(...)`. Một đường dẫn trong dấu ngoặc kép trên mỗi dòng, **không
có dấu phẩy**. Có thể trộn OPD/TA/CMT/IW; method và protocol được đọc từ
`resolved_config.yaml` cạnh từng checkpoint:

```bash
CHECKPOINTS=(
  "${STORAGE_ROOT}/nlp/minhpn19/BellmanOPD_analysis/outputs/alfworld_v2_opd_20261109_093150_093055241/opd_opd/checkpoint-000400"
  # "${STORAGE_ROOT}/nlp/minhpn19/BellmanOPD_analysis/outputs/TEN_RUN/cmt_opd/checkpoint-000400"
  # "${STORAGE_ROOT}/nlp/minhpn19/BellmanOPD_analysis/outputs/TEN_RUN/ta_opd/final"
)
SEEDS=(42 43 44)
```

Sau khi sửa danh sách, chạy trên hai GPU:

```bash
export STORAGE_ROOT=/workspace/storage-shared
cd "$STORAGE_ROOT/nlp/minhpn19/BellmanOPD_analysis"

# Kiểm tra đường dẫn, split và protocol; không load model/chưa dùng GPU.
ALFWORLD_EVAL_GPUS=0,1 bash scripts/eval_alfworld_multiseed.sh --dry-run

# Full ID/OOD, từng checkpoint lần lượt, seed 42/43/44.
ALFWORLD_EVAL_GPUS=0,1 bash scripts/eval_alfworld_multiseed.sh
```

Muốn bốn GPU: đổi `ALFWORLD_EVAL_GPUS=0,1` thành `ALFWORLD_EVAL_GPUS=0,1,2,3`.
Mỗi GPU giữ một bản model TP=1 và nhận một phần task; tám sample của cùng task
chạy lần lượt trên GPU nhận task đó. Tqdm ở terminal hiển thị checkpoint và
số **episode hoàn tất** của mỗi seed; log model/worker nằm trong folder kết quả.

Chế độ danh sách mặc định `EPISODES_PER_TASK=8`: với **mỗi seed**, mỗi task
được chạy tám episode simulator hoàn toàn mới. Trong tám lần thử, reset bằng
cùng environment seed để so trên cùng điều kiện khởi đầu; policy sampling seed
khác nhau và được ghi cùng `sample_index=0..7` trong trace. Mỗi lượt vẫn chỉ
sinh **một action**, không lấy tám action rồi chọn action tốt nhất. Không dùng
oracle, action projection hoặc replay của episode trước.

Với `c_i` là số episode thành công trong tám lần thử task `i`, tổng `N` task:

- `avg@8 = sum(c_i)/(8N)`; tương đương success rate trên toàn bộ tám episode/task.
- `pass@8 = sum(c_i > 0)/N`; thành công nếu ít nhất một trong tám episode thắng.
- Thành công đọc từ simulator `state.won`, giữ nguyên protocol/giới hạn của run.
- Tính cả hai metric từ **cùng tám episode**, theo từng split ID/OOD và task type.
- `±` dùng **sample standard deviation giữa seed 42/43/44, ddof=1**. Không gộp
  ba seed thành một tập 24 episode để tính pass@8.

Với bản dataset 140 ID +134 OOD, tổng là **6.576 episode/checkpoint** qua ba
seed. Đây là tám lần nhiều episode hơn giao thức cũ chỉ có SR một lần/task/seed.
Sampling mặc định đọc từ evaluation config của run (v2 thường temperature=1,
top_p=1); không tự đổi giao thức prompt/action hay giới hạn episode.

Tag tự sinh theo timestamp, ví dụ `alfworld_avg8_pass8_20261110_153045_123456789`.
Kết quả riêng cho checkpoint ví dụ của bạn:

```text
outputs/alfworld_v2_opd_20261109_093150_093055241/opd_opd/
  alfworld_eval/<tag>/checkpoint-000400/
    plan.json
    resolved_config.yaml
    summary.json
    summary.csv
    per_seed.csv
    task_results.jsonl
    step-000400/seed-42/episodes.jsonl
    step-000400/seed-42/summary.json
    step-000400/seed-42/rank-00000.log
    complete.json
```

`summary.json` có `aggregate.eval_in_distribution.avg_at_8`/`pass_at_8` và
`eval_out_of_distribution` trong từng phần tử `results`, gồm mean/std/n/display
và `display_percent`, ví dụ `45.20 ± 0.31`. CSV cũng có cột phần trăm này.
`task_results.jsonl` giữ summary theo task với `n=3` seed, không phải n=24 episode.

Summary **so sánh tất cả checkpoint trong danh sách** nằm tại:

```text
outputs/alfworld_eval/<tag>/summary.json
outputs/alfworld_eval/<tag>/summary.csv
```

Một checkpoint lỗi sẽ được ghi vào `failures` trong summary chung; chương trình
tiếp tục checkpoint kế tiếp và chỉ trả exit code 1 sau khi đã thử hết danh sách.
Mỗi checkpoint có namespace riêng nên nhiều step cùng run không đè kết quả nhau.
Tag đã tồn tại bị từ chối; dùng tag mới cho lần chạy khác hoặc lần bị gián đoạn.
Code không ghi đè history toán hoặc kết quả ALFWorld cũ, không tự trộn protocol.

Nếu muốn truyền danh sách trực tiếp thay vì sửa file:

```bash
ALFWORLD_EVAL_GPUS=0,1 bash scripts/eval_alfworld_multiseed.sh \
  --checkpoint-paths \
    "$PWD/outputs/TEN_RUN/opd_opd/checkpoint-000400" \
    "$PWD/outputs/TEN_RUN/opd_opd/checkpoint-000800"
```

### Cú pháp cũ theo một run vẫn được hỗ trợ

Ví dụ eval CMT mới trên 2 GPU:

```bash
bash scripts/eval_alfworld_multiseed.sh \
  --method cmt \
  --run-output "$PWD/outputs/alfworld_v2_cmt_s42/cmt_opd" \
  --checkpoints 50 150 final \
  --seeds 42 43 44 --gpus 0,1 --episodes-per-task 8 \
  --tag id_ood_v2_avg8_pass8_s42_43_44
```

4 GPU: thay `--gpus 0,1` bằng `--gpus 0,1,2,3`. Chỉ chọn checkpoint tồn tại.
Số step được đổi thành `checkpoint-000050`...; `final` phải có metadata step.

Eval cả bốn run đã train ở mục 4, lần lượt:

```bash
for METHOD in opd ta cmt iw; do
  bash scripts/eval_alfworld_multiseed.sh \
    --method "$METHOD" \
    --run-output "$PWD/outputs/alfworld_v2_${METHOD}_s42/${METHOD}_opd" \
    --checkpoints 50 150 final \
    --seeds 42 43 44 --gpus 0,1 --episodes-per-task 8 \
    --tag id_ood_v2_avg8_pass8_s42_43_44
done
```

Cú pháp `--run-output ... --checkpoints ...` không có `--episodes-per-task 8`
vẫn giữ chế độ cũ **1 episode/task/seed**, chỉ có SR và diagnostics. Các lệnh
ở trên đã ghi rõ `--episodes-per-task 8` để lấy avg@8/pass@8. Các probe student/
teacher vẫn chạy một episode/task; training cũng giữ nguyên seed/rollout.

Kết quả ở `<run-output>/alfworld_eval/<tag>/`:

```text
plan.json                          # protocol version/fingerprint + exact config
resolved_config.yaml
summary.json                       # từng checkpoint, ID/OOD, từng task category
summary.csv                        # scalar metric mean/std/n/display
per_seed.csv
task_results.jsonl                 # từng task ID qua các seed
step-000050/seed-42/episodes.jsonl  # raw response, exact IDs, parsed action, observations
step-000050/seed-42/summary.json
step-000050/seed-42/rank-00000.log
complete.json                      # chỉ có khi toàn bộ evaluation hoàn thành
```

Summary có SR, average turns, parser failure, simulator-invalid, total-invalid,
punctuation recovery, valid action, action diversity/count, look/first-look/repeat,
unchanged observation, actor token count và termination reasons. Top 20 raw
responses và normalized actions được lưu cho mỗi split/task category/seed và
aggregate. Invalid-rate denominator là tất cả agent lượt; unchanged observation
rate dùng số simulator steps; punctuation recovery rate dùng tất cả lượt. SR
vẫn đến từ state.won. Giảm parser failure không được gọi là tăng task success.

`±` là **sample standard deviation (ddof=1) giữa seed 42/43/44 của cùng checkpoint**,
không phải confidence interval hay biến động training seed. Top-K overlap/mass và
gradient norm có trong training JSONL/TensorBoard; standalone eval không load
teacher/backward nên ghi chúng là training-only/không có giá trị, không bịa số.

Tag đã tồn tại bị từ chối. Dùng tag mới cho lần chạy khác hoặc lần trước bị ngắt.
Không gộp old-protocol summaries vào new-protocol summaries.

## 6. Chẩn đoán checkpoint cũ bằng protocol mới

Eval mặc định dùng protocol của training run, không tự đổi v1 sang v2. Có thể
chủ động so checkpoint v1 dưới giao thức mới trong output namespace mới:

```bash
bash scripts/eval_alfworld_multiseed.sh \
  --run-output "$PWD/outputs/OLD_RUN/cmt_opd" --method cmt \
  --checkpoints final --seeds 42 43 44 --gpus 0,1 \
  --set alfworld.protocol_version=2 \
  --set alfworld.action_prompt_mode=admissible \
  --allow-protocol-change --tag legacy_weights_new_protocol_v2
```

Plan/summary ghi rõ training fingerprint, evaluation fingerprint và
`evaluation_protocol_changed=true`. Đây là **checkpoint cũ dưới eval protocol
mới**, không phải kết quả của model được train bằng v2. Không đổi optimizer hay
checkpoint cũ và không dùng cờ này để vượt kiểm tra resume.

## 7. Train diagnostics và tests

Trong mỗi method output:

```text
resolved_config.yaml                          # protocol manifest/fingerprint
metrics.jsonl                                 # alfworld metrics + selector/loss/cost
alfworld_episodes/rank-XXXXX/*.jsonl             # active episodes từng rank
alfworld_rollout_diagnostics/*.json             # rates/top actions, lưu trước guard
alfworld_failures/*.jsonl                       # zero-actor diagnostics khi fail-fast
training_cost.jsonl / training_cost_summary.json
tensorboard/                                  # alfworld/*, cost/*, selector metrics
checkpoint-N/ / final/
```

Actor mask/indexing trong tensors không bị thay đổi khi tắt token IDs trong trace
bằng `--set alfworld.trace_token_ids=false`. Model scoring dùng full history và
causal p-1 alignment. Dòng parser-error feedback là context, không phải actor target.

CPU/mock regression và real simulator test:

```bash
python3 -m pytest -q tests/test_alfworld_protocol_v2.py \
  tests/test_alfworld_ppo_normalizer.py tests/test_alfworld_multiturn.py \
  tests/test_alfworld_evaluation.py tests/test_alfworld_runtime.py

ALFWORLD_REAL_DATA="$STORAGE_ROOT/nlp/minhpn19/data/ALFworld" \
python3 -m pytest -q tests/test_alfworld_multiturn.py
```

Optional GPU mechanics test dùng **mô hình ngẫu nhiên nhỏ** và scripted real
trajectory, không phải trained-agent performance:

```bash
ALFWORLD_REAL_DATA="$STORAGE_ROOT/nlp/minhpn19/data/ALFworld" \
ALFWORLD_TEST_TOKENIZER="$STORAGE_ROOT/$STUDENT_MODEL" \
python3 -m pytest -q -s tests/test_alfworld_v2_gpu.py
```

Các file scientific objective/selector khác giữ nguyên. Audit có danh sách sửa,
phản ví dụ PPO, validation thực tế và các kiểm tra B200/model còn chưa xác minh.

Validation cuối: **404 passed, 7 failed, 1 skipped, 46 subtests passed**. Cả 7 lỗi
được tái hiện trên HEAD sạch trước sửa; chi tiết từng lỗi nằm trong audit. GPU
toy OPD/CMT và real scripted ID/OOD three-seed checks đã chạy, nhưng student/teacher
probe bằng model thật và vLLM/FSDP B200 smoke cần chạy ở máy đích.
