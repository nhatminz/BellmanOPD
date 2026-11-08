# ALFWorld multi-turn OPD trên B200

Nếu CMT lỗi `Global PPO minibatch has no real trajectory` hoặc simulator lỗi
`GLIBCXX_3.4.32`, xem [ALFWORLD_FIX_CMT_RUNTIME.md](ALFWORLD_FIX_CMT_RUNTIME.md)
cho bản sửa và lệnh chạy cặp Qwen3-14B → Qwen3-4B trên 2 GPU.

Pipeline này dùng **simulator TextWorld PDDL thật** để reset/step từng game ALFWorld.
Student sinh mọi lượt; teacher được freeze và chỉ chấm các prefix student đã đi qua.
Train/eval dùng chung giao thức action và lịch sử đầy đủ. Eval chạy riêng trên ID/OOD.
Các launcher/config thí nghiệm toán hiện có giữ nguyên.

## 1. Chuẩn bị môi trường và dữ liệu

Chạy trong môi trường B200 hiện đã cài dependencies của repository, bao gồm vLLM
với CUDA IPC weight transfer đang dùng cho các thí nghiệm cũ. Cài thêm simulator:

```bash
export STORAGE_ROOT=/workspace/storage-shared
cd "$STORAGE_ROOT/nlp/minhpn19/BellmanOPD_analysis"
export PYTHON_BIN="$(command -v python3)"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
"$PYTHON_BIN" -m pip install -r requirements-alfworld.txt
```

TextWorld 1.7.0 đã được kiểm tra với dữ liệu này. `textworld[pddl]` cài backend PDDL
và Fast Downward; source build cần C/C++ compiler, `make`, CMake và executable
`python` từ venv trên `PATH`. Không cần ALFWorld THOR/Unity hay tải toàn bộ assets
ALFWorld vì parquet đã nhúng domain/problem/grammar của từng game. Thiếu simulator
hoặc file game sẽ báo lỗi setup; không có fallback sang phản hồi tĩnh.

Đặt ba file tại:

```text
$STORAGE_ROOT/nlp/minhpn19/data/ALFworld/train-00000-of-00001.parquet
$STORAGE_ROOT/nlp/minhpn19/data/ALFworld/eval_in_distribution-00000-of-00001.parquet
$STORAGE_ROOT/nlp/minhpn19/data/ALFworld/eval_out_of_distribution-00000-of-00001.parquet
```

Nếu chưa tải, lệnh này lấy trực tiếp các file từ dataset bạn cung cấp, bỏ qua file
đã tồn tại:

```bash
"$PYTHON_BIN" - <<'PY'
import os
import urllib.request
from pathlib import Path
root = Path(os.environ["STORAGE_ROOT"]) / "nlp/minhpn19/data/ALFworld"
root.mkdir(parents=True, exist_ok=True)
for split in ("train", "eval_in_distribution", "eval_out_of_distribution"):
    name = f"{split}-00000-of-00001.parquet"
    path = root / name
    if not path.exists():
        temporary = path.with_suffix(".download")
        urllib.request.urlretrieve(
            f"https://huggingface.co/datasets/awawa-agi/alfworld-raw/resolve/main/data/{name}",
            temporary,
        )
        temporary.replace(path)
    print(path)
PY
```

Schema thực tế đã đọc: `id`, `task_type`, `game_file_path`, `game_content` đều là
string. `game_content` là JSON chứa `pddl_domain`, `pddl_problem`, `grammar`,
`solvable`, `walkthrough`. Bản tải kiểm tra có **3553 train, 140 ID, 134 OOD**;
cả ba split disjoint theo ID, đường dẫn game và hash nội dung game. Loader còn hỗ
trợ metadata mapping và đường dẫn game thật tương đối với parquet.

Loader bỏ `walkthrough`/solution khỏi task runtime và khỏi game cache của policy.
Không đưa danh sách action đúng hay planner vào policy. Entity names dùng cách
đánh số xác định tương thích `AlfredDemangler` của ALFWorld. Grammar của raw export
dùng `move cup 1 to table 1` để đặt đồ; không nên chỉ prompt `put ... in/on ...`.

Kiểm tra dữ liệu và real reset/step, không tải model lên GPU:

```bash
"$PYTHON_BIN" -m b200_experiment.alfworld_cli ready
"$PYTHON_BIN" -m b200_experiment.alfworld_cli smoke
```

`ready` kiểm tra tất cả split và gọi reset/`look` qua simulator subprocess.
`smoke` chạy thêm walkthrough của đúng một game **chỉ để xác minh simulator**,
kiểm tra simulator trả `success=true`; đây không phải kết quả đánh giá policy.
Mặc định là game ID đầu tiên, đã kiểm tra hoàn thành sau 4 action.
Có thể chọn `--split eval_out_of_distribution --task-index 0`.
`--output readiness.json` ghi một report mới và từ chối ghi đè.

## 2. Teacher Qwen3-4B và student Qwen3-1.7B

Launcher giữ convention đường dẫn đang có trong repo, relative với `STORAGE_ROOT`:

```bash
export TEACHER_MODEL=models/Qwen3-4B
export STUDENT_MODEL=nlp/tungdd11/stable-on-policy-distillation/OPD/model/Qwen3-1.7B-Base
```

Nếu chưa có teacher ở vị trí này, tải trước khi kiểm tra model:

```bash
hf download Qwen/Qwen3-4B --local-dir "$STORAGE_ROOT/$TEACHER_MODEL"
```

Student mặc định dùng asset Base cũ. Khi chuẩn bị asset Base mới, nguồn tương ứng
là `Qwen/Qwen3-1.7B-Base`; bản `Qwen/Qwen3-1.7B` cũng dùng được bằng cách đặt
`STUDENT_MODEL` tới đúng thư mục chứa bản đó.

`STUDENT_MODEL` mặc định trỏ vào student asset cũ, kể cả tên thư mục có hậu tố
`Base`. Nếu bạn dùng Qwen/Qwen3-1.7B ở vị trí khác, thay biến đó bằng đúng đường
dẫn local của bạn; không đổi teacher/student giữa bốn phương pháp. Cần model/tokenizer
đầy đủ, chat template của student, và token-to-ID mapping giống hệt teacher.
Kiểm tra trước khi train:

```bash
"$PYTHON_BIN" -m b200_experiment.alfworld_cli ready --check-models \
  --set "models.teacher_path=$TEACHER_MODEL" \
  --set "models.student_path=$STUDENT_MODEL"
```

Student tokenizer là tokenizer duy nhất khi render/sampling/scoring; không
decode rồi re-tokenize input cho teacher. `enable_thinking=false` dùng cho cả
lịch sử chia sẻ. Kiểm tra tương thích tokenizer hiện có của repo vẫn chạy khi
load model. Chỉ thay model path không thể bỏ qua kiểm tra này.

ALFWorld thêm native assistant end-of-turn vào stop IDs khi tokenizer có token
đó. Điều này cần cho Qwen3 Base: EOS là `<|endoftext|>`, còn chat template đóng
lượt bằng `<|im_end|>`. Thay đổi stop này chỉ bật cho ALFWorld; explicit
`models.stop_token_ids` vẫn có quyền ưu tiên.

## 3. Train bốn phương pháp với cùng cấu hình

Các biến sau dùng chung cho mọi phương pháp:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GLOBAL_BATCH_SIZE=8
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

Chạy lần lượt, hoặc chọn phương pháp bạn muốn chạy trước:

```bash
RUN_NAME=alfworld_qwen3_cmt_s42 bash scripts/train_alfworld_multiturn.sh cmt
RUN_NAME=alfworld_qwen3_opd_s42 bash scripts/train_alfworld_multiturn.sh opd
RUN_NAME=alfworld_qwen3_ta_s42  bash scripts/train_alfworld_multiturn.sh ta
RUN_NAME=alfworld_qwen3_iw_s42  bash scripts/train_alfworld_multiturn.sh iw
```

`GLOBAL_BATCH_SIZE` là số task toàn cục, `NUM_RESPONSES` là số episode độc lập mỗi
task. Ví dụ trên tạo 32 episode/rollout và tối đa 4 optimizer updates/rollout;
`PPO_MINI_BATCH_SIZE=8` là số episode toàn cục/update, không phải mỗi GPU.
`MAX_STEPS` là tổng optimizer steps. Nếu muốn consume epoch thay vì giới hạn step,
dùng `MAX_STEPS=-1 NUM_EPOCHS=...` như các launcher cũ.
Global PPO mini-batch phải >= số GPU. `TRAIN_NPROC_PER_NODE` có thể giới hạn worker
trong mask GPU; mặc định dùng toàn bộ GPU trong `CUDA_VISIBLE_DEVICES`.

Config nằm ở `configs/alfworld_multiturn.yaml`; có thể duplicate launcher để sửa
biến trực tiếp hoặc truyền `--set key=value` cuối lệnh. Ví dụ kiểm tra tích hợp
ngắn trên B200 trước khi train dài:

```bash
RUN_NAME=alfworld_cmt_check MAX_STEPS=2 SAVE_INTERVAL=1 \
  bash scripts/train_alfworld_multiturn.sh cmt
```

Một episode dừng khi simulator terminal, agent chỉ sinh EOS, đạt giới hạn lượt,
tổng actor token, hoặc context. `MAX_TURN_TOKENS` là giới hạn mỗi response;
`MAX_RESPONSE_LEN` là tổng actor token trong episode; `MAX_EPISODE_TOKENS` tính cả
task/observations/chat envelopes/actions. Lịch sử không bị truncate. Khi tăng
context phải tăng cả `rollout.vllm.max_model_len` (launcher tự đặt theo biến context).
Action sai vẫn được simulator xử lý; response nhiều dòng/nhiều lệnh hoặc sai format
nhận protocol error, không có bước game giả. Timeout được ghi rõ trong trace.
Episode không có actor token không được biến thành loss giả; training báo lỗi.

## 4. Resume

Output mặc định: `outputs/<RUN_NAME>/<method>_opd/`. Ví dụ resume CMT ở step 50:

```bash
RESUME_FROM_CHECKPOINT="$PWD/outputs/alfworld_qwen3_cmt_s42/cmt_opd/checkpoint-000050" \
MAX_STEPS=1000 bash scripts/train_alfworld_multiturn.sh cmt
```

Hoặc chọn checkpoint hoàn chỉnh mới nhất:

```bash
OUTPUT_DIR="$PWD/outputs/alfworld_qwen3_cmt_s42/cmt_opd" \
RESUME_FROM_CHECKPOINT=auto MAX_STEPS=1000 \
  bash scripts/train_alfworld_multiturn.sh cmt
```

Giữ các biến model/batch/sampling/seed/context/LR như lần đầu. Resume phục hồi
model, optimizer, RNG và vị trí PPO mini-batch bằng cơ chế hiện có; checkpoint cần
`optimizer.pt`. `MAX_STEPS` là mốc tổng cần đạt, không phải số step chạy thêm.
ALFWorld protocol/max turns/token budgets cũng được kiểm tra khi resume.
Episode traces cũ được giữ; các rollout tái sinh có tên UUID mới và ghi rõ
`rollout_index`/`first_optimizer_step`, nên không ghi đè hay coi trace cũ là dữ liệu train.

## 5. Eval độc lập checkpoint bất kỳ trên seed 42/43/44

Không chạy eval trong training, không cần `eval_history.jsonl` cũ hoặc baseline.
Chọn method folder có `resolved_config.yaml`, và một hoặc nhiều checkpoint:

```bash
bash scripts/eval_alfworld_multiseed.sh \
  --method cmt \
  --run-output "$PWD/outputs/alfworld_qwen3_cmt_s42/cmt_opd" \
  --checkpoints 50 150 final \
  --seeds 42 43 44 \
  --gpus 0,1,2,3 \
  --tag id_ood_s42_43_44_v1
```

Chỉ chọn các checkpoint thực sự tồn tại. `50` được đổi thành `checkpoint-000050`;
cũng có thể truyền `checkpoint-000150`, `final`, hoặc absolute checkpoint path
trực tiếp nằm trong method run đã chọn:

```bash
bash scripts/eval_alfworld_multiseed.sh \
  --method ta \
  --run-output "$PWD/outputs/alfworld_qwen3_ta_s42/ta_opd" \
  --checkpoints "$PWD/outputs/alfworld_qwen3_ta_s42/ta_opd/checkpoint-000050" \
  --seeds 42 43 44 --gpus 0,1,2,3 --tag selected_s42_43_44_v1
```

`--dry-run` kiểm tra selection và tất cả task/split, in kế hoạch rồi thoát trước
khi tạo output/model. Mỗi seed chạy **fresh reset/step** trên toàn bộ 274 task,
mỗi task đúng một episode; seed generation phụ thuộc seed đánh giá + task ID,
không phụ thuộc GPU partition. Không dùng replay cache, không dùng avg@8/pass@8.
Mỗi GPU là một vLLM TP=1 replica, nhận task shard riêng; mỗi replica gọi offline
`LLM.generate` tuần tự để tránh truy cập engine từ nhiều thread.

Eval giữ giới hạn lượt/context/actor tokens của run; mặc định sampling
temperature=1, top_p=1. Có thể cấu hình rõ `--set evaluation.temperature=0.7`;
hãy giữ cùng protocol cho mọi phương pháp. SR/invalid rate lưu dạng [0,1];
mean ± std dùng **sample std giữa các seed (ddof=1)**. Average turns tính trên
mọi episode, gồm cả episode không thành công; invalid rate = tổng response invalid
/ tổng lượt. Task type và từng task ID có thống kê riêng.

Eval xuất vào `alfworld_eval/<tag>/` trong run được chọn. Tag đã tồn tại sẽ báo lỗi;
đổi `--tag` để chạy fresh một lượt khác, kể cả lần trước bị gián đoạn. Bỏ `--tag`
để tự tạo timestamp. Thư mục chỉ có `complete.json` sau khi đủ mọi task/seed/checkpoint.

## 6. Tìm metrics, checkpoint, trace và cost

```text
outputs/alfworld_qwen3_cmt_s42/cmt_opd/
  resolved_config.yaml
  run_metadata.json
  metrics.jsonl                         # loss/selector/cost + alfworld metrics
  checkpoint-000050/                    # HF model/tokenizer + optimizer.pt
  final/
  alfworld_episodes/rank-00000/*.jsonl   # train episode traces, mỗi rank riêng
  training_cost.jsonl                   # cost logger hiện có
  training_cost_summary.json
  tensorboard/                         # loss/selector/cost + alfworld/*
  alfworld_eval/<tag>/
    plan.json
    resolved_config.yaml
    complete.json
    summary.json                       # mỗi checkpoint, split và task type
    summary.csv                        # mean/std/n/display cho từng metric
    per_seed.csv
    task_results.jsonl                 # mỗi task ID, thống kê qua seed
    step-000050/
      summary.json
      seed-42/
        episodes.jsonl                 # đủ ID + OOD, sorted theo split/task ID
        summary.json
        rank-00000.jsonl               # shard streaming, giữ được nếu bị ngắt
        rank-00000.log
```

Train traces có full input IDs, actor positions, từng response/action/observation/
reward/success, ranh giới lượt/episode và lý do dừng. Có thể tắt phần token IDs
trong trace bằng `--set alfworld.trace_token_ids=false` để giảm dung lượng, nhưng
tensor actor mask/positions dùng trong training vẫn đầy đủ.

Cost/tensorboard dùng logger cũ: rollout + simulator time nằm trong rollout wall
time, có scoring/backward/optimizer/checkpoint time và peak memory. So sánh overhead
với OPD bằng `COST_OPD_BASELINE=<opd output>/training_cost_summary.json` sau khi đã
có OPD baseline cùng workload. Signature so sánh đã tính cả cấu hình ALFWorld.

## 7. Actor clock và tính đúng của objective

Mỗi hàng trajectory là **một episode**. `input_ids [B,S]` giữ mọi token trong
full interaction. `actor_token_mask [B,S]` chỉ bật token do student sinh.
`actor_positions [B,A]` liệt kê vị trí vật lý của những token đó; `response_ids`
và `valid_mask [B,A]` là trục actor compact, padding bên phải. Token actor ở vị trí
`p` lấy logits tại `p-1`, với đúng history và observation trước nó. Scoring và
backward gọi chung một hàm indexing để tránh lệch causal shift.

OPD giữ weights đồng đều trên valid actor tokens. TA giữ local teachability và
hard top-rho toàn cục. IW giữ sampled-token PPO objective và prefix discrepancy
tích lũy qua mọi lượt trong cùng hàng. CMT giữ Student-Top-K local gain, union
common-mass kernel, root-centered successor excess và constrained Gibbs allocation
theo PPO group đang có. Teacher-only Top-K không được thêm vào OPD policy loss.

Thay đổi cần thiết cho multi-turn là **đồng hồ quyết định**: chỉ actor tokens
chiếm bước trong IW prefix/CMT scan. Từ actor token cuối lượt trước tới actor token
đầu lượt sau, observation/chat envelope là conditioning context có loss bằng 0;
không chèn tỷ số policy hay discount cho các token môi trường. Recurrence vì vậy
đi qua ranh giới lượt, nhưng luôn dừng tại ranh giới episode/hàng. Đây là rollout
ước lượng trên đường đi môi trường thực tế, không marginalize các observation
counterfactual. Không thay Student Top-K/PPO loss hoặc các công thức selector.

vLLM sync current student weights **một lần trước cả batch episode**, sleep sau
khi mọi local episode kết thúc. Không có FSDP collective giữa các lượt; các rank
có số lượt khác nhau vẫn đi qua đúng collective schedule của trainer hiện có.
Không dùng teacher-generated turn hay curriculum của Guided-OPD
([bài tham khảo](https://aclanthology.org/2026.emnlp-main.1424/)).

## 8. Tests

CPU/mock, không cần model weights/vLLM/GPU:

```bash
"$PYTHON_BIN" -m pytest -q tests/test_alfworld_multiturn.py tests/test_alfworld_evaluation.py
```

Thêm actual simulator smoke, dùng đúng ba raw parquet đã tải:

```bash
ALFWORLD_REAL_DATA="$STORAGE_ROOT/nlp/minhpn19/data/ALFworld" \
  "$PYTHON_BIN" -m pytest -q tests/test_alfworld_multiturn.py tests/test_alfworld_evaluation.py
```

Tests chạy mô hình Qwen3 nhỏ khởi tạo ngẫu nhiên trên CPU để so causal logits với
full forward, kiểm tra loss targets chỉ gồm actor và teacher không nhận gradient, backward bằng
trainer thật cho cả bốn phương pháp, CMT cross-turn/IW prefix reset, parquet/split
isolation, seed độc lập, checkpoint selection, output aggregation/no overwrite và
resume protocol. Không tự động khởi chạy full multi-GPU training.

Kết quả kiểm tra local ngày 2026-11-08: nhóm test liên quan đạt **134 passed**
(17 subtests); toàn repo **327 passed, 7 failed, 1 skipped** (44 subtests).
Cả 7 test fail cũng fail trên bản HEAD sạch `279511c`: legacy avg@16 CSV merge,
config RAC legacy, phiên bản math prompt protocol, 3 test GRPO reward và một
expectation về giá trị padding trung tính của IW. Không có thay đổi công thức
OPD/TA/IW/CMT trong phần triển khai ALFWorld.

Đã kiểm tra thêm 12 real simulator smoke game, đủ 6 task types ở cả ID và OOD,
bằng walkthrough dùng riêng cho diagnostic; cả 12 hoàn thành. Đây không phải
điểm success rate của model. Tokenizer/config chính thức Qwen3-4B và Qwen3-1.7B
cũng đã được tải riêng để kiểm tra tương thích token IDs và append-only chat.
