# Gemma‑2: launcher độc lập và resume sau OOM

Code cần được cập nhật đầy đủ lên B200, gồm `training_cost.py`, `trainer.py` và
các launcher mới. Chỉ chép file `.sh` sẽ không có phần sửa cost/resume mới.

**Traceback của run `cmt_20261108_103126_485124171`**

OOM xảy ra tại `loss.backward()` ở step 189, trên rank 2. GPU có 178.35 GiB,
còn 25.45 GiB, trong khi backward cần thêm một allocation 30.20 GiB; PyTorch
đang allocated 141.15 GiB. Đây là bằng chứng để giảm backward micro-batch trước.
Không có bằng chứng từ traceback này rằng rollout, scoring hay checkpoint bị OOM.
Độ dài trajectory thay đổi giữa các batch có thể làm peak tăng sau hơn 100 step;
traceback chưa chứng minh có memory leak.

Cả bốn file dưới đây là launcher **độc lập**, có block `USER CONFIG` để sửa trực tiếp,
được tạo từ launcher CMT đã gửi. Không gọi `train_gemma2_b200.sh`, không cần overlay mới:

```text
scripts/train_cmt_b200_gemma.sh
scripts/train_opd_b200_gemma.sh
scripts/train_ta_b200_gemma.sh
scripts/train_iw_b200_gemma.sh
```

| Tham số | File đã gửi | Bốn file mới |
|---|---:|---:|
| Teacher | `nlp/minhpn19/models/gemma-2-9b-it` | giống cũ |
| Student | `nlp/minhpn19/models/gemma-2-2b-it` | giống cũ |
| GPUs | 4 | 4 |
| Global prompt batch | 64 | 64 |
| Response/prompt | 4 | 4 |
| Global PPO mini-batch | 64 | 64 |
| Backward micro-batch/GPU | **16** | **4** |
| Scoring micro-batch/GPU | 8 | 8 |
| LR | `5e-6` | `5e-6` |
| Epochs Competition-MATH | 3 | 3 |
| Prompt/response cap | 1024 / 4096 | 1024 / 4096 |
| Rollout vLLM utilization/context | 0.50 / 5200 | 0.50 / 5200 |
| Save interval | 150 | 150 |
| Train-time eval | false | false |
| Token-score stats interval | 100 | 100 |

Với bốn GPU, mỗi PPO mini-batch 64 trajectory chia thành 16 trajectory/GPU.
Micro=4 xử lý thành bốn lượt forward/backward trước mỗi optimizer update. Denominator
vẫn là global weighted-token mass của cùng PPO group, nên không đổi allocation domain,
global batch hoặc lịch optimizer. Không cần đổi LR. Floating-point accumulation có thể
khác; đây không phải hứa hẹn tái lập bit-for-bit.

Nếu micro=4 vẫn OOM ở backward: thử `MICRO=2`, sau đó `MICRO=1`. Đặt giống nhau cho cả
bốn method khi so sánh cost. `SCORE_MICRO_BATCH_SIZE=2` chỉ cần nếu có OOM ở scoring;
giảm nó không trực tiếp giải quyết allocation đang thất bại trong backward. Không giảm
`PPO_MINI_BATCH_SIZE`, `NUM_RESPONSES` hoặc response cap trong lần xử lý đầu tiên vì
chúng thay đổi thí nghiệm, đặc biệt là domain phân bổ CMT.

**Resume run cũ trên B200**

```bash
cd /workspace/storage-shared/nlp/minhpn19/BellmanOPD_analysis
RUN_DIR=/workspace/storage-shared/nlp/minhpn19/BellmanOPD_analysis/outputs/cmt_20261108_103126_485124171/cmt_opd

# Kiểm tra checkpoint đã save xong. Với SAVE_INTERVAL=150, checkpoint-000150
# là ứng viên gần nhất trước OOM step 189; phải tồn tại trên máy B200.
ls -lh "$RUN_DIR/checkpoint-000150/config.json" \
       "$RUN_DIR/checkpoint-000150/optimizer.pt"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
RUN_NAME=cmt_20261108_103126_485124171 \
OUTPUT_DIR="$RUN_DIR" \
RESUME_FROM_CHECKPOINT=auto \
MAX_STEPS=2250 \
MICRO=4 \
  bash scripts/train_cmt_b200_gemma.sh
```

`auto` chọn checkpoint đã lưu của run này. Có thể thay bằng
`RESUME_FROM_CHECKPOINT="$RUN_DIR/checkpoint-000150"` để chọn rõ step 150.
True resume cần cả weights, `config.json` và `optimizer.pt`; code restore optimizer,
RNG và xác định tiếp vị trí rollout/PPO theo step. `MAX_STEPS=2250` là **tổng đích**,
không phải train thêm 2250 step. Phần sau checkpoint chưa được lưu sẽ chạy lại;
không thể khôi phục trực tiếp step 189 từ terminal progress. Checkpoint không lưu
pending rollout; nếu save giữa một rollout batch, phần tiếp tục sẽ generate lại
rollout từ model của checkpoint, nên resume không phải tái lập bit-for-bit phiên cũ.

Nếu không có checkpoint đủ model + optimizer thì không thể true-resume từ run này.
Không dùng thư mục `.checkpoint-*.tmp` dở dang sau crash. Chạy `auto` khi không có
checkpoint sẽ báo lỗi rõ, không tự khởi động fresh run.

Không cần `RESUME_ALLOW_CONFIG_MISMATCH=true` chỉ để đổi micro-batch. Guard khoa học
vẫn kiểm tra model/data, seed, rollout, LR và formulation CMT. Nếu guard báo khác các
trường đó, đối chiếu `resolved_config.yaml` cũ trước khi chạy tiếp.

Code resume hiện có rewind metrics/eval/selector logs về checkpoint được chọn, rồi
ghi tiếp trong cùng `OUTPUT_DIR`; `resolved_config.yaml` gốc vẫn được giữ và có thêm
resolved config của phiên resume. Cost raw JSONL/CSV giữ audit cũ. Summary cost mới chỉ
tính các record **cùng runtime signature** (micro=4), bỏ record micro=16 và record
vượt step resume của nhánh cũ. Summary trước khi đổi runtime được lưu thêm thành
`training_cost_summary_<signature-cũ>.json`. Warmup áp dụng lại cho rollout đầu của
phiên resume, kể cả nó đang ở step > 0.

**Log cost khi dùng launcher duplicate**

Với `common_b200.sh` và `trainer.py` đã cập nhật, launcher bạn gửi cũng đã bật cost
logging mặc định; không phụ thuộc wrapper. Bốn file mới export các biến này rõ ràng:

```bash
export COST_LOGGING_ENABLED=true
export COST_WARMUP_ROLLOUTS=1
export COST_MEMORY_SAMPLE_INTERVAL_SEC=0.25
export TENSORBOARD_ENABLED=true
export TENSORBOARD_LOG_INTERVAL=1
```

Các file ở ngay `$RUN_DIR`, cạnh checkpoint và eval:

```text
training_cost.jsonl
training_cost.csv
training_cost_summary.json
tensorboard/
```

Có train seconds/optimizer step (amortized theo số optimizer update thật), thời gian
rollout/scoring/selector/allocation, PyTorch allocated/reserved peaks và device-wide
memory peak lấy mẫu (gồm vLLM process). Thời gian save checkpoint, snapshot export và
periodic eval được tách khỏi train time. Peak lấy mẫu không đảm bảo bắt mọi spike ngắn.
`cost/*` cũng được ghi vào TensorBoard, trong đó tag step cost đúng là
`cost/train_seconds_per_optimizer_step`; tag legacy `system/step_time` vẫn có nghĩa cũ.

**Overhead % cần có OPD baseline đã đo**; file bạn gửi chưa khai báo baseline nên chưa
có overhead so với OPD. Có thể resume CMT ngay để không mất tiến độ, rồi bổ sung
overhead vào report và TensorBoard sau khi chạy xong OPD cùng micro=4.

**Train ba baseline, cùng cấu hình với CMT đã giảm micro**

```bash
RUN_NAME=opd_gemma_micro4_seed42 \
  bash scripts/train_opd_b200_gemma.sh

OPD_COST=/workspace/storage-shared/nlp/minhpn19/BellmanOPD_analysis/outputs/opd_gemma_micro4_seed42/opd

RUN_NAME=ta_gemma_micro4_seed42 COST_OPD_BASELINE="$OPD_COST" \
  bash scripts/train_ta_b200_gemma.sh

RUN_NAME=iw_gemma_micro4_seed42 COST_OPD_BASELINE="$OPD_COST" \
  bash scripts/train_iw_b200_gemma.sh
```

Mặc định tự đặt run name có timestamp nếu bỏ `RUN_NAME`. Chạy fresh cần run name/output
mới. Mỗi method vẫn có objective riêng: OPD uniform Student-Top-K, TA hard selection,
IW sampled-action prefix weighting và CMT groupwise allocation. Config chung giống nhau
không có nghĩa thay objective của IW thành Top-K hoặc bỏ selection của TA.

Nếu OPD đã xong trước khi resume CMT, thêm `COST_OPD_BASELINE="$OPD_COST"` vào lệnh
resume để overhead hiện ngay trong TensorBoard. Baseline micro=16, batch khác hoặc số
GPU khác bị từ chối; dùng một OPD run micro=4 cùng runtime với ba baseline mới.

Nếu CMT chạy trước OPD, bổ sung overhead **sau khi OPD đo xong**:

```bash
python3 -m b200_experiment.training_cost \
  --baseline "$OPD_COST" \
  --outputs "$RUN_DIR" \
  --tensorboard
```

Lệnh ghi `training_cost_comparison.json` vào `$RUN_DIR` và thêm tag
`cost/overhead_vs_opd_percent` vào TensorBoard của chính run đó. Nó chỉ dùng record
cùng signature, sau warmup, trong phạm vi step của summary hiện tại; không trộn
micro=16, không chỉnh history eval hoặc weights. Raw cost log vẫn là số đo tại thời
điểm train. Tránh gọi lại backfill nhiều lần với các baseline khác nhau nếu muốn
giữ một chuỗi overhead duy nhất trong TensorBoard.

Overhead là so sánh end-to-end giữa các run thực, nên còn chịu ảnh hưởng độ dài
trajectory và giai đoạn training. Summary cost sau khi đổi micro không đại diện
cho toàn bộ run từ step 0; dùng `runtime_settings`, `resume_step`, số step đo và
raw cost log để xác định phạm vi.

```bash
tensorboard --logdir "$RUN_DIR/tensorboard" --port 6006
```
