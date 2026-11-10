# Sửa CMT PPO partition và lỗi GLIBCXX khi chạy ALFWorld

## Timeout vLLM khi khởi động CMT ngày 09/11/2026

Run `alfworld_v2_cmt_20261109_175814_891588417/cmt_opd` dừng trong
`VLLMRolloutEngine.start()`: `/health` chưa ready trong timeout 600 giây.
Đây là giai đoạn trước khi tải HF teacher/student, scoring và optimizer update.
Lệnh khởi tạo server giống nhau cho OPD/TA/CMT/IW khi cùng cấu hình rollout;
việc OPD/TA trước đó chạy được không chứng minh lỗi nằm trong CMT objective.

Log được gửi chỉ có vài WARNING. Không đủ bằng chứng để kết luận vì sao engine
chậm/kẹt bên trong, ví dụ import/CUDA initialization hay bước khác. Không coi
`Unknown vLLM environment variable ... VLLM_LOGPROB_SANITY_FAIL` là fatal error.

Bản sửa:

- ALFWorld tăng health startup timeout từ 600 lên 1800 giây, có thể chỉnh
  `ROLLOUT_VLLM_STARTUP_TIMEOUT`. Math/Gemma giữ timeout YAML hiện có khi không
  đặt biến này. Cả launcher 4B/1.7B và 14B/4B đều hỗ trợ.
- ALFWorld dùng INFO server log, chỉnh bằng `ROLLOUT_VLLM_LOGGING_LEVEL`.
  In tiến độ khởi động mỗi 30 giây, lưu GPU/PID/timeout/HTTP state và giai đoạn
  vào `vllm_rollout_server.rank-00000.startup.json` (tương tự rank 1) ngay cạnh
  file `.log`, checkpoint/eval/training metrics.
- Loại các biến diagnostic riêng `VLLM_LOGPROB_SANITY_*` và
  `VLLM_RETURN_LOG_PROBS` khỏi môi trường server, sau khi launcher đã đọc chúng.
  Diagnostic trong trainer vẫn giữ nguyên giá trị.
- Khi đóng server, dọn đúng process group riêng của run kể cả API launcher đã
  thoát nhưng EngineCore/worker còn sống; giúp run nối tiếp nhận lại GPU.

Theo [hướng dẫn vLLM](https://docs.vllm.ai/en/v0.17.1/usage/troubleshooting/#enable-more-logging),
có thể dùng DEBUG để tìm bước bị kẹt. Bản sửa bổ sung khả năng quan sát và đủ
thời gian cho cold start; không tuyên bố tăng timeout sẽ sửa được mọi deadlock.

Sau khi cập nhật code, chạy riêng CMT mới trên **hai GPU đang rảnh**. Nếu IW
trong log vẫn chạy trên GPU 0,1, chờ IW hoàn tất trước khi dùng hai GPU đó:

```bash
export STORAGE_ROOT=/workspace/storage-shared
cd "$STORAGE_ROOT/nlp/minhpn19/BellmanOPD_analysis"
unset RUN_NAME OUTPUT_DIR RESUME RESUME_FROM_CHECKPOINT

CUDA_VISIBLE_DEVICES=0,1 TRAIN_NPROC_PER_NODE=2 \
TEACHER_MODEL=models/Qwen3-4B STUDENT_MODEL=models/Qwen3-1.7B \
BATCH_SIZE=8 GLOBAL_BATCH_SIZE=8 NUM_RESPONSES=4 PPO_MINI_BATCH_SIZE=8 \
MICRO_BATCH_SIZE_PER_GPU=1 SCORE_MICRO_BATCH_SIZE=1 \
SEED=42 ROLLOUT_SEED=42 MAX_STEPS=1000 NUM_EPOCHS=1 LR=5.0e-6 SAVE_INTERVAL=50 \
ROLLOUT_VLLM_STARTUP_TIMEOUT=1800 ROLLOUT_VLLM_LOGGING_LEVEL=INFO \
bash scripts/train_alfworld_multiturn.sh cmt
```

Tên run tự sinh theo timestamp. Run lỗi chưa train step nào nên không có
optimizer/checkpoint để resume từ lần đó. Objective, batch, rollout token budget,
PPO/CMT allocation và learning rate không thay đổi bởi bản sửa startup này.

Nếu vẫn timeout, lấy log INFO trong **run mới** (đường dẫn được in ngay lúc start)
và file `.startup.json`. Có thể đặt `ROLLOUT_VLLM_LOGGING_LEVEL=DEBUG` cho lần
chẩn đoán kế tiếp. Khi gửi log, giữ cả các dòng trước traceback; chúng cho biết
engine kẹt ở bước nào. Không cần cài lại venv hay sửa checkpoint để áp dụng patch.

Kiểm thử bổ sung trong `tests/test_vllm_rollout_startup.py`: fake cold start
700 giây vượt budget cũ, timeout/HTTP state, early server exit, process-group
cleanup, mapping timeout qua launcher và workload giống nhau giữa bốn method.
Có thêm test HTTP server/process group thật trên CPU; không coi đó là xác nhận
vLLM/CUDA IPC hoạt động trên B200.

## Hai nguyên nhân khác nhau trong log

Log trên `general-embedding-training-0-0` dừng ở `stage=train`: CMT sắp trajectory
theo response index rồi theo rank. Với global task batch=16, n=4, PPO batch=8 và
4 GPU, response group đầu tiên có 16 episode; PPO update đầu lấy 4 episode từ
rank 0 và 4 từ rank 1, còn rank 2/3 nhận 0. Guard collective vì thế báo lỗi trên
mọi rank. OPD đang dùng rank-interleaved partition nên mỗi rank nhận 2 episode
và vẫn chạy được.

Đã sửa `grouped_ppo_minibatch_partition`: giữ thứ tự response group, xen kẽ rank
bên trong từng group. Cấu hình này vẫn tạo **64 episode/rollout, 8 episode/update,
8 optimizer updates/rollout**; cả 4 rank đều nhận 2 episode/update. Với 2 GPU,
mỗi rank nhận 4. Không đổi objective, KL allocation budget hay số optimizer step.
Trường hợp legacy PPO batch bằng toàn bộ task batch vẫn giữ membership của mỗi
response group như trước.

Log trên `minz1-0-0` dừng ở `stage=rollout-vllm`, trong `env.reset`: bản native
`libdownward.so` cần symbol `GLIBCXX_3.4.32` nhưng node đang load `libstdc++.so.6`
thiếu symbol đó. Đây là sự không tương thích binary/runtime C++, xảy ra trước
teacher scoring. Cùng lỗi này có thể ảnh hưởng cả bốn phương pháp trên node đó;
việc đổi teacher sang 14B không sửa hay gây ra yêu cầu ABI này.
Xem [GCC ABI guidelines](https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html).

## Sau khi cập nhật code lên B200

OPD đang chạy bình thường có thể tiếp tục. Hai log CMT đều dừng trước optimizer
step đầu, nên chạy lại CMT với tên mới; chưa có checkpoint để true-resume từ hai
lần lỗi đó.

Chạy lại cặp 4B → 1.7B, giữ cấu hình được in trong log của bạn:

```bash
export STORAGE_ROOT=/workspace/storage-shared
cd "$STORAGE_ROOT/nlp/minhpn19/BellmanOPD_analysis"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
TEACHER_MODEL=models/Qwen3-4B STUDENT_MODEL=models/Qwen3-1.7B \
GLOBAL_BATCH_SIZE=16 NUM_RESPONSES=4 PPO_MINI_BATCH_SIZE=8 \
MICRO_BATCH_SIZE_PER_GPU=2 \
RUN_NAME=alfworld_cmt_s42_fixed \
bash scripts/train_alfworld_multiturn.sh cmt
```

## Trên node bị GLIBCXX, build native runtime một lần

Trong venv B200 hiện tại, trên chính node `minz1-0-0`:

```bash
export STORAGE_ROOT=/workspace/storage-shared
cd "$STORAGE_ROOT/nlp/minhpn19/BellmanOPD_analysis"
export PYTHON_BIN="$(command -v python3)"

bash scripts/rebuild_alfworld_native.sh
python3 -m b200_experiment.alfworld_cli smoke
```

Script build lại **đúng phiên bản fast-downward-textworld đã cài** từ source với
`/usr/bin/gcc` và `/usr/bin/g++` của node, kiểm tra `load_lib()` trong subprocess
mới rồi publish bản build riêng. Cần `gcc`, `g++`, `make`, `cmake` trên máy.
Có thể chọn compiler bằng `ALFWORLD_CC` / `ALFWORLD_CXX`, số build jobs bằng
`ALFWORLD_BUILD_JOBS` (mặc định 2).

Không cài đè torch/vLLM/TextWorld hay Fast Downward trong venv dùng chung, không
đổi system `libstdc++` và không đặt `LD_PRELOAD` cho trainer. Bản native nằm ở
`.alfworld-native/<hostname>/<python-tag>`, được simulator tự chọn trên đúng
host/Python đó. Build lỗi sẽ không thay bản đang dùng. Các bản cũ được giữ cho
worker còn sống. `.alfworld-native/` được git-ignore; máy khác build bản riêng.

Nếu cần đặt ngoài repo:

```bash
export ALFWORLD_NATIVE_SITE="$STORAGE_ROOT/nlp/minhpn19/alfworld-native/$(hostname)/current"
bash scripts/rebuild_alfworld_native.sh
python3 -m b200_experiment.alfworld_cli smoke
```

Giữ biến `ALFWORLD_NATIVE_SITE` trong shell train/eval nếu dùng đường dẫn tùy chọn.
Không cần biến này khi dùng vị trí mặc định. Mọi run train ALFWorld mới cũng kiểm
tra real reset/step trên từng rank trước khi tải HF/vLLM model, để lỗi ABI hiện ra
sớm. `smoke` chỉ dùng walkthrough để kiểm tra simulator, không phải policy score.

## Train cặp 14B → 4B

Đã thêm bản launcher đầy đủ `scripts/train_alfworld_multiturn_14b.sh`, có vùng
USER CONFIG để sửa trực tiếp như file bạn gửi. Mặc định: 2 GPU, global task batch
16, n=4, PPO=8, micro=4/GPU, score micro=1, LR=5e-6, max steps=1000, save mỗi 100
step. Cả bốn phương pháp dùng chung các giá trị đó.

Sau khi `smoke` thành công, chạy trên node 2 GPU:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export TEACHER_MODEL=models/Qwen3-14B
export STUDENT_MODEL=models/Qwen3-4B

RUN_NAME=alfworld_cmt_s42_14b_fixed bash scripts/train_alfworld_multiturn_14b.sh cmt
RUN_NAME=alfworld_opd_s42_14b_fixed bash scripts/train_alfworld_multiturn_14b.sh opd
RUN_NAME=alfworld_ta_s42_14b_fixed  bash scripts/train_alfworld_multiturn_14b.sh ta
RUN_NAME=alfworld_iw_s42_14b_fixed  bash scripts/train_alfworld_multiturn_14b.sh iw
```

Các biến environment đã export được ưu tiên hơn giá trị mặc định trong launcher.
`MAX_STEPS=1000` đặt mục tiêu tổng 1000 optimizer step; `NUM_EPOCHS=2` chỉ quyết
định thời lượng khi dùng `MAX_STEPS=-1`.

## Kiểm tra bản sửa

```bash
python3 -m pytest -q tests/test_cmt_grouped_distributed.py tests/test_alfworld_runtime.py
```

Test regression thực thi optimizer + constrained Gibbs với Gloo/DDP thật trên
CPU 2/4 rank, full multi-turn actor positions, 64 episode và PPO=8; xác minh
đúng 8 updates, coverage mỗi rank, không lặp/mất row và model đồng bộ giữa rank.
Test runtime xác minh lựa chọn native package theo host/Python và error ABI có
lệnh khắc phục cụ thể. Source rebuild và real one-episode smoke cũng đã chạy
thành công ở máy code; cần build lại trên node B200 bị lỗi để khớp runtime tại đó.
