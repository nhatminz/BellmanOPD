# ALFWorld eval: extra_special_tokens list / keys() error

## Nguyên nhân và bản sửa

Traceback gốc dừng ở `AutoTokenizer.from_pretrained`, trước model generation:

```text
self.SPECIAL_TOKENS_ATTRIBUTES + list(special_tokens.keys())
AttributeError: 'list' object has no attribute 'keys'
```

Trong schema Transformers 4.x, `extra_special_tokens` là mapping các named
special tokens. Schema mới còn lưu anonymous tokens dưới dạng list. Xem mã
nguồn chính thức [Transformers 4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/tokenization_utils_base.py)
và [5.0.0](https://github.com/huggingface/transformers/blob/v5.0.0/src/transformers/tokenization_utils_base.py).
Checkpoint có metadata dạng list có thể không đọc được trong môi trường 4.x.
Standalone eval không chạy selector TA/IW; sự khác nhau ở metadata checkpoint
không chứng minh thuật toán TA/IW bị lỗi.

`tokenizer_compat.prepare_tokenizer_view` chỉ tạo view khi gặp list:

- Chuyển list sang `additional_special_tokens`, hợp nhất với list đã có, giữ
  serialized AddedToken metadata. Empty list được bỏ khỏi extra field.
- Không đổi named mapping `extra_special_tokens` dạng dictionary.
- Giữ backend tokenizer, vocabulary, added-token IDs và chat templates. Asset
  được symlink vào thư mục eval riêng; model weights không bị copy hay thay đổi.
- Nếu list chứa token chưa có trong saved vocabulary, báo lỗi rõ, không tự thêm
  ID mới. Sau khi load, kiểm tra toàn bộ vocabulary/IDs và tokenizer length.
- **Cả HF loader và vLLM** đọc cùng tokenizer view. Chỉ override kwargs của
  `AutoTokenizer` sẽ chưa đủ vì vLLM tự load tokenizer thêm một lần.
- Model vẫn load weights từ checkpoint được chọn, không fallback sang base
  model hoặc teacher. Source checkpoint/config không bị chỉnh sửa.

Parent worker monitor cũng được sửa: worker thoát nhanh đồng thời không còn
bị báo chung chung `Incomplete ALFWorld worker`. Thông báo terminal và report
JSON chứa exit code/signal, phase, task/sample và log tail. Worker import,
tokenizer, engine startup, episode và cleanup errors đều được ghi; `.done` chỉ
được tạo sau khi episode output đóng và engine cleanup thành công.

## Chạy lại TA và IW, 2 GPU

Cập nhật các file code trong bản sửa lên B200, dùng venv hiện tại. Không cần
train lại, sửa checkpoint gốc hoặc đổi dependency version chỉ để xử lý lỗi này.

```bash
export STORAGE_ROOT=/workspace/storage-shared
cd "$STORAGE_ROOT/nlp/minhpn19/BellmanOPD_analysis"
export EVAL_TAG="alfworld_ta_iw_fixed_$(date +%Y%m%d_%H%M%S_%N)"

ALFWORLD_EVAL_GPUS=0,1 \
bash scripts/eval_alfworld_multiseed.sh \
  --checkpoint-paths \
  "$STORAGE_ROOT/nlp/minhpn19/BellmanOPD_analysis/outputs/alfworld_v2_ta_20261109_131350_168022203/ta_opd/checkpoint-000400" \
  "$STORAGE_ROOT/nlp/minhpn19/BellmanOPD_analysis/outputs/alfworld_v2_iw_20261109_182653_332980038/iw_opd/checkpoint-000400" \
  --seeds 42 43 44 --episodes-per-task 8
```

Lệnh chạy lần lượt checkpoint, mỗi checkpoint có seed 42/43/44 và 8 episodes/
task/seed. Mỗi GPU là một inference replica. Protocol, task/sample seeds,
generation settings, avg@8/pass@8 và sample standard deviation giữ nguyên.
Tag mới tạo output riêng, không ghi đè kết quả cũ. Có thể giữ list CHECKPOINTS
đã sửa trong launcher rồi chạy lại launcher bình thường thay vì dùng CLI list.

## Output và logs

Mỗi run:

```text
<run>/alfworld_eval/<tag>/checkpoint-000400/summary.{json,csv}
<run>/alfworld_eval/<tag>/checkpoint-000400/step-000400/seed-42/
  rank-00000.log
  rank-00000.status.json
  rank-00000.error.json          # chỉ có khi worker gặp exception
  rank-00000.tokenizer/          # chỉ có nếu metadata cần chuyển format
  rank-00000.jsonl
  rank-00000.done                # chỉ khi worker hoàn tất thành công
  worker_failures.json          # chỉ khi parent phát hiện failure/timeout
```

Seed 43/44 và rank 1 có cùng cấu trúc. File combined summary:

```text
outputs/alfworld_eval/<tag>/summary.json
outputs/alfworld_eval/<tag>/summary.csv
```

Checkpoint thất bại không được ghi thành điểm 0 hoặc summary hoàn chỉnh; batch
vẫn tiếp tục checkpoint sau và ghi failure rõ trong `summary.json`.

## Files và validation

- `b200_experiment/tokenizer_compat.py`: tokenizer view và kiểm tra IDs.
- `b200_experiment/alfworld_evaluation.py`: dùng view cho HF/vLLM, worker state,
  error propagation và completion checks.
- `tests/test_alfworld_worker_failures.py`: tái hiện lỗi với lớp
  `Qwen2TokenizerFast` thật, vocabulary nhỏ tạo local; kiểm tra empty/nonempty
  lists, serialized AddedToken, unchanged
  vocabulary/template, cả bốn methods, hai lần load tokenizer, fast worker
  failures, exit signals, timeout và failure phases.
- `ALFWORLD_EVAL_TOKENIZER_FIX.md`, `ALFWORLD_MULTITURN.md`: hướng dẫn.

Test không tải model Qwen lớn lên GPU; phần engine dùng stub đọc tokenizer thật.
Không tuyên bố đã eval checkpoint TA/IW thật trên máy B200 từ máy code local.

Kết quả: **134 passed, 2 skipped, 3 subtests passed** (11.44s); hai test cần raw
ALFWorld parquet thực được skip vì không bật `ALFWORLD_REAL_DATA`. Ruff,
`git diff --check` và Bash syntax check của launcher đều pass.

```bash
python3 -m pytest -q \
  tests/test_alfworld_worker_failures.py tests/test_alfworld_evaluation.py \
  tests/test_alfworld_sampled_evaluation.py tests/test_alfworld_multiturn.py \
  tests/test_alfworld_protocol_v2.py tests/test_alfworld_runtime.py \
  tests/test_alfworld_ppo_normalizer.py tests/test_models.py \
  --disable-warnings --tb=short
```
