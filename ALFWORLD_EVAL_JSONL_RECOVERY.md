# ALFWorld: Unicode JSONL error, recovery, and zero-success diagnostics

## Lỗi JSONL sau khi progress đạt 100%

Reader cũ dùng `read_text().splitlines()`. Python còn tách các ký tự Unicode
`U+0085`, `U+2028`, `U+2029`, trong khi những ký tự này được phép nằm nguyên
trong JSON string do `json.dumps(..., ensure_ascii=False)` sinh ra. Khi response
chứa chúng, một episode hợp lệ bị chẻ thành nhiều đoạn và gây `Unterminated
string`. Progress đếm byte LF nên vẫn có thể đạt đủ 2192 episodes.

Đã tái hiện lỗi bằng record hợp lệ dài trên 46.000 ký tự. Reader mới đọc từng
physical LF-delimited record bằng binary file iterator, giữ nguyên nội dung
Unicode và token IDs. JSONL mới escape Unicode; giá trị sau decode không đổi.
Nếu file thực sự hỏng/truncated, code báo đúng file và physical line, không bỏ
qua record hoặc tự điền điểm 0. Chưa có shard seed 43 của người dùng tại máy
code để khẳng định nó chứa ký tự nào; audit bên dưới xác minh được trực tiếp.

## Kết quả seed 42 đã cung cấp

Đã đọc file `summary.json` người dùng tải về cho IW checkpoint 750, seed 42:

| Thống kê | Giá trị quan sát |
|---|---:|
| Episode thành công / tổng episode | 0 / 2192 |
| Episode dừng do context budget | 1482 / 2192 = 67.61% |
| Episode dừng do max turns | 560 / 2192 = 25.55% |
| Episode dừng do agent EOS | 150 / 2192 = 6.84% |
| Turn không qua parser | 19520 / 53732 = 36.33% |
| Actor tokens trong invalid turns | 2375298 / 2567534 = 92.51% |

Có response rỗng, `1 1 1 ...`, `10 10 ...`, `</think>`, thiếu tên/số object
và nhiều lệnh trong một lượt. Summary ghi đúng zero simulator successes;
không phải việc seed 43 lỗi đọc JSON khiến seed 42 bị đổi thành 0. Các dấu
hiệu cho thấy hành vi sinh có vấn đề, nhưng aggregate summary không chứng minh
riêng nguyên nhân do training, prompting hay simulator. Không tự nới parser,
project action hoặc thay success label để tăng điểm.

## Audit dữ liệu đã chạy, không cần GPU/model

Sau khi cập nhật code lên B200:

```bash
export STORAGE_ROOT=/workspace/storage-shared
cd "$STORAGE_ROOT/nlp/minhpn19/BellmanOPD_analysis"
RUN_OUTPUT="$PWD/outputs/alfworld_v2_iw_20261109_182653_332980038/iw_opd"
OLD_TAG=alfworld_avg8_pass8_20261111_024933_002928366
OLD_STEP="$RUN_OUTPUT/alfworld_eval/$OLD_TAG/checkpoint-000750/step-000750"

bash scripts/eval_alfworld_multiseed.sh --audit-seed "$OLD_STEP/seed-43"
```

Đổi `seed-43` thành `seed-42` để audit seed 42. Đây là read-only CPU command:
không sinh episode, không ghi đè file cũ. Report có success counts, termination
counts/fractions, parser/simulator-invalid/repeat rates, invalid actor-token
fraction, top parser failures và kiểm tra episode/turn success consistency.
`response_unicode_line_separator_occurrences` cho biết response có ký tự nào
trong ba ký tự gây lỗi reader cũ. Dữ liệu thiếu hoặc sai task/seed/protocol bị
từ chối thay vì tạo summary không đầy đủ.

## Khôi phục 42/43 và chỉ chạy seed còn thiếu

Có thể giữ checkpoint paths trong mảng CHECKPOINTS. Với list chỉ chứa run IW
checkpoint 750 ở trên:

```bash
ALFWORLD_EVAL_GPUS=0,1 \
REUSE_TAG=alfworld_avg8_pass8_20261111_024933_002928366 \
EVAL_TAG="alfworld_recovered_$(date +%Y%m%d_%H%M%S_%N)" \
bash scripts/eval_alfworld_multiseed.sh
```

Hoặc chỉ định checkpoint ở CLI để chắc chắn chọn đúng run cần khôi phục:

```bash
ALFWORLD_EVAL_GPUS=0,1 \
REUSE_TAG=alfworld_avg8_pass8_20261111_024933_002928366 \
EVAL_TAG="alfworld_recovered_$(date +%Y%m%d_%H%M%S_%N)" \
bash scripts/eval_alfworld_multiseed.sh \
  --checkpoint-paths "$RUN_OUTPUT/checkpoint-000750" \
  --seeds 42 43 44 --episodes-per-task 8
```

Code đọc source tag dưới từng run/checkpoint. Nó kiểm tra method, checkpoint,
8-attempt protocol, toàn bộ task/sample IDs, policy/reset seeds, protocol
fingerprint và game fingerprint trước khi dùng lại. Seed đã có consolidated
episodes + summary, hoặc đủ worker shards + `.done`, được đọc lại. Seed thiếu
hoặc chưa hoàn tất được chạy mới trong namespace mới. Nếu 42/43 có dữ liệu đầy
đủ như progress đã báo và JSONL hợp lệ, chỉ seed 44 cần model inference.

Source tag không bị sửa. Output tag phải mới và khác source tag. Không cần
giữ nguyên số GPU khi tái sử dụng vì source topology được đọc từ old plan và
mọi episode seed đều được kiểm tra. Khi muốn sinh lại hoàn toàn, để REUSE_TAG
rỗng (`REUSE_TAG= bash scripts/eval_alfworld_multiseed.sh`).

Trong USER CONFIG của launcher có thêm `REUSE_TAG` để sửa trực tiếp nếu thích.
Namespace cũ không được dùng làm output. Cached samples vẫn thuộc checkpoint
và policy cũ đã xác minh; `reused_from` và `reuse_tag` ghi provenance rõ ràng.

Output mới:

```text
<run>/alfworld_eval/<new-tag>/checkpoint-000750/step-000750/seed-42/diagnostics.json
<run>/alfworld_eval/<new-tag>/checkpoint-000750/step-000750/seed-43/diagnostics.json
<run>/alfworld_eval/<new-tag>/checkpoint-000750/step-000750/seed-44/diagnostics.json
<run>/alfworld_eval/<new-tag>/checkpoint-000750/summary.csv
<run>/alfworld_eval/<new-tag>/checkpoint-000750/summary.json
outputs/alfworld_eval/<new-tag>/summary.csv
outputs/alfworld_eval/<new-tag>/summary.json
```

Summary vẫn tính avg@8/pass@8 mean ± sample std trên đủ 3 seeds. Reuse không
tính lại reward hay thay policy/actions; nó chỉ đọc lại episode outcomes.

## Phân biệt lỗi simulator với model không giải được task

Kiểm tra simulator bằng walkthrough smoke riêng, không phải điểm model:

```bash
python3 -m b200_experiment.alfworld_cli smoke --split eval_in_distribution
python3 -m b200_experiment.alfworld_cli smoke --split eval_out_of_distribution
```

Nếu smoke giải được task, probe teacher trên ít task trước khi quyết định
train lại student hoặc chạy thêm full evaluation:

```bash
python3 -m b200_experiment.alfworld_cli probe \
  --config "$RUN_OUTPUT/resolved_config.yaml" \
  --role teacher --split eval_in_distribution --limit 8 --seed 42 --gpus 0,1 \
  --output "$RUN_OUTPUT/probe_teacher_$(date +%Y%m%d_%H%M%S_%N)"
```

Probe dùng teacher/student assets và interaction protocol của run đã lưu.
Đây là diagnostic trên subset, không thay kết quả full-checkpoint evaluation.

## Files

- `b200_experiment/alfworld_eval_artifacts.py`: Unicode-safe JSONL reader,
  completed-artifact loading và observed-outcome diagnostics.
- `b200_experiment/alfworld_evaluation.py`: reader mới, audit/reuse CLI,
  source/episode validation, atomic consolidated JSONL và per-seed diagnostics.
- `scripts/eval_alfworld_multiseed.sh`: REUSE_TAG và audit passthrough; giữ
  checkpoint list, seed list, GPU defaults và fresh-run behavior hiện có.
- `tests/test_alfworld_artifact_recovery.py`: Unicode/corruption, parent reader,
  completed/partial seed recovery, guards, provenance, summary math, audit CLI.
- `ALFWORLD_EVAL_JSONL_RECOVERY.md`, `ALFWORLD_MULTITURN.md`: hướng dẫn.

Training objective, tokenizer compatibility fix, interaction protocol và
evaluation scoring không thay đổi. Chưa chạy checkpoint B200 thật tại máy code.

Validation: **148 tests passed, 3 subtests passed, 2 skipped** (15.59s). Hai
optional tests cần raw ALFWorld dataset được skip. Ruff, Bash syntax check và
`git diff --check` đều pass. Bộ test gồm regression suites cho tokenizer fix,
ALFWorld protocol, metrics, worker failures và PPO normalization.

```bash
python3 -m pytest -q \
  tests/test_alfworld_artifact_recovery.py tests/test_alfworld_worker_failures.py \
  tests/test_alfworld_evaluation.py tests/test_alfworld_sampled_evaluation.py \
  tests/test_alfworld_multiturn.py tests/test_alfworld_protocol_v2.py \
  tests/test_alfworld_runtime.py tests/test_alfworld_ppo_normalizer.py \
  tests/test_models.py --disable-warnings --tb=short
```
