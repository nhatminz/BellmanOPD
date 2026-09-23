# Thực nghiệm Pass@K (AIME24, AIME25, AMC23)

Pipeline này tái sử dụng nguyên evaluator hiện có của project:

- prompt được render bởi `render_evaluation_prompt()`;
- response được chấm bởi `grade_evaluation_response()`;
- mỗi bài lưu nguyên văn toàn bộ `responses` và vector `correct`;
- mọi Pass@K dùng lại `metric_problem_score()` với estimator
  `1 - C(n-c,k) / C(n,k)`.

Pipeline **không generate riêng cho từng K**. AIME24 và AIME25 được sample đúng
một lần với `N=128`; AMC23 được sample đúng một lần với `N=64`. Từ cùng vector
correctness đó, code tính AIME K = 8, 16, 32, 64, 128 và AMC23 K = 4, 8, 16,
32, 64.

## 1. Khai báo checkpoint

Mở `scripts/run_passk_experiment.sh` và sửa mảng `CHECKPOINT_SPECS` ở đầu file.
Mỗi dòng có format:

```text
MODEL_GROUP|METHOD|LABEL|CHECKPOINT[|CONFIG]
```

Ví dụ đầy đủ cho Qwen3-1.7B và Qwen3-4B:

```bash
CHECKPOINT_SPECS=(
  "qwen3_1.7b|opd|OPD|/workspace/storage-shared/nlp/minhpn19/BellmanOPD/outputs/opd_17b_run/opd/final"
  "qwen3_1.7b|cmt|CMT|/workspace/storage-shared/nlp/minhpn19/BellmanOPD_analysis/outputs/cmt_17b_run/cmt_opd/final"
  "qwen3_4b|opd|OPD|/workspace/storage-shared/nlp/minhpn19/BellmanOPD/outputs/opd_4b_run/opd/final"
  "qwen3_4b|cmt|CMT|/workspace/storage-shared/nlp/minhpn19/BellmanOPD_analysis/outputs/cmt_4b_run/cmt_opd/final"
)
```

`MODEL_GROUP` quyết định checkpoint thuộc figure nào. `LABEL` là tên curve trong
legend. Có thể thêm tùy ý checkpoint/method; nếu label trùng nhau, plot tự thêm
tên checkpoint để phân biệt. Trường `CONFIG` cuối là optional. Khi bỏ qua, method
`opd`, `cmt`, `ta`, `grpo`, `iw`, ... tự chọn config tương ứng trong `configs/`.

## 2. Cấu hình sampling/GPU

Các biến quan trọng đều nằm ở đầu `scripts/run_passk_experiment.sh`:

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
STORAGE_ROOT="/workspace/storage-shared"
PASSK_OUTPUT_ROOT="${REPO_DIR}/results/passk"
PASSK_TAG="eopd_figure7_8_v1"

PASSK_BENCHMARKS="AIME24 AIME25 AMC23"
AIME_K_VALUES="8 16 32 64 128"
AMC_K_VALUES="4 8 16 32 64"
AIME_NUM_SAMPLES=128
AMC_NUM_SAMPLES=64

PASSK_TEMPERATURE=1.0
PASSK_TOP_P=0.8
PASSK_MAX_NEW_TOKENS=7168
PASSK_MAX_MODEL_LEN=9216
PASSK_TENSOR_PARALLEL_SIZE=1
PASSK_WORLD_SIZE=0
PASSK_GPU_MEMORY_UTILIZATION=auto
PASSK_SEED=1234
```

Với `PASSK_WORLD_SIZE=0` và `TP=1`, code dùng toàn bộ GPU trong
`CUDA_VISIBLE_DEVICES`, mỗi GPU là một vLLM replica độc lập và benchmark được
shard deterministic. Nếu muốn chỉ dùng 4 GPU, đặt mask thành `0,1,2,3` hoặc đặt
`PASSK_WORLD_SIZE=4`.

## 3. Chạy toàn bộ experiment

Sau khi sửa mảng trong file:

```bash
cd /workspace/storage-shared/nlp/minhpn19/BellmanOPD_analysis
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PASSK_WORLD_SIZE=8 \
PASSK_TAG=eopd_figure7_8_v1 \
  bash scripts/run_passk_experiment.sh
```

Hoặc không sửa file, truyền từng spec trực tiếp vào command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PASSK_WORLD_SIZE=4 \
PASSK_TAG=eopd_figure7_8_v1 \
bash scripts/run_passk_experiment.sh \
  "qwen3_1.7b|opd|OPD|/abs/path/opd_17b/opd/final" \
  "qwen3_1.7b|cmt|CMT|/abs/path/cmt_17b/cmt_opd/final" \
  "qwen3_4b|opd|OPD|/abs/path/opd_4b/opd/final" \
  "qwen3_4b|cmt|CMT|/abs/path/cmt_4b/cmt_opd/final"
```

Script mặc định plot ngay sau khi tổng hợp. Đặt `PASSK_PLOT_AFTER_RUN=false` nếu
chỉ muốn inference/tổng hợp.

## 4. Cache, resume và raw outputs

Output có cấu trúc:

```text
results/passk/
  runs/<tag>/<model_group>/<method-run-checkpoint-hash>/
    generations/aime_n128/
      aime24_predictions.jsonl.gz
      aime25_predictions.jsonl.gz
      model_outputs_detailed.jsonl.gz
      summary.json
      passk_generation_manifest.json
    generations/amc23_n64/
      amc23_predictions.jsonl.gz
      model_outputs_detailed.jsonl.gz
      summary.json
      passk_generation_manifest.json
    passk_results.json
  summaries/
    passk_summary_<tag>.json
    passk_summary_<tag>.csv
  figures/
    passk_qwen3_1.7b_<tag>.png
    passk_qwen3_1.7b_<tag>.pdf
    passk_qwen3_4b_<tag>.png
    passk_qwen3_4b_<tag>.pdf
```

Mỗi prediction row giữ cả `responses` và `correct`. Manifest cache fingerprint
checkpoint snapshot, benchmark files, prompt/evaluator protocol, N, seed,
temperature, top-p và cấu hình vLLM. Chạy lại cùng command sẽ báo `CACHE HIT` và
không inference lại. Cache thiếu response, thiếu correctness hoặc sai N sẽ bị
từ chối. Artifact cache cũ không hợp lệ được đổi tên `.invalid-<timestamp>` sau
khi generation mới thành công, không bị xóa âm thầm.

Nếu job dừng giữa chừng, generation directory đang viết nằm trong thư mục
`.tmp-*`; cache hoàn chỉnh trước đó vẫn nguyên vẹn. Chạy lại command để tiếp tục;
checkpoint/benchmark đã hoàn tất hợp lệ sẽ được reuse.

## 5. Chỉ plot lại, không inference

```bash
PASSK_PLOT_TAG=paper_v1 \
bash scripts/plot_passk_experiment.sh \
  results/passk/summaries/passk_summary_eopd_figure7_8_v1.json
```

Có thể gộp nhiều summary độc lập trong một lần plot:

```bash
PASSK_PLOT_TAG=combined_v1 \
bash scripts/plot_passk_experiment.sh \
  results/passk/summaries/passk_summary_run_a.json \
  results/passk/summaries/passk_summary_run_b.json
```

Nếu tên figure đã tồn tại, plotter tự thêm timestamp/suffix nên không overwrite.
Mỗi model group tạo một figure 1×3 subplot, lưu PNG 300 dpi và PDF.

## 6. Smoke test rẻ (không phải kết quả paper)

Để kiểm tra path, prompt, grader, distributed merge và cache trước khi chạy đủ:

```bash
CUDA_VISIBLE_DEVICES=0 PASSK_WORLD_SIZE=1 \
PASSK_TAG=passk_smoke \
AIME_NUM_SAMPLES=8 AMC_NUM_SAMPLES=4 \
AIME_K_VALUES="1 4 8" AMC_K_VALUES="1 2 4" \
PASSK_PLOT_AFTER_RUN=false \
bash scripts/run_passk_experiment.sh \
  "qwen3_1.7b|cmt|CMT-smoke|/abs/path/cmt_opd/final"
```

Không dùng kết quả smoke này trong figure protocol chính vì N/K đã thay đổi.
