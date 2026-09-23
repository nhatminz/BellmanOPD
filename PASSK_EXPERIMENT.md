# Thực nghiệm Pass@K (AIME24, AIME25, AMC23)

Pipeline tái sử dụng evaluator/grader hiện có. Mỗi problem lưu toàn bộ
`responses` và vector `correct`; các Pass@K được tính bởi `metric_problem_score()`
với estimator `1 - C(n-c,k) / C(n,k)`.

- AIME24/AIME25 generate một lần với N=128, sau đó tính K=8,16,32,64,128.
- AMC23 generate một lần với N=64, sau đó tính K=4,8,16,32,64.

## 1. Quy ước một checkpoint = một run trong `outputs/`

Mỗi lần chạy một checkpoint sẽ tạo một thư mục độc lập trong:

```text
BellmanOPD_analysis/outputs/<PASSK_RUN_NAME>/
```

Tên mặc định được tạo tự động theo:

```text
<method>_checkpoint_<step>_<model_group>_<source_training_run>
```

Ví dụ:

```text
cmt_checkpoint_000600_qwen3_4b_cmt_20260918_112926
opd_checkpoint_000600_qwen3_4b_opd_20260918_104511
```

Nếu input là `final`, script đọc `latest.json` cạnh checkpoint để lấy step thật.
Nếu file đó không tồn tại, tên dùng hậu tố `final`. Có thể override bằng
`PASSK_RUN_NAME=...`, nhưng chỉ khi command chứa đúng một checkpoint.

Mỗi run chứa cả raw generation, cache manifest và summary:

```text
outputs/<PASSK_RUN_NAME>/
  runs/<PASSK_RUN_NAME>/<model_group>/<checkpoint-identity>/
    generations/aime_n128/
      aime24_predictions.jsonl.gz
      aime25_predictions.jsonl.gz
      model_outputs_detailed.jsonl.gz
      passk_generation_manifest.json
    generations/amc23_n64/
      amc23_predictions.jsonl.gz
      model_outputs_detailed.jsonl.gz
      passk_generation_manifest.json
  summaries/
    passk_summary_<PASSK_RUN_NAME>.json
    passk_summary_<PASSK_RUN_NAME>.csv
```

Chạy lại đúng command sẽ reuse raw data hợp lệ (`CACHE HIT`) và không inference
lại.

## 2. Chạy một checkpoint CMT

Format checkpoint spec:

```text
MODEL_GROUP|METHOD|LABEL|CHECKPOINT[|CONFIG]
```

Ví dụ checkpoint 600 của CMT/Qwen3-4B:

```bash
cd /workspace/storage-shared/nlp/minhpn19/BellmanOPD_analysis

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PASSK_WORLD_SIZE=4 \
bash scripts/run_passk_experiment.sh \
  "qwen3_4b|cmt|CMT|/workspace/storage-shared/nlp/minhpn19/BellmanOPD_analysis/outputs/cmt_20260918_112926/cmt_opd/checkpoint-000600"
```

Cuối command, terminal in rõ:

```text
Pass@K run name: cmt_checkpoint_000600_qwen3_4b_cmt_20260918_112926
Output: .../BellmanOPD_analysis/outputs/cmt_checkpoint_000600_qwen3_4b_cmt_20260918_112926
```

## 3. Chạy checkpoint OPD độc lập

Checkpoint có thể nằm trong repo `BellmanOPD` cũ; kết quả Pass@K vẫn được lưu
trong `BellmanOPD_analysis/outputs`:

```bash
cd /workspace/storage-shared/nlp/minhpn19/BellmanOPD_analysis

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PASSK_WORLD_SIZE=4 \
bash scripts/run_passk_experiment.sh \
  "qwen3_4b|opd|OPD|/workspace/storage-shared/nlp/minhpn19/BellmanOPD/outputs/opd_20260918_104511/opd/checkpoint-000600"
```

Run tự động có dạng:

```text
opd_checkpoint_000600_qwen3_4b_opd_20260918_104511
```

## 4. Chạy bằng cách sửa block trong file

Có thể sửa `CHECKPOINT_SPECS` ở đầu `scripts/run_passk_experiment.sh`. Mỗi lần chỉ
để lại một entry nếu muốn quản lý từng run riêng:

```bash
CHECKPOINT_SPECS=(
  "qwen3_4b|cmt|CMT|/abs/path/cmt_opd/checkpoint-000600"
)
```

Sau đó chạy:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PASSK_WORLD_SIZE=4 \
  bash scripts/run_passk_experiment.sh
```

Script vẫn chấp nhận nhiều specs trong một command, nhưng mỗi spec được chạy tuần
tự và tạo một thư mục `outputs/<run-name>` riêng.

## 5. Vẽ hình bằng tên hai run trong `outputs/`

Không cần truyền đường dẫn summary. Chỉ cần copy đúng hai tên thư mục trong
`outputs/`:

```bash
cd /workspace/storage-shared/nlp/minhpn19/BellmanOPD_analysis

bash scripts/plot_passk_experiment.sh \
  opd_checkpoint_000600_qwen3_4b_opd_20260918_104511 \
  cmt_checkpoint_000600_qwen3_4b_cmt_20260918_112926
```

Script tự tìm summary của hai run và tạo comparison directory:

```text
results/passk/
  opd_checkpoint_000600_..._vs_cmt_checkpoint_000600_.../
    passk_qwen3_4b_<comparison-name>.png
    passk_qwen3_4b_<comparison-name>.pdf
```

Mỗi model group tạo một figure gồm ba subplot AIME24, AIME25 và AMC23. Nếu figure
cùng tên đã tồn tại, timestamp/suffix được thêm tự động nên không overwrite.

Có thể so sánh nhiều hơn hai run:

```bash
bash scripts/plot_passk_experiment.sh \
  opd_checkpoint_000600_qwen3_4b_run_a \
  ta_checkpoint_000600_qwen3_4b_run_b \
  cmt_checkpoint_000600_qwen3_4b_run_c
```

Plotter vẫn hỗ trợ truyền trực tiếp một hoặc nhiều file summary JSON nếu cần.

## 6. Các biến cấu hình

Các defaults nằm đầu `scripts/run_passk_experiment.sh`:

```bash
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

`PASSK_WORLD_SIZE=0` nghĩa là dùng toàn bộ GPU trong `CUDA_VISIBLE_DEVICES` với
một vLLM TP=1 replica trên mỗi GPU. `PASSK_PLOT_AFTER_RUN=false` là mặc định;
comparison chỉ được vẽ khi gọi plot script với các run cần so sánh.

## 7. Smoke test rẻ

```bash
CUDA_VISIBLE_DEVICES=0 PASSK_WORLD_SIZE=1 \
AIME_NUM_SAMPLES=8 AMC_NUM_SAMPLES=4 \
AIME_K_VALUES="1 4 8" AMC_K_VALUES="1 2 4" \
PASSK_RUN_NAME=cmt_passk_smoke \
bash scripts/run_passk_experiment.sh \
  "qwen3_4b|cmt|CMT-smoke|/abs/path/cmt_opd/checkpoint-000600"
```

Không dùng smoke result cho figure chính vì N/K đã thay đổi.
