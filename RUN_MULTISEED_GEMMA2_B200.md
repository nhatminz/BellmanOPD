# Eval checkpoint nhiều seed và train Gemma‑2 trên B200

Tất cả lệnh dưới đây chạy trên máy B200, trong checkout đã được cập nhật bằng code mới.
Không dùng đường dẫn của máy RTX 3090 để resolve model hoặc dataset.

```bash
cd /workspace/storage-shared/nlp/minhpn19/BellmanOPD_analysis
export STORAGE_ROOT=/workspace/storage-shared
# Activate môi trường B200 đang dùng cho repo trước khi chạy.
python3 -m pip install -r requirements.txt
```

**1. Eval lại các checkpoint đã chọn, thêm seed 43 và 44**

Script mới đọc `resolved_config.yaml` của chính run cũ, kể cả run nằm trong checkout
`BellmanOPD` khác. Không cần teacher hoặc base student của run cũ được tải vào GPU:
evaluator chỉ tải checkpoint đã chọn. Các file dataset trong resolved config phải tồn tại.

Chọn checkpoint bằng tên thư mục, số step, hoặc `final`. Thay `150 300` trong ví dụ
bằng những step thực sự có trên máy; không tự động eval mọi checkpoint.

Kiểm tra kế hoạch trước, hoàn toàn không ghi artifact và không chạy generation:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval_multiseed_b200.sh \
  --run-output /workspace/storage-shared/nlp/minhpn19/BellmanOPD_analysis/outputs/cmt_20260922_114610_477530783/cmt_opd \
  --checkpoints 150 300 \
  --seeds 43 44 \
  --tag extra43_44 \
  --dry-run
```

Chạy thật: bỏ `--dry-run`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval_multiseed_b200.sh \
  --run-output /workspace/storage-shared/nlp/minhpn19/BellmanOPD_analysis/outputs/cmt_20260922_114610_477530783/cmt_opd \
  --checkpoints 150 300 \
  --seeds 43 44 \
  --tag extra43_44
```

Run TA trong checkout cũ:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval_multiseed_b200.sh \
  --run-output /workspace/storage-shared/nlp/minhpn19/BellmanOPD/outputs/ta_20260920_022254_977947377/ta_opd \
  --checkpoints checkpoint-000150 final \
  --seeds 43 44 \
  --tag extra43_44
```

Cũng có thể truyền thư mục run cha, ví dụ `.../cmt_20260922_114610_477530783`,
nếu nó chỉ chứa đúng một thư mục method có `resolved_config.yaml`.
Không chọn đồng thời `final` và `checkpoint-N` cùng đại diện một optimizer step.

Mỗi seed chạy đúng sáu dataset: Competition-MATH, MATH-500, AIME24, AIME25,
GPQA-Diamond, AMC23. Mỗi problem sinh 8 response **một lần**, tính cả:

- `avg@8`: tỷ lệ đúng trung bình trong 8 response.
- `pass@8`: problem có ít nhất một response đúng.

Sampling `temperature`, `top_p`, `max_new_tokens` được đọc từ evaluation cũ của
checkpoint. Không lấy training seed làm evaluation seed. Ví dụ run train với seed
42 có thể đã eval với seed **1234**. Dry-run hiển thị seed baseline thực sự.

Nếu artifact cũ không ghi seed, script yêu cầu khai báo rõ seed eval cũ:

```bash
# Chỉ thêm tùy chọn này nếu đã xác minh seed eval cũ đúng là 1234.
# ... các đối số trên ... --baseline-seed 1234
```

Ở chế độ mặc định, script dừng trước generation nếu thiếu baseline của một trong sáu dataset, sampling
protocol avg/pass cũ khác nhau, seed bị lặp, hoặc baseline không có 8 response/problem.
Cần hoàn thiện/kiểm tra kết quả cũ trước khi tổng hợp ba seed; không tự bịa baseline
hoặc trộn các protocol khác nhau.

Muốn chạy lại toàn bộ seed 42/43/44, không ghép thêm kết quả cũ, dùng
`--seeds 42 43 44 --fresh-seeds`. Sampling protocol lấy từ `training_evaluation`
trong resolved config của run; mỗi checkpoint/seed sinh 8 response/problem để
tính cả avg@8 và pass@8 trên đủ sáu dataset. Summary chỉ có đúng ba seed này,
sample std ddof=1; lịch sử cũ được giữ nguyên.

Bốn run Gemma được chọn, step 200/1050/1500, GPU 0/1:

```bash
cd /workspace/storage-shared/nlp/minhpn19/BellmanOPD_analysis
bash eval_gemma_selected_multiseed.sh
```

Các run chạy lần lượt, mỗi run dùng cả hai GPU; lỗi một run sẽ chuyển sang run
tiếp theo và báo danh sách lỗi cuối script. Cần có đủ `checkpoint-000200`,
`checkpoint-001050`, `checkpoint-001500` trong từng method directory. Kiểm tra
trước, không chạy GPU:

```bash
bash eval_gemma_selected_multiseed.sh --dry-run
```

File summary nằm ngay trong mỗi method directory:
`eval_summary_gemma_s42_43_44_steps200_1050_1500_ctx8192.json`, `.csv`, `.md`.
Mỗi file gồm cả ba step, sáu dataset, avg@8/pass@8, giá trị từng seed và mean ± std.
Chạy lại cùng command/tag sau gián đoạn để tiếp tục các seed chưa hoàn thành;
đổi `EVAL_TAG=...` nếu muốn sinh lại hoàn toàn một lần mới.

Nếu checkpoint đầu thực tế là step **300** như log eval mới, dùng:

```bash
EVAL_CHECKPOINTS="300 1050 1500" bash eval_gemma_selected_multiseed.sh
```

Launcher chỉ dùng GPU `0,1`, `--world-size 2`, `--max-model-len 8192`.
Default tag tự phản ánh đúng các checkpoint đã chọn và context 8192. Với file
launcher riêng như `eval_gemma.sh`, thêm `--max-model-len 8192` vào lệnh gọi
`eval_multiseed_b200.sh`; biến `VLLM_MAX_MODEL_LEN` của training launcher không
được script multiseed sử dụng.

Run Gemma cũ có thể lưu `training_evaluation.vllm.max_model_len=9216` do kế thừa
config Qwen. Evaluator nay kiểm tra `config.json` của checkpoint Gemma2 và giới
hạn engine theo `max_position_embeddings` (8192 với Gemma-2). Cả dry-run, request
lưu để resume, và vLLM workers đều dùng giới hạn đã resolve; không sửa config
cũ, checkpoint hay history. Giữ `max_new_tokens` (ví dụ 7168), temperature, top-p,
8 response/problem và stop tokens như protocol đã chọn. Context 8192 là tổng
prompt + response; generation vẫn phải dừng khi đạt giới hạn context của engine.
Không cần đặt `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`.

File mới nằm trực tiếp trong thư mục method, cạnh history và checkpoint cũ:

```text
cmt_opd/
  eval_history.jsonl                              # cũ, không sửa
  eval_history_pass_at_8.jsonl                    # cũ, không sửa
  eval_history_seed_43_extra43_44.jsonl            # avg@8 mới
  eval_history_pass_at_8_seed_43_extra43_44.jsonl   # pass@8 mới
  eval_history_seed_44_extra43_44.jsonl
  eval_history_pass_at_8_seed_44_extra43_44.jsonl
  eval_summary_extra43_44.json
  eval_summary_extra43_44.csv
  eval_summary_extra43_44.md
  eval_multiseed/extra43_44/checkpoint-000150/seed_43/
    summary.json
    request.json
    resolved_eval_config.yaml
    ... prediction/detail artifacts ...
```

Summary lưu điểm từng seed, tên checkpoint, step, dataset, metric, `n`, mean và
**sample standard deviation (`ddof=1`)**. Ví dụ `37.50 ± 12.50` là phần trăm ± điểm
phần trăm. Đó là biến động do **sampling seed khi eval cùng một checkpoint**, không
phải độ lệch chuẩn giữa ba lần train.

Nếu mất kết nối hoặc bị ngắt: chạy lại đúng command với cùng tag, seed, số GPU và
engine settings. Script tái sử dụng seed đã commit đầy đủ, chỉ chạy seed chưa xong.
Hai file metric cùng dùng một artifact generation. Đổi protocol hoặc checkpoint
đã bị ghi lại thì dùng tag mới; script không âm thầm thay kết quả đã commit.
Summary của tag tổng hợp các checkpoint được chọn trong lần gọi hiện tại.

Mặc định mỗi GPU visible chạy một evaluator TP=1, chia problem giữa các GPU.
Có thể giới hạn số worker bằng `--world-size 2`. Nếu thiếu VRAM, thêm
`--max-num-seqs 32` và giữ các thông số đó khi resume. Có thể chọn
`--gpu-memory-utilization 0.85`; mặc định dùng cơ chế `auto` của repo.

**2. Chuẩn bị đúng hai model Gemma‑2 chính thức**

Tài khoản Hugging Face cần có quyền tải model Google Gemma. Nếu cần, login trong
môi trường B200 trước khi download. Preset dùng đường dẫn dưới `STORAGE_ROOT`:

```bash
hf auth login
hf download google/gemma-2-9b-it \
  --local-dir /workspace/storage-shared/models/gemma-2-9b-it
hf download google/gemma-2-2b-it \
  --local-dir /workspace/storage-shared/models/gemma-2-2b-it
```

Kiểm tra assets, tokenizer và sáu dataset trên B200:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m b200_experiment.cli preflight \
  --config configs/qwen3_b200_cmt.yaml \
  --overlay configs/gemma2_b200.yaml \
  --output results/preflight_gemma2.json
```

Preset giữ strict tokenizer compatibility, Student Top‑16 OPD core, TA selector,
IW sampled-action objective, và CMT correction/groupwise bounded Gibbs allocation
của repo. `configs/gemma2_b200.yaml` chỉ bổ sung lựa chọn model/runtime.

Gemma‑2 có context 8192. Launcher mới đặt `max_model_len=8192` cho rollout/eval,
train prompt tối đa 1024, response tối đa 4096; eval response tối đa 4096.
Không dùng mặc định Qwen `max_model_len=9216`. Prompt eval không bị cắt âm thầm;
prompt cộng response budget phải vừa context của model.

Generation dừng tại cả `<eos>` và `<end_of_turn>` (`1`, `107`). HF scoring/training
giữ attention softcapping bằng FlashAttention‑2 nếu đã cài, nếu không thì eager.
Transformers 4.57 SDPA bỏ qua attention softcap nên code từ chối backend này với
Gemma‑2. Eager dùng nhiều memory hơn; vì vậy micro-batch mặc định là 1/GPU.

Nguồn model/runtime:
[Gemma 9B](https://huggingface.co/google/gemma-2-9b-it),
[Gemma 2B](https://huggingface.co/google/gemma-2-2b-it),
[Transformers Gemma2](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/gemma2).

**3. Train bốn phương pháp với cùng workload**

Chạy lần lượt OPD → TA → IW → CMT, tự gắn OPD đã đo làm baseline cho overhead:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
RUN_NAME=gemma2_compare_seed42 \
  bash scripts/train_gemma2_b200.sh all
```

Các thư mục kết quả:

```text
outputs/gemma2_compare_seed42/
  opd/
  ta_opd/
  iw/
  cmt_opd/
```

Mỗi thư mục chứa checkpoint, `eval_history*.jsonl`, TensorBoard và cost logs của
chính phương pháp đó. Launcher dùng cùng global prompt batch 64, `n=4`, PPO
mini-batch 64, micro-batch 1/GPU, score micro-batch 1/GPU, LR `5e-6`, một epoch,
train sampling `temperature=1`, `top_p=1`. Với rollout đủ 64 prompt, có 256
trajectory và 4 optimizer update; CMT vẫn chia group theo response index.
Eval/save mỗi 150 step; eval cuối run vẫn được thực hiện. Tất cả có thể override
bằng các environment variables đang có trong repo.

Thử ngắn trên B200 trước khi train toàn bộ:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
RUN_NAME=gemma2_smoke \
MAX_STEPS=8 TRAIN_EVAL_ENABLED=false \
  bash scripts/train_gemma2_b200.sh all
```

Train riêng từng phương pháp, vẫn gom vào cùng run cha:

```bash
RUN_NAME=gemma2_compare_seed42 bash scripts/train_gemma2_b200.sh opd

RUN_NAME=gemma2_compare_seed42 \
COST_OPD_BASELINE="$PWD/outputs/gemma2_compare_seed42/opd" \
  bash scripts/train_gemma2_b200.sh ta

RUN_NAME=gemma2_compare_seed42 \
COST_OPD_BASELINE="$PWD/outputs/gemma2_compare_seed42/opd" \
  bash scripts/train_gemma2_b200.sh iw

RUN_NAME=gemma2_compare_seed42 \
COST_OPD_BASELINE="$PWD/outputs/gemma2_compare_seed42/opd" \
  bash scripts/train_gemma2_b200.sh cmt
```

Nếu đã chạy phương pháp trong thư mục đó thì dùng run name mới hoặc resume;
không chạy fresh vào output đang có. Khi chạy `all`, dùng `OUTPUT_ROOT` để đổi
thư mục cha; `OUTPUT_DIR` và resume chỉ dành cho lệnh một method.

Ví dụ resume CMT, `MAX_STEPS` là tổng step đích:

```bash
RUN_NAME=gemma2_compare_seed42 \
OUTPUT_DIR="$PWD/outputs/gemma2_compare_seed42/cmt_opd" \
RESUME_FROM_CHECKPOINT="$PWD/outputs/gemma2_compare_seed42/cmt_opd/checkpoint-000150" \
COST_OPD_BASELINE="$PWD/outputs/gemma2_compare_seed42/opd" \
  bash scripts/train_gemma2_b200.sh cmt
```

Nếu đổi model/data/batch/GPU khi resume thì các kiểm tra hiện có vẫn áp dụng.
Không dùng `all` để resume đồng thời bốn phương pháp.

**4. Đọc training cost trong thư mục run**

```text
<method-output>/
  training_cost.jsonl
  training_cost.csv
  training_cost_summary.json
  tensorboard/
  metrics.jsonl
  train_metrics.csv
  eval_history.jsonl
  eval_history_pass_at_8.jsonl
  checkpoint-*/ hoặc final/
```

`training_cost.jsonl/csv` ghi từng optimizer step; log JSONL là audit append-only.
Summary lấy bản ghi mới nhất của mỗi logical step, bỏ những step vượt quá mốc
resume khi bắt đầu phiên resume, và loại rollout warmup mặc định đầu tiên.

Các giá trị chính:

- `cost_train_seconds_per_optimizer_step`: thời gian cả rollout batch, trừ
  checkpoint write và eval snapshot export, chia số optimizer update thực sự.
  Bao gồm rollout/weight sync, scoring, selector, optimization và training diagnostics;
  không bao gồm periodic evaluation. Các update cùng rollout chia sẻ giá trị
  amortized này, không phải bốn phép đo rollout độc lập.
- `cost_optimizer_update_seconds`: thời gian riêng từng PPO optimizer update,
  gồm allocation và phần chuẩn bị liên quan, không gồm callback save/eval.
- Scoring, selector, allocation, checkpoint, eval snapshot, periodic eval được
  log riêng; thời gian phân tán lấy giá trị chậm nhất giữa các rank.
- `cost_peak_torch_allocated_gib` / `reserved_gib`: peak PyTorch lớn nhất giữa
  các rank trong rollout, bao gồm snapshot/checkpoint export.
- `cost_peak_device_used_gib_sampled`: peak sử dụng **toàn GPU** lấy mẫu mỗi
  0.25 giây, bao gồm vLLM ở process khác. Có thể bỏ lỡ spike ngắn; nếu GPU có
  process khác thì nó cũng nằm trong con số này.
- `cost_selector_fraction_percent`: tỷ lệ thời gian selector + allocation trong
  train step; đây không phải overhead so với OPD.
- `cost_overhead_vs_opd_percent`: `100 * (method_step_time / OPD_mean_step_time - 1)`.
  Online so sánh từng step với mean OPD; summary so sánh hai mean sau warmup.
  Nếu chưa có OPD baseline phù hợp, field overhead để trống, không suy đoán.

Config signature kiểm tra model, data, rollout, số GPU và batch/runtime trước khi
so sánh. Overhead là phép đo end-to-end trên các run thực tế; độ dài response thay
đổi giữa phương pháp cũng ảnh hưởng con số, không phải benchmark selector trên
cùng một tensor cố định. Xem thêm response length và token throughput trong `metrics.jsonl`.

TensorBoard có các tag `cost/*`, ví dụ:

```text
cost/train_seconds_per_optimizer_step
cost/optimizer_update_seconds
cost/scoring_seconds_per_step
cost/selector_seconds_per_step
cost/peak_torch_allocated_gib
cost/peak_device_used_gib_sampled
cost/overhead_vs_opd_percent
cost/is_warmup
```

```bash
tensorboard --logdir outputs/gemma2_compare_seed42 --port 6006
```

Nếu các run được train độc lập, chưa khai báo baseline online, so sánh sau:

```bash
python3 -m b200_experiment.training_cost \
  --baseline outputs/gemma2_compare_seed42/opd \
  --outputs outputs/gemma2_compare_seed42/ta_opd \
            outputs/gemma2_compare_seed42/iw \
            outputs/gemma2_compare_seed42/cmt_opd
```

Lệnh này ghi `training_cost_comparison.json` vào **từng thư mục method**. Thêm
`--tensorboard` để bổ sung overhead từng step vào TensorBoard sau khi OPD đã đo xong.
Muốn overhead xuất hiện ngay lúc train thì chạy OPD trước và truyền `COST_OPD_BASELINE`.
Các run cũ chưa có cost logs không thể khôi phục chính xác time/memory bằng script này.

Các launcher Qwen hiện có cũng nhận cost logging mới. Có thể điều chỉnh bằng
`COST_WARMUP_ROLLOUTS`, `COST_MEMORY_SAMPLE_INTERVAL_SEC`, hoặc tắt bằng
`COST_LOGGING_ENABLED=false`. Trong checkout này tên method nghiên cứu vẫn là
`cmt`, thư mục `cmt_opd`; không tạo method `prow` mới.
