# Eval MMLU-Pro: nhiều checkpoint, ba seed, hai GPU

Sửa trực tiếp danh sách `CHECKPOINTS=(...)` trong
[`eval_mmlu_pro_multiseed.sh`](eval_mmlu_pro_multiseed.sh). Mỗi dòng là một
thư mục checkpoint Hugging Face đầy đủ, ví dụ:

```bash
CHECKPOINTS=(
  "${STORAGE_ROOT}/nlp/minhpn19/BellmanOPD_analysis/outputs/cmt_20260922_115452_504016128/cmt_opd/checkpoint-001500"
  "${STORAGE_ROOT}/nlp/minhpn19/BellmanOPD_analysis/outputs/TEN_RUN/opd/checkpoint-001050"
  "${STORAGE_ROOT}/nlp/minhpn19/BellmanOPD_analysis/outputs/TEN_RUN/ta_opd/final"
)
```

Không cần file `eval_history` cũ hay model teacher. Không hỗ trợ checkpoint chỉ
chứa LoRA adapter. Checkpoint phải có tokenizer/chat template riêng đã lưu cùng
model, giống checkpoint của pipeline hiện tại.

## Lệnh chạy trên B200

Dùng virtualenv đã chạy được vLLM của repo. Không cần thêm dependency ngoài
`requirements.txt` hiện có.

```bash
export STORAGE_ROOT=/workspace/storage-shared
cd "$STORAGE_ROOT/nlp/minhpn19/BellmanOPD_analysis"

# Kiểm tra dataset, đường dẫn và kế hoạch; chưa load model/chưa dùng GPU.
bash eval_mmlu_pro_multiseed.sh --dry-run

# Eval toàn bộ test, seed 42/43/44, GPU 0 và 1.
bash eval_mmlu_pro_multiseed.sh
```

Dataset mặc định:

```text
/workspace/storage-shared/nlp/minhpn19/data/mmlu-pro/test-00000-of-00001.parquet
```

Mỗi checkpoint chạy lần lượt. Bên trong một checkpoint, GPU 0 nhận các câu có
index chẵn, GPU 1 nhận các câu có index lẻ. Mỗi GPU load một bản model TP=1 và
giữ model cho cả ba seed; mỗi câu sinh đúng **8 completion cho mỗi seed**.
Có tqdm theo checkpoint, dựng prompt và số câu hoàn tất trên từng GPU/seed.
Không chạy training, không load optimizer/teacher.

Mặc định launcher chọn đúng GPU `0,1`, kể cả shell trước đó export bốn GPU.
Để chọn cặp GPU khác: `MMLU_PRO_GPUS=2,3 bash eval_mmlu_pro_multiseed.sh`.

Chạy thử ít câu trước, tách tag với kết quả full-test:

```bash
EVAL_TAG="mmlu_smoke_$(date +%Y%m%d_%H%M%S_%N)" \
  bash eval_mmlu_pro_multiseed.sh --limit 16
```

## Giao thức và cách tính

Đây là **zero-shot CoT, sampled evaluation**. Prompt dùng category, question và
toàn bộ options theo đúng thứ tự parquet, yêu cầu kết thúc bằng
`The answer is (X).`. Không đưa `answer`, `answer_index` hay `cot_content` của
test vào prompt. Không tải validation, không thêm few-shot ngầm. Mặc định
`enable_thinking=false` như các run Qwen3 trong repo; model vẫn được yêu cầu
giải thích trước khi trả lời. Gemma dùng chat template native của checkpoint
và stop token `<end_of_turn>`.

Schema được kiểm tra trên
[parquet chính thức của TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro):
12.032 câu, 14 category, `question_id`, `question`, `options`, `answer`,
`answer_index`, `cot_content`, `category`, `src`. Loader cũng chấp nhận chỉ có
một trong hai cột nhãn `answer`/`answer_index`; nếu có cả hai phải khớp nhau.
Giữ nguyên lựa chọn `N/A`, không shuffle/renumber đáp án.

Khác với [evaluator tham khảo](https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_local.py)
dùng few-shot CoT và greedy decoding, kết quả này có giao thức riêng được ghi
trong `request.json`/summary. Không so trực tiếp với điểm leaderboard theo
giao thức khác. Answer parser lấy đáp án cuối có nhãn rõ ràng, hỗ trợ thêm
`Answer: B`, `\boxed{B}` hoặc chỉ chữ `B`. Không chọn ngẫu nhiên khi parse
thất bại; trường hợp đó được tính sai và ghi `invalid_answer_rate`.

Với một seed, gọi `N` là số câu và `c_i` là số completion đúng trong 8 mẫu của
câu `i`:

- `avg@8 = sum(c_i) / (8N)`.
- `pass@8 = sum(c_i > 0) / N`.
- Hai metric lấy từ **cùng tám completion**, không generate hai lần.
- Overall tính trên toàn bộ câu; summary cũng có từng category.
- `±` là **sample standard deviation giữa ba seed**, `ddof=1`. Đây không phải
  confidence interval hay độ lệch giữa các câu hỏi.

Sampling mặc định: temperature `0.7`, top-p `0.95`, tối đa `2048` token/completion,
context `8192`. Mỗi cặp `(seed, index câu hỏi)` dùng một sampling seed xác định
bằng SHA-256, độc lập với cách chia GPU/batch; số gốc `42/43/44` vẫn được ghi
trong từng prediction. Sai khác số học của GPU/vLLM vẫn có thể ảnh hưởng bitwise
reproducibility. Không dùng temperature 0 cho thí nghiệm ba seed sampled này.

Code không truncate/drop câu dài. Nếu prompt + response budget vượt context,
checkpoint đó báo lỗi rõ ràng; chỉnh context/budget rồi chạy tag mới. Với
Gemma2, context được giới hạn theo native `max_position_embeddings` trước khi
tạo engine.

## Kết quả và resume eval

Mỗi lần chạy mặc định tự tạo tag timestamp, ví dụ
`mmlu_pro_20261109_153045_123456789`.

Kết quả mỗi checkpoint nằm cạnh checkpoint, cùng thư mục method output:

```text
outputs/<run>/<method>/mmlu_pro_eval/<tag>/checkpoint-001500/
  request.json
  seed_42_summary.json
  seed_43_summary.json
  seed_44_summary.json
  summary.json
  summary.csv
  shards/
    rank_0/seed_42.jsonl.gz    # tương tự seed 43, 44 và rank_1
    rank_0/first_prompt.txt
```

Prediction gzip giữ question ID/options/reference label, tám response đầy đủ,
đáp án đã parse, đúng/sai, finish reason, số token và sampling seed. Reference
labels chỉ dùng sau generation để chấm điểm.

Summary gộp **tất cả checkpoint**:

```text
outputs/mmlu_pro_eval/<tag>/summary.json
outputs/mmlu_pro_eval/<tag>/summary.csv
```

CSV có mean/std ở thang `0..1` và cột phần trăm `mean ± std`, ví dụ
`avg_at_8_percent = 45.20 ± 0.31`. JSON giữ cả điểm của từng seed, từng category,
tỷ lệ parse lỗi và tỷ lệ completion hết token budget.

Nếu một checkpoint lỗi, chương trình ghi traceback, tiếp tục checkpoint kế
tiếp, lưu danh sách lỗi vào summary chung và chỉ trả exit code 1 sau khi đã
thử hết danh sách.

Để tiếp tục eval đã gián đoạn, dùng lại đúng tag đã in ở đầu log:

```bash
EVAL_TAG=mmlu_pro_20261109_153045_123456789 \
  bash eval_mmlu_pro_multiseed.sh
```

Các seed/shard đã ghi hoàn chỉnh được kiểm tra và dùng lại; shard đang dở sẽ
chạy lại. Phải giữ nguyên danh sách checkpoint, seed, dữ liệu, GPU và các tham
số. Nếu thay đổi, dùng tag mới. Mỗi request lưu SHA-256 dataset và fingerprint
metadata checkpoint để tránh trộn kết quả. Không sửa các file `eval_history`
hoặc summary của eval toán/ALFWorld cũ.

Giảm bộ nhớ KV/cache khi cần:

```bash
MAX_NUM_SEQS=32 EVAL_BATCH_SIZE=16 GPU_MEMORY_UTILIZATION=0.80 \
  bash eval_mmlu_pro_multiseed.sh
```

Giới hạn scheduling này không thay số mẫu 8/câu. Đổi `MAX_NEW_TOKENS` làm thay
đổi giao thức; luôn dùng cùng budget cho các checkpoint cần so sánh.

## Lỗi token-ID kiểu chuỗi trong vLLM

Nếu bản cũ dừng ở `max_input_id > ...` với
`TypeError: '>' not supported between instances of 'str' and 'int'`, nguyên
nhân đã được tái hiện là evaluator truyền cả `BatchEncoding`/dictionary từ
`apply_chat_template` vào `prompt_token_ids`, thay vì lấy `input_ids`.
[API tokenizer](https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.apply_chat_template)
có thể trả dictionary; mặc định return_dict khác nhau giữa các phiên bản.

Bản sửa yêu cầu rõ `return_dict=True`, lấy `input_ids`, chuẩn hóa thành
`list[int]` và kiểm tra trước khi đo context/load vLLM. Prompt, seed, số mẫu,
budget và cách chấm không đổi. Không cần sửa checkpoint hoặc cài lại môi trường.

Sau khi cập nhật `b200_experiment/mmlu_pro_evaluation.py`, chạy lại:

```bash
MMLU_PRO_GPUS=0,1,2,3 bash eval_mmlu_pro_multiseed.sh
```

Có thể dùng lại `EVAL_TAG` của lần lỗi để giữ các seed/shard đã hoàn tất;
phải giữ đúng các tham số và GPU của lần đó. Code mới không thay request
schema của các kết quả cũ.

## Kiểm thử local

```bash
python3 -m pytest -q tests/test_mmlu_pro_evaluation.py
```

Test full schema trên parquet chính thức đã tải về là tùy chọn:

```bash
MMLU_PRO_REAL_DATA="$STORAGE_ROOT/nlp/minhpn19/data/mmlu-pro/test-00000-of-00001.parquet" \
  python3 -m pytest -q tests/test_mmlu_pro_evaluation.py
```

Các test kiểm tra schema thật, prompt không lộ nhãn/CoT, parser, hai shard,
ba seed, dual metric, sample std, cache, không ghi đè khác request, context
overflow, tiếp tục sau lỗi và shell dry-run. vLLM trong unit test được mock;
không coi các số test synthetic là điểm model trên MMLU-Pro.
