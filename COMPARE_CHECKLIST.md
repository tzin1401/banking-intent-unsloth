# Compare Checklist (Baseline Commit 7706247)

Mục đích: ghi lại các điểm cần đối chiếu giữa **bản hiện tại** và **commit mới** để hợp/merge nhanh, bám sát yêu cầu bài lab trong `requirement/`.

## 1) Baseline Snapshot (bản hiện tại)

- Commit baseline: `7706247` (`Make Kaggle pipeline auto-update repo before running.`)
- Cấu trúc chính đang có:
  - `scripts/`: `train.py`, `inference.py`, `preprocess_data.py`, `kaggle_train_and_package.sh`, `kaggle_push_artifacts.sh`, `package_artifacts.py`
  - `configs/`: `train.yaml`, `train_3b_t4x2.yaml`, `train_3b_t4x2_stable.yaml`, `inference.yaml`
  - File entrypoint: `train.sh`, `inference.sh`, `README.md`, `requirements.txt`
- Trạng thái thư mục dữ liệu/kết quả:
  - `sample_data/`: chưa thấy trong repo lúc check
  - `outputs/`: chưa thấy trong repo lúc check

## 2) Requirement Mapping (cần giữ để pass)

Theo đề bài, cần bảo đảm các nhóm sau:

1. **Data prep**
   - Dùng BANKING77
   - Có sampling subset (không train full không cần thiết)
   - Có preprocessing (normalize, label mapping)
   - Có train/test split
2. **Fine-tuning với Unsloth**
   - Có hyperparameters rõ ràng (batch size, lr, optimizer, epoch/steps, max seq len, regularization)
   - Có lưu checkpoint sau train
3. **Inference**
   - `IntentClassification` có đúng 2 method chính:
     - `__init__(self, model_path)`
     - `__call__(self, message)`
   - `model_path` trỏ tới file config có đường dẫn checkpoint
   - Dự đoán được 1 input đơn
4. **Source structure + README**
   - Các file/folder tối thiểu đúng format bài yêu cầu
   - `README.md` có hướng dẫn setup, data, train, infer
5. **Demo/video**
   - Có hướng dẫn inference + input mẫu + label dự đoán + final accuracy
   - Link Google Drive trong README (public)

## 3) Hyperparameters/Thông số quan trọng cần so sánh

Khi so với commit mới, ưu tiên diff các thông số này:

### A. Dữ liệu (`scripts/preprocess_data.py`)
- Nguồn dataset:
  - Hiện tại: parquet export của `PolyAI/banking77`
- Sampling:
  - `TRAIN_PER_CLASS` (default 50)
  - `TEST_PER_CLASS` (default 20)
  - `SAMPLE_SEED` (default 42)
- Text preprocessing:
  - lower + strip
- Output format:
  - cột `instruction`, `input`, `output` (Alpaca-style)
- Đường dẫn output:
  - `sample_data/train.csv`
  - `sample_data/test.csv`

### B. Train config 1B (`configs/train.yaml`)
- `model_name`: `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`
- `max_seq_length`: 512
- LoRA: `lora_r=16`, `lora_alpha=16`, `lora_dropout=0`
- Batching: `per_device_train_batch_size=8`, `gradient_accumulation_steps=4`
- `num_train_epochs`: 3
- `learning_rate`: `2.0e-4`
- `optimizer`: `adamw_8bit`
- `weight_decay`: 0.01
- Scheduler: `linear`, `warmup_steps=10`
- Save: `output_dir=./outputs`, `save_strategy=epoch`

### C. Train config 3B (`configs/train_3b_t4x2.yaml`)
- `model_name`: `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`
- `max_seq_length`: 512
- LoRA: `r=64`, `alpha=128`, `dropout=0.05`
- Batching: `batch_size=2`, `grad_acc=16`
- `num_train_epochs=6`
- `learning_rate=8.0e-5`
- Scheduler: `cosine`, `warmup_steps=50`
- Precision: `fp16=true`, `bf16=false`
- Eval generation:
  - `eval_max_new_tokens=12`
  - `eval_temperature=0.0`
  - `eval_do_sample=false`

### D. Inference (`scripts/inference.py`, `configs/inference.yaml`)
- Interface class:
  - `IntentClassification.__init__(model_path)`
  - `IntentClassification.__call__(message) -> predicted_label`
- Prompt template có đồng bộ với train hay không
- Load model:
  - dùng `model_path` từ YAML
  - `max_seq_length`, `load_in_4bit`
- Generation args:
  - `max_new_tokens` (default 32)
  - deterministic (`do_sample=False`)
- Parsing output:
  - tách sau `### Response:`

### E. Evaluation/Artifact flow
- `scripts/train.py`:
  - Có tính accuracy + classification report
  - Có ghi `outputs/eval_results.txt`
- `scripts/package_artifacts.py`:
  - Copy `outputs`, `sample_data`, `configs` vào `artifacts/run_*`
  - Tạo `manifest.json`, `LATEST.txt`
- `scripts/kaggle_train_and_package.sh`:
  - install -> preprocess -> train -> package

## 4) Quick Checklist trước khi hợp với commit mới

- [ ] `scripts/inference.py` vẫn dùng đúng interface 2-method bắt buộc
- [ ] Các đường dẫn trong config không bị vỡ (`train_data_path`, `test_data_path`, `model_path`)
- [ ] Hyperparameters chính không bị đổi ngoài ý muốn (LR, epoch, batch, seq_len, LoRA)
- [ ] Prompt train và prompt infer vẫn đồng bộ
- [ ] Vẫn lưu `eval_results.txt` sau train
- [ ] README vẫn mô tả đủ các bước setup/train/infer + kết quả
- [ ] Link video trong README không còn placeholder
- [ ] Cấu trúc folder tối thiểu đúng theo requirement

## 5) Lệnh để đối chiếu nhanh với commit mới

Dùng khi bạn checkout lại commit mới:

```bash
git diff --name-status 7706247..HEAD
git diff 7706247..HEAD -- configs/ scripts/ README.md
```

Nếu cần xem thay đổi chỉ ở hyperparameters:

```bash
git diff 7706247..HEAD -- configs/*.yaml
```

## 6) Ghi chú rủi ro hiện tại (để dễ bỏ sót)

- README đang để placeholder video: `YOUR_GOOGLE_DRIVE_LINK_HERE`.
- Tại thời điểm check, chưa thấy `sample_data/` và `outputs/` trong working tree.
- Có nhiều config train (1B/3B/stable): khi merge dễ bị lệch config đang dùng thực tế, cần ghi rõ bản nào là bản nộp chính.
