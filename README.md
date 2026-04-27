# Banking Intent Classification with Unsloth

Dự án fine-tune mô hình **Llama-3.2 (1B/3B)** bằng **Unsloth + QLoRA 4-bit** trên bộ dữ liệu **BANKING77** để phân loại ý định khách hàng ngân hàng.

Phiên bản hiện tại tập trung vào:
- prompt có **danh sách đầy đủ 77 labels**
- hậu xử lý dự đoán bằng **exact match + fuzzy match**
- pipeline **Kaggle train + package artifacts**

---

## 📁 Cấu trúc dự án

```text
banking-intent-unsloth/
├── scripts/
│   ├── preprocess_data.py
│   ├── train.py
│   ├── inference.py
│   ├── package_artifacts.py
│   ├── kaggle_train_and_package.sh
│   └── kaggle_push_artifacts.sh
├── configs/
│   ├── train.yaml
│   ├── train_1b_optimized.yaml
│   ├── train_3b_t4x2.yaml
│   ├── train_3b_t4x2_stable.yaml
│   └── inference.yaml
├── notebooks/
│   └── kaggle_pipeline.ipynb
├── train.sh
├── inference.sh
├── requirements.txt
└── README.md
```

Thư mục được tạo sau khi chạy:
- `sample_data/`: `train.csv`, `test.csv`, `labels.txt`
- `outputs/`: adapter/checkpoint + `eval_results.txt`
- `artifacts/`: gói kết quả theo từng run

---

## 🛠️ Cài đặt

### Yêu cầu
- Python 3.10+
- GPU CUDA (khuyến nghị T4 16GB hoặc cao hơn cho train)

### Cài thư viện
```bash
pip install -r requirements.txt
```

---

## 📊 Dữ liệu và tiền xử lý

- Dataset: [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)
- Script: `scripts/preprocess_data.py`

Script preprocess sẽ:
1. tải BANKING77 từ parquet export trên Hugging Face
2. normalize text (`strip().lower()`)
3. lấy mẫu cân bằng theo lớp
4. tạo dữ liệu Alpaca-style: `instruction`, `input`, `output`
5. sinh `sample_data/labels.txt` (danh sách 77 intent hợp lệ)

Biến môi trường chính:
- `TRAIN_PER_CLASS` (mặc định `50`)
- `TEST_PER_CLASS` (mặc định `20`)
- `SAMPLE_SEED` (mặc định `42`)

Ví dụ:
```bash
export TRAIN_PER_CLASS=110
export TEST_PER_CLASS=20
python scripts/preprocess_data.py
```

---

## 🧬 Fine-tuning

Script train chính: `scripts/train.py`  
Mặc định đọc config từ `TRAIN_CONFIG`, fallback về `configs/train.yaml`.

### Các config có sẵn
- `configs/train.yaml`: cấu hình mặc định 1B
- `configs/train_1b_optimized.yaml`: cấu hình 1B tối ưu (mục tiêu accuracy cao)
- `configs/train_3b_t4x2.yaml`: cấu hình 3B cho Kaggle T4x2
- `configs/train_3b_t4x2_stable.yaml`: cấu hình 3B ổn định hơn khi gặp OOM/instability

### Hyperparameters nổi bật (bản 3B hiện dùng nhiều)
- base model: `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`
- `max_seq_length=768`
- LoRA: `r=64`, `alpha=128`, `dropout=0.05`
- `per_device_train_batch_size=2`
- `gradient_accumulation_steps=16`
- `num_train_epochs=8`
- `learning_rate=8e-5`
- scheduler `cosine`, `warmup_steps=100`

### Chạy local nhanh
```bash
bash train.sh
```

### Chạy theo config cụ thể
```bash
export TRAIN_CONFIG=configs/train_3b_t4x2.yaml
export TRAIN_PER_CLASS=999
export TEST_PER_CLASS=999
python scripts/preprocess_data.py
python scripts/train.py
```

---

## ☁️ Workflow Kaggle (khuyến nghị)

### Cách 1: Notebook
1. Upload `notebooks/kaggle_pipeline.ipynb` lên Kaggle
2. Bật GPU + Internet
3. Run All để preprocess -> train -> eval -> package

### Cách 2: Script
```bash
bash scripts/kaggle_train_and_package.sh
```

Lưu ý: script Kaggle hiện tại gồm 3 bước:
1. preprocess
2. train
3. package artifacts

Sau khi chạy, kiểm tra:
- `artifacts/LATEST.txt`
- `artifacts/run_.../manifest.json`
- `artifacts/run_.../eval_results.txt` (nếu có)

Nếu muốn đẩy artifacts từ Kaggle về GitHub:
```bash
export GITHUB_TOKEN=<your_pat>
export GIT_USER_NAME=<your_name>
export GIT_USER_EMAIL=<your_email>
export TARGET_BRANCH=main
bash scripts/kaggle_push_artifacts.sh
```

---

## 🖥️ Inference

Script: `scripts/inference.py`

Interface bắt buộc theo đề:
```python
class IntentClassification:
    def __init__(self, model_path):
        ...
    def __call__(self, message):
        ...
        return predicted_label
```

Điểm chính của bản hiện tại:
- Đọc `labels_path` từ `configs/inference.yaml` (mặc định `./sample_data/labels.txt`)
- Prompt inference chèn danh sách nhãn hợp lệ
- Hậu xử lý output bằng exact match + fuzzy match (`difflib.get_close_matches`)

### Python API
```python
from scripts.inference import IntentClassification

classifier = IntentClassification("configs/inference.yaml")
label = classifier("I am still waiting on my card?")
print(label)
```

### CLI
```bash
bash inference.sh
# hoặc
python scripts/inference.py configs/inference.yaml
```

---

## 📈 Kết quả

`scripts/train.py` sẽ:
- evaluate trên `sample_data/test.csv`
- in accuracy + classification report
- lưu vào `outputs/eval_results.txt`

Kết quả phụ thuộc:
- config train đang dùng
- kích thước subset (`TRAIN_PER_CLASS`, `TEST_PER_CLASS`)
- tài nguyên GPU/runtime

---

## 🎬 Video demo (theo requirement)

[🔗 Demo Video (Google Drive)](YOUR_GOOGLE_DRIVE_LINK_HERE)

Video nên thể hiện:
1. cách chạy inference
2. ít nhất một input mẫu
3. nhãn dự đoán
4. accuracy cuối trên test set
