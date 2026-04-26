# Requirement Tổng Hợp Từ Đề Bài (4 Ảnh)

## 1. Project Requirements

Mục tiêu của lab là nghiên cứu và áp dụng kỹ thuật fine-tuning cho bài toán phân loại ý định (intent classification) trong lĩnh vực ngân hàng, sử dụng dataset `BANKING77` và `Unsloth`.

Sinh viên cần hoàn thành các nhiệm vụ:

- Lấy mẫu và xây dựng một tập con phù hợp từ `BANKING77` cho bài toán phân loại intent.
- Tiền xử lý dữ liệu và chia tập dữ liệu đã lấy mẫu thành train/test.
- Fine-tune mô hình phân loại văn bản bằng Unsloth, mô tả rõ toàn bộ hyperparameters, kỹ thuật và cấu hình đã dùng.
- Đánh giá, so sánh hiệu năng của mô hình fine-tuned trên tập test độc lập.
- Triển khai file inference độc lập: load checkpoint đã lưu và dự đoán nhãn intent cho input.

## 2. Task Requirements

### 2.1 Data Preparation and Processing

- Bắt buộc dùng dataset: `BANKING77` ([https://huggingface.co/datasets/PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)).
- Chỉ dùng **một phần (subset)** của dataset để bảo đảm training hoàn thành với tài nguyên tính toán sẵn có.
- Thực hiện các bước tiền xử lý cần thiết, gồm:
  - Chuẩn hóa văn bản (text normalization),
  - Ánh xạ nhãn (label mapping),
  - Làm sạch dữ liệu cơ bản nếu cần.
- Chuyển nhãn intent đã chọn sang định dạng phù hợp cho sequence classification.
- Chia dữ liệu thành train/test; có thể tách thêm validation từ train để chọn model.

### 2.2 Fine-Tuning with Unsloth

- Bắt buộc tham chiếu tài liệu fine-tuning chính thức của Unsloth.
- Có thể chạy trên Google Colab, Kaggle hoặc máy local.
- Phải ghi rõ toàn bộ hyperparameters đã dùng, tối thiểu gồm:
  - Batch size,
  - Learning rate,
  - Optimizer,
  - Số bước train hoặc số epoch,
  - Maximum sequence length,
  - Các kỹ thuật regularization/augmentation (nếu có).
- Lưu model checkpoint sau khi fine-tuning.

### 2.3 Inference Implementation

Sau khi training xong, phải có file inference độc lập.

Giao diện inference bắt buộc có đúng hai method chính:

```python
class IntentClassification:
    def __init__(self, model_path):
        pass

    def __call__(self, message):
        ...
        return predicted_label
```

Yêu cầu thêm:

- `model_path` phải trỏ đến file cấu hình có chứa ít nhất đường dẫn checkpoint đã lưu.
- File inference phải load được checkpoint đã train và dự đoán nhãn intent cho **một input đơn**.
- Cần có ví dụ sử dụng ngắn thể hiện cách gọi class inference sau training.

### 2.4 Source Code

Toàn bộ source cần đẩy lên GitHub và tổ chức tối thiểu theo format tương tự:

```text
banking-intent-unsloth/
|-- scripts
|   |-- train.py
|   |-- inference.py
|   |-- preprocess_data.py
|
|-- configs
|   |-- train.yaml
|   |-- inference.yaml
|
|-- sample_data
|   |-- train.csv
|   |-- test.csv
|
|-- train.sh
|-- inference.sh
|-- requirements.txt
|-- README.md
```

`README.md` phải có đầy đủ mô tả để setup môi trường, tải/chuẩn bị dữ liệu, train và chạy model.

### 2.5 Video Demonstration

- Nộp **một video ngắn** trình bày kết quả inference của model đã train.
- Video cần thể hiện rõ:
  - Cách chạy script inference,
  - Ít nhất một input message mẫu,
  - Nhãn intent dự đoán từ model,
  - Accuracy cuối cùng trên test set.
- Video không cần chỉnh sửa phức tạp; quay màn hình đơn giản là đủ.
- Thời lượng khuyến nghị: **2-5 phút**.
- Upload video lên Google Drive và gắn link vào `README.md` (bảo đảm link public).

## 3. Danh Sách Deliverables Bắt Buộc

- Mã nguồn theo đúng cấu trúc yêu cầu.
- File config train/inference và script tương ứng.
- Dữ liệu mẫu train/test đã xử lý.
- Mô tả đầy đủ trong README.
- Video demo public link.
- Kết quả đánh giá (ít nhất accuracy trên test set).
