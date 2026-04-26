# 📘 Hướng dẫn chi tiết: Train trên Kaggle + Giải thích Ollama

---

## PHẦN A: OLLAMA vs UNSLOTH TRỰC TIẾP — Khác nhau thế nào?

### Hình dung đơn giản

```
╔═══════════════════════════════════════════════════════════╗
║              CÁCH 1: UNSLOTH TRỰC TIẾP (bài lab dùng)    ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║   Python code ──→ Unsloth ──→ Model ──→ Kết quả          ║
║                                                           ║
║   classifier = IntentClassification("config.yaml")        ║
║   label = classifier("I lost my card")                    ║
║   # → "lost_or_stolen_card"                               ║
║                                                           ║
║   ✅ Code Python gọi THẲNG vào model                      ║
║   ✅ Không cần chạy server nào cả                          ║
║   ✅ Nhanh, gọn, ít phụ thuộc                              ║
╚═══════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════╗
║              CÁCH 2: OLLAMA (không dùng trong lab)        ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║   Ollama server (chạy nền) ←── HTTP API ──← Python code  ║
║       ↑                                                   ║
║   Model GGUF                                              ║
║                                                           ║
║   # Bước 1: Khởi động server                              ║
║   $ ollama serve                                          ║
║                                                           ║
║   # Bước 2: Gọi API                                       ║
║   response = requests.post("http://localhost:11434/api",  ║
║       json={"model": "my-model", "prompt": "..."})        ║
║                                                           ║
║   ⚠️ Phải chạy server riêng                                ║
║   ⚠️ Phải convert model sang GGUF                          ║
║   ⚠️ Giao tiếp qua HTTP (chậm hơn)                        ║
╚═══════════════════════════════════════════════════════════╝
```

### So sánh chi tiết

| | **Unsloth trực tiếp** | **Ollama** |
|---|---|---|
| **Cách hoạt động** | Python load model vào RAM/GPU trực tiếp | Chạy 1 server riêng, code gọi qua HTTP API |
| **Ví dụ thực tế** | Tự nấu ăn tại nhà | Gọi ship đồ ăn (phải qua app trung gian) |
| **Tốc độ** | ✅ Nhanh (gọi trực tiếp) | ❌ Chậm hơn (qua HTTP network) |
| **Cài đặt** | `pip install unsloth` | Cài Ollama + convert model sang GGUF |
| **Format model** | PyTorch / Safetensors (train ra sao dùng vậy) | Phải convert sang **GGUF** (thêm 1 bước) |
| **Prompt control** | ✅ Toàn quyền kiểm soát prompt template | ⚠️ Ollama tự thêm system prompt, khó kiểm soát |
| **Phù hợp cho** | Research, training, lab bài tập | Deploy chatbot cho nhiều người dùng |
| **Bài lab yêu cầu** | ✅ Đúng yêu cầu (class Python) | ❌ Không đúng interface đề bài |

### Tại sao Ollama KHÔNG phù hợp cho bài lab?

**Lý do 1 — Đề bài yêu cầu cụ thể:**
```python
# Đề bài BẮT BUỘC phải có class này:
class IntentClassification:
    def __init__(self, model_path):  ...
    def __call__(self, message):     return predicted_label
```
Ollama không cho phép tạo class Python kiểu này. Ollama là 1 server riêng, bạn phải gọi HTTP API.

**Lý do 2 — Prompt template:**
Khi train, ta dùng Alpaca format. Unsloth trực tiếp giữ nguyên prompt đó.
Ollama có thể **tự thêm/sửa** prompt → model bối rối → accuracy giảm.

**Lý do 3 — Thêm bước không cần thiết:**
```
Unsloth: Train → Save → Load → Predict         (3 bước)
Ollama:  Train → Save → Convert GGUF → Import Ollama → Start server → Call API → Predict (6 bước)
```

**Kết luận: Ollama hay, nhưng không phải cho bài lab này. Dùng Unsloth trực tiếp = đơn giản + đúng yêu cầu đề.**

---

## PHẦN B: HƯỚNG DẪN KAGGLE — TỪNG BƯỚC CÓ HÌNH

### Bước 1: Tạo tài khoản Kaggle

1. Vào [kaggle.com](https://www.kaggle.com/) → **Register** (dùng Google account)
2. Vào **Settings** → tìm phần **Phone Verification** → xác minh SĐT
   - ⚠️ **BẮT BUỘC** xác minh SĐT mới được dùng GPU miễn phí!

### Bước 2: Tạo Notebook mới

1. Nhấn **+ Create** → **New Notebook**
2. Đợi notebook load xong

### Bước 3: Bật GPU

1. Ở sidebar phải → tìm **Accelerator** (hoặc **Session Options**)
2. Chọn **GPU T4 x2** (hoặc **GPU T4 x1** nếu không có x2)
3. Nhấn **Save**
4. Đợi vài giây để Kaggle cấp GPU

> Nếu không thấy option GPU → bạn chưa xác minh SĐT ở Bước 1.

### Bước 4: Cài đặt Unsloth (Cell 1)

Paste vào cell đầu tiên và chạy (Shift+Enter):

```python
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers trl peft accelerate bitsandbytes
!pip install datasets pandas scikit-learn pyyaml
```

⏱️ Mất khoảng 3-5 phút. Nếu có warning đỏ → **bỏ qua**, không sao.

### Bước 5: Kiểm tra GPU (Cell 2)

```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

Kết quả mong đợi: `GPU: Tesla T4` và `VRAM: 15.x GB`

### Bước 6: Copy code training (Cell 3 trở đi)

Mở file `/home/zinnn/NLP/lab/lab2/banking-intent-unsloth/colab_training.py` trên máy bạn.

Copy từng phần (CELL 2 đến CELL 11) vào các cell trên Kaggle.
Chạy tuần tự từ trên xuống.

**Hoặc cách nhanh hơn:** Copy toàn bộ file vào 1 cell duy nhất, bỏ comment ở CELL 1:

```python
# Bỏ dấu # ở 3 dòng này (đã cài ở bước 4 rồi, nên giữ comment):
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers trl peft accelerate bitsandbytes
# !pip install datasets pandas scikit-learn pyyaml

# ... (paste toàn bộ phần còn lại từ CELL 2 đến CELL 11)
```

### Bước 7: Chờ training (~20-30 phút)

Bạn sẽ thấy output kiểu:

```
📥 Loading BANKING77 ...
   Train: 3850 samples  |  Test: 1540 samples

📦 Loading model ...
🔧 LoRA adapters attached

🚀 Training started ...
{'loss': 3.2145, 'learning_rate': 0.0002, 'epoch': 0.21}
{'loss': 1.8234, 'learning_rate': 0.00018, 'epoch': 0.41}
...
{'loss': 0.3210, 'learning_rate': 0.00002, 'epoch': 2.97}

✅ Done!  Loss: 0.4523  Time: 1234s

🎯 TEST ACCURACY: 72.34%
```

**Đừng tắt tab! Đừng đóng trình duyệt!**

> Kaggle ổn định hơn Colab, nhưng vẫn nên giữ tab mở.

### Bước 8: Download checkpoint về máy

Sau khi training xong, thêm **cell mới cuối cùng**:

```python
# Zip checkpoint
import shutil
shutil.make_archive("/kaggle/working/outputs", "zip", "./outputs")
shutil.make_archive("/kaggle/working/sample_data", "zip", "./sample_data")

print("✅ Đã zip xong!")
print("📁 Vào tab 'Output' bên phải → nhấn Download để tải về máy")
```

Chạy xong → nhìn **sidebar phải** → tab **Output** → bạn sẽ thấy:
- `outputs.zip` ← checkpoint model
- `sample_data.zip` ← data đã xử lý

Nhấn **Download** từng file.

### Bước 9: Giải nén và chạy inference trên máy local

```bash
cd ~/NLP/lab/lab2/banking-intent-unsloth/

# Giải nén
unzip ~/Downloads/outputs.zip -d ./outputs/
unzip ~/Downloads/sample_data.zip -d ./sample_data/

# Chạy inference
python scripts/inference.py configs/inference.yaml
```

---

## PHẦN C: XỬ LÝ SỰ CỐ KAGGLE

### Lỗi "GPU quota exceeded"
→ Bạn đã dùng hết 30h/tuần. Chờ đến thứ 2 tuần sau sẽ reset.
→ Hoặc chuyển sang Google Colab làm tạm.

### Lỗi "Out of memory"
→ Giảm batch size. Thêm cell mới trước khi train:
```python
CONFIG["per_device_train_batch_size"] = 4  # giảm từ 8 xuống 4
```

### Notebook bị disconnect
→ Kaggle tự lưu output. Nếu training chưa xong, chạy lại từ đầu.
→ Tip: bật **Internet ON** trong Settings để Kaggle tải model từ HuggingFace.

### Không thấy GPU option
→ Vào kaggle.com → Settings → Phone Verification → xác minh số điện thoại.

---

## PHẦN D: FLOW TỔNG QUAN (ĐÃ CẬP NHẬT)

```
                        ┌─────────────────┐
                        │   Máy local     │
                        │ (RTX 3050 4GB)  │
                        └───────┬─────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
              ┌─────▼──────┐         ┌──────▼──────┐
              │  Code đã   │         │  Download   │
              │  có sẵn    │         │  checkpoint │
              │  trong     │         │  từ Kaggle  │
              │  project   │         │  về đây     │
              └─────┬──────┘         └──────┬──────┘
                    │                       │
                    │    ┌──────────────┐    │
                    └───►│  Inference   │◄───┘
                         │  (local)    │
                         │  RTX 3050   │
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │  Demo +     │
                         │  Quay video │
                         └─────────────┘

           ┌──────────────────────────────────┐
           │          Kaggle (cloud)           │
           │         GPU T4 × 2 (32GB)        │
           │                                  │
           │  1. Cài Unsloth                  │
           │  2. Load BANKING77               │
           │  3. Preprocess data              │
           │  4. Load Llama-1B (4-bit)        │
           │  5. Gắn LoRA                     │
           │  6. Train 3 epochs (~25 phút)    │
           │  7. Evaluate → accuracy          │
           │  8. Save + Download checkpoint   │
           └──────────────────────────────────┘
```

**Tóm lại:**
1. **Kaggle** = "phòng gym" có máy mạnh → train ở đây
2. **Máy local** = nhà bạn → chạy inference + quay video demo ở đây
3. **Ollama** = không cần, vì đề bài yêu cầu dùng Unsloth Python class
