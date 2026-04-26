# 🧠 Giải thích chi tiết: TẠI SAO phải làm từng bước?

> File này dành cho người chưa có kiến thức nền. Mỗi bước đều trả lời câu hỏi **"Tại sao?"**

---

## 📖 Bức tranh toàn cảnh — Bài lab này làm gì?

Hãy tưởng tượng bạn là nhân viên ngân hàng. Mỗi ngày có hàng ngàn khách hàng nhắn tin hỏi:
- "Thẻ tôi đâu rồi?" → Đây là vấn đề **card_arrival**
- "Tôi muốn đổi mã PIN" → Đây là vấn đề **change_pin**
- "Tôi bị tính phí hai lần" → Đây là vấn đề **transaction_charged_twice**

Có tổng cộng **77 loại vấn đề** (gọi là **intent** - ý định). Bạn không thể thuê 1000 người ngồi đọc từng tin nhắn. Vậy ta cần **dạy máy tính tự phân loại**.

**Bài lab yêu cầu**: Lấy một con AI (LLM) có sẵn → "dạy" nó phân loại 77 loại ý định → kiểm tra nó đúng bao nhiêu %.

---

## PHẦN 1: DATA — Chuẩn bị dữ liệu

### 1.1 Tại sao phải dùng dataset BANKING77?

**Trả lời**: Để dạy AI, bạn cần **ví dụ mẫu**. Giống như dạy trẻ con:
- Chỉ con chó → nói "đây là chó"
- Chỉ con mèo → nói "đây là mèo"
- Lặp lại nhiều lần → trẻ tự nhận biết

BANKING77 chính là bộ **13,083 ví dụ mẫu** đã được con người gán nhãn sẵn:

```
"I am still waiting on my card?"  →  card_arrival
"How do I change my PIN?"         →  change_pin
```

Không có dữ liệu mẫu → không thể dạy AI. Đơn giản vậy thôi.

### 1.2 Tại sao phải "sample subset" (lấy một phần)?

**Trả lời**: Dataset gốc có **10,003 dòng train**. Nếu dùng hết:
- Train **rất lâu** (hàng giờ, tốn tiền GPU)
- Bạn đang dùng GPU miễn phí (Colab T4) → có giới hạn thời gian

Đề bài cũng nói rõ: *"sample and use only a subset... to ensure training can be completed with available computational resources"* (lấy subset để đảm bảo train được với tài nguyên có sẵn).

→ Ta lấy **50 mẫu/loại × 77 loại ≈ 3,850 dòng**. Đủ để AI học mà không quá lâu.

### 1.3 Tại sao phải "balanced sampling" (lấy đều)?

**Trả lời**: Nếu lấy ngẫu nhiên, có thể xảy ra:
- Loại A: 200 mẫu
- Loại B: 3 mẫu
- Loại C: 0 mẫu

→ AI sẽ **giỏi loại A, dở loại B, không biết loại C**. Giống như ôn thi mà chỉ học 1 chương, bỏ 76 chương còn lại.

**Balanced = lấy đều 50 mẫu mỗi loại** → AI được học đều tất cả 77 loại.

### 1.4 Tại sao phải "preprocess" (tiền xử lý) text?

```python
def preprocess_text(text):
    return text.strip().lower()
```

**Trả lời**: Đây là bước "làm sạch" dữ liệu:

| Vấn đề | Ví dụ | Giải thích |
|---------|-------|------------|
| `strip()` | `"  hello  "` → `"hello"` | Bỏ khoảng trắng thừa đầu/cuối |
| `lower()` | `"HELLO"` → `"hello"` | Viết thường hết |

Tại sao? Vì với máy tính, `"Hello"`, `"HELLO"`, `"hello"` là **3 từ khác nhau**. Nếu không chuẩn hóa, AI phải học 3 lần cho cùng 1 từ → lãng phí.

### 1.5 Tại sao phải "label mapping" (ánh xạ nhãn)?

**Trả lời**: Trong dataset gốc, nhãn là **số**: `0, 1, 2, ..., 76`. Nhưng số `11` vô nghĩa với con người.

Ta cần map: `11` → `"card_arrival"`. Như vậy:
- AI sẽ **trả lời bằng tên** (dễ hiểu) thay vì số
- Khi demo, output `"card_arrival"` dễ nhìn hơn `"11"`

### 1.6 Tại sao phải chia Train / Test?

**Trả lời**: Giống như học sinh ôn bài vs thi:
- **Train set** = sách giáo khoa (AI xem và học từ đây)
- **Test set** = đề thi (AI CHƯA BAO GIỜ thấy, dùng để chấm điểm)

Nếu không chia → bạn cho AI "thi" bằng chính đề nó đã "học" → nó thuộc lòng → điểm cao giả tạo → ra thực tế thì dở. Đây gọi là **overfitting**.

### 1.7 Tại sao format thành "Alpaca prompt"?

```
### Instruction:
You are a banking intent classifier...

### Input:
i am still waiting on my card?

### Response:
card_arrival
```

**Trả lời**: LLM (như ChatGPT, Llama) không hiểu bảng CSV. Nó chỉ hiểu **văn bản**. Alpaca prompt là một **format chuẩn** mà LLM được pre-train để hiểu:

- `Instruction` = nhiệm vụ cần làm (phân loại intent)
- `Input` = dữ liệu đầu vào (câu hỏi khách hàng)
- `Response` = đáp án đúng (nhãn intent)

Giống như bạn viết đề bài cho AI: *"Đây là nhiệm vụ, đây là input, đây là đáp án mẫu — hãy học theo."*

---

## PHẦN 2: MODEL — Chọn và chuẩn bị AI

### 2.1 Tại sao dùng Llama-3.2-1B?

**Trả lời**: "Llama" là một LLM do Meta (Facebook) tạo ra, miễn phí.

- **1B** = 1 tỷ tham số (parameter). Có bản 8B, 70B nhưng chúng quá lớn
- 1B đủ nhỏ để **train miễn phí trên Colab** (GPU T4, 16GB VRAM)
- Đủ thông minh để hiểu task phân loại 77 classes

Giống chọn xe: bạn không cần xe F1 để đi chợ. Xe nhỏ, tiết kiệm, đủ dùng.

### 2.2 Tại sao "4-bit quantization" (QLoRA)?

**Trả lời**: Model Llama-3.2-1B gốc dùng **16-bit** cho mỗi số:

```
Bộ nhớ gốc:   1 tỷ × 16 bit = ~2GB
Bộ nhớ 4-bit:  1 tỷ × 4 bit  = ~0.5GB   ← tiết kiệm 75%!
```

**4-bit** nghĩa là nén mỗi con số từ 16 bit xuống 4 bit. Như nén file ZIP — nhỏ hơn nhưng vẫn dùng được.

Tại sao cần? Vì **GPU của bạn chỉ có 4GB VRAM**. Không nén → không nhét vừa.

### 2.3 Tại sao dùng "Fine-tuning" mà không dùng ChatGPT trực tiếp?

**Trả lời**: ChatGPT hiểu ngôn ngữ tự nhiên, nhưng:

| | ChatGPT trực tiếp | Fine-tuning |
|---|---|---|
| Biết 77 intent cụ thể? | ❌ Không | ✅ Có (vì đã được dạy) |
| Output chuẩn format? | ❌ Có thể trả lời lan man | ✅ Chỉ output đúng tên intent |
| Tốc độ | ❌ Phải gọi API | ✅ Chạy offline, nhanh |
| Chi phí | ❌ Tốn tiền API | ✅ Miễn phí sau khi train |
| Tùy chỉnh | ❌ Khó | ✅ Dễ |

Fine-tuning = lấy AI "đa năng" → dạy thêm để nó thành **chuyên gia** trong 1 lĩnh vực.

### 2.4 Tại sao dùng LoRA mà không train toàn bộ model?

**Trả lời**: Llama-1B có **1 tỷ con số** (tham số). Nếu sửa tất cả:
- Cần GPU cực mạnh (A100 80GB, giá ~$10/giờ)
- Mất nhiều giờ
- Dễ "quên" kiến thức cũ

**LoRA** (Low-Rank Adaptation) = chỉ thêm **~1% tham số mới** (khoảng 10 triệu), giữ nguyên 99% còn lại.

```
Hình dung đơn giản:
┌────────────────────────────────────┐
│     Model gốc (1 tỷ tham số)       │  ← ĐÓNG BĂNG, không đụng vào
│                                      │
│  + [Ma trận A nhỏ] × [Ma trận B nhỏ] │  ← CHỈ TRAIN 2 ma trận nhỏ này
└────────────────────────────────────┘
```

Kết quả: Train nhanh gấp 10x, tốn RAM ít hơn 4x, hiệu quả gần bằng train full.

### 2.5 Giải thích các LoRA hyperparameters

```yaml
lora_r: 16          # "rank" - kích thước ma trận nhỏ
lora_alpha: 16      # hệ số scale
lora_dropout: 0     # xác suất bỏ random (Unsloth tối ưu = 0)
```

- **`r = 16`**: Ma trận LoRA có 16 hàng/cột. Số càng lớn → AI học được nhiều hơn, nhưng tốn RAM hơn. 16 là cân bằng tốt.
- **`alpha = 16`**: Quyết định LoRA ảnh hưởng bao nhiêu đến model gốc. Thường đặt bằng `r`.
- **`target_modules`**: Chỉ gắn LoRA vào các phần quan trọng nhất (attention layers: q, k, v, o + feed-forward: gate, up, down).

---

## PHẦN 3: TRAINING — Quá trình dạy AI

### 3.1 Tại sao cần train trên GPU (không dùng CPU)?

**Trả lời**: GPU xử lý song song hàng ngàn phép tính cùng lúc. CPU làm tuần tự.

```
CPU: Tính 1 phép → xong → tính phép tiếp → ...    (hàng ngày)
GPU: Tính 1000 phép CÙNG LÚC                       (vài phút)
```

AI training = hàng triệu phép nhân ma trận. Không có GPU → đợi vài ngày thay vì vài phút.

### 3.2 Giải thích từng Hyperparameter training

#### `per_device_train_batch_size: 8`
**Batch size = bao nhiêu ví dụ mẫu AI xem cùng lúc.**

- Giống lớp học: dạy 1 học sinh/lần (batch=1) rất chậm, dạy cả lớp 30 người/lần (batch=30) nhanh hơn
- **8** = mỗi lần, AI xem 8 câu hỏi cùng lúc rồi cập nhật kiến thức

#### `gradient_accumulation_steps: 4`
**Tích lũy gradient qua 4 bước trước khi cập nhật.**

- Effective batch = 8 × 4 = **32** (tương đương xem 32 mẫu mỗi lần update)
- Tại sao không để batch=32 thẳng? Vì **hết RAM GPU**! Trick này cho kết quả tương đương mà không tốn thêm RAM.

#### `num_train_epochs: 3`
**Epoch = 1 lần xem HẾT dataset.**

- Epoch 1: AI xem qua 3,850 mẫu → bắt đầu hiểu sơ sơ
- Epoch 2: Xem lại lần 2 → hiểu rõ hơn
- Epoch 3: Xem lại lần 3 → nắm vững

Giống ôn bài: đọc 1 lần chưa nhớ, đọc 3 lần thì nhớ. Nhưng đọc 10 lần thì **thuộc lòng** mà không hiểu → overfitting. Nên 3 là hợp lý.

#### `learning_rate: 2e-4` (= 0.0002)
**Tốc độ học — mỗi bước AI điều chỉnh bao nhiêu.**

```
Learning rate LỚN (0.01):   Bước nhảy to  → nhanh nhưng dễ "nhảy qua" đáp án tối ưu
Learning rate NHỎ (0.00001): Bước nhảy bé  → chính xác nhưng cực chậm
Learning rate VỪA (0.0002):  Cân bằng      → đủ nhanh, đủ chính xác
```

Hình dung: bạn tìm đáy hố trên ngọn đồi. Bước quá to → nhảy qua đáy. Bước quá bé → đi mãi không tới.

#### `optimizer: "adamw_8bit"`
**Bộ tối ưu — thuật toán quyết định cách cập nhật tham số.**

- **Adam** = thuật toán thông minh, tự điều chỉnh learning rate cho từng tham số
- **W** = Weight Decay (xem bên dưới)
- **8-bit** = nén bộ nhớ optimizer → tiết kiệm RAM

#### `weight_decay: 0.01`
**Regularization — phạt các tham số quá lớn.**

Tại sao? Ngăn overfitting. Nếu AI nhớ quá kỹ data train → nó chỉ hoạt động tốt trên data train, gặp data mới thì fail. Weight decay giống cô giáo nói: *"Đừng học vẹt, hãy hiểu bản chất!"*

#### `lr_scheduler_type: "linear"`
**Giảm learning rate dần theo thời gian.**

```
Bắt đầu:  LR = 0.0002  (bước lớn, học nhanh)
Giữa:     LR = 0.0001  (bước vừa)
Cuối:     LR = 0.0000  (bước bé, tinh chỉnh)
```

Giống lái xe: đầu tiên tăng tốc, gần đích thì giảm ga để dừng chính xác.

#### `warmup_steps: 10`
**10 bước đầu, learning rate tăng dần từ 0 lên max.**

Tại sao? Nếu nhảy thẳng vào LR=0.0002, model chưa ổn định → có thể "sốc" → loss nhảy lung tung. Warmup = khởi động nhẹ trước.

#### `max_grad_norm: 1.0`
**Gradient clipping — giới hạn "bước nhảy" tối đa.**

Đôi khi gradient (hướng cập nhật) quá lớn → model "nhảy" loạn xạ. Clipping = đặt rào chắn: *"Dù gradient lớn cỡ nào, bước nhảy tối đa = 1.0"*.

#### `seed: 42`
**Số random cố định — để kết quả lặp lại được.**

AI dùng random ở nhiều chỗ (khởi tạo weights, shuffle data). Nếu seed khác → kết quả hơi khác. Đặt seed=42 → ai chạy cũng ra cùng kết quả. (42 là con số nổi tiếng từ "Hitchhiker's Guide to the Galaxy" 😄)

### 3.3 Tại sao cần "EOS token"?

```python
text = ALPACA_PROMPT.format(instr, inp, out) + EOS_TOKEN
```

**EOS = End Of Sequence** (ký tự kết thúc).

Nếu không có EOS, AI không biết khi nào dừng → nó cứ sinh text mãi. EOS giống dấu chấm cuối bài: *"Hết rồi, dừng ở đây."*

### 3.4 "Loss" là gì? Tại sao theo dõi nó?

**Loss = độ sai** của AI. Loss cao → AI đoán sai nhiều. Loss thấp → AI đoán đúng nhiều.

```
Epoch 1: Loss = 3.2  (AI mới bắt đầu, đoán bừa)
Epoch 2: Loss = 1.0  (đang học, sai ít hơn)
Epoch 3: Loss = 0.4  (khá giỏi rồi)
```

Nếu loss **không giảm** → có vấn đề (LR quá cao, data lỗi...)
Nếu loss **= 0** → overfitting (thuộc lòng, không khái quát được)

---

## PHẦN 4: EVALUATION — Chấm điểm AI

### 4.1 Tại sao phải evaluate?

**Trả lời**: Loss thấp ≠ AI giỏi. Bạn cần kiểm tra trên **test set** (data AI chưa thấy bao giờ).

Giống thi cuối kỳ: ôn bài giỏi (loss thấp) chưa chắc thi giỏi (accuracy cao trên test).

### 4.2 Accuracy là gì?

```
Accuracy = Số câu đoán đúng / Tổng số câu × 100%
```

Ví dụ: Test 1,540 câu, đúng 1,155 câu → Accuracy = 75%

Với 77 classes, accuracy **65-80%** là kết quả tốt. Random guess = 1/77 ≈ 1.3%.

### 4.3 Classification Report là gì?

Cho biết AI **giỏi loại nào, dở loại nào**:

```
                    precision  recall  f1-score  support
card_arrival           0.85     0.90     0.87       20
change_pin             0.92     0.88     0.90       20
...
```

- **Precision**: Khi AI nói "card_arrival", đúng bao nhiêu %?
- **Recall**: Trong tất cả câu thực sự là "card_arrival", AI tìm ra bao nhiêu %?
- **F1**: Trung bình hài hòa của precision và recall

---

## PHẦN 5: INFERENCE — Sử dụng AI đã train

### 5.1 Tại sao cần file inference riêng?

**Trả lời**: 
- `train.py` = **dạy** AI (chỉ chạy 1 lần)
- `inference.py` = **sử dụng** AI đã dạy xong (chạy nhiều lần)

Giống: đào tạo bác sĩ 6 năm (train) → bác sĩ đi khám bệnh mỗi ngày (inference).

### 5.2 Tại sao class phải có `__init__` và `__call__`?

```python
class IntentClassification:
    def __init__(self, model_path):   # Load model 1 lần
    def __call__(self, message):      # Dùng lại nhiều lần
```

- **`__init__`**: Load model nặng (mất 20-30s). Chỉ cần làm **1 lần**.
- **`__call__`**: Predict nhanh (0.5s/câu). Gọi **nhiều lần**.

Nếu mỗi lần predict đều load model → chờ 30s/câu. Tách ra → load 1 lần, predict bao nhiêu lần cũng nhanh.

`__call__` còn cho phép dùng object như hàm: `classifier("hello")` thay vì `classifier.predict("hello")`.

### 5.3 Tại sao prompt inference phải GIỐNG prompt training?

**Trả lời**: AI đã được dạy với format Alpaca:

```
### Instruction: ...
### Input: ...
### Response: ...
```

Nếu inference dùng format khác (ví dụ: `"Question: ... Answer: ..."`), AI sẽ **bối rối** vì chưa thấy format này bao giờ → output rác.

Giống: bạn ôn thi trắc nghiệm, vào phòng thi lại gặp đề tự luận → hoảng.

### 5.4 `temperature = 0.0` là gì?

**Temperature** = mức độ "sáng tạo" của AI.

```
temperature = 0.0  → Luôn chọn đáp án chắc chắn nhất (deterministic)
temperature = 0.7  → Có chút ngẫu nhiên, sáng tạo
temperature = 1.5  → Rất ngẫu nhiên, hay nói bậy
```

Cho classification → **phải dùng 0.0**. Bạn muốn AI luôn cho cùng 1 đáp án cho cùng 1 câu hỏi, không đoán lung tung.

---

## PHẦN 6: CẤU TRÚC PROJECT

### 6.1 Tại sao tách thành nhiều file?

| File | Vai trò | Tại sao tách? |
|------|---------|--------------|
| `preprocess_data.py` | Xử lý data | Chỉ chạy 1 lần, có thể thay data khác |
| `train.py` | Training | Chạy trên Colab (GPU mạnh) |
| `inference.py` | Dự đoán | Chạy trên máy local |
| `train.yaml` | Config | Đổi hyperparameters mà không sửa code |
| `train.sh` | Shell script | Chạy 1 lệnh thay vì 3 lệnh |

Tách file = **dễ bảo trì, dễ debug, dễ tái sử dụng**. Nhét hết vào 1 file = hỗn loạn.

### 6.2 Tại sao dùng YAML config thay vì hard-code?

```yaml
# Đổi learning rate? Sửa 1 dòng trong YAML:
learning_rate: 1e-4
```

vs hard-code:
```python
# Phải mở code, tìm dòng, sửa, cẩn thận không phá code khác
trainer = SFTTrainer(... learning_rate=1e-4 ...)
```

YAML config = **tách biệt "cài đặt" và "logic"**. Ai cũng đọc được YAML, không cần biết Python.

---

## PHẦN 7: TÓM TẮT FLOW

```
┌──────────────────────────────────────────────────────┐
│  1. BANKING77 dataset (13,083 câu, 77 loại)          │
│       ↓                                              │
│  2. Preprocess: lowercase, sample 3,850 câu          │
│       ↓                                              │
│  3. Format thành Alpaca prompt (Instruction/Input/    │
│     Output)                                          │
│       ↓                                              │
│  4. Load Llama-3.2-1B (nén 4-bit để vừa GPU)        │
│       ↓                                              │
│  5. Gắn LoRA (chỉ train 1% tham số)                 │
│       ↓                                              │
│  6. Train 3 epochs (AI xem data 3 lần)               │
│       ↓                                              │
│  7. Save checkpoint (file nhỏ ~100MB)                │
│       ↓                                              │
│  8. Evaluate: chạy test set → tính accuracy          │
│       ↓                                              │
│  9. Inference: nhập câu hỏi → ra intent label       │
└──────────────────────────────────────────────────────┘
```

Mỗi bước đều có lý do. Bỏ bước nào cũng sẽ gây lỗi hoặc kết quả tệ.
