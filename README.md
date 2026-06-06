# Khóa luận tốt nghiệp

## Tên đề tài:
Ứng dụng mô hình EfficientNet trong phát hiện ung thư da liễu

## Sinh viên thực hiện:
*   Mã sinh viên: 22T1020755
*   Họ tên: Đào Quang Thuận
*   Lớp: Công nghệ thông tin K46C
*   Chuyên ngành: Công nghệ thông tin
*   Giảng viên hướng dẫn: ThS. Trần Việt Khoa

## Tóm tắt:
Khóa luận trình bày hệ thống hỗ trợ sàng lọc ung thư da liễu tự động dựa trên dòng họ mô hình học sâu EfficientNet (B0, B3, B5). Tập dữ liệu huấn luyện chính là ISIC 2019 với 8 lớp bệnh lý, đặc trưng bởi sự mất cân bằng dữ liệu nghiêm trọng. Để khắc phục, đề tài áp dụng cơ chế đánh trọng số lớp (class-weighted Cross-Entropy) ở Stage 2 và tinh chỉnh Focal Loss ở Stage 3. Khóa luận triển khai đánh giá đối chứng song song giữa hai phương pháp chia dữ liệu Image-level split và Patient-level split để nhận diện và triệt tiêu hiện tượng rò rỉ dữ liệu (data leakage) bệnh nhân. Kết quả thực nghiệm cho thấy mô hình EfficientNet-B5 trên Patient-level split đạt Balanced Accuracy 71.89%, tối ưu hóa Recall cho lớp u hắc tố ác tính Melanoma đạt 69.81% và ung thư tế bào vảy SCC đạt 59.57%, góp phần giảm tối đa tỷ lệ bỏ sót ca bệnh nguy hiểm trong lâm sàng.

## Abstract:
This thesis presents an automated dermatological cancer screening system based on the EfficientNet deep learning architecture (B0, B3, B5). The model is trained on the ISIC 2019 dataset containing 8 skin lesion classes characterized by severe class imbalance. To address this issue, we employ a staged fine-tuning strategy incorporating class-weighted Cross-Entropy at Stage 2 and Focal Loss tuning at Stage 3. Furthermore, we implement a comparative evaluation between image-level and patient-level splits to mitigate the impact of patient data leakage. Experimental results demonstrate that the EfficientNet-B5 model evaluated on the patient-level split achieves a Balanced Accuracy of 71.89%, optimizing the Recall for malignant melanoma (MEL) to 69.81% and squamous cell carcinoma (SCC) to 59.57%, thereby minimizing the risk of missing critical clinical cases.

---

# 🧠 SkinScan Backend (FastAPI API Engine)

API Engine chính thức phục vụ cho đề tài khóa luận tốt nghiệp. Backend được xây dựng trên nền tảng **FastAPI** (Python), hỗ trợ chạy suy luận (inference) thời gian thực bằng cách gọi API đến các mô hình PyTorch (EfficientNet B0/B3/B5) đã được huấn luyện và hosting trực tiếp trên **Hugging Face Space**, hoặc chạy ở chế độ giả lập (Mock Mode) phục vụ cho quá trình phát triển giao diện và kiểm thử nhanh.

---

## ⚡ Các Tính Năng Chính
*   **API REST Stateless:** Tối ưu hóa hiệu năng suy luận độc lập, dễ dàng tích hợp và mở rộng.
*   **Hai chế độ vận hành (Runtime Modes):**
    *   `Mock Mode`: Trả về dữ liệu giả lập ngẫu nhiên có phân phối gần giống thực tế, giúp lập trình viên frontend phát triển giao diện mà không cần GPU hay file weights nặng.
    *   `Real Mode`: Chạy forward pass trực tiếp trên ảnh đầu vào qua mô hình PyTorch thật thông qua thư viện `timm` và `torch`.
*   **Trích xuất đặc trưng trung gian (Activations):** Hỗ trợ xuất mảng kích hoạt qua `forward_hook` phục vụ mục đích trực quan hóa giải thích mô hình (XAI).
*   **Tự động tạo tài liệu:** Swagger UI tích hợp sẵn tại `/docs`.

---

## 🛠️ Cấu Hình Môi Trường (.env)

Tạo tệp `.env` dựa trên tệp [.env.example](file:///c:/KLTN/source/backend/.env.example):

```env
APP_NAME="SkinScan API"
APP_VERSION="1.0.0"
DEBUG=true

# Cấu hình suy luận
USE_MOCK_INFERENCE=false                       # Đặt thành true để bật chế độ Mock giả lập
DEFAULT_MODEL="b3"                             # Mô hình mặc định (b0, b3, b5)
MODEL_WEIGHTS_DIR="weights"
MODEL_DEVICE="cpu"

# Cấu hình Hugging Face Space (Real Mode)
HF_SPACE_ID="qthuan2604/skin-disease-classifier"  # Space ID chứa trọng số mô hình phục vụ suy luận

# Giới hạn tải lên
MAX_UPLOAD_SIZE_MB=10

# Security/CORS Settings
FRONTEND_ORIGINS="http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"
```

---

## 🚀 Hướng Dẫn Khởi Chạy Nhanh

### 1. Cài đặt môi trường ảo (Virtual Environment)
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt trên Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Kích hoạt trên macOS/Linux
source venv/bin/activate

# Cài đặt các thư viện dependencies
pip install -r requirements.txt
```

### 2. Cấu hình Model Weights & Hugging Face Space (Real Mode)
Trong chế độ chạy thực tế (`Real Mode`), Backend API không yêu cầu tải trực tiếp các tệp trọng số mô hình `.pt` (nặng hàng trăm MB) về máy cục bộ. Thay vào đó, toàn bộ các mô hình **EfficientNet-B0**, **EfficientNet-B3**, và **EfficientNet-B5** sau khi huấn luyện xong ở **Stage 3** đã được đóng gói và hosting trực tiếp dưới dạng một Gradio App trên **Hugging Face Spaces**:

*   **HF Space ID:** `qthuan2604/skin-disease-classifier`
*   **Cơ chế hoạt động:** Khi người dùng gửi yêu cầu chẩn đoán, FastAPI Backend đóng vai trò là một Proxy client (`RealPredictor`), sử dụng `gradio_client` để giao tiếp với Hugging Face Space, nhận kết quả phân tích cùng bản đồ kích hoạt trung gian để trả về cho Client.
*   **Lợi ích:** Tránh quá tải tài nguyên máy chủ cục bộ (không cần GPU mạnh hay RAM lớn để load weights) và giảm kích thước mã nguồn dự án.

Nếu bạn muốn tùy chỉnh hoặc thay đổi Space ID, hãy cập nhật biến môi trường sau trong tệp `.env`:
```env
HF_SPACE_ID="tên_tài_khoản/tên_space_của_bạn"
```

*(Lưu ý: Thư mục [weights/](file:///c:/KLTN/source/backend/weights) vẫn được định cấu hình trong `.gitignore` để đề phòng trường hợp bạn muốn phát triển thêm các tính năng suy luận ngoại tuyến offline cục bộ).*

### 3. Chạy Server Development
```bash
uvicorn app.main:app --reload --port 8000
```
Truy cập tài liệu API tự động tại: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🐳 Triển Khai Với Docker

Nếu muốn chạy toàn bộ hệ thống bằng Docker, hãy cấu hình đầy đủ các biến môi trường trong tệp `.env` (đảm bảo trỏ đúng đến Hugging Face Space mong muốn hoặc chạy ở Mock Mode) và thực hiện lệnh:
```bash
docker compose up --build
```

---

## 📊 Mô tả API Contract (Đầu ra hệ thống)

### 1. Health Check
*   **Endpoint:** `GET /api/health`
*   **Phản hồi:** Trạng thái hệ thống, cấu hình thiết bị phần cứng (`cpu`/`cuda`), chế độ suy luận (`real`/`mock`) và danh sách các Model ID đã sẵn sàng.

### 2. Danh sách Mô hình
*   **Endpoint:** `GET /api/models`
*   **Phản hồi:** Metadata chi tiết của 3 biến thể (B0, B3, B5) bao gồm số lượng tham số, độ phân giải đầu vào đề xuất, trạng thái khả dụng (`available=true` vì các mô hình đã được hosting sẵn trên Hugging Face Space).

### 3. Chẩn đoán hình ảnh (Predict)
*   **Endpoint:** `POST /api/predict`
*   **Payload (Multipart Form):**
    *   `file`: File ảnh tổn thương da liễu cần sàng lọc (định dạng JPG/PNG).
    *   `model`: ID mô hình lựa chọn (`efficientnet-b0`, `efficientnet-b3`, hoặc `efficientnet-b5`).
*   **Phản hồi JSON mẫu:**
    ```json
    {
      "predicted_class": "MEL",
      "confidence": 0.7143,
      "probabilities": {
        "AK": 0.02,
        "BCC": 0.05,
        "BKL": 0.08,
        "DF": 0.01,
        "MEL": 0.7143,
        "NV": 0.11,
        "SCC": 0.01,
        "VASC": 0.0057
      },
      "model_id": "efficientnet-b5",
      "latency_ms": 112.5
    }
    ```
    *Thứ tự 8 lớp đầu ra được ánh xạ chính xác theo thứ tự nhãn huấn luyện:* `AK, BCC, BKL, DF, MEL, NV, SCC, VASC`.
