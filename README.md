# D4S – Dự đoán Tiến độ Học tập
---
## Tên dự án
Nhóm D4S – Dự đoán Tiến độ Học tập 

---

## Tổng quan dự án
Dự án này nhằm mục tiêu **dự đoán tiến độ học tập của sinh viên** thông qua việc ước lượng số tín chỉ hoàn thành (`TC_HOANTHANH`) vào cuối mỗi học kỳ.
Mục tiêu chính của dự án là **hỗ trợ phát hiện sớm rủi ro học tập** và **giúp nhà quản lý giáo dục đưa ra quyết định kịp thời và chủ động**.
Thay vì coi đây chỉ là một bài toán dự đoán thuần túy, dự án được thiết kế như một **pipeline hỗ trợ ra quyết định**, trong đó đầu ra của mô hình được chuyển hóa thành các **chỉ báo rủi ro học vụ**, có thể tích hợp trực tiếp vào các dashboard quản lý.

---

## Tính năng chính
Dự án tập trung vào ba mục tiêu cốt lõi:

- **Dự đoán sớm tiến độ học tập**:  
  Dự đoán số tín chỉ mà sinh viên có khả năng hoàn thành (`TC_HOANTHANH`) vào cuối học kỳ dựa trên dữ liệu học tập lịch sử và số tín chỉ đăng ký.

- **Mô hình hóa theo thời gian, tránh rò rỉ dữ liệu**:  
  Áp dụng phương pháp chia tập huấn luyện/đánh giá/kiểm tra theo thời gian (BTC – Back-Time Consistent) nhằm đảm bảo mô hình chỉ sử dụng thông tin quá khứ, phản ánh đúng điều kiện triển khai thực tế.

- **Hỗ trợ ra quyết định có khả năng giải thích**:  
  Cung cấp các kết quả dễ diễn giải thông qua Feature Importance và các kỹ thuật Explainable AI như SHAP và LIME, phục vụ cho việc nhận diện sớm sinh viên có nguy cơ học tập thấp và đưa ra biện pháp can thiệp phù hợp.

---

## Yêu cầu hệ thống (Prerequisites)
- Python >= 3.10
- RAM tối thiểu: 8GB
- Hệ điều hành khuyến nghị: Windows / Linux / macOS

---

## Cài đặt (Installation)
### Bước 1: Tải mã nguồn dự án
```bash
git clone https://github.com/THUHA-3104/D4S.git
cd D4S

```
### Bước 2: Cài đặt thư viện 
```md
```bash
pip install -r requirements.txt
```
---

## Cách chạy
### Bước 1: Chuẩn bị dữ liệu
Đặt các file sau vào thư mục `data/`:
- admission.csv
- academic_records.csv
- test.csv
- sample_submission.csv

### Bước 2: Tiền xử lý dữ liệu
```bash
python pythondata/fixdata.py
```
### Bước 3: Huấn luyện mô hình 
```bash
python pythondata/train.py
```
### Bước 4: Dự đoán 
```bash
python pythondata/final.py
```

### Bước 5: Hiển thị kết quả
Các biểu đồ phân tích, kết quả (Feature Importance, SHAP) được lưu tại thư mục `result/`

---

## Cấu trúc dự án (Project Structure)
```text
.
├── pythondata/
│   ├── fixdata.py      # Tiền xử lý dữ liệu
│   ├── train.py        # Huấn luyện mô hình
│   └── final.py        # Dự đoán
├── data/               # Dữ liệu đầu vào
├── result/            # Kết quả dự đoán & biểu đồ
├── requirements.txt
└── README.md

```
### Đóng góp 
- Phan Thi Đinh
- Trieu Phuong Anh
- Nguyen Thi Thu Ha
