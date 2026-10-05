# 🎙️ SEG — Speech Analysis App

> **Ứng dụng phân tích giọng nói tiếng Anh & Tiếng Việt**  
> Nhận dạng giọng nói → Phát hiện ngôn ngữ → Dịch thuật → Phân tích bài nói

---

## 👤 TÁC GIẢ

| | |
|---|---|
| **Tác giả** | **HUYTKING** |
| **Zalo** | **0396241674** |
| **GitHub** | [@shirmtry](https://github.com/shirmtry) |
| **Email** | *dintanhuy547@gmail.com* |
| **Năm** | 2026 |

---

## ⚖️ BẢN QUYỀN

```
================================================================
                    BẢN QUYỀN HUYTKING
================================================================

  © 2025 HUYTKING. All rights reserved.

  Dự án "SEG — Speech Analysis App" này được phát triển
  và sở hữu bởi HUYTKING.

  📞 Liên hệ: Zalo 0396241674

  ⚠️ MỌI HÀNH VI SAO CHÉP, CHỈNH SỬA, PHÂN PHỐI
     MÀ KHÔNG CÓ SỰ ĐỒNG Ý BẰNG VĂN BẢN CỦA TÁC GIẢ
     ĐỀU BỊ NGHIÊM CẤM.

  Nếu bạn muốn sử dụng code này cho mục đích:
    - Học tập       → Được phép (ghi rõ nguồn)
    - Đồ án         → Được phép (ghi rõ nguồn)
    - Thương mại    → PHẢI LIÊN HỆ ZALO 0396241674
    - Bán lại        → NGHIÊM CẤM

================================================================
```

---

## 📌 GIỚI THIỆU

**SEG** là ứng dụng web phân tích giọng nói, hỗ trợ:

- 🎤 Nhận dạng giọng nói tiếng Anh (faster-whisper)
- 🇻🇳 Nhận dạng giọng nói tiếng Việt (PhoWhisper)
- 🌐 Dịch tự động Anh ⇄ Việt (MarianMT)
- 📊 Phân tích bài nói: WPM, từ đệm, vốn từ, cấu trúc câu
- 💬 Feedback tự động theo tiêu chuẩn public speaking

---

## 🚀 CÀI ĐẶT

### Yêu cầu

- Python 3.10+
- FFmpeg (đặt trong thư mục gốc hoặc cài global)

### Các bước

```bash
# 1. Clone repo
git clone https://github.com/shirmtry/seg.git
cd seg

# 2. Tạo môi trường ảo
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Cài dependencies
pip install -r requirements.txt

# 4. Chạy app
python app.py
```

Mở trình duyệt: http://localhost:5000

---

## 📂 CẤU TRÚC DỰ ÁN

```
seg/
├── app.py                  # Flask server chính
├── requirements.txt        # Dependencies
├── README.md               # File này
├── LICENSE                 # Bản quyền HUYTKING
├── .gitignore
└── templates/
    └── index.html          # Giao diện web
```

---

## 🛠️ CÔNG NGHỆ SỬ DỤNG

| Thành phần | Công nghệ |
|------------|-----------|
| Backend | Flask, Flask-CORS |
| ASR (EN) | faster-whisper (base) |
| ASR (VI) | vinai/PhoWhisper-base |
| Dịch thuật | Helsinki-NLP/opus-mt-en-vi, opus-mt-vi-en |
| Audio | pydub, librosa |
| Deep Learning | PyTorch, Transformers |

---

## 📞 LIÊN HỆ

Mọi thắc mắc, báo lỗi, hoặc xin phép sử dụng:

- **Zalo:** 0396241674
- **GitHub:** https://github.com/shirmtry

---

## 📜 GIẤY PHÉP

Dự án này được phát hành dưới **Giấy phép HUYTKING Custom License**.  
Xem file [LICENSE](./LICENSE) để biết chi tiết.

---

<p align="center">
  <b>© 2025 HUYTKING — Zalo: 0396241674</b><br>
  <i>Made with ❤️ in Vietnam</i>
</p>
```