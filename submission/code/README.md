# Nhận diện ngôn ngữ ký hiệu (ASL) thời gian thực — môn Xử lý ảnh số

Webcam → MediaPipe landmark → phân đoạn theo vận tốc → đặc trưng 24D → DTW so với bank mẫu → ghép câu (EN/VI).
Chạy hoàn toàn trên CPU.

## Chạy (Windows, Python 3.12/3.13, có webcam)
Nhấp đôi **`chay.bat`** (hoặc gõ `chay.bat` trong cmd). Lần đầu tự tạo môi trường `.venv`, cài thư viện, rồi mở nhận diện
webcam; các lần sau chạy ngay.

Phím: `q` thoát · `c` xóa câu · `s` đánh vần · `d` cửa sổ 5 bước DIP · `u` đổi giao diện · `n`/`x` mẫu âm.

## Các lệnh khác (thêm sau `chay.bat`)
| Việc | Lệnh |
|---|---|
| Xem luồng xử lý chính trên video mẫu (không cần camera) | `.venv\Scripts\python pipeline.py --video demo/demo_clip.mp4` |
| Giao diện cửa sổ chọn chế độ | `chay.bat` rồi đóng, hoặc `.venv\Scripts\python dip.py` |
| Dạy từ mới | `chay.bat hieu-chinh --moi "Ten" --tu HUNGRY SORRY` |
| Bản web | `chay.bat web` (http://127.0.0.1:8000) |
| Kiểm thử | `.venv\Scripts\python -m pytest -q` |

## Đọc code ở đâu
`pipeline.py` (~270 dòng) trình bày 8 bước của phương pháp; `predict_sentence_webcam.py` là vòng lặp đầy đủ;
`dynamic_dtw_engine.py` chứa phân đoạn + DTW; `spotter_*.py` loại cử chỉ vu vơ.

## Giới hạn
Bank chỉ từ một người ký (18 từ, mỗi từ vài mẫu); leave-one-session-out 187/189, người lạ 210/242;
ảnh cháy sáng nhận kém; đánh vần 14 chữ cái.
