# Nhận diện ngôn ngữ ký hiệu ASL thời gian thực (DIP / Xử lý ảnh số)

Hệ thống nhận diện **từ ký hiệu động** và ghép thành câu, chạy hoàn toàn trên CPU với webcam thường.
Phương pháp chính: **MediaPipe landmark → phân đoạn theo vận tốc → đặc trưng 24D → DTW so với bank mẫu**,
kèm spotter CNN 1D (loại cử chỉ vu vơ) và bộ lọc hình tay từ landmark. Không cần huấn luyện để thêm từ mới
(chỉ cần ký 3 lần: `py dip.py hieu-chinh`).

## 1. Cài đặt (Python 3.12/3.13, Windows)
```bash
py -m pip install -r requirements.txt
```
Cần webcam; kiểm tra bằng `py dip.py camera`.

## 2. Chạy nhanh
| Muốn làm | Lệnh |
|---|---|
| Xem luồng xử lý chính trên video có sẵn (không cần camera) | `py pipeline.py --video demo/demo_clip.mp4` |
| Giao diện cửa sổ (chọn chế độ bằng nút) | `py dip.py` |
| Nhận diện trực tiếp bằng webcam | `py dip.py nhan-dien` |
| Dạy từ mới cho một người dùng khác | `py dip.py hieu-chinh --moi "Ten" --tu HUNGRY SORRY` |
| Bản web | `py dip.py web` (mở http://127.0.0.1:8000) |
| Thu mẫu âm / huấn luyện lại spotter / chấm | `py dip.py quay-mau-am`, `py dip.py huan-luyen-spotter`, `py dip.py cham-mau-am` |
| Xem mọi lệnh | `py dip.py --help` |

Phím trong cửa sổ nhận diện: `s` đánh vần chữ cái, `n` đánh dấu cử chỉ vu vơ (ghi mẫu âm), `x` từ vừa chốt là sai,
`c` xóa câu, `u` đổi giao diện, `f` đảo lật gương, `q` thoát.

## 3. File quan trọng nhất: `pipeline.py`
Chỉ khoảng 270 dòng, đọc từ trên xuống thấy rõ **8 bước** và phương pháp của từng bước:

```
khung hình → 1 PERCEIVE → 2 SEGMENT → 3 SPOT → 4 FEATURIZE → 5 MATCH → 6 VERIFY → 7 COMMIT → 8 COMPOSE
              MediaPipe     vận tốc +     CNN 1D    đặc trưng 24D   DTW       hình tay      khóa 1 lần   gloss → câu
              landmark      máy trạng thái "đang ký"  K=32 khung    Sakoe-Chiba landmark
```
Đây là bản đọc hiểu của vòng lặp sản xuất `predict_sentence_webcam.py`: dùng cùng các thành phần và tham số `CONFIG`,
bỏ phần giao diện, giọng nói, dạy từ, đánh vần. Đã đối chiếu trên 150 giây của một phiên thu: cho cùng các câu với vòng lặp
sản xuất (12 so với 13 từ được chốt, khác nhau do vòng sản xuất có thêm bước chốt sớm).

## 4. Cấu trúc thư mục
| Thư mục / file | Nội dung |
|---|---|
| `pipeline.py` | luồng xử lý chính, rút gọn |
| `dip.py`, `dip_app.py` | bộ khởi chạy một lệnh và giao diện cửa sổ |
| `predict_sentence_webcam.py` | vòng lặp nhận diện đầy đủ (giao diện, giọng nói, dạy từ, đánh vần) |
| `dynamic_dtw_engine.py`, `dynamic_features.py` | phân đoạn, đặc trưng 24D, bộ so khớp DTW |
| `spotter_*.py`, `handshape_*.py`, `hand_pose_features.py`, `letter_landmark_classifier.py` | spotter, hình tay, chữ cái |
| `sentence_composer.py`, `fingerspell.py` | ghép gloss thành câu, đánh vần |
| `collect_*.py`, `camera_diag.py`, `train_spotter.py`, `eval_*.py`, `scoreboard.py` | thu dữ liệu, huấn luyện spotter, đánh giá |
| `webcam_templates.json` | bank mẫu sản xuất: 13 từ của một người ký (Đạt) |
| `demo_templates.json`, `profiles/` | bank 18 từ (thêm 5 từ của Phi) và hồ sơ cá nhân |
| `spotter_model.pth`, `letter_landmark_model*.pt`, `handshape_gate_table.json`, `*.task`, `*.tflite` | mô hình đã huấn luyện |
| `tests/` | kiểm thử tự động (`py -m pytest -q`) |
| `demo/demo_clip.mp4` | đoạn video phiên thu 45 giây để chạy thử `pipeline.py` |

## 5. Kết quả đo và giới hạn (nói thật)
- Bank sản xuất chỉ từ **một người ký**; mọi số đo vẫn lạc quan so với người lạ.
- Chia leave-one-session-out trên bank: top-1 187/189 (99%). Người lạ (4 người, 242 truy vấn, bank của Đạt, không dạy trước):
  top-1 210/242 (87%); từ yếu nhất của người lạ là HUNGRY.
- Phát lại một phiên ký liên tục 29 từ: 25 từ đúng, 6 báo nhầm, 4 bỏ sót (cấu hình mặc định).
- Ảnh quá sáng (cháy sáng) làm hệ thống gần như không nhận được; hệ thống có cảnh báo "Quá chói" trên màn hình.
- Chưa kiểm chứng trên nhiều camera và nhiều người ký.
- 5 từ mới (LIKE, NO, WANT, MY, NAME) chỉ có 8 mẫu từ một người; NO còn yếu (top-1 ~47% trong đo chia khối).
- Đánh vần chỉ có 14 chữ (A B C D E F G H I K L O S W).
