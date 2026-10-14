# ĐỀ CƯƠNG BÁO CÁO DỰ ÁN XỬ LÝ ẢNH SỐ (DIP PROJECT OUTLINE)

**ĐỀ TÀI:** HỆ THỐNG NHẬN DIỆN THỦ NGỮ MỸ (ASL) THỜI GIAN THỰC: TỪ DÁNG TAY TĨNH ĐẾN CHUỖI TỪ ĐỘNG VÀ GHÉP CÂU
**MÔN HỌC:** XỬ LÝ ẢNH SỐ (DIGITAL IMAGE PROCESSING - DIP)
**SINH VIÊN:** TÔ PHƯƠNG ĐẠT | **MSSV:** 24110086
**CẬP NHẬT:** 2026-10-14 (bản cũ chỉ mô tả 5 từ và 5 chữ cái A, B, C, L, V)

---

## I. TỔNG QUAN ĐỀ TÀI

### 1. Vấn đề và mục tiêu
* **Mục tiêu:** dịch ngôn ngữ ký hiệu ASL thành văn bản và giọng nói theo thời gian thực, chạy trên **webcam và CPU laptop phổ thông**, không cần GPU.
* **Hướng tiếp cận:** không dùng mạng học sâu video (nặng, cần nhiều dữ liệu). Thay vào đó dùng:
  1. Xử lý ảnh số và landmark bàn tay để chuẩn hóa mọi biến dạng vật lý.
  2. So khớp chuỗi thời gian (DTW) trên quỹ đạo đặc trưng, nên chỉ cần **vài mẫu mỗi từ**.
  3. Bộ học nhẹ chỉ ở vai trò phụ trợ (spotter, lọc dáng tay).

### 2. Phạm vi hiện tại
| Hạng mục | Hiện trạng |
|---|---|
| Từ vựng động | **13 từ**: HELLO, ME, YOU, THANK_YOU, BYE, EAT, DRINK, WATER, HUNGRY, SORRY, PLEASE, YES, BAD |
| Dữ liệu | 10 hồ sơ người ký trong `profiles/`; 4 người có bộ dữ liệu đủ đánh giá (Đạt, Mai, Phi, Vuong) |
| Dạy từ mới | Có: `py dip.py hieu-chinh`, 1–3 mẫu/từ, không train lại |
| Chữ cái tĩnh | Bộ phân loại landmark 16 lớp (`letter_landmark_model*.pt`), dùng làm **bộ lọc dáng tay** (handshape gate). Đánh vần ghép từ chưa là tính năng chính |
| Ghép câu | Có: gom từ, chốt câu khi hạ tay > 1,8 s hoặc đủ 8 từ, ra câu tự nhiên song ngữ (SPEC 14) |
| Giao diện | Cửa sổ ứng dụng (`py dip.py`), panel bên phải + panel dưới; bản web `py dip.py web` |

---

## II. KIẾN TRÚC HỆ THỐNG

```
Webcam
  └─ MediaPipe Hands: 21 landmark / tay
       ├─ Chuẩn hóa: lật tay trái -> phải (chirality), chia theo palm_len
       ├─ Lọc EMA cổ tay
       ├─ Phân đoạn động học (KineticPauseSegmenter)
       │     IDLE -> RECORDING -> PENDING_COMMIT -> IDLE (chốt nét)
       ├─ Spotter DL + cổng thời lượng [0,4 s ; 2,5 s]
       ├─ Đặc trưng quỹ đạo 24D, nội suy về K = 32 khung
       ├─ GoldenTriadDTWMatcher (DTW dải Sakoe-Chiba, trọng số theo nhóm)
       │     ACCEPTED / UNCERTAIN / REJECTED + kiểm tra biên độ top-2
       ├─ Handshape gate (loại ứng viên sai dáng tay)
       ├─ Xác nhận người dùng (y / 2 / n) khi UNCERTAIN
       └─ SentenceComposer -> câu, TTS không chặn luồng
```

Tệp chính: `predict_sentence_webcam.py` (vòng lặp), `dynamic_dtw_engine.py` (phân đoạn và DTW),
`dynamic_features.py` (24D), `handshape_gate.py`, `spotter_*.py`, `sentence_composer.py`, `ui_side_panel.py`.

---

## III. VAI TRÒ CỦA XỬ LÝ ẢNH SỐ (PHẦN TRỌNG TÂM CỦA MÔN)

Cần nói rõ để không bị hiểu nhầm: **vòng lặp nhận diện từ động hiện chạy bằng landmark, không dùng CNN trên ảnh** (`enable_static_shape_cnn = False`). Xử lý ảnh cổ điển nằm ở các khâu sau.

| Khâu | Kỹ thuật DIP | Tệp | Dùng ở đâu |
|---|---|---|---|
| Cắt bàn tay thông minh | Hộp cắt neo ở cổ tay, loại cẳng tay, lật gương tay trái | `improved_dip_pipeline.py` (`extract_smart_hand_crop`) | Thu dữ liệu, huấn luyện chữ cái, mô hình `_image` |
| Chuẩn hóa ánh sáng và nhiễu | Lọc song phương, cân bằng trắng Gray-World, CLAHE | `improved_dip_pipeline.py` | Như trên |
| Đặc trưng ảnh 3 kênh | CLAHE + AutoCanny + Sobel (128×128×3) | `improved_dip_pipeline.py` (`process_lean_dip_tensor`) | Dữ liệu và mô hình chữ cái dạng ảnh, nhánh `TwoStreamHandCNN` trong `train.py` |
| Chuẩn hóa hình học | Mô-men ảnh xoay trục tay; chuẩn hóa theo palm_len | `improved_dip_pipeline.py`, `pipeline_common.py` | Ảnh và quỹ đạo |
| Phân đoạn thời gian | Vận tốc cổ tay chuẩn hóa `v_norm = Δw / palm_len`, lọc EMA, dừng có thời gian ân hạn 0,20 s | `dynamic_dtw_engine.py` | Vòng lặp nhận diện |
| Chuẩn hóa không-thời gian | Neo gốc nét, nội suy tuyến tính K=32 | `dynamic_dtw_engine.py` | So khớp |
| So khớp | DTW Sakoe-Chiba, hàm chi phí lai nhiều nhóm đặc trưng | `dynamic_dtw_engine.py` | So khớp |

> Điểm cần chốt trong báo cáo: phần DIP "ảnh" được đo trên bài toán chữ cái tĩnh (mục V); phần "chuỗi thời gian" mới là lõi của hệ thống nhận diện từ.

---

## IV. NHẬN DIỆN TỪ ĐỘNG VÀ GHÉP CÂU

### 1. Phân đoạn và chống kích hoạt nhầm
* FSM 4 trạng thái; khi dừng tay vào `PENDING_COMMIT` 0,20 s để cho phép nối ký hiệu nhiều pha.
* Cổng thời lượng [0,4 s ; 2,5 s]; cổng hình học riêng cho cú gạt trái (xóa từ cuối).
* Spotter DL và `handshape gate` giảm báo nhầm khi người dùng làm việc thường ngày (đứng yên, gãi mặt, uống nước).

### 2. So khớp và quyết định
* Đặc trưng 24D (vị trí, hướng, góc ngón, nhóm khác), nội suy K = 32.
* Trọng số theo nhóm cố định trong `configs/G1p90.json`; ngưỡng toàn cục `global_k = 1.9`.
* Ba mức: chấp nhận, nghi vấn (hỏi lại bằng phím), từ chối.

### 3. Dạy từ mới không train
* `py dip.py hieu-chinh --moi "Tên" --tu HELLO ME ...`: 3 mẫu mỗi từ rồi nhận diện ngay.
* Hồ sơ lưu riêng theo người trong `profiles/`.

### 4. Ghép câu
* Gom từ vào câu; chốt câu khi hạ tay > 1,8 s hoặc đủ 8 từ; từ thứ 9 mở câu mới.
* `sentence_composer.py` biến chuỗi gloss thành câu tự nhiên song ngữ.
* Đọc bằng TTS trong luồng riêng, không chặn webcam.

---

## V. KẾT QUẢ THỰC NGHIỆM ĐÃ CÓ (nguồn: `docs/`, `results/`)

| Thí nghiệm | Kết quả | Nguồn |
|---|---|---|
| DTW so với 1-NN, MLP, GRU, LOSO 7 lượt, 13 từ | DTW LOSO 100%, người lạ (Mai) 92,2%; GRU ngang về đúng-từ, **DTW hơn về từ chối từ lạ** (0/175 so với 12/175 nhận nhầm) | `docs/SO_SANH_BASELINE.md` |
| Thêm mẫu của chính người ký (k = 1, 2, 3) | Mai: 65% -> 91,6 / 94,8 / 95,1%; Phi: 58,5% -> 84,4 / 93,7 / 100% | `docs/SO_SANH_BASELINE.md` |
| Bộ chữ cái bằng landmark, 16 lớp | 84,8% trên 434 ảnh thật; yếu ở `a`, `b`, `1` | `results/letter_landmark_eval.json` |
| Handshape gate trên 242 truy vấn | Số nhận đúng 117 -> 123, nhận sai 2 -> 3 | `results/handshape_gate_eval.json` |

### Thử người lạ trực tiếp trên camera
| Người | Chưa dạy (đúng/25) | Sau dạy 3 mẫu/từ (đúng/25) | Ghi chú |
|---|---|---|---|
| (cần điền từ ghi chú buổi thử) | | | |

### Giới hạn phải nêu trong báo cáo
* Trọng số và hệ số k của DTW do người viết tự chỉnh; baseline học máy chỉ dùng siêu tham số mặc định.
* Ít người ký, 13 từ, mẫu của người lạ chỉ vài chục.
* Kết quả "thêm mẫu của chính họ" đo khi mẫu enroll và truy vấn thu cùng buổi.
* Chữ cái: chưa đạt mức đủ tin cậy cho đánh vần tự do.

---

## VI. KẾT LUẬN
1. **Vai trò DIP:** chuẩn hóa hình học và ánh sáng, phân đoạn theo chuyển động và biến mọi người ký về cùng một không gian đặc trưng. Phần học máy chỉ ở mức nhẹ.
2. **Điểm mạnh được số liệu ủng hộ:** từ chối từ lạ tốt, thích nghi người mới chỉ với 1–3 mẫu, không cần huấn luyện, giải thích được.
3. **Điểm chưa đạt:** chưa hơn học sâu về độ chính xác thuần, chữ cái còn yếu, dữ liệu ít người.
4. **Hướng phát triển:** thêm người ký, mở rộng từ vựng, đánh vần ổn định hơn.
