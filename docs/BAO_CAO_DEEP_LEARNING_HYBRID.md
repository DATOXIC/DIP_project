# Báo cáo: Tầng Deep Learning song song với DTW (Hybrid Gated)

Tác giả: Tô Phương Đạt (24110086) — DIP Project. Ngày: 2026-09-30.

## 1. Câu hỏi đặt ra

Có nên thêm một tầng Deep Learning (mạng `LightweightGestureNet`: 1D-CNN + Bi-GRU + Attention,
đã có sẵn trong `train_dynamic_dl.py`) chạy song song với DTW, rồi "cân đo đong đếm" để chọn kết
quả phù hợp — nhằm cải thiện khả năng nhận diện với **người lạ** (mục tiêu bảo vệ đồ án)?

## 2. Phát hiện lỗi ở bản thử nghiệm đầu

Bản `eval_dynamic_hybrid.py` ban đầu kết hợp 2 kết quả bằng **blend xác suất alpha cố định**
(`hybrid = 0.65*dl_probs + 0.35*dtw_scores`) và **bỏ qua hoàn toàn `status` của DTW**
(ACCEPTED/UNCERTAIN/REJECTED). Hậu quả đo được: ở các từ DTW đã đúng nhưng DL sai (BYE, EAT,
DRINK), `hybrid_accuracy` trùng khớp tuyệt đối `dl_accuracy` — tức là ghi đè oan quyết định đúng
của DTW ở MỌI trường hợp, không hề "cân đo" gì cả.

**Đã sửa**: thay bằng cơ chế **GATE theo trạng thái** (`dl_fusion.py`), cùng nguyên tắc đã kiểm
chứng cho `handshape_fusion.py` (fusion DRINK/WATER trước đó):
1. DTW `ACCEPTED` → luôn giữ DTW, không hỏi DL.
2. DTW `UNCERTAIN`/`REJECTED` → hỏi DL, chỉ nhận nếu độ tin cậy softmax vượt ngưỡng đã hiệu chỉnh.
3. Không giới hạn từ DL phải nằm trong top-candidate của DTW (khác handshape_fusion) — vì giá trị
   lớn nhất của DL nằm đúng ở các từ DTW sai hẳn (PLEASE, HUNGRY từng đo 0% với DTW).

## 3. Kết quả chính — 4 người lạ thật (LOSO)

| Người lạ | DTW đơn | DL đơn | Hybrid GATED |
|---|---|---|---|
| Dang (n=42) | 73.8% | 95.2% | 92.9%* |
| Mai (n=100) | 87.0% | 93.0% | 95.0% |
| Phi (n=48) | 100% | 100% | 100% |
| Vuong (n=52) | 84.6% | 96.2% | 96.2% |

*Xem mục 4 — 92.9% dùng ngưỡng notch, 95.2% dùng ngưỡng đã sửa (0.38).

Ở Mai, Hybrid **vượt cả DTW lẫn DL riêng lẻ** (DRINK: DTW 17/18, DL 16/18 → Hybrid 18/18) — bằng
chứng synergy thật, không chỉ là lấy max.

## 4. Hiệu chỉnh ngưỡng tin cậy DL — tránh rò rỉ và tránh "hõm lõm"

Ngưỡng ban đầu (0.55, đoán) sai vì đây là bài toán 13 lớp — confidence của dự đoán ĐÚNG cũng chỉ
quanh 0.50-0.60. Sweep thô trên 3 người → 0.32, nhưng khi quét đầy đủ (bước 0.02, 0.10-0.90) phát
hiện đường cong accuracy trung bình có **2 miền ổn định** (0.20-0.26 và 0.34-0.42, đều 96.6%)
nhưng **tụt xuống 96.0% đúng ở khoảng 0.28-0.32** (1 mẫu EAT của Dang nằm đúng hõm này). 0.32 vô
tình rơi vào hõm. Đã sửa `sweep_confidence_threshold()` để chọn **điểm giữa của đoạn ổn định dài
nhất** thay vì điểm đầu tiên chạm max — chốt **`DEFAULT_CONF_THRESHOLD = 0.38`**.

**Kiểm chứng không rò rỉ (nested cross-validation, `tools/calibrate_dl_threshold_nested.py`)**: với mỗi
người lạ, ngưỡng được chọn CHỈ từ dữ liệu của 3 người còn lại (không đụng vào chính họ) — kết quả
trung bình **96.6%**, khớp chính xác với con số dùng ngưỡng cố định toàn cục. Ngưỡng 0.38 KHÔNG bị
overfit vào các người cụ thể đã dùng để chọn nó.

## 5. Độ ổn định qua nhiều lần train (multi-seed)

Train lại DL 2 seed nữa (123, 2024) cho mỗi người, cộng với seed gốc (42):

| Người | seed 42 | seed 123 | seed 2024 | mean ± std |
|---|---|---|---|---|
| Dang | 95.24% | 97.62% | 100.00% | 97.62% ± 1.94 |
| Mai | 93.00% | 91.00% | 98.00% | 94.00% ± 2.94 |
| Phi | 100.00% | 100.00% | 100.00% | 100.00% ± 0.00 |
| Vuong | 96.15% | 98.08% | 96.15% | 96.79% ± 0.91 |

Độ lệch chuẩn nhỏ (0-2.9 điểm %) — độ chính xác DL không phải may rủi của 1 lần khởi tạo.

## 6. Loại lỗi Hybrid KHÔNG cứu được (có chủ đích)

Chỉ **2/242 truy vấn (0.8%)** là DTW "tự tin nhưng sai" (status=ACCEPTED, top1 sai) — loại lỗi
DUY NHẤT gate không đụng vào (luật: không bao giờ ghi đè quyết định tự tin của DTW). Cả 2 ca đều
liên quan đến từ **HUNGRY** (nhầm với PLEASE ở Mai, với THANK_YOU ở Vuong) — một ca ngay cả DL
cũng đồng ý sai (không thể cứu bằng bất kỳ cách nào ở kiến trúc hiện tại), ca còn lại DL đúng
(94% tin cậy) nhưng bị chặn bởi luật an toàn. Tần suất thấp (0.8%) nên chưa cần điều tra sâu như
đã làm với DRINK/WATER trước đây.

## 7. Enrollment + Hybrid cộng gộp — KHÔNG đơn thuần cộng dồn

Đo cùng lúc 2 đòn bẩy (enrollment k mẫu/từ + Hybrid gate) trên 4 người:

| Người | k=0 | k=1 | k=2 | k=3 |
|---|---|---|---|---|
| Dang | 73.8→95.2 (**+21.4**) | 90.6→98.1 (+7.5) | 95.5→99.1 (+3.6) | 100→100 (0, trần) |
| Mai | 87.0→95.0 (**+8.0**) | 98.4→98.0 (**-0.4**) | 98.8→96.8 (**-2.0**) | 98.9→96.6 (**-2.3**) |
| Phi | 100→100 (0, trần) | 98.9→98.9 (0) | 99.3→99.3 (0) | 100→100 (0) |
| Vuong | 84.6→96.2 (**+11.5**) | 89.3→94.9 (+5.6) | 94.1→97.1 (+2.9) | 94.4→98.4 (+4.0) |

**Phát hiện quan trọng**: giá trị của DL-gate LỚN NHẤT khi CHƯA enroll (k=0: +8 đến +21 điểm %) và
**giảm dần** khi enrollment tăng — vì enrollment đã tự đóng phần lớn lỗ hổng mà DL lẽ ra sửa. Ở
Mai, khi k≥1, gate còn khiến kết quả **tệ hơn một chút** (-0.4 đến -2.3 điểm) — enrollment đã đưa
DTW lên rất cao (98%+), phần UNCERTAIN/REJECTED còn lại là các ca biên mà DTW thường đúng nhờ may
rủi margin, trong khi DL đôi khi đoán sai với confidence vừa đủ ngưỡng. **Kết luận: 2 lớp bổ trợ
nhau tốt nhất khi CHƯA có enrollment (người dùng lần đầu, chưa xác nhận mẫu nào); giá trị giảm dần
khi đã enroll nhiều — cân nhắc tắt hoặc siết ngưỡng DL-gate cao hơn khi người dùng đã có ≥2 mẫu
enroll.**

## 8. Giới hạn cần biết trước khi trích dẫn số liệu

- Cỡ mẫu nhỏ (42-100/người, có từ chỉ 4-5 mẫu) — khoảng tin cậy thực tế rộng hơn số liệu trông có vẻ.
- So sánh không hoàn toàn cùng điều kiện: DTW chỉ dùng bank Đạt (189 mẫu), DL train trên pool toàn
  bộ người khác (~400+ mẫu) — đúng với ý định triển khai thực tế, không phải so sánh thuật toán thuần túy.
- Quần thể "người lạ" hẹp: 4 người cùng nhóm tình nguyện viên dự án, có thể cùng phong cách ký được hướng dẫn giống nhau — chưa test với người hoàn toàn ngoài nhóm.
- `best_gesture_model_none.pth` (checkpoint production, train --test_signer none) tự báo cáo
  100% accuracy — **con số này VÔ NGHĨA**, vì tập test là random-split sau khi đã gộp hết, không
  còn ai lạ với chính nó. KHÔNG dùng số này để báo cáo/bảo vệ.

## 9. Việc còn thiếu, cần người dùng thực hiện (không code được)

- Tìm 1-2 người lạ THẬT SỰ ngoài nhóm tình nguyện viên hiện tại để kiểm chứng cuối cùng.
- Thu thêm mẫu cho các từ đang ít (BAD, YES, HUNGRY chỉ 4-5 mẫu/người).

## 10. File liên quan

- `dl_fusion.py` — module gate (mới).
- `eval_dynamic_hybrid.py` — script so sánh DTW/DL/Hybrid theo từng người (đã sửa).
- `tools/calibrate_dl_threshold_nested.py` — hiệu chỉnh ngưỡng không rò rỉ (mới).
- `eval_enroll_hybrid.py` — đo enrollment + hybrid cộng gộp (mới).
- `dynamic_dtw_engine.py`, `predict_sentence_webcam.py`, `dip.py` — wiring production (`--dl-model`, `--dl-nguong`).
- `results_hybrid_{Dang,Mai,Phi,Vuong}.json`, `results/baselines/dl_threshold_nested_cv.json`,
  `results/baselines/enroll_hybrid.json`, `results/dl_seeds/*.json` — số liệu thô.
