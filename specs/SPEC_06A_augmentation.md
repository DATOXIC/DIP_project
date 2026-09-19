# SPEC 06A — Tăng cường dữ liệu từ stroke thô (chạy TRƯỚC SPEC 04)

> Đọc `specs/_PREAMBLE.md` trước. **Điều kiện tiên quyết:** SPEC 05 xong (bank webcam +
> `raw_strokes/` schema v1.2 có `session_id`, `face_box`). **Không phụ thuộc SPEC 04** —
> augmentation chỉ cần stroke thô, không cần trọng số đã dò. Ước lượng: 3–4 giờ.

## Mục tiêu và giới hạn (đọc kỹ để không kỳ vọng sai)

Sinh biến thể từ stroke thô để **bank dò ổn định hơn**. Dữ liệu phía sau SPEC 04 sẽ dò
trọng số/ngưỡng trên bank này.

**Nó KHÔNG làm được:** chống ánh sáng xấu hay nền phức tạp. Những lỗi đó xảy ra ở
MediaPipe **trước** khi tới DTW, và biến đổi trên landmark/vector không tạo lại được chúng.
Cái chống được là thu hai phiên khác điều kiện (SPEC 05).

**Nó KHÔNG tạo thông tin mới:** 8 biến thể của cùng một stroke **không độc lập**. Nó làm việc
dò ổn định hơn, không thay được phiên thu thứ hai. Vì vậy mọi phép đánh giá phải tách
theo `session_id` VÀ theo `augment_of`.

## Cách dùng đã chốt: chế độ A — chỉ để dò, KHÔNG làm template

| Chế độ | Dùng biến thể làm gì | Trạng thái |
|---|---|---|
| **A (spec này)** | Chỉ làm phía *tham chiếu* khi dò trọng số/ngưỡng. Template runtime vẫn là **mẫu gốc** | **Làm** |
| B | Đưa biến thể vào bank làm template runtime | Chỉ thử ở SPEC 06B, có điều kiện |

Lý do: chế độ A không đổi hành vi runtime (không chậm thêm, không tăng nhận nhầm), chỉ làm
số đo đáng tin hơn.

---

## NHIỆM VỤ 1 — `augment_templates.py`

Đọc `raw_strokes/<TỪ>/*.json` (KHÔNG đọc `__NEG__`, không tăng cường mẫu âm). Sinh **7
biến thể** mỗi stroke gốc. Giữ cả bản gốc → 8 mẫu/stroke.

**Phải augment ở mức stroke THÔ, không phải vector 24D đã resample**, vì resample và đạo
hàm phải tính lại sau mỗi biến đổi để giữ tính nhất quán vật lý.

| Biến đổi | Khoảng | Cách làm |
|---|---|---|
| Co giãn thời gian | ±20% | Nội suy lại chuỗi frame về `round(n * hệ_số)` frame trước khi tính đặc trưng |
| Co giãn tỉ lệ | ±15% | Nhân toàn bộ toạ độ pixel landmark và `palm_len` với cùng hệ số |
| Xoay | ±10° | Quay toạ độ landmark 2D quanh cổ tay (landmark 0) |
| Lệch điểm khởi đầu | ±10% palm_len | Dịch toàn bộ chuỗi một offset không đổi |
| Dropout frame | 5–10% frame | Xoá ngẫu nhiên frame rồi nội suy bù (mô phỏng MediaPipe mất dấu) |
| Nhiễu góc | sigma = 0.02 rad | Cộng nhiễu Gauss lên landmark rồi **tính lại** 5 góc gập |
| Kết hợp | — | 1 biến thể trộn nhẹ 2–3 phép trên |

Mặt: hộp mặt trong stroke thô được **giữ nguyên** (đầu không bị biến đổi theo tay); chỉ
tọa độ tay bị biến đổi. Tính lại `body_anchor` và `z_norm` sau biến đổi.

### CẤM TUYỆT ĐỐI

- **KHÔNG lật gương ngang.** Nó đảo chirality — đúng thứ đã mất công chuẩn hoá ở SPEC 00.
- Không xoay quá 15° — vượt mức đó pháp tuyến lòng bàn tay không còn đúng giải phẫu.
- Không co giãn thời gian ra ngoài cổng `[STROKE_DURATION_MIN_SEC, STROKE_DURATION_MAX_SEC]`
  (hằng số trong `dynamic_dtw_engine.py`). Mẫu sinh ra phải là thứ pipeline live **tạo được**.
- Không biến đổi làm stroke bị loại bởi `is_movement_epenthesis` — kiểm và loại biến thể đó.

### Yêu cầu kỹ thuật

- Ghi ra file **MỚI** `webcam_templates_augmented.json`, không đụng `webcam_templates.json`.
- Mỗi mẫu có: `"augment_of"` (id stroke gốc), `"augment_kind"`, `"is_original"` (bool),
  và **thừa hưởng `session_id`** của stroke gốc.
- `numpy.random.default_rng(seed)`, seed cố định (`--seed`, mặc định 20260919), in ra.
  **Phải tái lập được.**
- Áp `canonicalize_chirality` cho mọi mẫu sinh ra, y như đường live.
- Tính lại `rep_count` cho từng biến thể; nếu `rep_count` đổi so với gốc, đếm và báo cáo tỉ
  lệ (dấu hiệu tham số phát hiện nhịp chưa bền).
- Cờ `--strength` (mặc định 1.0) nhân với mọi khoảng biến đổi, để thử 0.5× và 1.0×.

## NHIỆM VỤ 2 — Kiểm định: augmentation CÓ giúp không? (bắt buộc)

Đây là phép đo quyết định có giữ augmentation hay không. Đừng bỏ qua.

Điều kiện: cần **>= 2 phiên** trong `session_id`. Nếu chỉ có 1 phiên, dùng tách theo khối
(mẫu 1–2 vs 3–4 trong mỗi từ) và **nói rõ đây là đánh giá yếu hơn**.

Quy trình (leave-one-session-out, luân phiên):
1. Query = **chỉ mẫu gốc** của phiên bị giữ lại.
2. Tham chiếu = mẫu của các phiên còn lại, theo 2 cách:
   - (i) **chỉ mẫu gốc**
   - (ii) **gốc + biến thể** (biến thể **chỉ** sinh từ stroke của các phiên còn lại)
3. Trọng số/ngưỡng: dùng cấu hình production hiện tại (chưa dò), giống nhau cho cả hai.

Báo cáo bảng:

| Cách tham chiếu | top-1 | margin TB | margin từ yếu nhất | nhận nhầm trên mẫu âm | UNCERTAIN trên mẫu dương |
|---|---|---|---|---|---|
| (i) chỉ gốc | | | | | |
| (ii) gốc + biến thể, strength 1.0 | | | | | |
| (ii) gốc + biến thể, strength 0.5 | | | | | |

**Quy tắc quyết định:** nếu (ii) không cải thiện margin/top-1 rõ ràng hơn (i), hoặc làm tăng
nhận nhầm trên mẫu âm, báo cáo "augmentation không giúp" và **đề xuất bỏ** — đừng giữ chỉ vì
spec bảo làm.

## Tiêu chí hoàn thành

- Số mẫu sinh ra và số stroke gốc theo từ.
- Bảng nhiệm vụ 2, kèm cách tách tập đã dùng (**phải nêu rõ đã loại mọi biến thể cùng
  `augment_of` với query**; nếu con số ~100% thì đó là dấu hiệu đã quên quy tắc này).
- Tỉ lệ biến thể có `rep_count` đổi so với gốc.
- Xác nhận mọi biến thể qua được cổng thời lượng và cổng epenthesis.
- Diff FAILED: **không yêu cầu** chạy test suite (người dùng đã yêu cầu bỏ để tiết kiệm);
  nếu bạn nghĩ có nguy cơ hồi quy thật sự thì nêu ra trong báo cáo.

## TUYỆT ĐỐI KHÔNG LÀM

- Không lật gương. Không augment ở mức vector đã resample.
- Không augment mẫu âm.
- Không ghi đè `webcam_templates.json` hay `dynamic_templates.json`.
- Không đưa biến thể vào template runtime (chế độ B thuộc SPEC 06B).
- Không dò trọng số (đó là SPEC 04).
- Không dùng Soft-DTW hay mô hình neural sequence (xem lý do ở SPEC 06B).
