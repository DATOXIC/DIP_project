# SPEC 05A — Chuẩn bị collector TRƯỚC KHI người dùng thu mẫu webcam

> Đọc `specs/_PREAMBLE.md` trước. Dành cho agent. Ước lượng: 1–2 giờ.
> **Phải xong TRƯỚC SPEC 05** (người dùng thu mẫu). Nếu thiếu bước nào ở đây mà người
> dùng đã thu xong, họ sẽ phải thu lại — đúng điều cả kế hoạch đang tránh.

## Bối cảnh — quyết định của người dùng

Người dùng **không dùng template ASL Citizen nữa**. Bank production = **chỉ webcam của
chính họ**, và mọi bước dò (trọng số, ngưỡng) phải tác động trực tiếp lên bank đó.
Quy tắc thu đã chốt là **phương án A: dừng rồi ký** — đưa tay vào vị trí, đứng yên
~0.25s (segmenter báo `SAN SANG`), rồi mới ký. Không có ngoại lệ "ký liền từ tay hạ".

Vì bank webcam chỉ có ~80 mẫu (10 từ × 8) mà cần dò ~8 trọng số, nguy cơ overfit là
lớn. Cách chống duy nhất trung thực là **tách theo phiên thu** (session). Nên collector
phải ghi nhãn phiên vào từng mẫu **ngay từ đầu**.

**Ghi chú về trạng thái code hiện tại:** SPEC 00/01/02/03 đã được thực hiện xong bởi
các agent trước. Segmenter đã có `READY`, `require_ready_pose`, `max_phase_merges`,
`is_movement_epenthesis`; `collect_dynamic_templates.py` đã ghi `raw_strokes/` schema
`raw_stroke_v1.1` (có `face_box`), `rep_count`, `body_anchor_valid`, `handedness`.
**Trước khi sửa, đọc lại code hiện tại** — chuỗi grep dưới đây được viết dựa trên bản
đã có, nếu không khớp thì DỪNG VÀ BÁO CÁO.

---

## NHIỆM VỤ 1 — Bank webcam MỚI, tách khỏi bank cũ

Người dùng muốn bank webcam **bắt đầu sạch** (các mẫu cũ đã bị nhiễm pha nhấc tay).
KHÔNG được sửa hay xóa `dynamic_templates.json`.

1. Backup `dynamic_templates.json` theo quy ước PREAMBLE (`...backup_before_webcam_bank_<ts>.json`).
2. Đảm bảo `collect_dynamic_templates.py --templates webcam_templates.json` **tạo file
   mới rỗng đúng schema** nếu file chưa tồn tại (xem cách `HybridDTWEngine` /
   `save_templates` xử lý file thiếu; tìm chuỗi `load_templates`). Nếu hiện tại nó
   crash hoặc nạp file khác, sửa cho đúng — **chỉ cộng thêm**.
3. `feature_dim` trong metadata phải là **24** khi có mẫu 24D (SPEC 00 đã sửa suy từ dữ
   liệu — kiểm lại bằng cách lưu thử 1 mẫu giả lập).
4. Thêm cờ `--templates` vẫn hoạt động cho `predict_sentence_webcam.py` (đã có; chỉ kiểm).
5. **KHÔNG** đổi `CONFIG["templates_path"]` mặc định ở bước này. Người dùng sẽ đổi sau
   khi bank vượt cổng go/no-go.

## NHIỆM VỤ 2 — Nhãn phiên thu (`session_id`)

1. Mỗi lần chạy `collect_dynamic_templates.py` sinh **một** `session_id` dạng
   `YYYYMMDD_HHMMSS` (thời điểm khởi động) và cho phép ghi đè bằng cờ
   `--session <tên>` (ví dụ `--session sang_den_yeu`).
2. Ghi `session_id` vào **dict mẫu** (cùng cấp `traj_features`, `hand_shapes`,
   `handedness`, `rep_count`) và vào **metadata top-level** của record trong
   `raw_strokes/`. Tăng `RAW_STROKE_SCHEMA_VERSION` lên `raw_stroke_v1.2` và đảm bảo
   `rebuild_templates_from_raw.py` đọc được v1.0, v1.1 **và** v1.2 (thiếu `session_id`
   thì đặt `"unknown"` và cảnh báo).
3. Hiện `session_id` và số mẫu của phiên hiện tại lên HUD của collector (chữ không dấu).

## NHIỆM VỤ 3 — Bảng phím 10 từ + thu mẫu ÂM (negative)

1. Trong `DEFAULT_VOCAB` (tìm chuỗi `DEFAULT_VOCAB`), thay bằng **đúng 10 từ**, theo
   thứ tự và phím sau:

   ```
   '1' HELLO      '2' BYE       '3' ME        '4' YOU       '5' THANK_YOU
   '6' EAT        '7' DRINK     '8' WATER     '9' SORRY     '0' PLEASE
   ```
   Giữ nguyên tên hiển thị dạng `"Hello / Xin chào"` và mô tả ngắn như định dạng cũ.
   Bỏ các từ hai tay (GO, COME, WANT, MORE, HELP, NAME, STOP, GOOD).
   **Kiểm tra các từ mới (EAT, DRINK, SORRY, PLEASE...) có mô tả tay đúng ASL** —
   nếu không chắc, ghi mô tả trung tính `"Xem huong dan ASL"` thay vì bịa.
2. Thêm **phím `n`** = chế độ thu mẫu ÂM: cử chỉ KHÔNG thuộc từ vựng nào (gãi mặt,
   vuốt tóc, với tay lấy chuột, vẫy tay vô nghĩa, đưa tay lên rồi hạ xuống).
   - Mẫu âm đi qua **đúng cùng** segmenter + cổng thời lượng + cổng epenthesis.
   - Lưu vào file riêng `negative_strokes.json` (KHÔNG vào `vocabulary` của bank) cùng
     schema mẫu (24D, `session_id`, `handedness`, `rep_count`, ...), và cũng ghi
     `raw_strokes/__NEG__/`.
   - HUD hiển thị số mẫu âm; mục tiêu 20.
   - Mục đích: đo **tỉ lệ nhận nhầm** (false accept) khi dò ngưỡng. Không có dữ liệu
     âm thì không thể biết ngưỡng có làm demo nhận bừa hay không.
3. Phím `x` (xóa toàn bộ mẫu từ) và `r` (undo) giữ nguyên, nhưng **thêm xác nhận**:
   `x` phải bấm hai lần liên tiếp trong 2 giây, lần đầu chỉ hiện cảnh báo đỏ. (Lý do:
   một lần bấm nhầm là mất cả 8 mẫu vừa thu.)

## NHIỆM VỤ 4 — Bộ đếm mục tiêu trên HUD

HUD hiển thị cho từ đang chọn `n_mau/8` và tô xanh khi đủ 8. Dòng chân trang hướng dẫn
phím cập nhật cho khớp bảng phím mới. **Chữ không dấu** (OpenCV không vẽ được tiếng
Việt có dấu).

## NHIỆM VỤ 5 — Kiểm chứng (không có webcam)

Không có camera trong môi trường agent, nên:
1. `py_compile` mọi file đã sửa.
2. Viết script mô phỏng: phát một chuỗi frame tổng hợp qua đúng khối chốt stroke của
   collector (như các agent trước đã làm), lưu 3 mẫu thường + 2 mẫu âm với 2 giá trị
   `session_id` khác nhau vào file tạm; xác nhận:
   - `webcam_templates.json` tạo mới đúng schema, `feature_dim = 24`.
   - Mẫu có `session_id`; mẫu âm nằm trong `negative_strokes.json`, **không** nằm trong
     `vocabulary`.
   - `rebuild_templates_from_raw.py` đọc được raw v1.2 và tái tạo vector khớp (sai lệch
     ≤ 1e-4).
3. **Nói rõ trong báo cáo**: đường webcam thật KHÔNG kiểm chứng được. Người dùng phải
   tự chạy thử một lần (xem SPEC 05 bước 0).

## Tiêu chí hoàn thành

- Kết quả 5 kiểm tra ở nhiệm vụ 5, kèm bằng chứng.
- `dynamic_templates.json` **không bị thay đổi** (so sánh hash trước/sau).
- Danh sách file đã sửa, tên hàm/class đã chạm.
- Mọi chỗ spec lệch code thật; mọi quyết định tự đưa ra; mọi thứ không kiểm chứng được.

## TUYỆT ĐỐI KHÔNG LÀM

- Không đổi `CONFIG["templates_path"]` mặc định.
- Không chạm `dynamic_templates.json` hay bất kỳ `*.backup*.json`.
- Không đổi tham số segmenter, cổng thời lượng, trọng số DTW.
- Không dò trọng số (SPEC 04).
- Không chạy pytest/e2e/tune_thresholds (người dùng đã yêu cầu bỏ các phép kiểm hồi
  quy tốn kém); nếu bạn nghĩ một thay đổi có nguy cơ hồi quy thật sự, **nêu ra trong báo
  cáo** thay vì tự chạy cả bộ test.
