# SPEC 05 — Quy trình thu mẫu webcam (dành cho NGƯỜI, không phải agent)

> File này là quy trình cho **Tô Phương Đạt tự thực hiện**, không giao cho agent.
> Điều kiện tiên quyết: **SPEC 05A đã xong** (collector có bảng phím 10 từ, nhãn phiên,
> phím `n` cho mẫu âm, bank mới `webcam_templates.json`).
> Thời gian: **~60 phút**, chia làm **2 phiên ngắn** (xem lý do bên dưới).

## Quyết định đã chốt

- Bank production = **chỉ template webcam của bạn**. ASL Citizen không còn là template.
- Quy tắc thu = **phương án A: dừng rồi ký.** Mọi mẫu đều theo cùng một quy tắc, không có
  ngoại lệ, để bank nhất quán với segmenter lúc demo.
- Mọi bước dò (trọng số, ngưỡng — SPEC 04) tác động trực tiếp lên bank này.

## Vì sao chia 2 phiên

SPEC 04 cần dò ~8 trọng số trên ~80 mẫu. Dò và đánh giá trên cùng mẫu cho con số đẹp vô
nghĩa. Cách trung thực duy nhất là **giữ lại nguyên một phiên để kiểm định**. Hai phiên
khác nhau về ánh sáng, tư thế ngồi, độ mỏi tay — đúng những thứ thay đổi giữa lúc thu và
lúc demo. Tổng thời gian không đổi (vẫn ~4 mẫu/từ mỗi phiên), chỉ tốn thêm việc ngồi lại một lần.

- **Phiên 1**: 4 mẫu/từ, đủ 10 từ. Ban ngày hoặc ánh sáng bạn hay dùng.
- **Phiên 2**: 4 mẫu/từ, đủ 10 từ. **Cố ý khác**: giờ khác hoặc ánh sáng khác, ngồi lại
  từ đầu (đừng giữ nguyên vị trí ghế/camera).

Đặt tên phiên bằng cờ `--session` (ví dụ `--session ngay` và `--session toi`).

---

## Bước 0 — Chạy thử collector MỘT LẦN (5 phút, ĐỪNG BỎ QUA)

Agent không có webcam nên **chưa ai kiểm chứng đường live**. Chạy:

```bash
py collect_dynamic_templates.py --templates webcam_templates.json --session thu_nghiem
```

Kiểm tra 4 thứ, mất khoảng 5 phút:

1. Đưa tay vào khung, đứng yên ~0.3s → HUD phải đổi sang **`SAN SANG`**.
2. Ký thử 1 từ: mẫu được lưu, HUD tăng số mẫu.
3. Bộ dò mặt chạy: FPS không tụt rõ rệt. Nếu FPS tụt dưới ~20, báo lại — bộ dò mặt
   (`CachedFaceDetector`) có thể cần tinh chỉnh.
4. Có thư mục `raw_strokes/` và file trong đó có trường `face_box` và `session_id`.

**Sau đó xóa sạch mẫu thử nghiệm:** xóa file `webcam_templates.json` và thư mục
`raw_strokes/` (chỉ những thứ vừa tạo ở bước này). Bắt đầu thu thật từ file trống.

Nếu bất kỳ mục nào trong 4 mục trên sai → **DỪNG, báo lại**. Đừng thu tiếp.

---

## Bộ từ vựng: 10 từ, một tay

```
1 HELLO    2 BYE    3 ME    4 YOU    5 THANK_YOU
6 EAT      7 DRINK  8 WATER  9 SORRY  0 PLEASE
```

Đã loại: GO, COME, WANT, MORE, HELP, NAME, STOP (hai tay); GOOD, FATHER, MOTHER, SEE, LIKE,
YES, BAD (va nhau hoặc quá tán — đo trên Citizen). **Lưu ý:** nếu bảng ablation cho thấy
mốc cơ thể cứu được FATHER/MOTHER trên **dữ liệu webcam của bạn**, có thể thêm lại sau.
Đừng thêm trước khi có bằng chứng trên dữ liệu của chính bạn.

---

## Bước 1 — Cổng go/no-go ở phút thứ ~10 (ĐỪNG BỎ QUA)

Phiên 1, thu **3 từ trước**: `HELLO`, `ME`, `EAT`, **4 mẫu mỗi từ**. Rồi chạy:

```bash
py audit_sign_quality.py --all --templates webcam_templates.json
```

(Nếu `audit_sign_quality.py` chưa có cờ `--templates`, báo lại — đó là việc nhỏ cho agent.)

**Điều kiện đi tiếp:** LOO-CV **100%** (12/12) và không có cảnh báo lệch mẫu.

**Nếu không đạt → DỪNG.** Đừng thu thêm. Vấn đề nằm ở pipeline, không nằm ở dữ liệu. Báo
lại số liệu để chẩn đoán. Đây chính là cái cổng ngăn bạn mất cả buổi rồi mới phát hiện.

---

## Bước 2 — Thu nốt phiên 1 (~15 phút)

7 từ còn lại, 4 mẫu mỗi từ. **Cố tình biến thiên nhẹ giữa các mẫu** (miễn phí, và làm
ngưỡng tổng quát hóa được):

| Mẫu | Cách thực hiện |
|---|---|
| 1 | Tốc độ bình thường |
| 2 | Chậm hơn ~30% |
| 3 | Nhanh hơn ~30% |
| 4 | Lệch nhẹ vị trí (sang trái/phải ~10cm) |

Sau mỗi từ, chạy `py audit_sign_quality.py --word <TỪ>`. Từ nào LOO < 100% → thu lại từ đó
ngay tại chỗ, đừng để dồn.

## Bước 3 — Thu cử chỉ ÂM (5 phút)

Bấm **`n`**, thu **20 cử chỉ KHÔNG thuộc từ nào**: gãi mặt, vuốt tóc, với tay lấy chuột,
vẫy tay vô nghĩa, đưa tay lên rồi hạ xuống, chỉ tay vào màn hình, xoa tay...

Đây là dữ liệu để đo **tỉ lệ nhận nhầm** — điều quyết định demo có nhận bừa hay không.
Không có nó, bạn chỉ biết hệ thống nhận đúng, không biết nó có nhận sai.

## Bước 4 — Nghỉ, rồi thu phiên 2 (~20 phút)

Nghỉ ít nhất vài giờ (hoặc sang ngày hôm sau). Ngồi lại từ đầu, thay đổi ánh sáng nếu được.
Chạy lại với `--session <tên khác>`. 4 mẫu/từ, 10 từ, cùng cách biến thiên như bước 2.
Không cần thu thêm mẫu âm ở phiên 2.

## Bước 5 — Chốt

Nhắn lại cho tôi: bank đã thu xong (`webcam_templates.json` + `negative_strokes.json`).
Tôi sẽ giao SPEC 04 (dò trọng số và ngưỡng trên bank này) cho agent.

**Chưa đổi `CONFIG["templates_path"]` sang bank mới cho tới khi SPEC 04 chạy xong.**

---

## Mẹo thực hành

- **Luôn đưa tay vào vị trí, dừng ~0.3s (thấy `SAN SANG`), rồi mới ký.** Đây là quy tắc A
  và là tính năng, không phải lỗi. Đừng ký liền từ lúc hạ tay xuống.
- **Ký từ vị trí xuất phát nhất quán** giữa các mẫu của cùng một từ. Chính điều này làm
  DTW khớp tốt.
- Ánh sáng: tránh ngược sáng (cửa sổ sau lưng). MediaPipe mất dấu khi tay tối.
- Nền: tránh nền có màu da hoặc chuyển động.
- Sai một mẫu → bấm `r` (undo) ngay. Đừng để lại mẫu rác. Nút `x` (xóa hết) cần bấm 2 lần
  trong 2 giây để chắc chắn.
- Bấm `o` để xem overlay quỹ đạo các mẫu cũ — giúp ký nhất quán.
- **Nghỉ tay 5 phút giữa phiên.** Mỏi tay làm cử chỉ biến dạng, và biến dạng đó đi thẳng
  vào template.

## Tổng chi phí thời gian

| Việc | Thời gian |
|---|---|
| Bước 0: chạy thử collector | 5 phút |
| Phiên 1: 3 từ + cổng kiểm tra | ~10 phút |
| Phiên 1: 7 từ còn lại | ~15 phút |
| Mẫu âm | 5 phút |
| Phiên 2 (sau khi nghỉ) | ~20 phút |
| **Tổng** | **~55 phút, chia 2 lần ngồi** |

Nhờ `raw_strokes/`, **mọi lần đổi đặc trưng sau này chỉ cần chạy lại
`rebuild_templates_from_raw.py`** — không bao giờ phải thu lại nữa.
