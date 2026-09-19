# SPEC 05 — Quy trình thu mẫu webcam (dành cho NGƯỜI, không phải agent)

> File này là quy trình cho **Tô Phương Đạt tự thực hiện**, không giao cho agent.
> Điều kiện tiên quyết: **SPEC 05A và SPEC 05B đã xong** (collector có bảng phím 10 từ, nhãn
> phiên, phím `n` cho mẫu âm, **chạy ≥ 28 FPS**, bộ lọc cổ tay đồng bộ với predict).
> **Bank cũ (thu ở ~14 FPS) đã được lưu thành `*.backup_14fps_*` và KHÔNG dùng nữa — thu lại từ đầu.**
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

## Bước -1 — Chuẩn bị camera (5 phút, TRƯỚC khi thu)

Đã đo: bank thu lần trước chạy ~14 FPS, trong khi predict chạy ~30 FPS. Nguyên nhân có thể
nằm ở camera. Trước khi thu:

1. **Khung hình:** thấy từ đầu tới bụng, mặt chiếm ~20–25% chiều cao khung (lùi ra ~80–90 cm),
   để vùng ngực (ME, PLEASE, SORRY) nằm trong khung.
2. **Tắt hiệu ứng camera của Windows** (auto-framing, Studio Effects) nếu có.
3. **Đủ sáng.** Thiếu sáng làm camera kéo dài thời gian phơi sáng và **hạ FPS**.
4. Cố định camera và ghế. Giữ nguyên trong cả phiên.

## Bước 0 — Chạy thử collector MỘT LẦN (5 phút, ĐỪNG BỎ QUA)

Agent không có webcam nên **chưa ai kiểm chứng đường live**. Chạy:

```bash
py collect_dynamic_templates.py --templates webcam_templates.json --session thu_nghiem
```

Thêm cờ `--profile` để xem FPS và ms từng tầng:

```bash
py collect_dynamic_templates.py --templates webcam_templates.json --session thu_nghiem --profile
```

Kiểm tra 5 thứ, mất khoảng 5 phút:

0. **FPS trượt trên HUD ≥ 28** khi có tay trong khung. Dưới 24 thì collector tự từ chối lưu
   (HUD báo `FPS THAP - KHONG LUU`). Nếu thấp: xem bảng `--profile`; nếu ms tính toán nhỏ mà FPS
   vẫn thấp thì camera là nguyên nhân (thử thêm ánh sáng; xem dòng in FOURCC/FPS lúc mở camera).
   **Đừng thu khi FPS chưa đạt — dữ liệu 14 FPS là lý do phải thu lại lần này.**
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

**Cách ký YOU đã đổi (SPEC 05B đã đo thực nghiệm):** chỉ ngón trỏ về phía camera **kèm gật cổ tay 1 nhịp trong mặt phẳng ảnh** (lên/xuống >= 0.5 độ dài bàn tay ~32 px, trong thời gian < 0.3 giây, v_norm đỉnh trung bình 0.092 > v_start = 0.08). Đạt 100% tỷ lệ kích hoạt stroke hợp lệ và 0% kích hoạt nhầm trên nhiễu. Chỉ đẩy thẳng về camera thì cổ tay gần như không dịch trong ảnh 2D nên segmenter không ghi nhận. Mức tối thiểu cụ thể này đã được cập nhật trực tiếp vào mô tả phím `4` của collector — **làm theo con số đó**.

Đã loại: GO, COME, WANT, MORE, HELP, NAME, STOP (hai tay); GOOD, FATHER, MOTHER, SEE, LIKE,
YES, BAD (va nhau hoặc quá tán — đo trên Citizen). **Lưu ý:** nếu bảng ablation cho thấy
mốc cơ thể cứu được FATHER/MOTHER trên **dữ liệu webcam của bạn**, có thể thêm lại sau.
Đừng thêm trước khi có bằng chứng trên dữ liệu của chính bạn.

---

## NGUYÊN TẮC CỐT LÕI: Phân biệt hai loại cổng kiểm tra

| Loại cổng | Ví dụ | Dùng để loại mẫu? |
|---|---|---|
| **Khách quan** | Qua cổng thời lượng [0.4s, 2.5s], không epenthesis, không mất dấu tay, HUD không báo mẫu lệch bất thường | **Có (loại ngay khi thu)** |
| **Theo kết quả nhận diện** | "predict_sentence_webcam có nhận ra mẫu này không" | **TUYỆT ĐỐI KHÔNG** |

> ⚠️ **CẢNH BÁO THIÊN LỆCH CHỌN LỌC (Selection Bias):**
> Tuyệt đối **KHÔNG** bật `predict` sau từng từ để thử xem nó có nhận hay không rồi mới thu lại!
> - Nếu làm vậy, bạn sẽ vô thức chỉ giữ lại những mẫu "dễ nhận" và nịnh hệ thống.
> - Bank sẽ bị co hẹp phương sai, mất đi tính tự nhiên, và khi demo thật sẽ xịt.
> - Hơn nữa, lúc bank mới có 1–3 từ, nó chưa có từ đối thủ và ngưỡng chưa được dò (SPEC 04), test lúc đó không có giá trị phân biệt.

---

## Bước 1 — Cổng go/no-go nhóm đầu (Nhóm 1: 3 từ đầu, ~10 phút)

Phiên 1, thu **3 từ đầu**: `HELLO`, `ME`, `EAT`, **4 mẫu mỗi từ**. Rồi chạy:

```bash
py audit_sign_quality.py --all
```

**Điều kiện đi tiếp:** LOO-CV **100%** (12/12) và không có cảnh báo lệch mẫu hay va chạm.

**Smoke Test (DUY NHẤT 1 LẦN):**
Bật `predict_sentence_webcam.py` lên kiểm tra nhanh (1–2 phút):
- Camera mở mượt, FPS >= 24, HUD báo `MAT: OK`, nhận được chữ lên màn hình.
- **Mục đích:** Kiểm tra đường ống live hoạt động bình thường.
- **TUYỆT ĐỐI KHÔNG** ghi nhớ kết quả đúng/sai và **KHÔNG** thu lại mẫu vì lần test này.

---

## Bước 2 — Thu các nhóm từ tiếp theo & Audit theo nhóm (~15 phút)

Thu 7 từ còn lại chia làm 2 nhóm, mỗi từ 4 mẫu:
- **Nhóm 2 (4 từ):** `BYE`, `YOU`, `THANK_YOU`, `DRINK` -> Chạy `py audit_sign_quality.py --all` kiểm tra va chạm liên lớp với nhóm 1.
- **Nhóm 3 (3 từ):** `WATER`, `SORRY`, `PLEASE` -> Chạy `py audit_sign_quality.py --all` kiểm tra toàn bộ 10 từ (chú ý cặp `SORRY` vs `PLEASE`).

**Cố tình biến thiên nhẹ giữa 4 mẫu** (miễn phí, và làm ngưỡng tổng quát hóa được):

| Mẫu | Cách thực hiện |
|---|---|
| 1 | Tốc độ bình thường |
| 2 | Chậm hơn ~30% |
| 3 | Nhanh hơn ~30% |
| 4 | Lệch nhẹ vị trí (sang trái/phải ~10cm) |

Sau mỗi từ, chạy `py audit_sign_quality.py --word <TỪ>`. Từ nào LOO < 100% → thu lại từ đó ngay tại chỗ.

---

## Bước 3 — Thu cử chỉ ÂM (5 phút)

Bấm **`n`**, thu **20 cử chỉ KHÔNG thuộc từ nào**: gãi mặt, vuốt tóc, với tay lấy chuột,
vẫy tay vô nghĩa, đưa tay lên rồi hạ xuống, chỉ tay vào màn hình, xoa tay...

Đây là dữ liệu để đo **tỉ lệ nhận nhầm** — điều quyết định demo có nhận bừa hay không.
Không có nó, bạn chỉ biết hệ thống nhận đúng, không biết nó có nhận sai.

---

## Bước 4 — Nghỉ, rồi thu phiên 2 (~20 phút)

Nghỉ ít nhất vài giờ (hoặc sang ngày hôm sau). Ngồi lại từ đầu, thay đổi ánh sáng nếu được.
Chạy lại với `--session <tên khác>` (ví dụ `--session toi`). 4 mẫu/từ, 10 từ, cùng cách biến thiên như bước 2.
Không cần thu thêm mẫu âm ở phiên 2. Tổng cộng bạn có **8 mẫu/từ**.

---

## Bước 5 — Chốt & Kế hoạch Test Live Phiên 3 (Sau SPEC 04)

1. Nhắn lại cho agent: bank đã thu xong (`webcam_templates.json` + `negative_strokes.json`).
2. Agent sẽ chạy **SPEC 04** (dò trọng số và ngưỡng tối ưu trên bank này).
3. **Test live thật — Phiên 3 độc lập (Unseen Session)** sau khi SPEC 04 xong:
   - Một buổi test riêng biệt, không dùng dữ liệu này để tune trọng số hay ngưỡng.
   - Ký 10 lần ngẫu nhiên cho mỗi từ + xen kẽ cử chỉ âm.
   - Bật cờ log:
     ```bash
     py predict_sentence_webcam.py --templates webcam_templates.json --log-session logs/live_test_session3.csv
     ```
   - Hệ thống tự động ghi lại toàn bộ ma trận nhầm lẫn để đưa vào báo cáo đồ án một cách trung thực, khoa học nhất!

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
