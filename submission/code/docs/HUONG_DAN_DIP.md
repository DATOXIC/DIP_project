# Hướng dẫn dùng `dip.py` — một lệnh cho mọi việc

Chạy trong thư mục dự án:

```bash
py dip.py
```

Hoặc trên Windows, bấm đúp `dip.bat` / gõ `dip`. Menu hiện các việc, chọn bằng số. Mọi việc cũng có lệnh trực tiếp:

| Việc | Lệnh |
|---|---|
| Nhận diện (Đạt, bank sản xuất) | `py dip.py nhan-dien` |
| Nhận diện bằng hồ sơ của Mai | `py dip.py nhan-dien --ho-so Mai` |
| Nhận diện bank demo 18 từ (Đạt + 5 từ mới của Phi), có đọc giọng nói | `py dip.py nhan-dien -- --cam 0 --spotter rule --templates demo_templates.json` |
| Dạy từ mới / thu lại một từ (không phải Đạt) | `py dip.py hieu-chinh --ho-so Phi --tu WANT --so-mau 8 -- --cam 0 --mute` |
| Thu dữ liệu cho người có sẵn | `py dip.py thu --ho-so Phi` |
| Thu cho người mới (tạo hồ sơ luôn) | `py dip.py thu --moi "Nguyễn An"` |
| Đạt thu điều kiện môi trường | `py dip.py thu --ho-so Đạt --dieu-kien dark` (dark, bright, far, lefthand, bg2) |
| Kiểm tra dữ liệu đã thu | `py dip.py kiem-tra --ho-so Mai` |
| Đánh giá | `py dip.py danh-gia open --ho-so Mai` (loso, open, baseline, hoc, hybrid) |
| Xem / tạo hồ sơ | `py dip.py ho-so` , `py dip.py ho-so --moi "Tên"` |
| Kiểm thử | `py dip.py test` |
| Chẩn đoán camera | `py dip.py camera` |

Tham số thêm cho chương trình con: đặt sau `--`, ví dụ `py dip.py nhan-dien -- --time-norm`. Chọn camera: `-- --cam 0`.
Tham số đặt sau `--` được nối vào cuối nên ghi đè mặc định của `dip.py` (ví dụ `-- --spotter rule` thay cho `--spotter dl`).

## Người ký và nơi lưu dữ liệu

Từ 2026-09-28 dùng tên thật: V01 → **Mai**, V02 → **Phi**, chủ dự án → **Đạt** (mã `DAT`). Danh sách ở `profiles/people.json`.

| Người | Bank / hồ sơ | Nét thô | session_id |
|---|---|---|---|
| Đạt | `webcam_templates.json` (bank sản xuất, đã đóng băng) | `raw_strokes/` | theo phiên |
| Đạt, điều kiện môi trường | `profiles/DAT_cond.json` | `raw_volunteers/` | `C_<điều kiện>_<ngày>` |
| Tình nguyện viên | `profiles/<mã>.json` | `raw_volunteers/` | `<mã>_<ngày>` |

- Thu cho tình nguyện viên **luôn** ghi nét thô vào `raw_volunteers/`, không lẫn vào `raw_strokes/`.
- Thu vào bank của Đạt phải gõ `YES` xác nhận, vì nó làm các số đo trước đó không còn so sánh được.
- Mỗi buổi thu tự thêm một dòng vào `profiles/notes.csv` (hỏi nhanh tay thuận, khoảng cách, ánh sáng).
- Mã hồ sơ là chữ Latinh không dấu (`Nguyễn An` → `NguyenAn`), tên có dấu vẫn được lưu để hiển thị.

## Thu dữ liệu cho một người mới (các bước)

1. `py dip.py thu --moi "Tên"` (hoặc menu → 2 → N).
2. Làm mẫu từng từ cho họ, cho luyện 1 phút quy tắc "đưa tay vào, giữ yên nửa giây, rồi ký". Mỗi từ 3–5 lần (nét âm là tùy chọn, không bắt buộc). Mặt phải nằm trong khung hình khi thu: mốc cơ thể (cổ tay so với hộp mặt, `w_b = 0.8`) dùng hộp mặt; không thấy mặt thì mẫu đó bị đánh dấu `body_anchor_valid = False` và yếu hơn. Phiếu hướng dẫn riêng (`PHIEU_THU_TINH_NGUYEN_VIEN.md`) hiện không còn trong repo.
3. `py dip.py kiem-tra --ho-so <Tên>`: báo ĐẠT hoặc CHƯA ĐẠT và liệt kê từ cần thu bù.
4. Có dữ liệu thì chạy `py dip.py danh-gia open --ho-so <Tên>` để xem hệ thống nhận người đó thế nào.

## Lưu ý về dữ liệu và quyền riêng tư

- Các tình nguyện viên đầu tiên được thông báo "dữ liệu ẩn danh (V01...)". Nay dữ liệu dùng tên thật; nên báo lại và xin đồng ý với họ, và người mới nên được thông báo đúng cách dữ liệu sẽ được ghi.
- **Cảnh báo (đối chiếu 2026-10-14):** `profiles/*.json` và `raw_volunteers/` hiện **đang được git theo dõi** (844 file), chỉ riêng `profiles/_gop_*.json` bị bỏ qua. Nghĩa là dữ liệu theo tên thật của tình nguyện viên có thể đã lên GitHub. Cần xin đồng ý của họ, hoặc đưa hai thư mục này vào `.gitignore` và gỡ khỏi git (`git rm -r --cached`).
- Bản sao lưu trước khi đổi tên: `archive/backups/ten_nguoi_ky_20260928_184735/`.

## Chế độ hỏi xác nhận và bank gộp

```
py dip.py nhan-dien --xac-nhan                    # bank của Đạt + hỏi lại khi không chắc
py dip.py nhan-dien --ho-so Mai --xac-nhan        # bank riêng của Mai + hỏi xác nhận
py dip.py nhan-dien --ho-so Mai --gop-dat         # bank gộp Đạt + Mai (file tạm profiles/_gop_Mai.json)
py dip.py nhan-dien --ho-so Mai --gop Phi Đạt     # gộp nhiều hồ sơ
```

Khi bật `--xac-nhan`, cử chỉ bị UNCERTAIN hoặc REJECTED (có ứng viên) hiện banner "Is it X?" trong 8 giây:
`[y]` đúng (ứng viên 1) · `[2]` đúng (ứng viên 2) · `[n]` bỏ qua · `[z]` hoàn tác mẫu vừa thêm.
Mẫu xác nhận được thêm vào bank của phiên (tối đa 3 mẫu/từ) và từ đó được ghi vào câu. Cuối phiên các mẫu đã thêm lưu ở
`profiles/xacnhan_<ngày_giờ>.json` (dùng lại bằng `--templates`). Bank trên đĩa không bị sửa. Mã: `session_enroll.py`.

## Thu dữ liệu tình nguyện viên: chỉ cần nhớ MỘT lệnh

```
py dip.py                        # menu, mục 1 = cả buổi thu
py dip.py tinh-nguyen            # tương tự, vào thẳng phiên thu
py dip.py tinh-nguyen --moi "Nguyễn An"     # người mới
py dip.py tien-do                # đã có bao nhiêu người đạt / mục tiêu 8
```

Phiên thu tự làm lần lượt: chọn hoặc tạo người → hướng dẫn cho người ký → (tùy chọn) kiểm camera → ghi chú buổi thu
(tay thuận, khoảng cách, ánh sáng, camera) → thu → kiểm tra chất lượng, thiếu gì thì hỏi thu bù ngay → thử nhận diện với
bank Đạt + mẫu của họ, có hỏi xác nhận → in bảng tiến độ. Tất cả cùng một cửa sổ, không cần nhớ lệnh nào khác.

## Cửa sổ ứng dụng (một lệnh duy nhất)

```
py dip.py          # mở cửa sổ ứng dụng; py dip.py --menu = menu chữ cũ
```

MỘT màn hình: bên trái danh sách người ký (kèm tình trạng dữ liệu, "+ Người mới"); bên phải ba nút lớn cho người đang chọn:
**▶ Nhận diện**, **● Thu dữ liệu** (ghi chú → thu → tự kiểm tra → hỏi thử nhận diện), **✓ Kiểm tra dữ liệu**.
Phía dưới là nhật ký chữ và nút Dừng. Việc ít dùng nằm ở menu **Nâng cao**: đánh giá, miễn từ, chẩn đoán camera, kiểm thử,
hướng dẫn cho người ký. Cửa sổ camera vẫn là cửa sổ OpenCV riêng; mỗi lúc chỉ chạy một việc. Mã: `dip_app.py`.
Nét âm là TÙY CHỌN: hồ sơ không có nét âm vẫn đạt (từ 2026-09-29).

## Từ mới, bank demo và đánh vần (cập nhật 2026-10-14)

**Từ vựng.** Bank Đạt (`webcam_templates.json`) có 13 từ. Hồ sơ `profiles/Phi.json` có thêm 5 từ quy ước 1 tay: `LIKE`, `NO`, `WANT`, `MY`, `NAME`.
`NICE`, `MEET` mới có tên hiển thị và luật ghép câu, chưa có mẫu. Đây là ký hiệu đơn giản hóa từ ASL (nhiều từ vốn 2 tay), cần ghi rõ như vậy khi báo cáo.

**Bank demo.** `demo_templates.json` = bank Đạt (giữ nguyên) + mẫu của Phi cho 5 từ mới (18 từ). Đạt chạy bằng lệnh ở bảng đầu trang.
Từ mới là mẫu của Phi nên có thể nhận kém hơn với tay Đạt; muốn tốt hơn thì thu mẫu của chính mình vào một hồ sơ riêng bằng `hieu-chinh`
(hồ sơ Đạt bị từ chối vì là bank gốc đóng băng).

**Thu lại một từ.** `hieu-chinh` luôn NỐI mẫu mới vào cuối danh sách, không thay thế. Muốn thu lại thì phải xóa mẫu cũ của từ đó khỏi
`profiles/<tên>.json` trước (nên sao lưu vào `archive/backups/`). Chỉ ghi vào file khi thoát bằng phím `q`.

**Spotter chặn từ mới.** Với `--spotter dl` (mặc định khi có `spotter_model.pth`), spotter chỉ được huấn luyện trên 13 từ gốc nên có thể chặn từ mới
(console in `[OneShot] Chặn ... (no_span)`). Cách tạm: `-- --spotter rule` hoặc `-- --lock-soft-prob 0.05`. Cách đúng: thu phiên liên tục có chứa từ mới
rồi `py dip.py huan-luyen-spotter`. Model DL dự phòng (`best_gesture_model_none.pth`) chỉ có 13 lớp nên không biết từ mới.

**Đánh vần.** Trong cửa sổ nhận diện bấm `s`: ký từng chữ cái tĩnh, giữ yên ~0,7 s mỗi chữ; `Backspace` xóa chữ cuối; bấm `s` lần nữa
(hoặc hạ tay > 3 s) để đưa tên vào câu. Hỗ trợ 14 chữ: `A B C D E F G H I K L O S W`. Dữ liệu chữ cái hiện chỉ từ một người ký.
Chạy riêng để xem nhận chữ: `py tools/predict_letters_live.py --camera 0`.

**Người lạ.** Đã đo (một người, 5 từ) rằng mẫu của người lạ khớp kém bank Đạt dù ký đúng quy ước. Cách giữ độ chính xác: dạy 1–3 mẫu mỗi từ
bằng `hieu-chinh --moi "Tên" --tu ... --so-mau 3`, hoặc dùng chế độ hỏi xác nhận, hoặc gộp nhiều hồ sơ (`--gop`).

