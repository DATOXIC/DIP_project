# Danh sách chạy thử trực tiếp trên camera (khoảng 20 phút)

Mục đích: kiểm chứng những thứ mới mà test và phát lại video **không** bắt được (giao diện, camera thật, thao tác tay).
Ghi lại ở mỗi bước: **Đạt / Lỗi (mô tả ngắn, chụp màn hình nếu được)**. Gửi lại danh sách này cho Claude để sửa.

Chuẩn bị: cắm camera, ánh sáng đủ, ngồi cách camera ~0,5–1 m, thấy rõ tay và mặt.

## 0. Camera (1 phút)
```bash
py dip.py camera
```
- [ ] Thấy hình, FPS ≥ 20, không báo lỗi camera.

## 1. Cửa sổ ứng dụng (2 phút)
```bash
py dip.py
```
- [ ] Cửa sổ mở, thấy nút "Nhận diện" và nút "✎ Dạy từ mới".
- [ ] Menu "Nâng cao" có "Quay mẫu âm…", "Huấn luyện lại spotter…", "Chấm báo nhầm…".

## 2. Nhận diện bình thường, hồ sơ Đạt (5 phút)
```bash
py dip.py nhan-dien
```
Kỳ vọng: giao diện panel bên phải + panel dưới; dòng khởi động nên có `--one-shot`, spotter DL và `[HandshapeGate] BẬT`.
- [ ] Panel hiển thị bình thường, không đè/tràn chữ.
- [ ] Ký 5 từ riêng lẻ (ví dụ HELLO, ME, YOU, THANK_YOU, BYE), mỗi từ **một lần**: từ hiện đúng, **không** đọc lặp 2–4 lần.
- [ ] Ký hai từ liền nhau (ví dụ ME rồi YOU) không dừng tay: cả hai đều vào câu (không bị `dup_in_span`).
- [ ] Hạ tay xuống rồi ký lại cùng từ: vẫn nhận (không bị `no_span`).
- [ ] Đứng yên, gãi mặt, uống nước, vẫy tay (không ký): **không** nên có từ nào chốt (ghi lại nếu có, kèm từ nào).
- [ ] Ký liên tục 10 từ không hạ tay: câu tự chốt khi đủ 8 từ (viền xanh), từ thứ 9 mở câu mới; câu không tràn khỏi panel.
- [ ] Hạ tay > 1,8 s: câu chốt, có câu tự nhiên song ngữ (SPEC 14) nếu bật.
- [ ] Mẫu âm trực tuyến: làm một cử chỉ vu vơ cho đến khi một từ bị chốt nhầm, bấm **n**: từ đó bị xóa khỏi câu, hiện "REMOVED", và có thư mục mới `continuous_data/_online/*_onlineneg`. Bấm **x** khi một từ vừa chốt là sai từ: từ bị xóa, thêm dòng vào `continuous_data/_online/log.jsonl`.

## 3. Dạy từ mới / hiệu chỉnh (5 phút) — phần dùng để trình bày cho giảng viên
Phải dùng hồ sơ **không phải Đạt** (Đạt bị từ chối vì bank gốc đang đóng băng):
```bash
py dip.py hieu-chinh --moi "Thu" --tu HUNGRY SORRY
```
- [ ] Banner "TEACH: sign 'HUNGRY' (1/3)" hiện; ký 3 lần, mỗi lần được ghi nhận (log `[DayTu] Lưu mẫu`).
- [ ] Phím `z` hoàn tác được 1 mẫu.
- [ ] Sau khi dạy xong tự chuyển sang nhận diện; ký lại HUNGRY và SORRY: nhận đúng.
- [ ] Thoát (q) rồi chạy lại `py dip.py nhan-dien --ho-so Thu`: các từ đã dạy **vẫn còn** (đã lưu vào hồ sơ).
- [ ] Nút "✎ Dạy từ mới" trong cửa sổ `py dip.py` cũng làm được việc tương tự.

Từ **hoàn toàn mới** (không có trong bank), ví dụ:
```bash
py dip.py hieu-chinh --ho-so Thu --tu XIN_CHAO
```
- [ ] Dạy 3 lần xong ký lại: nhận ra, hiện tên từ.

## 3b. Từ mới của hồ sơ Phi và bank demo (10 phút) — thêm 2026-10-14
Từ mới `LIKE`, `NO`, `WANT`, `MY`, `NAME` nằm trong `profiles/Phi.json` (ký hiệu quy ước 1 tay, không phải ASL chuẩn).
```bash
py dip.py nhan-dien --ho-so Phi -- --cam 0 --spotter rule
```
- [ ] Ký từng từ mới riêng lẻ, mỗi từ 5 lần: ghi **số lần đúng / 5** và từ nào bị nhầm (ví dụ `MY` với `ME`/`PLEASE`, `NAME` với `YES`).
- [ ] Ký `ME LIKE YOU`, `ME WANT WATER`, `ME NO LIKE YOU`: câu tự nhiên hiện đúng ("I like you.", "I want water.", "I don't like you.").
- [ ] Với `--spotter dl` mặc định (bỏ `--spotter rule`): từ mới có bị chặn không? Console có dòng `[OneShot] Chặn ... no_span` hay không (ghi lại).
- [ ] Bank demo của Đạt: `py dip.py nhan-dien -- --cam 0 --spotter rule --templates demo_templates.json`: 5 từ mới có nhận được bằng tay của Đạt không (ghi số lần đúng / 5).

## 3c. Đánh vần chữ cái (5 phút) — thêm 2026-10-14
Trong cửa sổ nhận diện bấm `s`. Chữ hỗ trợ: `A B C D E F G H I K L O S W` (khuyến nghị demo: `D B W I L`).
- [ ] Banner vàng "SPELL: ..." hiện; ký một chữ, giữ yên ~0,7 s: chữ vào ("SPELL: D_").
- [ ] Chữ đôi (ví dụ `BOB`, `DEED`): phải hạ tay hoặc đổi hình tay giữa hai chữ giống nhau mới chốt thêm.
- [ ] `Backspace` xóa chữ cuối.
- [ ] Ký `MY`, `NAME`, bấm `s`, ký 3–4 chữ, bấm `s`, hạ tay > 1,8 s: câu "My name is …" (không phát từ nào trong lúc đánh vần).
- [ ] Nếu chữ nhận kém hoặc nhầm hàng loạt: bấm `f` lật camera rồi thử lại (dữ liệu chữ cái thu với khung hình lật gương) và ghi kết quả.

## 4. Bản web của đồng đội (3 phút)
```bash
py dip.py web
```
- [ ] Trình duyệt mở http://127.0.0.1:8000, tải được video, chọn mốc bắt đầu, ra kết quả nhận diện.

## 5. Người lạ (nếu có thể, 10 phút) — quan trọng nhất cho báo cáo
Nhờ 1–2 bạn (không phải Đạt) ngồi trước camera:
```bash
py dip.py hieu-chinh --moi "TenBan" --tu HELLO ME YOU THANK_YOU BYE
```
- [ ] Bạn dạy 3 lần mỗi từ, rồi ký lại từng từ 5 lần: ghi **số lần đúng / 25**.
- [ ] So sánh với `py dip.py nhan-dien --ho-so TenBan` (không dạy trước): ghi số lần đúng. Đây là con số chứng minh lợi thế "không cần train".

## Kết quả ghi lại
| Bước | Kết quả | Ghi chú / lỗi |
|---|---|---|
| 0 Camera | | |
| 1 Cửa sổ | | |
| 2 Nhận diện (Đạt) | | |
| 3 Dạy từ mới | | |
| 3b Từ mới của Phi / bank demo | | |
| 3c Đánh vần | | |
| 4 Bản web | | |
| 5 Người lạ | | |
