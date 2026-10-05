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
| 4 Bản web | | |
| 5 Người lạ | | |
