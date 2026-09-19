# SPEC 06B — DBA và thử bank mở rộng (chạy SAU SPEC 04)

> Đọc `specs/_PREAMBLE.md` trước. **Điều kiện tiên quyết:** SPEC 06A và SPEC 04 đã xong
> (đã có bank augmented, `tuned_global_weights.json`, `negative_strokes.json`).
> Ước lượng: 3–4 giờ. **Spec này là TÙY CHỌN** — chỉ làm nếu sau SPEC 04 vẫn có từ yếu
> hoặc muốn giảm chi phí khớp. Nếu SPEC 04 đã cho kết quả tốt, có thể dừng ở đây.

## Vì sao đặt sau SPEC 04

DBA tính barycenter bằng một hàm khoảng cách; nếu tính bằng khoảng cách chưa dò thì
barycenter sai. Nên phải có trọng số đã chốt trước.

## Ba cấu hình phải đo (đều dùng trọng số đã dò)

| Cấu hình | Template dùng để khớp | Ghi chú |
|---|---|---|
| **0 — hiện tại** | toàn bộ mẫu **gốc** (min qua N mẫu) | Mốc so sánh |
| **1 — DBA hybrid** | 1 centroid/từ; **giữ mẫu thô chỉ để tính `intra_max`** | Cảnh báo bên dưới |
| **2 — bank mở rộng (chế độ B)** | mẫu gốc + biến thể từ SPEC 06A | Rủi ro nhận nhầm |

**Cảnh báo cho cấu hình 1:** DBA **bình quân hoá mất** biến thiên nội lớp, mà ngưỡng per-word
lại sống nhờ `intra_max`. Nếu chỉ giữ centroid, `intra_max` tụt về ~0 và ngưỡng quá chặt.
Vì vậy phải là **hybrid**: centroid để khớp, mẫu thô để chỉnh ngưỡng.

**Cảnh báo cho cấu hình 2:** bank rộng hơn nghĩa là cử chỉ lạ dễ lọt gần một biến thể nào đó
→ **nhận nhầm tăng**. Phải đo trên `negative_strokes.json`.

## NHIỆM VỤ 1 — `dba_templates.py`

Công trình tham chiếu: Petitjean, Ketterlin & Gançarski (2011), *A global averaging method
for dynamic time warping, with applications to clustering*. **Người dùng cần tự xác thực
tên và năm trước khi đưa vào báo cáo** (trích từ trí nhớ).

1. Khởi tạo barycenter = **medoid** của từ (mẫu có tổng khoảng cách DTW tới các mẫu khác nhỏ
   nhất). Không khởi tạo ngẫu nhiên — DBA nhạy với khởi tạo.
2. Lặp tối đa 15 vòng: với mỗi mẫu, tính đường warp DTW tới barycenter hiện tại; với mỗi
   frame `k` của barycenter, gom mọi frame được warp vào `k`, đặt frame `k` mới = **trung bình**.
   Dừng sớm nếu dịch chuyển < 1e-4.
3. **Chuẩn hoá lại pháp tuyến (cột 9:12) về vector đơn vị** sau khi lấy trung bình — trung
   bình các vector đơn vị **không** là vector đơn vị. Bỏ bước này thì `w_n` bơm lỗi vào mọi
   khoảng cách.
4. Cột `body` (18:20), `z_norm`, `abduction`: lấy trung bình bình thường. `hand_shapes`:
   điền `"Unknown"` (`lambda_shape = 0`). `rep_count`: lấy trung vị.
5. Nếu một từ có **2 lối thực hiện khác nhau rõ rệt** (dấu hiệu: `intra_max` cao bất thường),
   thử `k=2` cụm cho riêng từ đó và báo cáo.
6. Xác nhận mọi centroid có pháp tuyến norm = 1.0 (sai số < 1e-5).

## NHIỆM VỤ 2 — Đo ba cấu hình (leave-one-session-out)

Cùng quy trình tách tập như SPEC 06A nhiệm vụ 2. Báo cáo:

| Cấu hình | top-1 | margin TB | margin từ yếu nhất | nhận nhầm (mẫu âm) | UNCERTAIN (mẫu dương) | thời gian khớp 1 query (ms) | số template |
|---|---|---|---|---|---|---|---|
| 0 hiện tại | | | | | | | |
| 1 DBA hybrid | | | | | | | |
| 2 bank mở rộng | | | | | | | |

**Quy tắc quyết định:** chọn cấu hình 0 trừ khi cấu hình khác **tốt hơn rõ ràng** (margin
vượt 1 sai số chuẩn) **và không tăng nhận nhầm**. Nếu 1 hoặc 2 kém hơn 0, báo cáo và đề xuất
giữ 0. Nhanh hơn không đủ làm lý do nếu độ chính xác giảm.

## TUYỆT ĐỐI KHÔNG LÀM

- Không dùng **Soft-DTW**: chỉ có giá trị khi cần khả vi để backprop trong mạng neural (dự án
  này không có), thêm siêu tham số `gamma`, và đổi ngữ nghĩa khoảng cách nên làm vô hiệu mọi
  ngưỡng đã dò. Trích dẫn ở phần related work thì được.
- Không huấn luyện mô hình neural sequence (GRU/LSTM/Transformer/ST-GCN) — 8 mẫu gốc/từ sẽ
  overfit. Nêu ở phần future work.
- Không ghi đè `webcam_templates.json`. Kết quả ra file mới.
- Không tự áp dụng cấu hình 1 hoặc 2 vào production khi chưa được người dùng xác nhận.
