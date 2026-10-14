# SPEC 03 — Feature registry + 4 nhóm đặc trưng mới (17D → 24D + 1 hằng số chuỗi)

> Đọc `specs/_PREAMBLE.md` trước. **Bắt buộc làm sau SPEC 00 và SPEC 01.**
> Ước lượng: 6–10 giờ. Đây là spec dài nhất — đọc hết trước khi bắt đầu.

## Mục tiêu

Thêm 4 nhóm thông tin mà hệ thống hiện đang **thiếu về mặt cấu trúc**, theo cách
**cộng thêm, đo được từng nhóm, và tắt được**.

## Cơ sở: vì sao cần, và vì sao KHÔNG phải "càng nhiều càng tốt"

Ablation đã đo trên bank Citizen 139 mẫu / 19 từ:

| Cấu hình | Top-1 |
|---|---|
| Đầy đủ (17D hiện tại) | **79.9%** |
| Bỏ 5 góc gập ngón (`w_a=0`) | 62.6% |
| Bỏ pháp tuyến lòng bàn tay (`w_n=0`) | 68.3% |
| Chỉ còn quỹ đạo (`w_a=0, w_n=0`) | **33.8%** |

→ Thông tin **cấu hình bàn tay** là kênh giá trị nhất. Nên mở rộng theo hướng đó.

**NHƯNG**: thêm chiều KHÔNG đơn điệu tăng độ chính xác. Bằng chứng trong chính dự án
này: thêm nhãn C_Shape (một kênh thông tin) làm 79.9% → 54.7%. Hàm chi phí là tổng có
trọng số của các nhóm; thêm một nhóm nhiễu là **pha loãng ngân sách** của các nhóm
đang có ích. Vì vậy spec này yêu cầu **đo từng nhóm một** và **chỉ giữ nhóm nào làm
tăng margin**.

## Hai điểm mù cấu trúc đang tồn tại (đã đo)

1. **Không có mốc quy chiếu theo cơ thể.** Cột 0:2 là `(wrist - wrist_frame_đầu)/palm_len`,
   nên thông tin *vị trí tuyệt đối trên cơ thể* bị mất hoàn toàn. `FATHER` (trán) và
   `MOTHER` (cằm) có cùng dáng tay, cùng chuyển động, khác duy nhất vị trí → **không
   thể phân biệt về mặt toán học**. Đo được: `MOTHER` LOO 2/4, `FATHER` 8/11, và từ gần
   nhất của mỗi bên chính là bên kia.
2. **Không đếm được số nhịp lặp.** Sakoe-Chiba `r=4` trên `K=32` về mặt toán học không
   thể warp 1 nhịp thành 2 nhịp. Đo được: `YES` ↔ `BYE` va nhau (cả hai đều lặp).

---

## LAYOUT ĐÍCH — 24 chiều/frame + 1 hằng số/chuỗi

| Nhóm | Cột | Khóa trọng số | Chuẩn hóa chia | Ghi chú |
|---|---|---|---|---|
| Vị trí (neo stroke) | 0:2 | `w_p` | `sqrt(2)` | **GIỮ NGUYÊN** |
| Vận tốc | 2:4 | `w_m` | `sqrt(2)` | **GIỮ NGUYÊN** |
| 5 góc gập ngón | 4:9 | `w_a` | `sqrt(5)` | **GIỮ NGUYÊN** |
| Pháp tuyến lòng bàn tay | 9:12 | `w_n` | cosine | **GIỮ NGUYÊN** |
| Vận tốc góc | 12:17 | `w_a` | `sqrt(5)` | **GIỮ NGUYÊN** |
| **z_norm (độ sâu ảo)** | **17:18** | `w_z` | `1.0` | MỚI |
| **Mốc cơ thể** | **18:20** | `w_b` | `sqrt(2)` | MỚI |
| **Độ xòe ngón** | **20:24** | `w_s` | `sqrt(4) = 2.0` | MỚI |
| **Số nhịp lặp** | *không phải cột* | `w_r` | — | MỚI, xem bên dưới |

**Nguyên tắc chuẩn hóa bắt buộc**: mỗi nhóm phải chia `sqrt(số_chiều_của_nhóm)`, y
hệt cách code hiện tại làm (tìm chuỗi `SQRT2` và `SQRT5` trong
`compute_golden_triad_dtw_distance`). Nếu không làm, nhóm nhiều chiều tự động được ưu
tiên hơn một cách âm thầm.

---

## NHIỆM VỤ 1 — Tạo module `dynamic_features.py` (file MỚI)

Tạo file mới `dynamic_features.py` ở gốc dự án. **Không** nhồi vào
`dynamic_dtw_engine.py` (file đó đã 1200+ dòng và đang bị ràng buộc additive).

### 1a. Registry khai báo

Một danh sách/dict khai báo các nhóm, mỗi nhóm gồm:
`(tên_nhóm, cột_bắt_đầu, cột_kết_thúc, khóa_trọng_số, kiểu_khoảng_cách, hệ_số_chuẩn_hóa)`

`kiểu_khoảng_cách` là một trong: `"euclid"`, `"cosine"`, `"scalar_abs"`.

Mục đích: thêm/bớt một nhóm trở thành sửa config, không phải sửa code — và bảng
ablation cho báo cáo sinh ra tự động.

### 1b. Các hàm trích xuất đặc trưng

Viết đúng theo công thức sau, **không được tự sáng tạo**:

**`compute_z_norm(palm_len_seq)`** → mảng 1 chiều, độ dài = số frame
```
z[t] = clip((palm_len[t] - palm_len[0]) / palm_len[0], -1.0, 1.0)
```
- `palm_len[0]` là của frame đầu stroke. Nếu `palm_len[0] < 1e-6` → trả về toàn 0.
- **Việc clip vào [-1, 1] là BẮT BUỘC.** `palm_len` có thể tăng gấp đôi khi tay đẩy sát
  camera; không clip thì nhóm này sẽ áp đảo mọi nhóm khác.
- Ý nghĩa: tay đẩy về camera → bàn tay to lên → `z > 0`. Tay kéo về ngực → `z < 0`.
  Đây là thứ phân biệt `YOU` (chỉ ra) với `ME` (chỉ vào mình).

**`compute_finger_abduction(landmarks_3d)`** → mảng 4 phần tử
Hướng của từng ngón = vector đã chuẩn hóa `tip - mcp`, dùng đúng các cặp landmark:
- ngón cái: `(4, 2)`
- ngón trỏ: `(8, 5)`
- ngón giữa: `(12, 9)`
- ngón nhẫn: `(16, 13)`
- ngón út: `(20, 17)`

4 giá trị trả về = góc giữa các cặp hướng **liền kề**, chia `pi` để về [0, 1]:
1. góc(ngón cái, ngón trỏ)
2. góc(ngón trỏ, ngón giữa)
3. góc(ngón giữa, ngón nhẫn)
4. góc(ngón nhẫn, ngón út)

Dùng `arccos(clip(dot(u, v), -1, 1))`. Nếu vector nào có norm `< 1e-6` → đặt góc đó = 0.

Lý do cần: 5 góc gập hiện tại chỉ đo *ngón cong bao nhiêu*, **không đo ngón cách nhau
bao xa**. Nên `V` (2 ngón xòe) và `U` (2 ngón khép) đang không phân biệt được.

**`compute_repetition_count(xs, ys)`** → 1 số nguyên
1. `speed[t] = hypot(xs[t]-xs[t-1], ys[t]-ys[t-1])`, `speed[0] = speed[1]`.
2. Làm mượt bằng trung bình trượt cửa sổ 3.
3. Đếm số **cực đại địa phương** thỏa: giá trị `>= 0.30 * max(speed_đã_mượt)` VÀ cách
   cực đại được chấp nhận trước đó `>= 4` frame.
4. Trả về số đếm được, kẹp vào `[1, 6]`.

**Đây KHÔNG phải một cột per-frame.** Xem Nhiệm vụ 3.

**`compute_body_anchor(wrist_xy, face_box)`** → 2 phần tử
```
(wx - face_cx) / face_w,  (wy - face_cy) / face_w
```
- `face_cx, face_cy` = tâm hộp mặt; `face_w` = **bề rộng** hộp mặt (pixel).
- **Chia bề rộng mặt, KHÔNG chia `palm_len`.** `palm_len` co giãn theo tư thế và góc
  quay bàn tay; bề rộng mặt thì ổn định.
- Kẹp mỗi giá trị vào `[-6.0, 6.0]` để chống giá trị bệnh lý khi hộp mặt bị lỗi.
- Nếu `face_box` là `None` → trả về `(0.0, 0.0)` **và** báo lại cờ "không có mặt" để
  phía gọi biết mà đặt `w_b = 0` cho stroke đó (xem Nhiệm vụ 2).

### 1c. Bộ dò mặt có cache

Class `CachedFaceDetector` trong cùng file:
- Dùng `mp.tasks.vision.FaceDetector` với model `blaze_face_short_range.tflite`.
- Tự tải nếu thiếu, theo **đúng khuôn** mà `tools/inspect_and_trim_asl_citizen.py` đã làm cho
  `hand_landmarker.task` (tìm chuỗi `urllib.request.urlretrieve` trong file đó để copy
  cách làm). URL:
  `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite`
- `detect_every_n_frames: int = 8`. Giữa các lần detect thì **trả lại hộp mặt cache**.
  Lý do: đầu gần như bất động khi ký, nên chi phí khấu hao chỉ còn ~1/8.
- Nếu detect thất bại: giữ hộp cache cũ, nhưng nếu đã quá 30 frame không thấy mặt thì
  trả `None`.
- `reset()` để xóa cache.

---

## NHIỆM VỤ 2 — Mở rộng hàm khoảng cách (CỘNG THÊM, mặc định bit-identical)

Trong `dynamic_dtw_engine.py`, hàm `compute_golden_triad_dtw_distance`:

1. Thêm 4 tham số vào **cuối** danh sách, **tất cả mặc định 0.0**:
   ```
   w_z: float = 0.0,
   w_b: float = 0.0,
   w_s: float = 0.0,
   w_r: float = 0.0,
   ```
   Thêm 2 tham số nữa ở cuối cho số nhịp lặp:
   ```
   q_rep_count: Optional[int] = None,
   t_rep_count: Optional[int] = None,
   ```

2. Phát hiện khả dụng theo **đúng khuôn có sẵn** (tìm chuỗi `has_template_delta_angle`):
   ```
   has_z    = (t_traj.shape[1] >= 18 and q_traj.shape[1] >= 18)
   has_body = (t_traj.shape[1] >= 20 and q_traj.shape[1] >= 20)
   has_abd  = (t_traj.shape[1] >= 24 and q_traj.shape[1] >= 24)
   ```

3. Trong vòng lặp tính `step_cost`, thêm (vectorized theo NumPy như code hiện có):
   - `d_z = abs(q_row[17] - t_slice[:, 17])` (chia 1.0, tức không chia)
   - `d_body = hypot(q_row[18]-t_slice[:,18], q_row[19]-t_slice[:,19]) / SQRT2`
   - `d_abd = norm(q_row[20:24] - t_slice[:, 20:24], axis=1) / 2.0`
   - Cộng vào: `+ w_z * d_z + w_b * d_body + w_s * d_abd`
   - Nhóm nào `has_* == False` thì đặt mảng 0 (đúng như cách `d_angle`/`d_norm` đang làm).

4. **Số nhịp lặp cộng MỘT LẦN sau khi đã tính đường DTW**, ngay cạnh chỗ đã có
   `lambda_shape_global` (tìm chuỗi `lambda_shape_global > 0.0`):
   ```
   if w_r > 0.0 and q_rep_count is not None and t_rep_count is not None:
       base_distance += w_r * abs(q_rep_count - t_rep_count)
   ```
   **Vì sao không làm cột per-frame**: số nhịp lặp là thuộc tính **toàn cục của chuỗi**.
   Nhồi thành cột hằng số thì DTW chỉ cộng một offset không đổi — vô dụng. Code đã có
   sẵn đúng khuôn này (`lambda_shape_global`), đi theo nó.

5. `GoldenTriadDTWMatcher`: thêm `w_z`, `w_b`, `w_s`, `w_r` vào `__init__` (cuối danh
   sách, mặc định 0.0), lưu vào `self.`, truyền xuống hàm khoảng cách. Trong `match()`,
   tính `rep_count` của query và đọc `rep_count` của từng template từ dict mẫu (khóa
   `"rep_count"`, mặc định `None` nếu thiếu).
   Cũng phải mở rộng `_get_weights_for_sign` / phần blend ASL-LEX: **các trọng số mới
   KHÔNG có trong file ASL-LEX**, nên chúng dùng giá trị global từ `__init__`, không đi
   qua blend. Ghi comment tiếng Việt nói rõ điều này.

### Tiêu chí bất biến (BẮT BUỘC kiểm)

Với `w_z=w_b=w_s=w_r=0.0` và template 17D, khoảng cách phải **giống hệt đến từng bit**
so với trước khi sửa. Kiểm bằng cách: lưu ma trận khoảng cách 139x139 trên bank Citizen
trước khi sửa, sửa xong tính lại, báo cáo **sai số tuyệt đối lớn nhất**.
**Yêu cầu: chính xác 0.0.**

---

## NHIỆM VỤ 3 — Trích xuất đặc trưng mới ở 4 CHỖ, nhất quán

Cùng một công thức phải được dùng ở cả 4 nơi. Nếu lệch nhau, sẽ tái diễn đúng thảm họa
lệch miền đã xảy ra với chirality.

Tìm chuỗi `column_stack` trong từng file để định vị chỗ ghép vector:

| # | File | Ghi chú |
|---|---|---|
| 1 | `predict_sentence_webcam.py` | đường live (query) |
| 2 | `collect_dynamic_templates.py` | đường live (thu mẫu) |
| 3 | `tools/inspect_and_trim_asl_citizen.py` | đường video |
| 4 | `rebuild_templates_from_raw.py` | đường tái tạo offline |

Thứ tự ghép mới (giữ nguyên 17 cột đầu, chỉ thêm vào sau):
```
[xs, ys, dxs, dys, angles(5), normals(3), dangles(5), z_norm(1), body(2), abduction(4)]
```

Và lưu `rep_count` vào **dict mẫu** (cùng cấp với `traj_features`, `hand_shapes`,
`duration_sec`), không nhồi vào `traj_features`.

### 3a. CẢNH BÁO QUAN TRỌNG — archive stroke thô đang THIẾU hộp mặt

`rebuild_templates_from_raw.py` tái tạo vector từ `raw_strokes/`. Archive hiện lưu 21
landmark + `palm_len` → tính lại được `z_norm` và `abduction`. Nhưng **KHÔNG lưu hộp
mặt**, nên **không thể tái tạo được mốc cơ thể**.

Bắt buộc: thêm `face_box` (dạng `[cx, cy, w, h]` hoặc `null`) vào record **từng frame**
trong `write_raw_stroke_archive()` của `collect_dynamic_templates.py`, và **tăng
`RAW_STROKE_SCHEMA_VERSION`**. `rebuild_templates_from_raw.py` phải xử lý được cả
schema cũ (không có `face_box` → đặt `w_b = 0` cho stroke đó và ghi cảnh báo).

Nếu bỏ qua bước này, người dùng sẽ phải thu lại mẫu lần nữa — đúng điều toàn bộ kế
hoạch này đang cố tránh.

---

## NHIỆM VỤ 4 — Trích xuất lại bank Citizen KHÔNG cần cắt tay lại

Đây là bước tiết kiệm công sức lớn nhất, **đừng bỏ qua**.

Mỗi mẫu trong `asl_citizen_templates.json` **đã lưu sẵn** `source_video` và
`trim_range`. Video gốc vẫn còn trong `ASL_Citizen_videos/<TỪ>/`.

Viết script mới `tools/reextract_citizen_batch.py`:
- Đọc file template Citizen hiện có, lấy `(source_video, trim_range)` của từng mẫu.
- Mở đúng video đó, chạy lại MediaPipe **và bộ dò mặt** trên đúng khoảng
  `trim_range`, tính vector 24D mới + `rep_count`.
- Ghi ra file mới (mặc định `asl_citizen_templates_24d.json`), **giữ nguyên** file cũ.
- Áp `canonicalize_chirality` như các đường khác.
- Giữ lại toàn bộ metadata gốc của từng mẫu (`source_video`, `trim_range`,
  `quality_flag`, `dropout_ratio`, `max_gap_frames`, `created_at`).
- In tiến độ và tổng kết: bao nhiêu mẫu thành công, bao nhiêu mẫu **không tìm thấy mặt**
  (con số này rất quan trọng — video ASL Citizen có thể khung hình chỉ lấy phần thân trên).

**Nếu tỉ lệ không tìm thấy mặt > 20%**, nhóm mốc cơ thể sẽ không đo được trên bank này
→ **DỪNG VÀ BÁO CÁO**, đừng tự ý bỏ nhóm đó hoặc điền số giả.

---

## NHIỆM VỤ 5 — CỔNG ĐO TỪNG NHÓM (phần quan trọng nhất của spec)

Viết script `tools/ablate_feature_groups.py`:

- Nạp `asl_citizen_templates_24d.json`.
- Chạy LOO top-1 + margin trung bình cho **từng** cấu hình dưới đây. Trọng số khởi
  điểm cho nhóm mới: `w_z=0.5`, `w_b=0.8`, `w_s=0.4`, `w_r=0.15`.

| Cấu hình | `w_z` | `w_b` | `w_s` | `w_r` |
|---|---|---|---|---|
| Mốc 17D | 0 | 0 | 0 | 0 |
| chỉ z_norm | 0.5 | 0 | 0 | 0 |
| chỉ mốc cơ thể | 0 | 0.8 | 0 | 0 |
| chỉ độ xòe ngón | 0 | 0 | 0.4 | 0 |
| chỉ số nhịp lặp | 0 | 0 | 0 | 0.15 |
| tất cả | 0.5 | 0.8 | 0.4 | 0.15 |

- In bảng: cấu hình / top-1 / margin trung bình / **và** LOO riêng của các từ va nhau
  đã biết: `FATHER`, `MOTHER`, `THANK_YOU`, `GOOD`, `YES`, `BYE`, `ME`, `COME`.

### Kỳ vọng cụ thể (để tự đánh giá kết quả có hợp lý không)

- **Mốc cơ thể** phải cải thiện rõ `FATHER` và `MOTHER` (mốc: 8/11 và 2/4). Nếu không,
  hoặc bộ dò mặt sai, hoặc công thức sai → điều tra, đừng báo cáo "không hiệu quả".
- **Số nhịp lặp** phải cải thiện `YES`/`BYE` (mốc: 8/10 và 9/10).
- **z_norm** phải cải thiện `ME`/`YOU` (mốc: 4/5 và 3/4).
- **Độ xòe ngón** không có kỳ vọng cụ thể — đây là nhóm đầu tiên bị loại nếu không giúp gì.

### Quy tắc quyết định (bắt buộc tuân theo)

Nhóm nào **không làm tăng margin trung bình** thì **đặt trọng số của nó về 0.0** ở cấu
hình production và **báo cáo rõ là đã loại**. Đừng mang theo chiều chết. Code vẫn giữ
lại (không xóa) để có thể bật lại thử nghiệm sau.

---

## Tiêu chí hoàn thành TOÀN SPEC

1. Sai số bất biến khi tất cả trọng số mới = 0: **chính xác 0.0**.
2. Diff danh sách test FAILED: **RỖNG**.
3. Bảng ablation đầy đủ như Nhiệm vụ 5.
4. Tỉ lệ không tìm thấy mặt trên bank Citizen (con số).
5. LOO nội miền 34 mẫu webcam (vẫn 12D, chưa có nhóm mới) phải giữ **32/34 = 94.1%**.
6. `raw_strokes` schema đã tăng phiên bản và có `face_box`;
   `rebuild_templates_from_raw.py` đọc được **cả hai** phiên bản schema.

---

## TUYỆT ĐỐI KHÔNG LÀM trong spec này

- Không dùng `landmark.z` thô của MediaPipe làm đặc trưng độ sâu. Nó không đáng tin cậy.
  `palm_len` làm proxy (tức `z_norm`) là lựa chọn đã chốt.
- Không lật gương (mirror) bất cứ thứ gì — chirality đã được chuẩn hóa ở SPEC 00.
- Không thêm nhãn từ CNN vào đặc trưng dưới bất kỳ hình thức nào. Đã đo: có hại.
- Không đổi `lambda_shape`, `lambda_shape_global` khỏi 0.0.
- **KHÔNG tự ý dò/đặt trọng số tối ưu.** Việc đó thuộc SPEC 04. Ở spec này chỉ dùng
  đúng các giá trị khởi điểm đã cho.
- Không xóa cột nào trong 17 cột hiện có, không đổi thứ tự cột.
- Không sửa `improved_dip_pipeline.py` hay `train.py`.
