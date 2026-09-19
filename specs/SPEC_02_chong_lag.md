# SPEC 02 — Chống lag: đo trước, rồi bỏ việc vô ích ra khỏi vòng lặp

> Đọc `specs/_PREAMBLE.md` trước. Độc lập với SPEC 01, có thể làm song song
> **nếu** người điều phối xác nhận không có agent nào khác đang sửa
> `predict_sentence_webcam.py`. Ước lượng: 2–4 giờ.

## Mục tiêu

Tăng FPS và giảm độ trễ, dựa trên **đo lường**, không phỏng đoán. Thắng lợi lớn nhất
đã được chứng minh bằng số liệu: **DIP tensor + CNN đang chạy mỗi frame nhưng kết quả
của nó không được engine DTW dùng.**

## Cơ sở đo lường (đã thực hiện, không cần làm lại)

LOO trên bank ASL Citizen 139 mẫu / 19 từ, với hàm khoảng cách thật của dự án:

| Cấu hình | Top-1 |
|---|---|
| `lambda_shape=0, lambda_shape_global=0` (hiện tại) | **79.9%** |
| `lambda_shape=0.2` | 74.8% |
| `lambda_shape=0.5` | 67.6% |
| `lambda_shape_global=0.3` | 58.3% |

Thêm nữa: độ đồng thuận nhãn C_Shape giữa các mẫu **cùng một từ** chỉ **45%** (từ
`HELLO`: 5 mẫu ra 5 nhãn khác nhau). Tỉ lệ frame `Unknown` chỉ 0.1% — nghĩa là CNN
**tự tin nhưng sai**, không phải không phân loại được.

Kết luận: nhãn dáng tay từ CNN đóng góp **0 (thậm chí âm)** cho nhận diện cử chỉ động.
Mọi chi phí CPU bỏ ra để tính nó trong vòng lặp động là chi phí vô ích.

---

## NHIỆM VỤ 1 — Đo thời gian theo tầng (LÀM TRƯỚC MỌI TỐI ƯU)

Trong `predict_sentence_webcam.py`, thêm cơ chế đo thời gian cho các tầng sau, mỗi
tầng một biến tích lũy riêng:

1. `t_mediapipe` — quanh lời gọi detector MediaPipe (tìm chuỗi `detect` trong vòng lặp chính).
2. `t_dip_cnn` — quanh toàn bộ khối bắt đầu từ chuỗi `run_dip_cnn = (frame_counter % 2 == 0)`
   cho tới hết nhánh `if run_dip_cnn:`.
3. `t_features` — quanh `compute_joint_flexion_angles` + `compute_palm_normal_vector`.
4. `t_dtw` — quanh lời gọi `dtw_engine.match(`.
5. `t_draw` — quanh toàn bộ phần vẽ overlay/HUD/trail, cho tới trước `cv2.imshow`.
6. `t_total` — toàn bộ một vòng lặp.

Yêu cầu:
- Dùng `time.perf_counter()`, **không** dùng `time.time()`.
- Giữ trung bình trượt trên **60 frame gần nhất** (dùng `collections.deque(maxlen=60)`).
- Thêm cờ dòng lệnh `--profile`. Khi bật: vẽ một bảng nhỏ ở góc màn hình liệt kê từng
  tầng theo **ms** và FPS thực đo, đồng thời in cùng thông tin ra console mỗi 60 frame.
- Khi `--profile` **tắt**: chi phí đo phải bằng 0 — bọc các lời gọi `perf_counter()` sau
  một biến boolean, đừng gọi rồi mới bỏ.

**Báo cáo bắt buộc**: bảng ms từng tầng ở trạng thái hiện tại (trước khi tối ưu gì),
chạy khoảng 300 frame, kèm FPS trung bình. Đây là mốc so sánh cho mọi việc phía sau.

---

## NHIỆM VỤ 2 — Cho phép TẮT DIP + CNN trong vòng lặp động

### Việc cần làm

1. Thêm khóa vào `CONFIG` (tìm chuỗi `"templates_path": "dynamic_templates.json"` để
   định vị dict `CONFIG`):
   ```
   "enable_static_shape_cnn": False,
   ```
   Và thêm cờ dòng lệnh `--with-cnn` để bật lại khi cần (mặc định = theo CONFIG = tắt).

2. Khi tắt, bỏ qua **toàn bộ** các việc sau trong vòng lặp:
   - `extract_smart_hand_crop(...)`
   - `compute_finger_count_from_defects(...)`
   - `process_lean_dip_tensor(...)`
   - forward CNN (`cnn_model(t_pt)`, `F.softmax`, `apply_soft_morphological_prior`)

   Và đặt `hand_shape = "Unknown"`.

3. **Không được xóa code đó.** Chỉ đưa vào nhánh `if`. Chế độ nhận diện chữ cái tĩnh
   (Stage 1 của đồ án) vẫn cần nó, và đó là một deliverable riêng.

4. Xử lý các biến trở thành `None`/không xác định khi tắt: `cached_box`, `hull_pts`,
   `valid_defects`, `n_fingers`, `cached_hand_shape`. Tìm mọi chỗ **vẽ** chúng và bọc
   `if ... is not None`. Nếu không làm, chương trình sẽ crash ở phần vẽ.

### Vì sao việc này ĐÚNG về mặt toán học (phải tự kiểm chứng, đừng tin suông)

`hand_shape = "Unknown"` chảy vào `GestureFrame` → `stroke_shapes` → số hạng `d_shape`
trong hàm khoảng cách. Trong `compute_golden_triad_dtw_distance`, tìm chuỗi `is_unknown`:
code **đã trung lập hóa** hình phạt khi một trong hai phía là `"Unknown"` (đặt về 0.0).
Thêm nữa `lambda_shape` mặc định là `0.0`. Và `_compute_dominant_shape` trả về
`"Unknown"` khi không có nhãn nào khác → nhánh `lambda_shape_global` bị bỏ qua.

**Hãy tự đọc code và xác nhận cả 3 điều trên trước khi làm.** Nếu có điều nào không
đúng như mô tả, **DỪNG VÀ BÁO CÁO**.

### Tiêu chí hoàn thành (bất biến rất mạnh, phải kiểm)

Viết script kiểm chứng: lấy 34 mẫu webcam + 139 mẫu Citizen, chạy
`compute_golden_triad_dtw_distance` **hai lần** — một lần với `hand_shapes` thật, một
lần với toàn bộ `hand_shapes` bị thay bằng `"Unknown"`. Với `lambda_shape=0.0` và
`lambda_shape_global=0.0`, khoảng cách phải **giống hệt nhau đến từng bit**.

Báo cáo: sai số tuyệt đối lớn nhất giữa hai lần chạy. **Yêu cầu: chính xác 0.0.**
Nếu khác 0, nghĩa là mô tả ở trên sai → DỪNG VÀ BÁO CÁO.

Và: bảng ms từng tầng **sau** khi tắt, so với mốc ở Nhiệm vụ 1.

---

## NHIỆM VỤ 3 — Chạy MediaPipe trên frame đã thu nhỏ

### Việc cần làm

1. Thêm vào `CONFIG`: `"detect_scale": 0.5`.
2. Trước khi đưa frame vào MediaPipe, thu nhỏ bằng `cv2.resize(frame, None, fx=s, fy=s,
   interpolation=cv2.INTER_AREA)`. Chỉ frame đưa vào **detector** được thu nhỏ.
3. **Điểm then chốt**: landmark MediaPipe trả về ở dạng **chuẩn hóa [0, 1]** so với
   ảnh đầu vào. Cùng một vị trí vật lý sẽ cho **cùng** giá trị chuẩn hóa dù ảnh to hay
   nhỏ. Nên khi nhân ra pixel, **vẫn nhân với `w_f`, `h_f` của frame GỐC** — không cần
   chia lại cho `detect_scale`. Đừng "sửa" thêm gì, sẽ sai.
4. **Giữ frame GỐC** cho: `extract_smart_hand_crop`, `lk_fallback.update_reference`,
   `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)`, và toàn bộ phần vẽ. Chỉ detector dùng
   frame nhỏ.
5. Nếu `detect_scale >= 0.999` thì bỏ qua resize hoàn toàn (tránh chi phí vô ích).

### Tiêu chí hoàn thành

- Báo cáo `t_mediapipe` ms trước và sau.
- Kiểm chứng độ chính xác landmark: giữ tay **bất động** trước camera, thu 100 frame ở
  `detect_scale=1.0` và 100 frame ở `0.5`, so sánh `palm_len` trung bình và vị trí cổ
  tay trung bình. Báo cáo sai lệch theo **phần trăm**. Nếu lệch quá **2%**, hạ
  `detect_scale` lên 0.75 và đo lại.
- Nếu môi trường không có camera: nói rõ là **không kiểm chứng được**, và thay bằng
  kiểm chứng trên file video có sẵn trong `ASL_Citizen_videos/` (lấy 1 video, chạy
  detector ở 2 mức scale, so sánh landmark từng frame).

---

## NHIỆM VỤ 4 — Giảm chi phí vẽ

1. Tìm chuỗi `show_features = True` trong `predict_sentence_webcam.py` và đổi mặc định
   thành `False`. (Phím `d` vẫn bật/tắt được như cũ — không đổi phần đó.)
   Làm tương tự trong `collect_dynamic_templates.py`.
2. Đo `t_draw` trước/sau. Nếu `t_draw` vẫn là tầng tốn nhiều thứ 2 sau MediaPipe, báo
   cáo con số và **dừng lại ở đó** — đừng tự ý tái cấu trúc phần vẽ.

---

## Tiêu chí hoàn thành TOÀN SPEC

Một bảng duy nhất, đo trên **cùng một điều kiện** (cùng số frame, cùng ánh sáng nếu có
camera; nếu không có camera thì cùng một video từ `ASL_Citizen_videos/`):

| Tầng | Mốc ban đầu (ms) | Sau NV2 | Sau NV3 | Sau NV4 |
|---|---|---|---|---|
| mediapipe | | | | |
| dip_cnn | | | | |
| features | | | | |
| dtw | | | | |
| draw | | | | |
| **total** | | | | |
| **FPS** | | | | |

Cộng thêm:
- Sai số bất biến `hand_shapes` thật vs `"Unknown"`: phải **0.0**.
- Diff danh sách test FAILED: phải **RỖNG**.

---

## TUYỆT ĐỐI KHÔNG LÀM trong spec này

- **KHÔNG tái cấu trúc sang đa luồng (threading/multiprocessing).** Không thuộc spec
  này. Nếu bạn nghĩ nó cần, hãy **báo cáo đề xuất** chứ đừng làm.
- Không xóa code DIP/CNN — chỉ bọc vào nhánh `if`.
- Không đổi `lambda_shape` hay `lambda_shape_global` khỏi `0.0`.
- Không đổi `pause_frames`, `grace_window_sec`, hay bất kỳ tham số segmenter nào
  (đó là SPEC 01).
- Không thay model MediaPipe sang biến thể khác (`hand_landmarker.task` giữ nguyên).
- Không sửa `improved_dip_pipeline.py` — file bất biến.
