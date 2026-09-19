# SPEC 01 — Sửa Segmenter: pha nhấc tay không được thành cử chỉ

> Đọc `specs/_PREAMBLE.md` trước. Làm SPEC 00 trước spec này. Ước lượng: 3–5 giờ.
> **ĐÂY LÀ SPEC QUAN TRỌNG NHẤT.** Nó chặn việc thu mẫu webcam: mọi mẫu thu trước
> khi sửa xong đều là dữ liệu bẩn.

## Bối cảnh — 4 nguyên nhân độc lập

Triệu chứng người dùng: *"mỗi lần đưa tay lên để chuẩn bị làm động tác thì hệ thống
nhận đó là một cử chỉ, phải để nó dự đoán xong rồi mới được làm tiếp."*

Đã truy ra 4 nguyên nhân riêng biệt. Bằng chứng đo được: template webcam có
`duration` trung bình **1.62s** và `pos_span` **1.28**, trong khi bank ASL Citizen là
**0.43s** / **0.82**. Chênh lệch đó **không** phải do người ký khác nhau — đó là pha
nhấc tay đang nằm bên trong chính template. 4 mẫu `YOU` dài 3.0–3.6s là bằng chứng rõ nhất.

**Nguyên tắc tương thích bắt buộc cho toàn spec này:** mọi thay đổi hành vi FSM phải
**bật/tắt được bằng tham số hàm khởi tạo, mặc định = GIỮ HÀNH VI CŨ**, và chỉ được bật
lên ở `predict_sentence_webcam.py` + `collect_dynamic_templates.py`. Lý do: 49 test
đang chạy trên FSM cũ; đổi mặc định sẽ làm số test fail tăng lên.

---

## NHIỆM VỤ A — `reset()` để sót `prev_wrist` (bug thật, sửa 1 dòng)

### Vấn đề

Trong `dynamic_dtw_engine.py` tìm chuỗi:

```
def reset(self):
```

(trong class `KineticPauseSegmenter`). Nó xóa `state`, `buffer`, `pause_counter`,
`recent_wrists`, `_pending_deadline` — nhưng **KHÔNG xóa `self.prev_wrist`**.

Trong khi đó `predict_sentence_webcam.py` gọi `segmenter.reset()` khi mất tay
(tìm chuỗi `if hand_lost_frames >= 4:`). Nên khi người dùng hạ tay ra khỏi khung rồi
đưa lên lại ở vị trí khác, frame đầu tiên tính:

```
v_norm = |wrist_moi - wrist_cu_da_cu| / palm_len
```

→ một cú "dịch chuyển tức thời" khổng lồ → vượt `v_start_ratio = 0.08` ngay → `RECORDING`.

### Việc cần làm

Thêm `self.prev_wrist = None` vào `reset()`.

Đây là thay đổi hành vi **không** bật/tắt được — nó là sửa bug, không phải tính năng.
Nhưng vì nó có thể động tới test: chạy diff danh sách FAILED trước/sau theo đúng cách
trong PREAMBLE. Nếu có test **mới** fail vì thay đổi này, **DỪNG và BÁO CÁO** kèm tên
test và nội dung assert, đừng tự sửa test.

---

## NHIỆM VỤ B — Thêm trạng thái `READY` (cổng "đã sẵn sàng")

### Vấn đề

Tìm chuỗi:

```
if self.state == KineticState.IDLE:
```

Bên trong chỉ có duy nhất `if v_norm >= self.v_start_ratio:` → vào `RECORDING`.
Không có tiền đề nào. Động tác nhấc tay từ lòng lên vùng ký **luôn** vượt ngưỡng này.

### Việc cần làm

1. Thêm thành viên mới vào enum `KineticState` (tìm chuỗi `class KineticState`):
   ```
   READY = "READY"
   ```
   Enum này khai báo là `str, Enum` nên thêm thành viên là an toàn.

2. Thêm 2 tham số vào **cuối** `KineticPauseSegmenter.__init__`, đúng thứ tự này:
   ```
   require_ready_pose: bool = False,     # Mac dinh False = giu nguyen FSM cu cho tests
   ready_stable_frames: int = 8,         # Tay phai dung yen 8 frame (~0.25s @30fps) moi coi la san sang
   ```
   Lưu cả hai vào `self.`, và khởi tạo `self.ready_counter = 0`.

3. Sửa nhánh `IDLE` thành:
   - Nếu `self.require_ready_pose` là **False**: giữ **NGUYÊN Y HỆT** logic cũ. Không
     được thay đổi một ký tự nào của đường đi cũ.
   - Nếu **True**:
     - Nếu `v_norm <= self.v_stop_ratio`: `self.ready_counter += 1`. Khi
       `self.ready_counter >= self.ready_stable_frames` → `self.state = KineticState.READY`.
     - Ngược lại (`v_norm > v_stop_ratio`): `self.ready_counter = 0` và **Ở LẠI IDLE**.
       Đây là điểm cốt lõi: **cử động khi chưa sẵn sàng thì KHÔNG được bắt đầu ghi.**

4. Thêm nhánh `READY` mới (chỉ tồn tại khi `require_ready_pose=True`):
   - Nếu `v_norm >= self.v_start_ratio` → `self.state = KineticState.RECORDING`,
     `self.buffer = [curr_frame]`, `self.pause_counter = 0`.
   - Ngược lại: ở lại `READY` (tay vẫn đang chờ).

5. Trong `reset()`: thêm `self.ready_counter = 0`. State vẫn về `IDLE` như cũ.

6. Bật lên ở production: tại nơi khởi tạo `KineticPauseSegmenter(...)` trong
   `predict_sentence_webcam.py` và `collect_dynamic_templates.py`, truyền
   `require_ready_pose=True`.

7. **UI**: người dùng phải thấy được trạng thái này, nếu không họ sẽ tưởng máy treo.
   Ở `predict_sentence_webcam.py` tìm chuỗi `"KHUNG TAY"` — đó là chỗ vẽ nhãn trạng
   thái. Thêm nhãn cho `READY`, ví dụ `"SAN SANG"` màu xanh lá, khác rõ với `IDLE`
   (`"CHO TAY"`). Dùng chữ **không dấu** như các chuỗi `cv2.putText` khác trong file
   (OpenCV không vẽ được tiếng Việt có dấu).

---

## NHIỆM VỤ C — Grace window đang DÁN pha nhấc tay vào từ thật

### Vấn đề

Tìm chuỗi:

```
elif self.state == KineticState.PENDING_COMMIT:
```

Bên trong, khi `v_norm >= self.v_start_ratio` thì FSM quay lại `RECORDING` và **GỘP**
vào cùng một cử chỉ. Chuỗi thực tế của người dùng:

> nhấc tay (RECORDING) → ổn định vào vị trí (khựng 11 frame = 0.35s → PENDING_COMMIT)
> → bắt đầu ký thật trong vòng 0.20s → **gộp thành 1 stroke = [nhấc tay] + [từ thật]**

Nếu nghỉ lâu hơn 0.20s thì thay vào đó nhận được một cử chỉ rác. **Cả hai lối đều sai.**

Nhiệm vụ B đã chặn phần lớn vấn đề (pha nhấc tay không còn vào RECORDING nữa). Nhưng
vẫn cần chặn việc gộp vô hạn và gộp quá dài.

### Việc cần làm

1. Thêm 1 tham số vào **cuối** `__init__`:
   ```
   max_phase_merges: int = 999,   # Mac dinh 999 = gan nhu vo han = giu hanh vi cu
   ```
   Khởi tạo `self.merge_count = 0`.

2. Trong nhánh `PENDING_COMMIT`, tại chỗ gộp lại `RECORDING`, **chỉ gộp khi CẢ HAI
   điều kiện sau đều đúng**:
   - `self.merge_count < self.max_phase_merges`
   - Thời lượng buffer nếu gộp vẫn còn hợp lệ: `(timestamp - self.buffer[0].timestamp)`
     phải `<= STROKE_DURATION_MAX_SEC` (hằng số này đã có sẵn trong file — tìm chuỗi
     `STROKE_DURATION_MAX_SEC`).

   Khi gộp thành công: `self.merge_count += 1`.

   Khi **không** thỏa điều kiện gộp: coi như hết grace window → **chốt cử chỉ ngay**
   (đi theo đúng nhánh chốt đã có sẵn bên dưới), rồi để frame kế tiếp bắt đầu cử chỉ mới.

3. Reset `self.merge_count = 0` ở 2 chỗ: trong `reset()`, và mỗi khi một stroke được
   chốt xong (chỗ `self.state = KineticState.IDLE` sau khi gán `completed_stroke`).

4. Ở production truyền `max_phase_merges=1`. Lý do: từ vựng ASL trong dự án này tối đa
   2 pha (ví dụ `THANK_YOU`: chạm cằm rồi đẩy ra), tức tối đa 1 lần gộp.

5. Cũng ở production, giảm `grace_window_sec` từ `0.20` xuống `0.15`. **KHÔNG đổi giá
   trị mặc định trong `__init__`** — chỉ truyền giá trị khác ở chỗ khởi tạo.

---

## NHIỆM VỤ D — Loại chuyển động chuyển tiếp theo HÌNH DẠNG quỹ đạo

### Cơ sở

Pha nhấc tay là một đường **gần thẳng, đơn điệu, không đảo chiều**. Hầu như không có
stroke ASL nào như vậy. Trong ngôn ngữ học ký hiệu hiện tượng này gọi là
**movement epenthesis** (chuyển động chêm giữa các ký hiệu).

### Việc cần làm

1. Thêm hàm mới ở cấp module trong `dynamic_dtw_engine.py` (CỘNG THÊM, đặt gần
   `canonicalize_chirality`):

   ```
   def compute_trajectory_straightness(xs: np.ndarray, ys: np.ndarray) -> float
   ```
   - `path_length` = tổng `hypot` của các bước liên tiếp.
   - `net_displacement` = `hypot(xs[-1]-xs[0], ys[-1]-ys[0])`.
   - Trả về `net_displacement / path_length`. Nếu `path_length < 1e-6` trả về `0.0`.
   - Giá trị ~1.0 = đường thẳng tuyệt đối. Giá trị nhỏ = quỹ đạo có cong/đảo/lặp.

   ```
   def is_movement_epenthesis(stroke_frames, straightness_max: float,
                              min_path_palm_ratio: float = 0.8) -> bool
   ```
   - Lấy `xs`, `ys` từ `stroke_frames` (mỗi phần tử là `GestureFrame`, có `.x`, `.y`,
     `.palm_len`), chuẩn hóa bằng `palm_len` của frame ĐẦU TIÊN.
   - Chỉ coi là epenthesis khi **CẢ HAI**:
     - `path_length_da_chuan_hoa >= min_path_palm_ratio` (chặn trường hợp rung tay tại
       chỗ: quỹ đạo cực ngắn thì tỉ lệ straightness vô nghĩa), VÀ
     - `straightness > straightness_max`.
   - Docstring tiếng Việt giải thích movement epenthesis.

2. **KHÔNG gọi hàm này bên trong FSM.** Gọi ở phía production, tại chỗ xử lý stroke đã
   chốt. Ở `predict_sentence_webcam.py` tìm chuỗi:
   ```
   if is_stroke_duration_valid(stroke_duration):
   ```
   Thêm điều kiện loại epenthesis ngay sau cổng thời lượng. Khi bị loại: **không match,
   không thêm từ vào câu**, chỉ hiện một thông báo ngắn kiểu `"BO QUA: chuyen dong
   chuyen tiep"` rồi tiếp tục.
   Làm điều tương tự trong `collect_dynamic_templates.py` (không lưu mẫu, hiện toast đỏ).

3. **CÁCH CHỌN `straightness_max` — PHẢI DỰA TRÊN DỮ LIỆU, KHÔNG ĐƯỢC ĐOÁN.**

   Viết một script đo (đặt trong scratchpad) làm đúng các bước sau:
   - Nạp **toàn bộ** mẫu từ `asl_citizen_templates.backup_before_bye_delete_20260918_183236.json`
     (139 mẫu) và các mẫu webcam (không có `source_video`) từ
     `dynamic_templates.backup_before_12d_delete_20260918_212502.json` (34 mẫu).
   - Với mỗi mẫu, lấy `traj_features` cột 0 và 1 (đây là x, y **đã chuẩn hóa** sẵn) và
     tính `compute_trajectory_straightness`.
   - In bảng: mỗi từ một dòng, kèm min / trung bình / **max** straightness.
   - In ra **giá trị straightness LỚN NHẤT trên toàn bộ 173 mẫu ký hiệu thật**.

   Chọn `straightness_max = max_quan_sat_duoc + 0.03`, rồi **kẹp vào khoảng [0.85, 0.97]**.
   Nếu `max_quan_sat_duoc` vượt quá 0.94 → **DỪNG VÀ BÁO CÁO**: nghĩa là có ký hiệu
   thật gần như đường thẳng, và bộ lọc này sẽ gây hại. Đừng tự chọn bừa một số.

   Đặt giá trị chốt được thành hằng số module có tên rõ ràng, kèm comment tiếng Việt
   ghi lại **con số quan sát được** và **ngày đo**.

### Tiêu chí hoàn thành cho nhiệm vụ D

- **Không một mẫu nào** trong 173 mẫu ký hiệu thật bị `is_movement_epenthesis` trả về
  `True`. Đây là yêu cầu cứng — báo cáo con số `0/173`.
- In bảng straightness theo từ vào báo cáo.

---

## NHIỆM VỤ E — Dự đoán tạm để khử cảm giác chờ

### Vấn đề

`pause_frames = 11` (0.35s) + `grace_window` (0.15s) = **~0.50s** mới ra từ. Đó là
**độ trễ**, không phải tụt FPS. Hạ ngưỡng sẽ đánh đổi với việc gộp ký hiệu đa pha.

### Việc cần làm

Ở `predict_sentence_webcam.py`, khi FSM **vừa chuyển vào** `PENDING_COMMIT` (buffer đã
có đủ dữ liệu ở thời điểm đó), chạy match trên buffer hiện tại và hiển thị kết quả
dạng **tạm thời, màu xám nhạt, kèm chữ `?`**. Khi stroke được chốt thật, thay thế
bằng kết quả chính thức (màu như hiện tại).

Ràng buộc bắt buộc:
- Dự đoán tạm **KHÔNG được** thêm từ vào câu, **KHÔNG được** gọi TTS, **KHÔNG được**
  ghi vào `last_committed_word` / `last_commit_time`.
- Chạy **nhiều nhất 1 lần** cho mỗi lần vào `PENDING_COMMIT` (dùng một cờ boolean),
  không chạy mỗi frame — nếu không sẽ tụt FPS.
- Nếu match trả về `REJECTED`, không hiện gì cả (đừng nhá chữ "Unknown" gây nhiễu mắt).

---

## Tiêu chí hoàn thành TOÀN SPEC (số liệu bắt buộc báo cáo)

1. Diff danh sách test FAILED trước/sau: **phải RỖNG**. Nếu không rỗng, báo cáo tên
   từng test mới fail và nội dung assert của nó.
2. Hành vi FSM khi `require_ready_pose=False`, `max_phase_merges=999`: chứng minh
   **không đổi**. Cách chứng minh: viết một script phát một chuỗi frame tổng hợp
   (synthetic) qua segmenter với cấu hình mặc định, trước và sau khi sửa, và so sánh
   chuỗi `(state, completed_stroke is None, is_swipe_left)` từng frame — phải **giống
   hệt**. Lưu ý: chỉ nhiệm vụ A (`prev_wrist`) được phép làm lệch kết quả này; nếu
   lệch, phải chỉ rõ là do A.
3. Với `require_ready_pose=True`: viết script synthetic mô phỏng đúng tình huống lỗi —
   (a) tay xuất hiện đột ngột ở vị trí mới sau khi mất dấu, (b) một cú nhấc tay thẳng
   0.5s, (c) tay đứng yên 0.3s, (d) một cử chỉ thật có cong/đảo chiều. Yêu cầu: **chỉ
   (d) sinh ra `completed_stroke`**. Báo cáo kết quả từng tình huống.
4. `0/173` mẫu ký hiệu thật bị loại bởi bộ lọc epenthesis, kèm bảng straightness theo từ.
5. LOO nội miền webcam vẫn **>= 32/34 = 94.1%** (spec này không chạm hàm khoảng cách,
   nên con số phải không đổi — đây là phép kiểm tra "không làm hỏng gì").

---

## TUYỆT ĐỐI KHÔNG LÀM trong spec này

- **Không đổi giá trị mặc định** của `v_start_ratio`, `v_stop_ratio`, `pause_frames`,
  `min_stroke_frames`, `max_stroke_frames`, `grace_window_sec`, `swipe_*` trong
  `__init__`. Chỉ truyền giá trị khác ở chỗ khởi tạo phía production.
- Không chạm hàm `compute_golden_triad_dtw_distance` hay bất kỳ trọng số nào.
- Không thêm chiều đặc trưng nào (đó là SPEC 03).
- Không tái cấu trúc luồng (threading) — không thuộc spec này.
- Không sửa 49 test đang fail sẵn.
- Không xóa `KineticState.PAUSE_DETECTED` dù nó đã không còn dùng — có test tham chiếu tới nó.
