# SPEC 15 — Chế độ đánh vần chữ cái tĩnh (CNN DIP mở rộng)

> **TRẠNG THÁI THỰC HIỆN (2026-10-14): đã cài theo PHƯƠNG ÁN DỰ PHÒNG (landmark), không theo CNN như đặc tả dưới đây.**
> - Phím `s` trong `predict_sentence_webcam.py` bật/tắt chế độ đánh vần; `Backspace` xóa chữ cuối; hạ tay > 3 s tự kết thúc.
> - Bộ phân loại: `letter_landmark_model.pt` (MLP trên landmark 3D chuẩn hóa, `hand_pose_features.py`), **14 chữ**
>   `A B C D E F G H I K L O S W` + 2 lớp hình tay `1`, `5` (bỏ qua khi đánh vần). Không có lớp NONE.
> - Logic ghép chữ thuần Python ở `fingerspell.py` (nhiệm vụ A3), test ở `tests/test_fingerspell.py` (A6). Chữ được chốt khi giữ ổn định
>   0,7 s ở tin cậy ≥ 0,6; chữ đôi cần hạ tay hoặc đổi hình tay ≥ 0,35 s.
> - Kết quả là token `SPELL:<TÊN>`; `sentence_composer.py` ghép "My name is …" / "Tên tôi là …".
> - **Chưa làm:** A1 (`fingerspell_letters.json`), A2 (`collect_fingerspell.py`), A4 (`train_fingerspell.py`), cổng GO/NO-GO 15B,
>   đo độ chính xác trên người lạ. Dữ liệu chữ cái hiện chỉ từ một người ký (`letter_landmarks/DAT`); `C`/`V` đã được ghi nhận yếu ở phần dưới.
> - Phần còn lại của file này là đặc tả gốc, giữ làm lịch sử và làm cơ sở nếu sau này muốn quay lại hướng CNN.

> Đọc `specs/_PREAMBLE.md` trước. Dành cho agent **và người dùng** (có bước thu dữ liệu).
> Ước lượng: 15A agent 5–7 giờ → **Bước 15 người ~45–60 phút** → 15B agent 3–5 giờ.
> **Chạy SAU SPEC 14** (cùng sửa `predict_sentence_webcam.py`, `ui_side_panel.py`). SPEC 13 đã bị bỏ.
> Cấu trúc 3 giai đoạn giống SPEC 05A → 05 → 04: **đừng bắt người dùng thu mẫu trước khi 15A xong**
> (nếu thiếu bước nào ở 15A mà người dùng đã thu xong, họ sẽ phải thu lại).

## ĐỐI CHIẾU HIỆN TRẠNG VÀ RỦI RO (2026-10-02, đã kiểm tra code + `docs/SO_SANH_BASELINE.md`) — ĐỌC TRƯỚC

**Không trùng:** hiện KHÔNG có chế độ đánh vần/ghép chữ cái ở đâu trong dự án (đã grep). Nhưng có các sự thật sau làm
cổng GO/NO-GO của spec này **có nguy cơ cao không đạt**:

1. **CNN chữ cái hiện có yếu trong điều kiện thật.** Đo ngay trên ảnh đã train (511 ảnh): `V` chỉ đúng 63% (20/54 bị
   nhầm thành `L`), `C` 93.9% (6 → `L`); live (2026-09-30) dáng tay `C` "phần lớn hiện sai thành L hoặc B". Nhóm đã kết luận
   "lỗi nằm ở chính model", nghi ngờ thêm độ phân giải ảnh DIP 128×128 có đủ bắt số ngón dựng hay không.
2. **Nhóm đã tự thu hẹp phạm vi chữ cái** về `A, B, C, L, V` + `W` ("KHÔNG mở rộng toàn bộ 26 chữ"). Spec này đề xuất 12 chữ
   (thêm D F I O U Y), trái hướng đó nếu không có bằng chứng mới.
3. **Nhóm đã chuyển dáng tay sang đặc trưng góc ngón từ landmark** (`handshape_fusion.py`: góc cong 5 ngón, `angles` cột 4:9
   của vector 24D) và đo được có ích (cứu 11/13 lỗi DRINK↔WATER); CNN CIELab cũ đã bị tắt trong vòng lặp động
   (`enable_static_shape_cnn: False`). Tức hướng "landmark/góc ngón" đã có bằng chứng thực nghiệm, hướng "CNN DIP" thì chưa.
4. Hậu quả: nếu giữ nguyên hướng CNN 12 chữ, khả năng 15B ra **NO-GO** là đáng kể. Hai cách giảm rủi ro (người dùng chọn):
   (a) giữ CNN nhưng **thử nghiệm nhỏ trước** (6 chữ `A B C L V W` + `NONE`, đúng phạm vi nhóm khuyến nghị) rồi mới mở rộng;
   (b) chuyển sang bộ phân loại **góc ngón/landmark** (phương án dự phòng ở dưới, nhẹ, cần ít mẫu hơn, đã có bằng chứng).
   Nếu người dùng chưa quyết định, agent **KHÔNG tự chọn**: làm xong phần dùng chung (A1, A3, A5, A6, test) rồi dừng và hỏi.

## Bối cảnh — quyết định của người dùng (đã chốt)

- Mục tiêu: thêm **chế độ đánh vần** để ghi từ ngoài từ vựng 13 từ (tên riêng, từ lạ), bằng cách
  ký từng chữ cái **tĩnh**.
- **Hướng kỹ thuật đã chốt: MỞ RỘNG CNN DIP** — giữ `TwoStreamHandCNN` + tensor DIP 3 kênh
  (`extract_smart_hand_crop` → `process_lean_dip_tensor`) đúng chủ đề môn Xử Lý Ảnh Số, huấn luyện
  một model **MỚI, file riêng**, cho một **tập con chữ cái**. Phương án dự phòng (phân loại theo
  landmark MediaPipe) **chỉ** được chuyển sang khi người dùng đồng ý sau khi 15B báo cáo không đạt.

Hiện trạng cần biết:
- `vsl_dataset/models/best_model*.pth` chỉ có **5 lớp (A, B, C, L, V)**, 451 mẫu, val_acc 92.3% (v3).
  **KHÔNG sửa/ghi đè** các file này, `vsl_dataset/`, `train.py`, `improved_dip_pipeline.py`
  (PREAMBLE). Việc import từ chúng là được.
- CNN hiện **TẮT** trong vòng lặp từ động (`CONFIG["enable_static_shape_cnn"] = False`,
  `lambda_shape = 0.0`) vì khi tay đang chuyển động nhãn CNN chỉ nhất quán 30–49%. Chế độ đánh vần
  khác: tay **giữ yên** nên CNN đáng tin hơn — nhưng phải **đo**, không giả định.
- `predict_webcam.py` đã có nhận dạng chữ cái tĩnh với bộ bỏ phiếu `SlidingWindowVoter`
  (cửa sổ 7 khung) — tái sử dụng ý tưởng và lớp này.
- `train.py` chia train/val/test **ngẫu nhiên theo mẫu** (`stratified_split`). Với ảnh webcam liên
  tiếp gần giống hệt nhau, cách chia này gây **rò rỉ** (val cao giả tạo). Dự án đã có nguyên tắc
  "tách theo phiên thu" (SPEC 05A). Spec này BẮT BUỘC chia **theo phiên**.

### Chữ cái được chọn (tập con ban đầu, 12 chữ + 1 lớp nền)

`A B C D F I L O U V W Y` và lớp `NONE` (tay thả lỏng / đang chuyển / dáng không phải chữ nào).

Lý do loại: `J`, `Z` (có chuyển động); `E S T M N` (đều là nắm tay, rất dễ nhầm với `A`);
`G H P Q` (hướng ngang/chúc xuống, khác phân bố ảnh sau chuẩn hóa xoay); `K R X` (gần `V`/`U`/`D`).
`U` và `V` **được giữ nhưng nằm trong danh sách theo dõi nhầm lẫn** (xem 15B). Danh sách lưu trong
`fingerspell_letters.json` (một nguồn duy nhất) để đổi được mà không sửa code.

---

# GIAI ĐOẠN 15A — Chuẩn bị (AGENT, không cần dữ liệu)

## NHIỆM VỤ A1 — `fingerspell_letters.json` và cấu trúc dữ liệu riêng

1. File mới `fingerspell_letters.json`:

   ```json
   {
     "schema_version": "fingerspell_letters_v1",
     "letters": ["A","B","C","D","F","I","L","O","U","V","W","Y"],
     "none_class": "NONE",
     "descriptions": { "A": "...", "...": "..." },
     "watch_confusions": [["U","V"], ["D","F"], ["A","O"]]
   }
   ```
   `descriptions`: mô tả ngắn dáng tay ASL bằng tiếng Việt không dấu (cho HUD thu mẫu). Chữ nào
   bạn không chắc đúng chuẩn ASL ⇒ ghi `"Xem huong dan ASL"`, **không bịa**.
2. Dữ liệu chữ cái đặt ở thư mục **MỚI** `fingerspell_dataset/` (cấu trúc
   `fingerspell_dataset/samples/<LETTER>/*.npy`), **không** dùng `vsl_dataset/` (model 5 lớp đang
   phụ thuộc vào nó). Thêm `fingerspell_dataset/` vào `.gitignore` **chỉ nếu** người dùng muốn —
   mặc định KHÔNG đụng `.gitignore`; hỏi/ghi chú trong báo cáo.
3. **Tên file bắt buộc chứa phiên:** `webcam_<session_id>_<LETTER>_<NNNNN>.npy`
   (ví dụ `webcam_20260922_201500_A_00012.npy`). Tên phải chứa chuỗi `webcam` để
   `VSLDataset._is_webcam_sample` nhận ra. `session_id` dạng `YYYYMMDD_HHMMSS` hoặc tên do người
   dùng đặt qua `--session`.

## NHIỆM VỤ A2 — Công cụ thu mẫu chữ cái `collect_fingerspell.py` (file MỚI)

Tách khỏi `collect_webcam.py` để **không làm vỡ** collector 5 lớp. Có thể **import và tái dùng**
hàm từ `collect_webcam.py`/`improved_dip_pipeline.py` (chất lượng mẫu, crop, tensor DIP, mediapipe)
— đọc `run_webcam_collector` trước, bám đúng quy trình DIP đã có (cùng bước crop, cân bằng trắng,
xoay trục chính, tensor 3 kênh, **cùng lật tay trái → phải**), để mẫu thu và mẫu lúc chạy **giống hệt**
về tiền xử lý.

Yêu cầu:
1. Đọc danh sách chữ từ `fingerspell_letters.json`. Chọn lớp bằng **chính phím chữ cái** tương ứng
   (nhấn `a` để thu chữ A, ...) nếu không xung đột với phím điều khiển của collector; nếu xung đột,
   dùng phím khác nhưng **ghi rõ bảng phím trên HUD** và trong `HUONG_DAN_THU_CHU_CAI.md`.
   Có một phím riêng cho lớp `NONE`.
2. Cờ `--session <tên>` (mặc định `YYYYMMDD_HHMMSS` lúc khởi động) — ghi vào **tên file** (A1.3) và
   hiện trên HUD kèm số mẫu của phiên theo từng chữ.
3. Chế độ thu: giữ phím cho chữ cần thu, hệ thống tự lưu mỗi ~0.15 s khi **tay ổn định** (độ rung
   thấp, không chạm mép, độ nét đạt) — **tái dùng các thước đo chất lượng mẫu có sẵn** của
   `collect_webcam.py` (độ nét cạnh, ổn định, an toàn mép). Chống ghi đè (dùng
   `DIPDatasetBuilder.save_sample` có sẵn hoặc cơ chế tương đương không bao giờ ghi đè).
4. HUD (chữ không dấu): chữ đang thu, mô tả dáng tay, **số mẫu của phiên × từng chữ** và **tổng toàn
   bộ**, cảnh báo nếu chữ nào < mục tiêu (80 mẫu/chữ **tổng**, ≥ 2 phiên).
5. Phím `u` hoàn tác mẫu cuối (xóa file vừa lưu) — **chỉ** file do chính phiên này tạo.
6. Không bao giờ ghi vào `vsl_dataset/`.
7. Viết `HUONG_DAN_THU_CHU_CAI.md` (tiếng Việt, ngắn): quy trình thu (xem Bước 15 dưới), cách đặt
   tay/ánh sáng, danh sách dáng tay theo `descriptions`, mục tiêu số mẫu.

## NHIỆM VỤ A3 — Logic thuần Python `fingerspell.py` (không cv2/mediapipe/torch)

File **mới**. Test được không cần webcam/model.

```python
class LetterHoldDetector:
    def __init__(self, hold_sec=0.6, conf_min=0.80, release_sec=0.25, repeat_hold_sec=1.4,
                 none_label="NONE"): ...
    def update(self, label: Optional[str], conf: float, now: float, hand_present: bool) -> Optional[str]: ...
    def progress(self, now: float) -> float: ...        # 0..1 tiến độ giữ yên cho HUD
    @property
    def candidate(self) -> Optional[str]: ...           # chữ đang được giữ (chưa chốt)
    def reset(self) -> None: ...

class SpellBuffer:
    def __init__(self, max_len: int = 24): ...
    def add(self, letter: str) -> bool: ...
    def backspace(self) -> Optional[str]: ...
    def commit(self) -> Optional[str]: ...              # trả từ (HOA) rồi xóa buffer; rỗng -> None
    def clear(self) -> None: ...
    @property
    def text(self) -> str: ...
    def __len__(self) -> int: ...
```

Hành vi `LetterHoldDetector.update` (mỗi dòng cần test):
1. Chỉ coi một nhãn là **ứng viên** khi `hand_present`, `label` không `None`, không phải `NONE`,
   và `conf >= conf_min`. Mỗi khung không thỏa ⇒ ứng viên bị hủy ngay (đồng hồ giữ về 0).
2. Ứng viên giữ **liên tục ≥ `hold_sec`** (cùng một nhãn, theo dấu thời gian thật `now`, **không**
   đếm khung) ⇒ trả chữ đó **một lần** (chốt).
3. Sau khi chốt, **không** chốt lại cùng chữ cho tới khi **nhả**: nhãn đổi, hoặc `NONE`, hoặc mất
   tay, kéo dài ≥ `release_sec` (cho phép chữ đôi `LL`: nhả rồi giữ lại). Ngoại lệ: nếu giữ **nguyên
   một chữ** thêm ≥ `repeat_hold_sec` kể từ lần chốt thì chốt lặp một lần (chữ đôi mà không nhả);
   mỗi `repeat_hold_sec` chốt thêm một lần.
4. Đổi nhãn giữa chừng (A→B) ⇒ ứng viên mới bắt đầu đếm lại từ 0.
5. Mất tay ⇒ hủy ứng viên; `progress()` về 0.
6. Dấu thời gian không tăng (`now` lùi) ⇒ không ném lỗi, coi như không đổi.
7. Tất định, không dùng đồng hồ hệ thống bên trong (nhận `now` từ ngoài).

`SpellBuffer`: `add` chỉ nhận chữ A–Z (một ký tự), từ chối khi đầy; `backspace` bỏ chữ cuối (rỗng ⇒
`None`); `commit` trả chuỗi HOA rồi xóa; `text` hiển thị dạng liền (`CAT`).

**Hàm tiện ích** trong cùng file: `spell_key(word: str) -> str` trả `"SPELL:" + word.upper()`;
`is_spell_key(key) -> bool`; `spell_display(key) -> str` trả dạng `Title` (`SPELL:ANNA` → `Anna`).
(SPEC 14 dùng chung quy ước token này.)

## NHIỆM VỤ A4 — Script huấn luyện `train_fingerspell.py` (file MỚI) — chia theo PHIÊN

Import (không sửa) `TwoStreamHandCNN`, `VSLDataset`, các hàm train/eval từ `train.py`
(kiểm tra rằng `import train` **không** chạy huấn luyện — `main()` nằm sau
`if __name__ == "__main__":`; nếu import có tác dụng phụ, DỪNG VÀ BÁO CÁO).

Yêu cầu:
1. Đọc `fingerspell_dataset/samples/<NHÃN>/*.npy`; nhãn = 12 chữ + `NONE`. Suy `session_id` từ tên
   file (`webcam_<session_id>_<LETTER>_<NNNNN>.npy`).
2. **Chia theo phiên:** dùng cờ `--val-sessions`, `--test-sessions` (danh sách tên phiên). Mặc định
   (không chỉ định): phiên **muộn nhất** = test (niêm phong, chỉ chạy một lần ở cuối cùng bằng cờ
   `--final-test`), phiên muộn thứ hai = val, còn lại = train. Nếu < 3 phiên: **in cảnh báo rõ**
   "chưa đủ phiên để chia sạch; kết quả val có thể lạc quan" và vẫn chạy (không crash).
   In bảng: số mẫu mỗi chữ × {train, val, test}.
3. **Không** rò rỉ: assert không file nào xuất hiện ở hơn một tập; không phiên nào ở hơn một tập.
4. Dùng augmentation sẵn có của `VSLDataset` (đã loại lật ngang). **Không lật ngang** (tay trái đã
   được chuẩn hóa về tay phải ở khâu crop).
5. Cân bằng lớp (trọng số theo nghịch tần suất hoặc sampler); lớp `NONE` không được áp đảo.
6. Lưu checkpoint **cùng schema** với `best_model_v2.pth` (khóa: `epoch`, `model_state`,
   `num_classes`, `id_to_label`, `val_acc`, `args`) **cộng** `letters` (danh sách chữ), `none_label`,
   `split_info` (phiên của từng tập), `seed`, vào **`vsl_dataset/models/fingerspell_v1.pth`**.
   **Chống ghi đè:** nếu file đã có, ghi `fingerspell_v1_<timestamp>.pth` và báo. Không đụng
   `best_model*.pth`.
7. Xuất `docs/FINGERSPELL_EVAL.md` (tiếng Việt): bảng độ chính xác **từng chữ** trên val (và test khi
   `--final-test`), **ma trận nhầm lẫn**, các cặp nhầm hàng đầu, kiểm tra riêng các cặp trong
   `watch_confusions`, và độ trễ suy luận.
8. Seed cố định; chạy lại cho kết quả giống (cùng seed ⇒ cùng độ chính xác ± nhiễu số thực).

## NHIỆM VỤ A5 — Tích hợp vào `predict_sentence_webcam.py` và `ui_side_panel.py`

Định vị bằng chuỗi:

| Việc | Chuỗi grep |
|---|---|
| Nạp model tĩnh hiện có (tham khảo) | `model_path = 'vsl_dataset/models/best_model_v2.pth'` |
| Phím điều khiển | `elif key in [ord('j'), ord('J')]:` (thêm nhánh mới cạnh các nhánh phím) |
| Xử lý Swipe-Left | `if is_swipe:` |
| Khối xử lý cử chỉ hoàn thành | `if completed_stroke is not None:` |
| Cấu hình | `"camera_health": {` |
| Trích tay + tensor DIP | `from improved_dip_pipeline import` (trong `predict_webcam.py` có sẵn cách dùng `extract_smart_hand_crop` + `process_lean_dip_tensor`) |
| Panel giao diện thật | `class SidePanelState:`, `def draw_bottom_panel(`, `def draw_right_panel(` trong `ui_side_panel.py`; nơi dựng: `panel_state = ui_side_panel.SidePanelState(` trong `predict_sentence_webcam.py` |
| Nhật ký sự kiện | `event_log.push(` (nhiều chỗ) |
| `get_asl_phrase` | `def get_asl_phrase(` |

1. **Cấu hình** (cuối `CONFIG`):

   ```python
   "fingerspell": {
       "enabled": True,                        # cho phép vào chế độ đánh vần bằng phím [s]
       "model_path": "vsl_dataset/models/fingerspell_v1.pth",
       "hold_sec": 0.6, "conf_min": 0.80, "release_sec": 0.25, "repeat_hold_sec": 1.4,
       "commit_idle_sec": 1.0,                 # tay vắng mặt ngần này giây => chốt từ đang đánh vần
       "infer_every_n_frames": 2,              # chỉ suy luận CNN mỗi N khung
       "speak_mode": "word",                   # "word" = đọc cả từ | "letters" = đọc từng chữ cái
   },
   ```
   Cờ `--fingerspell / --no-fingerspell` (`BooleanOptionalAction`, mặc định `None`).
2. **Phím `s`** bật/tắt chế độ đánh vần (kiểm `ord('s')` đang trống — nếu đã dùng, DỪNG VÀ BÁO
   CÁO). Vào chế độ:
   - Nếu file model không tồn tại hoặc không nạp được ⇒ **không vào**, hiện toast/banner
     "Chua co mo hinh danh van (fingerspell_v1.pth)" và in cảnh báo; **không crash**, **không**
     ảnh hưởng phiên đang chạy.
   - **Nạp model lười (lazy)** lần đầu vào chế độ (không nạp lúc khởi động ⇒ khởi động và FPS chế độ
     từ **không đổi**). `torch.no_grad()`, `model.eval()`, thiết bị `cpu`.
   - Vào chế độ: hủy stroke đang ghi, xóa `trail_points`, và hủy câu hỏi xác nhận đang chờ của
     `--confirm-enroll` (`enroller.clear()` nếu `enroller is not None`).
   - **Xung đột phím:** `--confirm-enroll` (mặc định BẬT) dùng `y/2/n/z`; chế độ đánh vần dùng `s`, Backspace, Enter ⇒
     không trùng. Trong chế độ đánh vần, nhánh phím của enroller **không được kích hoạt** (nhiều chữ `y/n/z` là phím điều
     khiển, không phải chữ cái, nên chỉ cần chắc chắn không đánh vần bằng bàn phím).
3. **Trong chế độ đánh vần:**
   - **Bỏ qua** xử lý stroke/DTW (đặt cờ `spell_mode` bảo vệ khối `if completed_stroke is not None:`
     và nhánh swipe; segmenter vẫn có thể chạy nhưng kết quả bị bỏ — ưu tiên cách **ít sửa nhất**
     mà không làm đổi hành vi khi `spell_mode == False`).
   - Mỗi `infer_every_n_frames` khung, nếu có tay: crop + tensor DIP (**cùng hàm** như lúc thu mẫu) ⇒
     CNN ⇒ `softmax` ⇒ `(label, conf)` ⇒ `LetterHoldDetector.update(...)`. Khi detector trả chữ ⇒
     `SpellBuffer.add`.
   - Phím **Backspace (mã 8)**: xóa chữ cuối của buffer (buffer rỗng ⇒ không làm gì).
     **Swipe-Left** trong chế độ này cũng xóa chữ cuối.
   - Phím **Enter (mã 13)** hoặc tay vắng mặt ≥ `commit_idle_sec` với buffer không rỗng ⇒ **chốt từ**:
     `word = buffer.commit()`; thêm vào câu: `sentence_words.append(spell_display(key))`,
     `sentence_keys.append(spell_key(word))`, `sentence_finalized = False`; banner/toast `accepted`;
     phát âm theo `speak_mode` (`"word"`: `speech_service.speak_word(key, ui_lang, spell_display(key))`;
     `"letters"`: đọc các chữ cách nhau dấu cách; luôn bọc try/except).
   - Phím `c` xóa cả câu **và** buffer đánh vần.
   - Phím `s` lần nữa: thoát chế độ; buffer còn chữ ⇒ **chốt** (không bỏ mất) rồi thoát.
4. **Không phá thứ khác:** `get_asl_phrase("SPELL:ANNA", lang)` phải trả `Anna` (hiện trả `default`/chính khóa) —
   thêm nhánh nhận `is_spell_key` (cộng thêm, không đổi hành vi với khóa thường). **Không sửa `ui_render.py`**
   (`DemoHUD` chỉ chạy ở chế độ `legacy_demo` không còn chọn được).
5. **Giao diện:** `SidePanelState` thêm các trường **cuối**, mặc định giữ giao diện cũ: `spell_mode: bool = False`,
   `spell_text: str = ""`, `spell_candidate: Optional[str] = None`, `spell_progress: float = 0.0`. Khi `spell_mode`:
   - **Panel dưới** (`draw_bottom_panel`): thay dòng `CAU: "..."` bằng `DANH VAN: C A _` (các chữ đã ghi, ký tự
     chờ là `_`) kèm chữ ứng viên và thanh tiến độ giữ yên; dòng phím tắt đổi thành `[Backspace] xoa  [Enter] chot  [s] thoat`.
     Khung viền đổi màu (`COL_WARN` → màu khác dễ phân biệt) để người dùng biết đang ở chế độ đánh vần.
   - **Panel phải:** thêm một khối nhỏ "CHU CAI" hiện top-3 chữ do CNN dự đoán và độ tin cậy (dùng lại kiểu thanh `_bar`
     của khối TOP-K ứng viên), thay cho khối DTW trong lúc đánh vần.
   - Mọi chữ trong panel viết **không dấu** (cv2.putText); **không** thêm khung lồng khung.
   - Vào/ra chế độ: `event_log.push(...)`.
6. **Tương thích SPEC 14:** chỉ cần token `SPELL:<CHỮ>` đúng quy ước; **không** import
   `sentence_composer` trong code đánh vần. Nếu SPEC 14 chưa có thì câu hiển thị gloss như thường.
7. **Hiệu năng:** đo và báo cáo thời gian DIP + CNN mỗi lần suy luận (ms, trung vị/p95) và FPS vòng
   lặp trong chế độ đánh vần. Mục tiêu: ≥ 15 FPS trong chế độ đánh vần; **FPS chế độ từ động không
   giảm** (theo `acceptance.json`, độ trễ tăng ≤ 20%).
8. Chế độ `--video` / headless: chế độ đánh vần không có phím, mặc định tắt; không được ném lỗi.

## NHIỆM VỤ A6 — Test (15A)

1. `tests/test_fingerspell.py` (thuần Python, **không cần model/webcam**), tối thiểu:
   - giữ đủ `hold_sec` ⇒ đúng một lần chốt; giữ thiếu ⇒ không chốt;
   - `conf < conf_min` ⇒ không chốt; nhãn `NONE` ⇒ không chốt;
   - đổi nhãn giữa chừng ⇒ đếm lại; mất tay ⇒ hủy;
   - sau chốt, giữ tiếp **không** chốt lại; nhả ≥ `release_sec` rồi giữ lại ⇒ chốt chữ đôi;
   - giữ liên tục ≥ `repeat_hold_sec` ⇒ chốt lặp đúng một lần mỗi chu kỳ;
   - `now` lùi/đứng yên ⇒ không ngoại lệ;
   - `SpellBuffer`: add/backspace/commit/đầy/ký tự không hợp lệ;
   - `spell_key`/`is_spell_key`/`spell_display` (`SPELL:ANNA` → `Anna`, khóa thường ⇒ `False`).
2. Test kiểm schema checkpoint **chỉ chạy nếu** `fingerspell_v1.pth` tồn tại (dùng
   `pytest.mark.skipif`): đủ khóa `model_state, num_classes, id_to_label, letters, none_label,
   split_info`.
3. **Hồi quy:** diff danh sách FAILED trước/sau **rỗng** (PREAMBLE).
4. Chứng minh `spell_mode == False` ⇒ hành vi y hệt cũ: chạy bộ phát lại headless có sẵn trước/sau,
   chuỗi sự kiện giống hệt.

## Tiêu chí hoàn thành 15A

- [ ] `fingerspell_letters.json`, `collect_fingerspell.py`, `HUONG_DAN_THU_CHU_CAI.md`,
      `fingerspell.py`, `train_fingerspell.py` đã có; **không file cấm nào bị sửa**.
- [ ] `tests/test_fingerspell.py` PASS; diff FAILED rỗng.
- [ ] `train_fingerspell.py` chạy thử được trên dữ liệu **tổng hợp** (vài mẫu giả cho vài chữ, tạo
      ở thư mục tạm của scratchpad, **không** ở `fingerspell_dataset/`) ⇒ không crash, in được
      bảng chia theo phiên, lưu được checkpoint đúng schema ở thư mục tạm.
- [ ] Phím `s` khi chưa có model ⇒ thông báo, không crash.
- [ ] Khởi động/FPS chế độ từ động không đổi; `spell_mode == False` ⇒ y hệt cũ.
- [ ] Báo cáo nêu rõ điều **không kiểm chứng được** (không có webcam/mediapipe trong môi trường
      agent ⇒ tích hợp trực tiếp chưa chạy thật).

---

# BƯỚC 15 — Thu mẫu (NGƯỜI DÙNG, ~45–60 phút) — chỉ làm sau khi 15A xong

Quy tắc đã chốt giống SPEC 05 (phương án A, tách theo phiên):

1. **≥ 2 phiên**, cách nhau (khác buổi hoặc khác ánh sáng/vị trí ngồi/nền). **Tốt nhất 3 phiên**
   (train/val/test sạch). Mỗi phiên dùng `--session <tên>` khác nhau (ví dụ `sang_den`, `toi_den`,
   `ban_cong`).
2. Mỗi chữ: **≥ 80 mẫu tổng** (≥ 25–30 mẫu/phiên), thay đổi nhẹ **góc cổ tay, khoảng cách, tay
   thuận/không thuận nếu bạn sẽ dùng cả hai**. Giữ dáng tay **đúng chuẩn ASL** (xem
   `HUONG_DAN_THU_CHU_CAI.md`) — mẫu sai nhãn làm hỏng model nhiều hơn thiếu mẫu.
3. Lớp **`NONE` ≥ 150 mẫu** gồm: bàn tay thả lỏng, bàn tay đang chuyển từ chữ này sang chữ khác,
   nắm hờ, xòe lỏng, các dáng không phải chữ nào. Đây là lớp chống "máy luôn đoán ra một chữ".
4. Không trộn mẫu cũ của `vsl_dataset/` vào; **không** xóa thư mục đó. Có thể **sao chép** các
   mẫu webcam A, B, C, L, V hiện có sang `fingerspell_dataset/` **nếu** muốn tái dùng — nhưng chúng
   không có nhãn phiên ⇒ đặt tên với `session_id = legacy_v1` và xem là **một phiên riêng**
   (chỉ để làm train, không dùng cho val/test).
5. Báo lại cho agent: tên các phiên và số mẫu từng chữ (HUD collector có hiện).

---

# GIAI ĐOẠN 15B — Huấn luyện, đo, cổng quyết định (AGENT, sau Bước 15)

## NHIỆM VỤ B1 — Huấn luyện và đo

1. Chạy `train_fingerspell.py` với chia theo phiên (val = phiên muộn thứ hai, test niêm phong).
2. **Cổng quyết định** (đặt trước, không chỉnh sau khi thấy kết quả):
   - **GO:** độ chính xác trung bình các chữ trên **val ≥ 85%**, **mỗi chữ ≥ 70%**, và với lớp
     `NONE` tỉ lệ nhận nhầm `NONE` → một chữ (false letter) **≤ 10%**.
   - Các cặp trong `watch_confusions`: báo cáo tỉ lệ nhầm hai chiều; cặp nào > 20% ⇒ liệt vào
     "đề xuất loại một chữ khỏi tập".
   - **Chỉ sau khi chốt cấu hình trên val** mới chạy `--final-test` **một lần** trên phiên niêm phong;
     báo cáo số đó trung thực dù thấp hơn val.
3. Nếu **NO-GO**: **không** cố vượt cổng bằng cách tinh chỉnh trên test. Báo cáo: chữ nào yếu, nghi
   nguyên nhân (thiếu mẫu, miền khác nhau giữa phiên, nhầm cặp), và liệt kê lựa chọn cho người dùng
   (thu thêm cho chữ yếu; loại chữ nhầm; hoặc chuyển phương án landmark — **chỉ khi người dùng
   đồng ý**).

## NHIỆM VỤ B2 — Đo trực tiếp phần cứng thật

1. Đo `ms` DIP + CNN mỗi lần suy luận và FPS chế độ đánh vần trên **máy người dùng** (hoặc báo rõ
   không đo được nếu agent không có webcam).
2. Nếu có thể chạy: nhờ người dùng thử đánh vần 5 từ chỉ gồm chữ trong tập
   (`A B C D F I L O U V W Y`): `COLD`, `WILD`, `OVAL`, `FLY`, `BOWL` — và ghi từ đúng/sai theo
   từng chữ. Báo cáo tỷ lệ chữ đúng ở **điều kiện thật** — số này quan trọng
   hơn val.

## NHIỆM VỤ B3 — Chốt cấu hình

1. Nếu GO: điền `fingerspell_v1.pth` làm mặc định (`CONFIG["fingerspell"]["model_path"]` đã trỏ sẵn).
2. Điều chỉnh `hold_sec` / `conf_min` **chỉ** dựa trên phiên val + thử trực tiếp (không trên test).
3. Cập nhật `specs/README.md` (mục SPEC 15) và `PROJECT.md` (mục Feature Inventory/Code Layout:
   thêm chế độ đánh vần, file mới, ràng buộc không sửa model 5 lớp).

## Tiêu chí hoàn thành 15B

- [ ] `docs/FINGERSPELL_EVAL.md` có bảng từng chữ, ma trận nhầm lẫn, kết luận GO/NO-GO theo cổng đặt
      trước, và số liệu test niêm phong (chạy đúng **một lần**).
- [ ] Nếu GO: đánh vần được các từ thử ở B2 với tỷ lệ chữ đúng được báo cáo thật.
- [ ] Chế độ từ động: FPS/độ trễ không đổi (≤ 20% theo `acceptance.json`); diff FAILED rỗng.

## Ngoài phạm vi (KHÔNG làm)

- Nhận diện **J, Z** (chữ động), toàn bộ 26 chữ, đánh vần hai tay; sửa/ghi đè model 5 lớp
  `best_model*.pth`; trộn dữ liệu chữ cái vào `vsl_dataset/`; tự động sửa chính tả / gợi ý từ.
- Đoán chữ khi tay đang chuyển động (CNN chỉ dùng khi tay **giữ yên**).
- Chuyển sang phương án landmark mà không có sự đồng ý của người dùng.

## Cách thử trực tiếp (người dùng, ~10 phút, sau 15B)

1. `py predict_sentence_webcam.py` → nhấn `s` → HUD chuyển sang đánh vần.
2. Giữ chữ `C` yên ~0.6 s → thấy `C _`; làm `A`, `B` (hoặc chữ trong tập) → `C A B`.
3. Nhấn Backspace → mất `B`. Nhấn Enter → từ `Ca` vào câu và được đọc.
4. Nhấn `s` để thoát → quay lại nhận từ động; ký `ME` → vẫn nhận bình thường.
5. Rút tay ≥ 1 s khi đang có chữ trong buffer → từ tự chốt.
