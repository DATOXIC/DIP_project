# SPEC 04 — Dò trọng số và ngưỡng TRỰC TIẾP trên template webcam của người dùng

> Đọc `specs/_PREAMBLE.md` trước. **Điều kiện tiên quyết:** SPEC 05A đã xong VÀ người
> dùng đã thu xong bank webcam (SPEC 05) vào `webcam_templates.json`, kèm
> `negative_strokes.json`. **Nên chạy SPEC 06A trước** để có `webcam_templates_augmented.json`
> (nếu chưa có, SPEC 04 vẫn chạy được trên bank gốc, nhưng dễ overfit hơn). Ước lượng: 4–6 giờ.

## Quyết định của người dùng (đã chốt, không bàn lại)

**Bank production = chỉ template webcam của người dùng.** ASL Citizen không còn là template,
không tham gia hàm mục tiêu, không tham gia chọn trọng số hay ngưỡng. Mọi bước dò tác động
**trực tiếp** lên bank webcam — đúng bank sẽ dùng để so khớp lúc demo.

File ASL Citizen (`asl_citizen_templates*.json`) vẫn được giữ và có thể dùng lại cho việc
khác, nhưng spec này **không đọc chúng**, trừ mục "Phụ lục: kiểm tra độc lập" (tùy chọn,
chỉ báo cáo).

## Mục tiêu

1. Chọn bộ trọng số global tốt nhất cho bank webcam.
2. Chọn chế độ trọng số: per-sign ASL-LEX, global đã dò, hay pha trộn.
3. Chọn ngưỡng per-word.
4. Báo cáo **độ chính xác kỳ vọng thật sự** (không bị thổi phồng) và **tỉ lệ nhận nhầm**
   trên cử chỉ ngoài từ vựng.

## Rủi ro chính: overfit (BẮT BUỘC đọc)

Bank chỉ có khoảng 10 từ × 8 mẫu = 80 mẫu, mà có ~8 trọng số cần dò. Dò và đánh giá trên
cùng những mẫu đó cho ra con số đẹp vô nghĩa. Ba biện pháp bắt buộc:

1. **Tách theo phiên thu, không tách ngẫu nhiên.** Mỗi mẫu có `session_id` (SPEC 05A).
   Cross-validation "giữ lại một phiên" (leave-one-session-out): dò trên các phiên còn
   lại, đánh giá trên phiên bị giữ lại, luân phiên.
   - Nếu chỉ có **1 phiên**: tách theo khối trong từng từ (mẫu số 1–4 là tập dò, 5–8 là
     tập kiểm; rồi đổi vai). Báo cáo rõ đây là đánh giá yếu hơn vì cùng phiên.
   - Nếu có **>= 2 phiên**: dùng leave-one-session-out. Đây là phép đo đáng tin nhất
     vì phiên khác nhau khác về ánh sáng, tư thế, tốc độ.
2. **Quy tắc một sai số chuẩn (1-SE).** Không chọn cấu hình có margin cao nhất. Chọn cấu
   hình **đơn giản nhất** (ít nhóm trọng số khác 0 nhất) có margin CV nằm trong 1 sai số
   chuẩn của cấu hình tốt nhất. Lý do: khác biệt nhỏ trên 80 mẫu là nhiễu.
3. **Không được dùng mẫu âm để dò trọng số.** Mẫu âm chỉ dùng ở bước ngưỡng và báo cáo.
4. **Nếu có `webcam_templates_augmented.json` (SPEC 06A):** biến thể chỉ được dùng ở phía
   **tham chiếu**, và **query luôn là mẫu gốc** (`is_original = true`) của phiên bị giữ lại.
   Khi một mẫu là query, phải loại **mọi** biến thể cùng `augment_of` với nó khỏi tham chiếu.
   Biến thể sinh từ stroke của phiên bị giữ lại **không được** có mặt phía tham chiếu.
   Kết quả top-1 gần 100% là dấu hiệu đã vi phạm quy tắc này.

## NHIỆM VỤ 1 — `tune_weights.py`

### Hàm mục tiêu

Tối ưu **margin**, không tối ưu top-1 (accuracy là hàm bậc thang, margin trơn và chính là
đại lượng mà logic ngưỡng tiêu thụ).

Với mỗi mẫu `i` của tập đánh giá, khi so với các mẫu của tập dò (chỉ khi khác `session`
hoặc khác khối):
- `d_correct(i)` = khoảng cách nhỏ nhất tới mẫu **cùng từ**
- `d_wrong(i)` = khoảng cách nhỏ nhất tới mẫu của **từ khác**
- `margin_norm(i) = (d_wrong(i) - d_correct(i)) / max(d_wrong(i), 1e-6)`

Mục tiêu = **trung bình `margin_norm`**. Báo cáo kèm top-1 và **margin của từ yếu nhất**
(min theo từ) — một từ tệ là một từ demo hỏng, trung bình che mất điều đó.

### Chống suy biến thang đo

`margin_norm` bất biến theo tỉ lệ, nên **cố định `w_a = 1.0`** làm mốc thang đo và chỉ dò
các trọng số còn lại: `w_p, w_m, w_n, w_z, w_b, w_s, w_r`. Ghi comment tiếng Việt giải thích.

### Không gian tìm kiếm (giữ nhỏ để chống overfit)

| Trọng số | Khoảng | Ghi chú |
|---|---|---|
| `w_p, w_m, w_n` | [0.0, 2.0] | |
| `w_z, w_b, w_s` | [0.0, 1.5] | |
| `w_r` | [0.0, 0.3] | phạt cộng thẳng; đo trên Citizen cho thấy có thể **có hại**, để nó dò tự do |
| `asllex_blend_alpha` | {0.0, 0.25, 0.5, 0.75, 1.0} | 0.0 = global đã dò; 1.0 = ASL-LEX thô |
| `flip_body_x` (cột 18) | {True, False} | xem "Câu hỏi mở" bên dưới |

### Thuật toán

1. Random search **300 điểm**. 2. Lấy 5 điểm tốt nhất, chạy coordinate descent (mỗi
trọng số thử `{×0.6, ×0.8, ×1.25, ×1.6}`, tối đa 6 vòng). 3. Áp quy tắc 1-SE ở mục trên.
4. In tiến độ để người dùng biết chưa treo.

Tăng tốc: cache theo nhóm nếu làm được; nếu phức tạp thì tính thô — với ~80 mẫu vẫn chạy
được. Báo cáo bạn chọn cách nào.

### Câu hỏi mở cần trả lời bằng dữ liệu (đừng đoán)

Agent SPEC 03 phát hiện trên Citizen: (a) trọng số **ASL-LEX per-sign cho 62.6%** trong khi
**global cho 79.9%**; (b) **không lật cột 18** khi lật chirality đo tốt hơn (MOTHER 4/4,
margin +0.195 so với +0.172 khi lật); (c) mốc cơ thể làm **ME xấu đi** (4/5 → 1/5) khi dùng
riêng. Cả ba đo trên miền khác. **Phải đo lại trên bank webcam** và báo cáo từng cái:
- `asllex_blend_alpha` nào tốt nhất trên webcam?
- `flip_body_x` True hay False? (Cần thêm một tùy chọn cộng thêm vào
  `canonicalize_chirality`, mặc định giữ hành vi hiện tại.)
- Nhóm `body` có làm hại ME/YOU trên webcam không? Nếu có, xem hộp mặt bằng mắt trước khi
  kết luận (vẽ hộp mặt lên vài frame từ `raw_strokes/` và lưu ảnh).

## NHIỆM VỤ 2 — Ngưỡng per-word và tỉ lệ nhận nhầm

Sau khi chốt trọng số:
1. Chạy `tune_thresholds.py` trên bank webcam với trọng số đã chốt (preview trước).
2. **Tỉ lệ nhận nhầm (false accept rate):** lấy `negative_strokes.json` (~20 cử chỉ
   ngoài từ vựng), chạy `match()` với ngưỡng đề xuất. Báo cáo bao nhiêu mẫu âm bị
   `ACCEPTED` (mục tiêu 0), bao nhiêu bị `UNCERTAIN`, bao nhiêu `REJECTED`.
3. Nếu có mẫu âm bị `ACCEPTED`: **hạ ngưỡng từ bị nhận nhầm** cho tới khi 0 nhận nhầm,
   rồi báo cáo cái giá phải trả: bao nhiêu mẫu dương của từ đó bị `UNCERTAIN`/`REJECTED`
   thêm. Đây là đánh đổi thật của demo — người dùng cần thấy con số để quyết định.
4. Chỉ ghi ngưỡng (`--apply`) sau khi người dùng xác nhận. Mặc định chỉ in xem trước.

## NHIỆM VỤ 3 — Ghi kết quả ra file MỚI

- `tuned_global_weights.json`: bộ trọng số, `asllex_blend_alpha`, `flip_body_x`, ngày dò,
  mô tả bank (số mẫu, số từ, các `session_id`), kiểu tách tập, giá trị mục tiêu đạt được,
  và cấu hình 1-SE đã chọn.
- `GoldenTriadDTWMatcher.__init__`: thêm `tuned_weights_path: Optional[str] = None` ở
  **cuối**. `None` → hành vi y như cũ.
- **TUYỆT ĐỐI KHÔNG sửa `ASL_ALEX_2.0/sign_weights_asllex.json`** (dữ liệu tra cứu ngôn
  ngữ học, không phải tham số dò được).

## Tiêu chí hoàn thành (bảng bắt buộc)

| Chỉ số | Cấu hình hiện tại (production) | Cấu hình đã dò |
|---|---|---|
| margin TB (kiểm định, CV) | | |
| top-1 (kiểm định, CV) | | |
| margin của từ yếu nhất | | |
| từ yếu nhất là từ nào | | |
| tỉ lệ nhận nhầm trên mẫu âm | | |
| tỉ lệ UNCERTAIN trên mẫu dương | | |

Kèm: bảng theo từ, kiểu tách tập đã dùng (leave-one-session-out hay theo khối), cấu hình
1-SE, và câu trả lời cho ba câu hỏi mở.

**Nếu margin CV của cấu hình đã dò KHÔNG cao hơn production một cách rõ ràng (vượt 1 sai
số chuẩn) thì báo cáo "không có cải thiện có ý nghĩa" và đề xuất giữ production.** Đừng áp
dụng chỉ vì spec nói phải dò.

## Phụ lục: kiểm tra độc lập trên ASL Citizen (TÙY CHỌN, chỉ báo cáo)

Chỉ làm nếu người dùng yêu cầu. Lấy cấu hình đã chốt, chạy trên `asl_citizen_templates_24d.json`
với các từ trùng, **chỉ để báo cáo** một con số "người ký khác". Không được phản hồi kết
quả này vào việc chọn trọng số. Lưu ý mẫu Citizen bắt đầu khi đang chuyển động (~49%) không
tương thích với quy tắc thu A — nên chỉ lấy các từ bắt đầu từ trạng thái đứng yên
(HELLO, YOU, THANK_YOU, DRINK) để phép so sánh có nghĩa.

## TUYỆT ĐỐI KHÔNG LÀM

- Không đọc hay dùng ASL Citizen trong hàm mục tiêu, chọn trọng số hay ngưỡng.
- Không dò trên tập rồi báo cáo kết quả trên chính tập đó.
- Không dò `lambda_shape` / `lambda_shape_global` — đã đo, giữ 0.0.
- Không dùng mẫu âm để dò trọng số.
- Không sửa `ASL_ALEX_2.0/sign_weights_asllex.json`.
- Không tự ý `--apply` ngưỡng hay đổi `CONFIG` production khi chưa được người dùng xác nhận.
