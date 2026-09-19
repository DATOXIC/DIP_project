# SPEC 04 (bản rút gọn) — Chọn cấu hình bằng bảng điểm, KHÔNG dò trọng số bằng tìm kiếm ngẫu nhiên

> Đọc `specs/_PREAMBLE.md` trước. Dành cho agent.
> **Điều kiện tiên quyết:** (1) SPEC 04A (`scoreboard.py`) đã xong và tự kiểm tra đạt;
> (2) người dùng đã thu xong bank webcam ở 30 FPS (SPEC 05B + 05), gồm `negative_strokes.json`
> và ít nhất 2 phiên. Nên có SPEC 06A (bank augmented) nhưng không bắt buộc.
> Ước lượng: 3–4 giờ.

## Quyết định của người dùng (đã chốt, không bàn lại)

- Bank production = **chỉ template webcam của người dùng**. ASL Citizen không tham gia dò
  trọng số, chọn cấu hình hay ngưỡng.
- Mọi so sánh đi qua `scoreboard.py`; so sánh **ghép cặp**; tiêu chí chấp nhận đặt **trước khi chạy**.
- Người dùng muốn `sign_weights_asllex.json` "thực sự có ích" — **nhưng chỉ nhận nó nếu bảng
  điểm chứng minh**. Không tin trước.

## Vì sao KHÔNG còn tìm kiếm ngẫu nhiên 300 điểm

Bank ~80 truy vấn, ~8 trọng số. Dò liên tục trong không gian đó dễ overfit và không kiểm
chứng được. Bằng chứng đo trước đó (Citizen, 139 mẫu): đòn bẩy lớn là **bật nhóm đặc trưng
nào** (margin trung bình 0.10 → 0.23 khi thêm z_norm + mốc cơ thể + độ xòe ngón), không phải
chỉnh 0.6 thành 0.7. Nên dò **một số ít cấu hình rời rạc, có tên, so sánh ghép cặp**.

## Bằng chứng về ASL-LEX (đã đo trước đó — đừng đo lại, đừng diễn giải quá mức)

Trên 10 từ của bank (mẫu Citizen làm đại diện, 68 mẫu), khi so trọng số nhóm của
`sign_weights_asllex.json` với dữ liệu:

| Phép đo | Kết quả | Ý nghĩa |
|---|---|---|
| Trọng số nhóm ASL-LEX vs nhóm thực sự tách từ (loại nhóm N) | tương quan hạng +0.05, p = 0.63 | không khác ngẫu nhiên |
| Độ giống âm vị ASL-LEX vs khoảng cách DTW giữa 36 cặp từ | tương quan +0.05, p = 0.61 | không dự đoán được cặp dễ nhầm |
| Nhấn mạnh nhóm mà ASL-LEX cho là khác nhau của **từng cặp** | Δ log(inter/intra) = −0.020, so với −0.022 khi gán nhãn ngẫu nhiên | không tốt hơn ngẫu nhiên; tốt cho SORRY–PLEASE (+0.15) nhưng **không giúp** các cặp khó nhất (YOU–WATER, THANK_YOU–PLEASE, ME–YOU) |
| Thay hoàn toàn bằng trọng số ASL-LEX (`asllex_blend_alpha = 1`) | top-1 62.6% so với 79.9% (bộ toàn cục) | có hại |

Giới hạn: 9/10 từ có `EntryID` (EAT không khớp), mẫu Citizen là đại diện, cỡ mẫu nhỏ. Đây là
**"không có bằng chứng hữu ích"**, không phải "chứng minh vô dụng". Cần đo lại trên bank
webcam — đó chính là việc của spec này.

---

## NHIỆM VỤ 0 — Tệp mục tiêu (người dùng có thể sửa TRƯỚC khi chạy)

Tạo `acceptance_targets.json` với giá trị mặc định dưới đây (người dùng chỉnh nếu muốn).
**Agent KHÔNG được sửa file này sau khi bắt đầu chạy.**

```json
{
  "top1_min": 0.90,
  "clean_accept_min": 0.80,
  "wrong_accept_max_rate": 0.02,
  "false_accept_neg_max": 1,
  "fps_min": 28
}
```

`false_accept_neg_max` tính trên tập mẫu âm (~20).

## NHIỆM VỤ 1 — Hai mốc và QUY TẮC DỪNG

Chạy trên bảng điểm (LOSO):
- **C0_production**: cấu hình production hiện tại, y nguyên.
- **C1_global**: trọng số toàn cục `w_p=0.6, w_a=1.0, w_m=0.4, w_n=0.8`, không nhóm mới,
  `asllex_mode = none`.

**Quy tắc dừng:** nếu C0 hoặc C1 đã đạt **mọi** mục trong `acceptance_targets.json` (trừ `fps_min`
đo riêng), thì **dừng ở đây**: chỉ chạy bước 6 (ngưỡng) và bước 7 (sealed) cho cấu hình đạt.
Báo cáo "không cần dò thêm". Đừng chạy các bước 2–5 chỉ vì spec có chúng.

## NHIỆM VỤ 2 — Ablation nhóm đặc trưng (so với C1, ghép cặp)

| Cấu hình | Khác C1 |
|---|---|
| C2_z | `w_z = 0.5` |
| C3_body | `w_b = 0.8` (thử `flip_body_x` True **và** False → C3a, C3b) |
| C4_abd | `w_s = 0.4` |
| C5_all | `w_z = 0.5, w_b = 0.8, w_s = 0.4` |
| C6_rep | `w_r = 0.15` (một mình) |

`flip_body_x`: nếu `canonicalize_chirality` hiện luôn lật cột 18 khi lật chirality, thêm một
tham số **cộng thêm** `flip_body_x: bool = True` (mặc định giữ hành vi hiện tại) và cho phép
cấu hình chọn. Đã có gợi ý từ bank Citizen rằng **không lật** có thể tốt hơn; cần đo trên webcam.

Mỗi cấu hình `compare` với C1 bằng `acceptance.json` của SPEC 04A. Ghi verdict
(ACCEPT / REJECT / INCONCLUSIVE) vào bảng tổng kết.

## NHIỆM VỤ 3 — Chọn cấu hình nhóm đặc trưng

Trong các cấu hình có verdict ACCEPT (so với C1), chọn cấu hình **đơn giản nhất** (ít nhóm
bật nhất). Nếu nhiều cấu hình ACCEPT với độ phức tạp bằng nhau, chọn cấu hình có
`margin_mean` cao hơn. Nếu **không có** cấu hình nào ACCEPT: giữ C1 (hoặc C0 nếu C0 tốt hơn C1
theo `compare`). Kết quả gọi là **C_best**.

## NHIỆM VỤ 4 — ASL-LEX như một bộ điều biến, thử có kiểm soát

### 4a. Cài chế độ `modulate` (CỘNG THÊM vào `GoldenTriadDTWMatcher`)

Chế độ mới `asllex_mode = "modulate"`, tham số `asllex_lambda` và `asllex_mapping`. Với từ ứng viên
`s` và nhóm khóa trọng số `k` (`w_p, w_a, w_m, w_n, w_b, w_s`):

```
share_g   = trọng số ASL-LEX của nhóm g cho từ s   (g ∈ {P, A, JM, N}, JSON đã cộng bằng 1)
factor_k  = clip( 1 + λ * (share_g(k) / 0.25 − 1),  0.5, 1.5 )
w_eff[k]  = w_global[k] * factor_k
```

rồi **chuẩn hóa lại để tổng `w_eff` bằng tổng `w_global`** (giữ nguyên thang khoảng cách, để
so sánh cộng bằng và ngưỡng không lệch). `w_z` và `w_r` không bị điều biến. `λ = 0` phải cho
kết quả **giống hệt bit** chế độ `none`.

Hai ánh xạ nhóm ASL-LEX → khóa trọng số:

| | P | A | JM | N |
|---|---|---|---|---|
| `legacy` (production hiện tại) | `w_p` | `w_a` | `w_m` | `w_n` |
| `semantic` | `w_b` (vị trí trên cơ thể) | `w_a`, `w_s` | `w_p`, `w_m` | `w_n` |

Chế độ `replace` hiện có (`asllex_blend_alpha`) **giữ nguyên**, chỉ dùng cho C0.
Từ nào không có trong JSON thì `factor = 1` cho mọi nhóm (không điều biến) — không được lỗi.

### 4b. Thí nghiệm

Trên C_best, chạy `λ ∈ {0.25, 0.5, 1.0}` × `mapping ∈ {legacy, semantic}` = 6 cấu hình. `compare`
từng cấu hình với **C_best** (không phải C1).

**Kết luận phải viết đúng theo verdict:**
- Có cấu hình ACCEPT → ASL-LEX có ích ở mức đo được; báo cáo λ và ánh xạ.
- Toàn INCONCLUSIVE/REJECT → **ASL-LEX không thắng trọng số học từ dữ liệu**; C_best giữ nguyên
  `asllex_mode = none`. Đây là một kết quả hợp lệ và đáng đưa vào báo cáo. **Đừng chọn cấu hình
  "nhìn có vẻ tốt hơn" khi verdict là INCONCLUSIVE.**

## NHIỆM VỤ 5 — Phân tích ASL-LEX làm nội dung báo cáo (không ảnh hưởng chọn cấu hình)

Chuyển ba script phân tích đã dùng ở phiên trước vào dự án: thư mục `analysis/`
(`prior_vs_data.py`, `lex_pairs.py`, `pairwise_lex.py` — người dùng hoặc điều phối viên sẽ cung
cấp; nếu chưa có, viết lại theo mô tả ở bảng "Bằng chứng về ASL-LEX" phía trên). Sửa để đọc **bank
webcam** thay vì Citizen, chạy lại, và ghi 3 con số (tương quan trọng số, tương quan cặp,
Δ so với ngẫu nhiên) vào `results/asllex_analysis.md`. Thêm: đối chiếu độ giống ASL-LEX của từng
cặp với **ma trận nhầm lẫn thực đo** của C_best trên bảng điểm.

## NHIỆM VỤ 6 — Ngưỡng và tỉ lệ nhận nhầm

Trên C_best:
1. Chạy `tune_thresholds.py` (xem trước, không `--apply`) với bank webcam và trọng số của C_best.
2. **Nhận nhầm mẫu âm:** chạy bảng điểm với ngưỡng đề xuất. Mục tiêu theo `false_accept_neg_max`.
3. Nếu vượt: hạ ngưỡng từ bị nhận nhầm tới khi đạt, rồi báo cáo **cái giá**: bao nhiêu mẫu
   dương của từ đó chuyển sang UNCERTAIN/REJECTED thêm.
4. Chỉ ghi ngưỡng (`--apply`) **sau khi người dùng xác nhận**. Mặc định chỉ in xem trước.

## NHIỆM VỤ 7 — Kiểm tra kín (sealed) đúng MỘT lần

Khi (và chỉ khi) người dùng có phiên 3: chạy `scoreboard.py sealed` cho **một** cấu hình cuối cùng
(C_best + ngưỡng đã chốt), rồi so với `acceptance_targets.json`. Báo cáo nguyên văn các con
số. **Không** dùng kết quả này để chọn lại cấu hình. Nếu không đạt mục tiêu: báo cáo, nhìn
ma trận nhầm lẫn để gợi ý đòn bẩy, và **dừng chờ người dùng quyết định**.

## NHIỆM VỤ 8 — (TÙY CHỌN, mặc định TẮT) Tinh chỉnh liên tục

Chỉ làm nếu, sau Nhiệm vụ 3–7, bảng điểm vẫn cho thấy khoảng cách **rõ ràng** so với
`acceptance_targets.json` VÀ ma trận nhầm lẫn chỉ ra một cặp từ cụ thể. Khi đó, dò tối đa **2
trọng số** liên quan đến cặp đó bằng lưới rời rạc (3–4 giá trị mỗi trọng số), đánh giá bằng
**LOSO lồng nhau** (chọn trên các phiên dò, chấm trên phiên bị giữ lại), và chỉ nhận nếu
`compare` cho ACCEPT. Ghi rõ đây là ngoại lệ.

## Tiêu chí hoàn thành

- `results/spec04_summary.md`: bảng **mọi** cấu hình đã chạy (tên, khác gì C1/C_best, verdict, top-1,
  clean_accept, wrong_accept, margin ± khoảng tin cậy, nhận nhầm mẫu âm).
- Cấu hình cuối `configs/C_final.json` + lý do chọn (theo quy tắc đơn giản nhất).
- Kết luận về ASL-LEX theo đúng verdict (Nhiệm vụ 4b).
- `results/asllex_analysis.md` (Nhiệm vụ 5).
- Nếu quy tắc dừng đã kích hoạt: nói rõ đã dừng ở bước nào và vì sao.
- Danh sách file đã sửa, mọi chỗ spec lệch code thật, mọi quyết định tự đưa ra, mọi thứ không
  kiểm chứng được.

## TUYỆT ĐỐI KHÔNG LÀM

- Không tìm kiếm ngẫu nhiên / dò liên tục nhiều trọng số (ngoại lệ có điều kiện ở Nhiệm vụ 8).
- Không đọc hay dùng ASL Citizen trong việc chọn cấu hình hay ngưỡng.
- Không sửa `ASL_ALEX_2.0/sign_weights_asllex.json` hay các CSV/script tạo ra nó.
- Không sửa `acceptance_targets.json` hay `acceptance.json` sau khi thấy kết quả.
- Không dùng phiên 3 ngoài lệnh `sealed`, và không dùng quá một lần.
- Không chọn cấu hình có verdict INCONCLUSIVE vì "nhìn có vẻ tốt hơn".
- Không tự `--apply` ngưỡng hay đổi `CONFIG` production khi chưa được người dùng xác nhận.
- Không chạy pytest / e2e (người dùng yêu cầu bỏ để tiết kiệm token).
