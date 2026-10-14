# So sánh DTW với baseline học máy (2026-09-28)

Công cụ: `eval_baselines.py` (chạy: `py eval_baselines.py --dtw-config configs/G1p90.json --dtw-config configs/C11_base.json --seeds 3`, ~13 phút CPU).
Cùng bank (13 từ, 175 mẫu), cùng cách chia (7 lượt LOSO theo phiên), cùng đặc trưng quỹ đạo 32x24, chỉ CPU. Kết quả JSON: `results/baselines/eval_baselines.json` (thư mục results/ không đưa lên git).

## Kết quả

Phần 1 - độ chính xác khi ép chọn một từ, và khả năng phân biệt từ đã biết với từ lạ bằng điểm tin cậy (AUROC, không phụ thuộc ngưỡng):

| Phương pháp | LOSO top-1 | Mai top-1 | AUROC tất cả | AUROC từ lạ (LOWO) | AUROC nét âm |
|---|---|---|---|---|---|
| DTW G1p90 (mặc định) | 100% | 92.2% | 0.990 | 0.990 | 0.994 |
| DTW C11_base (cũ) | 100% | 92.2% | 0.990 | 0.989 | 0.991 |
| 1-NN Euclid (không DTW) | 95.4% | 68.6% | 0.804 | 0.812 | 0.769 |
| MLP nhỏ (3 hạt giống) | 97.5±0.7% | 72.5±0.0% | 0.918±0.010 | 0.903±0.011 | 0.980±0.007 |
| GRU hai chiều nhỏ (3 hạt giống) | 100±0.0% | 94.1±5.5% | 0.974±0.006 | 0.969±0.007 | 0.994±0.003 |

Phần 2 - điểm vận hành: ngưỡng của baseline chỉnh để LOSO nhận-đúng = 152 (bằng DTW G1p90):

| Phương pháp | LOSO sai | Mai nhận đúng (/51) | Từ lạ nhận nhầm (/175) | Nét âm nhận nhầm (/41) |
|---|---|---|---|---|
| DTW G1p90 | 0 | 37 | 0 | 0 |
| 1-NN | 8 | 34 | 107 | 35 |
| MLP | 0 | 31.3 | 28.7 | 1.3 |
| GRU | 0 | 29.0 | 12.0 | 0 |

## Kết luận (giới hạn trong dữ liệu này)

- Về ĐÓNG (chọn từ đúng khi ép chọn): GRU ngang DTW (LOSO 100%, Mai 94.1±5.5% so với 92.2%, chênh lệch nằm trong nhiễu). DTW KHÔNG hơn học sâu ở mặt này.
- Về TỪ CHỐI từ lạ: DTW tốt hơn rõ (AUROC 0.990 so với 0.974 GRU và 0.918 MLP; ở điểm vận hành: 0/175 so với 12/175 GRU và ~29/175 MLP nhận nhầm từ lạ; người lạ nhận đúng 37 so với 29 GRU).
- 1-NN Euclid kém hẳn: cần căn chỉnh thời gian (DTW) hoặc mô hình chuỗi.

## Điều KHÔNG được nói quá

- DTW có lợi thế do người viết tự chỉnh: trọng số theo từng ký hiệu, hệ số k chọn trên chính LOSO này. Baseline chỉ có siêu tham số mặc định, chưa được tinh chỉnh công bằng.
- Chỉ 1 người ký cho bank, 1 tình nguyện viên (Mai, 51 mẫu), 13 từ, 7 lượt LOSO: khoảng tin cậy rộng (GRU Mai dao động ±5.5).
- Baseline chỉ dùng quỹ đạo 24D, không dùng nhãn hình bàn tay (DTW có dùng lambda_shape).
- Chưa đo: đường cong học (số mẫu mỗi từ ít thì ai thắng), nhiều người ký, bộ dữ liệu ngoài.

---

# Đường cong học (2026-09-28)

Công cụ: `eval_learning_curve.py` (`py eval_learning_curve.py --ns 1 2 3 5 8 --repeats 3`, ~23 phút CPU). Mỗi lượt LOSO chỉ giữ n mẫu gốc ngẫu nhiên mỗi từ ở phía tham chiếu/huấn luyện; 3 lần lặp, cùng tập mẫu con cho mọi phương pháp. Biểu đồ: `results/baselines/learning_curve.png`.

| n mẫu/từ | LOSO top-1: DTW | GRU | MLP | 1-NN | Mai top-1: DTW | GRU | MLP | 1-NN |
|---|---|---|---|---|---|---|---|---|
| 1 | 94.1 | 96.4 | 85.3 | 84.2 | 89.5 | 97.4 | 75.8 | 81.0 |
| 2 | 97.0 | 98.3 | 95.2 | 92.0 | 91.5 | 96.7 | 80.4 | 89.5 |
| 3 | 97.3 | 98.7 | 96.6 | 94.1 | 92.2 | 98.0 | 80.4 | 84.3 |
| 5 | 99.4 | 99.0 | 97.5 | 93.7 | 91.5 | 97.4 | 76.5 | 72.5 |
| 8 | 99.6 | 100.0 | 98.1 | 94.9 | 92.8 | 93.5 | 72.5 | 69.3 |

AUROC nét âm (từ đã biết vs nét âm): DTW 0.968 → 0.992, GRU 0.963 → 0.995 (n=1 → 8): ngang nhau; MLP 0.865 → 0.971; 1-NN ~0.71–0.77.

## Kết luận

- Giả thuyết "DTW hơn học sâu khi ít mẫu" **KHÔNG được ủng hộ** với GRU: GRU ngang hoặc hơn DTW ở mọi n về chọn-đúng-từ, kể cả n=1 (Mai: 97.4% so với 89.5%). DTW chỉ hơn MLP và 1-NN.
- Từ chối nét âm (AUROC): DTW và GRU ngang nhau ở mọi n.
- Ưu thế còn lại của DTW: từ chối TỪ LẠ ở dữ liệu đầy đủ (LOWO 0/175 so với GRU 12/175 ở cùng số nhận-đúng) — chưa đo LOWO ở n nhỏ vì phải huấn luyện lại 13 lần mỗi mức.
- Không cần huấn luyện và giải thích được vẫn là ưu điểm kỹ thuật của DTW, nhưng không phải ưu điểm về độ chính xác.

## Giới hạn
- Mai chỉ 51 mẫu (1 mẫu = ~2%), độ lệch chuẩn GRU ở n=8 là ±6.5: không suy diễn từng con số nhỏ.
- Kết quả Mai của GRU không đơn điệu theo n (97-98% ở n nhỏ, 93.5% ở n=8): dấu hiệu nhiễu/quá khớp, không phải "ít dữ liệu tốt hơn".
- "Nhận đúng" của DTW ở ngưỡng riêng không đơn điệu theo n vì độ rộng nội lớp ước lượng kém khi n nhỏ (n=1 không tính được): dùng top-1 và AUROC, không dùng cột này để so sánh.
- Vẫn 1 người ký cho bank, 13 từ, GRU chưa tinh chỉnh siêu tham số: đây là kết quả sơ bộ, cần nhiều người ký để kết luận.

---

# Thử kết hợp GRU + DTW (2026-09-28)

Công cụ: `eval_hybrid.py` (`py eval_hybrid.py --seeds 3`, ~18 phút CPU). Bank đầy đủ, cùng 4 phép đo; GRU trung bình 3 hạt giống; DTW = configs/G1p90.json.
Số liệu: số truy vấn (LOSO 175, Mai 51, từ lạ LOWO 175, nét âm 41).

| Biến thể | LOSO nhận đúng | LOSO sai | Mai top-1 | Mai nhận đúng | Mai sai | Từ lạ nhận nhầm | Nét âm nhận nhầm |
|---|---|---|---|---|---|---|---|
| dtw (hiện tại) | 152 | 0 | 92.2% | 37 | 0 | 0 | 0 |
| gru tự từ chối (cùng LOSO đúng) | 152 | 0 | 94.1% | 29 | 0 | 12 | 0 |
| hybrid_cong (nhãn GRU, cổng DTW) | 152 | 0 | 94.1% | 35.3 | 1.7 | 0 | 0 |
| hybrid_dong_y (DTW nhận và cùng từ với GRU) | 152 | 0 | 94.1% | 35.3 | 0 | 0 | 0 |
| cứu UNCERTAIN, GRU >= 0.9 | 164.7 | 0 | 94.1% | 35.3 | 1.7 | 0 | 0 |
| cứu UNCERTAIN, GRU >= 0.8 | 169.7 | 0 | 94.1% | 37.0 | 1.7 | 2 | 0 |
| cứu UNCERTAIN, GRU >= 0.6 | 170.3 | 0 | 94.1% | 40.0 | 1.7 | 6 | 0 |
| cứu cả REJECTED, GRU >= 0.6 | 174.3 | 0 | 94.1% | 44.0 | 1.7 | 23 | 2.7 |

("Cứu" = DTW chưa nhận nhưng GRU đồng ý với từ DTW chọn và xác suất >= ngưỡng thì nhận.)

## Kết luận
- Cổng DTW + nhãn GRU (hybrid_cong / dong_y) KHÔNG hơn DTW đơn lẻ về số nhận đúng (35.3 so với 37 ở Mai); nhãn GRU đôi khi sai khi DTW đã nhận đúng (1.7 nhận sai ở hybrid_cong).
- Cơ chế "cứu" tăng số nhận đúng của chính người ký bank (LOSO 152 -> 165-170, tức 87% -> 94-97%) với 0 nhận sai, giá là 0-2 (0-1%) từ lạ bị nhận nhầm ở ngưỡng 0.8-0.9. Với người lạ Mai hầu như không lợi (37 -> 37 ở ngưỡng 0.8; 40 chỉ khi hạ ngưỡng xuống 0.6 và trả 6/175 = 3.4% nhận nhầm từ lạ).
- Chưa nên đưa vào hệ thống: (1) ngưỡng GRU được thử 3 giá trị trên chính dữ liệu đo (lạc quan); (2) Mai chỉ 51 mẫu; (3) thêm phụ thuộc torch + phải huấn luyện lại khi đổi bank; (4) nhãn nên lấy từ DTW khi DTW đã nhận (chưa đo phiên bản này).

---

# Người lạ: thêm mẫu của chính họ có giúp không? (2026-09-28)

Công cụ: `eval_enroll.py` (`py eval_enroll.py --targets Mai Phi --ks 1 2 3 --repeats 5`, ~2 phút). DTW cấu hình G1p90. Với mỗi người P, chọn ngẫu nhiên k mẫu/từ làm enroll (5 lần lặp), các mẫu còn lại là truy vấn (cùng tập cho mọi điều kiện). Điều kiện: `dat` (bank Đạt), `dat+own` (Đạt + k mẫu của P), `own` (chỉ k mẫu của P), `pool` (Đạt + người còn lại, không có mẫu của P), `pool+own`.

Nhận đúng (ACCEPTED và đúng từ), trung bình 5 lần lặp:

| Người | k | dat | dat+own | own | pool | pool+own |
|---|---|---|---|---|---|---|
| Mai (~90 truy vấn) | 1 | 65.1% | 91.6% | 90.2% | 69.3% | 91.6% |
| Mai | 2 | 66.5% | 94.8% | 96.0% | 70.5% | 95.0% |
| Mai | 3 | 67.1% | 95.1% | 94.6% | 70.9% | 95.1% |
| Phi (~27 truy vấn) | 1 | 58.5% | 84.4% | 88.1% | 47.4% | 79.3% |
| Phi | 2 | 62.1% | 93.7% | 95.8% | 50.5% | 91.6% |
| Phi | 3 | 70.0% | 100% | 100% | 56.7% | 98.3% |

Nét âm (/41) bị nhận nhầm: dat+own 0–1; own 0.8–5.4 (Phi k=3: 5.4); dat 0; pool 0–1.

## Kết luận
- Enroll chính người đó là cách hiệu quả nhất: +26 đến +30 điểm với k=1, ~+28 với k=2, +30 với k=3 (Mai: 65 -> 91.6/94.8/95.1; Phi: 58.5 -> 84.4/93.7/100), và nhận sai gần như 0.
- Bank gộp người khác (không có mẫu của P) gần như không giúp (Mai +4 điểm) và có thể hại (Phi 58.5 -> 47.4).
- `dat+own` an toàn hơn `own` (bank Đạt làm "vùng đối chứng"): `own` cho nhiều nét âm nhận nhầm hơn, nhất là k lớn.
- Chỉ 1-3 mẫu/từ (khoảng 3-5 phút) đã đủ để lên ~92-100%.

## Giới hạn
- Enroll và truy vấn cùng buổi thu (mẫu của một người thu gần như cùng ngày): có thể lạc quan hơn dùng thật vài ngày sau.
- Chỉ 2 người, Phi chỉ ~27 truy vấn và 8 từ (thiếu ME, YOU, YES, BAD): các con số nhỏ dao động lớn.
- Cả hai người lạ là người khác giới/tay khác Đạt (Mai thuận trái): chưa biết có khái quát hóa sang mọi người.

## Đo nhanh trên ASL Citizen (2026-09-28, `eval_citizen.py`)

Bank Đạt + cấu hình G1p90, truy vấn = 226 video Citizen đã cắt (24D), không chỉnh gì. Chỉ để đo; biên đoạn ký của Citizen không đáng tin.

- 12 từ trùng bank (86 mẫu): top-1 **61,6%**, nhận đúng **22,1%**, nhận sai **1,2%**, không chắc 25,6%, từ chối 51,2%.
- Từ ngoài bank (7 từ, 53 mẫu): nhận nhầm **2 (3,8%)**, cả hai thành THANK_YOU.
- Theo từ: WATER 4/4, DRINK 7/7, PLEASE 6/10 nhận đúng; BYE 0/10, YES 0/10, YOU 0/4, HELLO 0/5.
- Kết luận: lệch miền vẫn lớn (nhận đúng 22% so với 58–65% của Mai/Phi) nhưng hệ thống **từ chối thay vì nhận sai**. Citizen chỉ dùng làm bộ đo thô.

## Tự thích nghi khi dùng (2026-09-28, `eval_adapt.py`)

Bank Đạt; người lạ ký lần lượt, mẫu ACCEPTED có `r_ratio` ≤ g được thêm dưới nhãn DỰ ĐOÁN (tối đa 3/từ); 3 lần lặp thứ tự ngẫu nhiên.
- Mai (100 mẫu): tĩnh 66,0% nhận đúng / 2,0% nhận sai; g=1,2: **69,3%** / 1,3% (thêm 17 mẫu, 0 nhầm nhãn); g=0,8: 66,0% / 2,3%.
- Phi (35 mẫu): không đổi (57,1%), chỉ thêm 3 mẫu.
- Không gate (g=∞, một lần chạy thử, Mai): 66→73% nhưng nhận sai 2→10% (3/24 mẫu học nhầm nhãn) — không an toàn.
- Kết luận: tự học chỉ từ mẫu đã được nhận tự tin nên không học được đúng những mẫu khó (bị từ chối); lợi rất nhỏ. Enroll có nhãn (1–3 mẫu/từ) đạt 92–100%.

## Mô phỏng phiên có hỏi xác nhận (2026-09-28, `eval_confirm.py`)

Bank Đạt; người lạ ký lần lượt (thứ tự ngẫu nhiên, 2 lần lặp); khi không ACCEPTED và có ứng viên thì hỏi, người dùng trả lời đúng nhãn thật
(đáp án thật phải nằm trong top-2). Quyết định đo TRƯỚC khi hỏi. Cấu hình G1p90:
- Mai (100 mẫu): tĩnh 66,0% nhận đúng → hỏi (UNCERTAIN+REJECTED) **87,5%**, nhận sai 1,0%, chỉ **11,5 câu hỏi**/100 lần ký; phần cuối phiên 98%.
- Phi (35 mẫu): tĩnh 57,1% → **77,1%**, nhận sai 0%, 8 câu hỏi/35 lần ký.
- Chỉ hỏi khi UNCERTAIN: Mai 83,5%, Phi 67,1% (kém hơn, nhất là Phi).
- cap 2 và cap 3 mẫu/từ cho cùng kết quả (mỗi từ hiếm khi cần > 2 mẫu). Nét âm nhận nhầm sau phiên: ~1/41.
- G2p00 (ngưỡng rộng hơn) gần như y hệt: Mai 89,0%, Phi 78,6%, nhưng nét âm nhận nhầm 1,5.
Giới hạn: người dùng mô phỏng luôn trả lời đúng; 2 người; câu hỏi ở đầu phiên nhiều hơn cuối phiên.

---

# Học sâu train trên dữ liệu GỘP nhiều người có giúp chống người lạ không? (2026-09-29)

Câu hỏi xuất phát từ nghi vấn: kết luận "hybrid GRU+DTW không hơn DTW" ở trên (`eval_hybrid.py`) huấn luyện GRU **chỉ trên bank của Đạt** (LOSO theo phiên, một người) — chưa từng thấy người khác lúc train, nên không phải bằng chứng "học sâu không giúp cho người lạ". Bốn pilot dưới đây kiểm định lại đúng câu hỏi này bằng cách train GRU trên dữ liệu GỘP nhiều người.

## Pilot 1–2: gộp Đạt + 1 tình nguyện viên (± ASL Citizen), `eval_deep_pilot.py` / `eval_deep_pilot2.py`

Top-1 ép chọn (13 từ), target = người bị giữ lại hoàn toàn (0 mẫu enroll):

| Target | DTW (Đạt) | DTW (gộp 2 người) | DTW (gộp 2 người + Citizen) | GRU (Đạt) | GRU (gộp 2 người) | GRU (gộp 2 người + Citizen) |
|---|---|---|---|---|---|---|
| Đạt | — | 60,0% | 64,3% | — | 61,4±1,1% | **86,1±1,3%** |
| Mai | 83,0% | 83,0% | 84,0% | 81,2±6,9% | 81,8±4,5% | 85,0±3,0% |
| Phi | 100,0% | 97,1% | 97,1% | 98,3±2,3% | 97,7±2,1% | 94,9±2,8% (giảm) |

Thêm Citizen luôn tăng "nhận-đúng" của DTW (ví dụ Đạt 26,5%→51,9%, Phi 48,6%→68,6%) nhưng đổi lại "nhận-sai" cũng tăng (0%→6,5–9%) — ngưỡng G1p90 chưa hiệu chỉnh lại cho bank có Citizen.

## Pilot 3: kiểm định tách bạch "đa dạng thật" hay chỉ "nhiều dữ liệu hơn"? `eval_deep_pilot3.py`

Với mỗi target, so 3 kịch bản GRU cùng tập test: `others` (2 người còn lại), `others+Citizen`, và `others+nhân-bản` (đối chứng: nhân bản ngẫu nhiên chính dữ liệu `others`, ĐÚNG số mẫu mà Citizen thêm vào mỗi từ — cùng khối lượng, không có gì mới).

| Target | others | +Citizen | +nhân-bản (đối chứng) | Tăng do Citizen | Tăng do CHỈ-KHỐI-LƯỢNG | **Đóng góp thật của đa dạng** |
|---|---|---|---|---|---|---|
| Đạt | 60,9±0,9% | 85,5±2,3% | 61,0±0,2% | +24,7 | +0,1 | **+24,6** |
| Mai | 82,2±3,8% | 80,8±2,8% | 81,3±1,9% | −1,5 | −1,0 | −0,5 |
| Phi | 96,1±2,4% | 95,4±1,4% | 95,4±0,3% | −0,7 | −0,7 | −0,0 |

Đối chứng nhân-bản cho thấy phần tăng của Đạt (+24,6 điểm) là đóng góp THẬT từ đa dạng dữ liệu, không phải ảo giác khối lượng. Với Mai/Phi (tập train nền vốn đã dày, 220–285 mẫu) thì Citizen không giúp gì, cả hai kịch bản đều phẳng.

**Thử loại BYE/YOU khỏi Citizen** (hai từ khớp 0% khi so trực tiếp với bank Đạt qua DTW — nghi có biến thể ký khác) — kết quả **không xác nhận giả thuyết**: đóng góp thật của Đạt giảm còn +15,6 (không tăng), Mai/Phi vẫn phẳng (−1,2 và +1,1). Kết luận: nhiễu nhãn BYE/YOU không phải nguyên nhân chính khiến Mai/Phi không hưởng lợi từ Citizen.

**Quy luật rút ra**: lợi ích của dữ liệu ngoài miền (Citizen) **tỉ lệ nghịch với độ giàu dữ liệu bản địa** — giúp nhiều khi tập train còn mỏng (Đạt, 135 mẫu nền), gần như không giúp khi tập train đã đủ dày (Mai/Phi, 220+ mẫu nền). Vì bank sẽ dày dần khi thu đủ 8 tình nguyện viên, giá trị của Citizen sẽ tự giảm — không đáng đầu tư công cắt thêm video Citizen thủ công (mỗi video phải cắt tay bằng `tools/inspect_and_trim_asl_citizen.py`, không có đường tắt tự động vì từng gây lệch miền).

## Pilot 4: gộp NGƯỜI THẬT (không Citizen) — đường cong theo số người tham chiếu, `eval_deep_pilot4.py`

5 người thật cùng pipeline thu (Đạt/Mai/Phi/Đăng/Vương). Với mỗi người bị giữ lại làm "người lạ" (0 mẫu enroll), đo top-1 khi tập tham chiếu gồm k = 1..4 người còn lại (trung bình 2 tổ hợp ngẫu nhiên mỗi k, GRU 3 hạt giống):

| Target | k=1 | k=2 | k=3 | k=4 (tất cả) |
|---|---|---|---|---|
| Đạt | dtw 50,0% / gru 58,9% | dtw 63,2% / gru 63,8% | dtw 63,2% / gru 65,4% | dtw 63,8% / gru 66,3% |
| Mai | dtw 83,0% / gru 82,5% | dtw 79,5% / gru 86,2% | dtw 83,5% / gru 94,7% | dtw 84,0% / gru 93,0% |
| Phi | dtw 83,3% / gru 84,4% | dtw 92,7% / gru 93,1% | dtw 97,9% / gru 96,5% | dtw 97,9% / gru 97,9% |
| Đăng | dtw 75,0% / gru 80,6% | dtw 89,3% / gru 87,7% | dtw 95,2% / gru 92,9% | dtw 95,2% / gru 88,1% |
| Vương | dtw 81,7% / gru 76,0% | dtw 84,6% / gru 88,5% | dtw 84,6% / gru 90,7% | dtw 84,6% / gru 94,2% |

**Kết luận — khác hẳn Citizen, đây là tín hiệu sạch và nhất quán**: gộp thêm người tham chiếu THẬT (cùng pipeline thu, không lệch miền) cải thiện độ chính xác trên người lạ ở **cả 5/5 người**, không chỉ trường hợp dữ liệu mỏng. DTW tăng mạnh từ k=1→3 rồi chững lại (Phi/Vương/Đăng gần bão hòa ở k=3); GRU tiếp tục nhích nhẹ tới k=4 ở hầu hết trường hợp. Với 13 từ vựng hiện tại, **~3 người tham chiếu đã lấy được phần lớn lợi ích** — khớp với khuyến nghị thống kê trước đó ("5-8 người là đủ để kết luận đáng tin"), giờ có bằng chứng thực nghiệm.

## Giới hạn chung của 4 pilot
- Chỉ 3-5 người thật, số tổ hợp lấy mẫu ít (2 tổ hợp ngẫu nhiên mỗi k ở pilot 4): khoảng tin cậy còn rộng, cần xác nhận lại khi đủ 8 người.
- Top-1 ép chọn, không tính ngưỡng nhận/từ chối cho GRU (GRU chưa có cơ chế từ chối như DTW) — số liệu chỉ so sánh khả năng phân biệt từ, chưa phải hệ thống hoàn chỉnh.
- Kết quả JSON: `results/baselines/deep_pilot.json`, `deep_pilot2.json`, `deep_pilot3.json`, `deep_pilot3_filtered.json`, `deep_pilot4.json`.

---

# Chuẩn hóa segmenter theo thời gian: `time_norm` (2026-09-29)

Đo bằng `tools/eval_segmenter_fps.py` (159 stroke thật × 7 biến thể fps × 4 chế độ, không cần camera). So sánh chế độ sản xuất hiện tại (`prod`) với `prod+tn` (bật `time_norm`: quy vận tốc bắt đầu/dừng + dò Swipe-Left về THỜI GIAN thay vì SỐ KHUNG, để xử lý camera có fps thật khác 30).

| Biến thể | prod (cũ) | prod+tn (mới) |
|---|---|---|
| 60fps | đúng 61,6% / nhớ 34,6% | đúng **79,9%** / nhớ **6,9%** |
| 15fps | đúng 91,8% / swipe-nhầm 9 | đúng **97,5%** / swipe-nhầm **0** |
| 45fps | đúng 76,7% / nhớ 17,0% | đúng **86,8%** / nhớ **5,7%** |
| khung trùng (dedup) | swipe-nhầm 9 | swipe-nhầm **0** |
| 30fps (gốc, tham chiếu) | 90,6% | 90,6% (giống hệt, không hồi quy) |

Bài thử Swipe-Left giả lập ở 60fps: `prod` bỏ lỡ hoàn toàn (0/1 phát hiện), `prod+tn` bắt đúng (1/1).

**Kết luận**: bật `time_norm` cải thiện rõ ở mọi fps khác 30, không hồi quy ở fps gốc, sửa được 1 lỗi thật (bỏ lỡ Swipe ở 60fps). Đánh đổi duy nhất: tỉ lệ "cắt đôi" tăng nhẹ ở 60fps (3,8%→13,2%) — chấp nhận được vì đổi lại giảm mất-hẳn-đoạn nhiều hơn (34,6%→6,9%).
**Đã áp dụng**: bật mặc định trong `predict_sentence_webcam.py` (`CONFIG["time_norm"]["enabled"] = True`) từ 2026-09-29; `tests/test_robustness_step.py` (23 bài) vẫn pass.

---

# Hiện trạng CNN dáng tay (2026-09-29, cập nhật 2026-09-30)

Kiểm tra trước khi bàn "mở rộng dùng handshape":
- CNN dáng tay (`best_model_v3.pth`) chỉ học **5/26 chữ cái** (A, B, C, L, V — 451 mẫu ảnh tĩnh: A=125, B=155, C=64, L=53, V=54), độ chính xác nội bộ 92,3% nhưng đo trên ảnh tĩnh tạo dáng sẵn, 5 chữ dễ phân biệt nhất.
- **CNN chưa từng chạy trong bất kỳ phiên thu thật nào (tính đến 2026-09-29)**: quét toàn bộ `raw_strokes/` (10.292 khung, 14 từ) → 0 khung có nhãn dáng tay thật (tất cả "Unknown", `hand_shape_conf=0`). Cờ `enable_static_shape_cnn`/`--with-cnn` có sẵn nhưng mặc định tắt.
- `raw_strokes/` chỉ lưu tọa độ landmark, không lưu ảnh gốc → không thể chạy lại CNN cho dữ liệu cũ, phải thu mới nếu muốn có nhãn thật.
- Thử nghiệm cũ (`lambda_shape_global`, trên Citizen) cho thấy nhãn CNN không ổn định giữa các mẫu cùng từ — đang tắt trong production.

## Đối chiếu handshape thật của 13 từ trong vocab với 5 chữ đã train

Sửa lại nhận định ban đầu ("5 chữ có thể không đại diện được vocab") — đối chiếu handshape ASL thật của từng từ với 5 chữ cái đã train cho thấy **độ phủ tốt hơn tưởng**:

| Từ | Handshape ASL thật | Khớp với 5 chữ đã train? |
|---|---|---|
| DRINK | C (khum tay cầm cốc) | ✅ khớp C |
| SORRY | A (nắm đấm, xoa ngực) | ✅ khớp A |
| THANK_YOU | B (bàn tay phẳng, từ cằm ra) | ✅ khớp B |
| PLEASE | B (bàn tay phẳng, xoa ngực) | ✅ khớp B |
| HUNGRY | C-cong/claw (kéo dọc ngực) | ~ gần C |
| HELLO | B/5 (vẫy tay) | ~ gần B |
| YES | S (nắm đấm gật) | ~ gần A (S không có trong 5 chữ) |
| BAD, WATER, YOU, ME, EAT, BYE | 5 / W / 1 / 1 / O-phẳng / 5 | ❌ không khớp chữ nào đã train |

→ Ít nhất 4 từ (DRINK, SORRY, THANK_YOU, PLEASE) có đáp án đúng rõ ràng để kiểm tra tính ổn định của CNN — không cần đợi mở rộng thêm chữ cái mới mới kiểm tra được.

## Kiểm tra sống đầu tiên (2026-09-30): CNN KHÔNG nhận đúng dáng tay C

Người dùng thu sống với `--with-cnn`, làm đúng dáng tay chữ C (như DRINK) — **hệ thống không nhận diện được dáng tay C**. Đây là bằng chứng thực nghiệm đầu tiên (dù chỉ 1 phiên, chưa định lượng số liệu) rằng phần nhận diện handshape hiện tại **hoạt động không hiệu quả trong điều kiện thu thật**, dù đo nội bộ trên ảnh tĩnh đạt 92,3%. Nguyên nhân khả dĩ (cần điều tra thêm, chưa xác nhận cái nào là chính):
1. **Lệch miền ảnh train vs ảnh thu thật**: 451 ảnh train là ảnh tĩnh tạo dáng sẵn (Kaggle/studio, có augmentation physics mô phỏng webcam) — khác điều kiện ánh sáng/góc tay/nền thật khi đang ký hiệu động (motion blur, tay chuyển động, góc nghiêng khác).
2. **C chỉ có 64 mẫu train** (ít nhất trong 5 chữ, so với B=155, A=125) — dễ là chữ yếu nhất trong 5 chữ.
3. **Chưa lọc khung ổn định/tin cậy trước khi phân loại** (ý tưởng đã đề xuất ở trên nhưng chưa triển khai) — nếu CNN chạy trên TẤT CẢ khung kể cả khung chuyển động mờ, kết quả sẽ nhiễu bất kể model tốt hay dở.
4. **`enable_static_shape_cnn`/`--with-cnn` mới lần đầu được bật thật** — có thể còn lỗi tích hợp (tiền xử lý ảnh, crop tay, chuẩn hóa) chưa được kiểm chứng bằng dữ liệu thật trước đây, khác với lỗi bản thân model.

## Điều tra nguyên nhân (2026-09-30): 2 lỗi tích hợp tìm được bằng đọc code, khớp với quan sát của người dùng

Người dùng cho biết: **giai đoạn test CNN tĩnh riêng lẻ (giữ yên dáng tay) chạy ổn**, nhưng tích hợp vào phiên thu động (`--with-cnn`) thì dáng tay C không được nhận ra. Đọc `collect_dynamic_templates.py` tìm thấy 2 lỗi thật, khớp chính xác với sự khác biệt "tĩnh vs động" này:

1. **Model cũ, quên cập nhật (đã sửa)**: dòng 596 hardcode `model_path = 'vsl_dataset/models/best_model_v2.pth'` (val_acc 90,79%) trong khi checkpoint tốt nhất thực tế là `best_model_v3.pth` (val_acc 92,29%, huấn luyện sau v2 ~3,5 tiếng cùng ngày 2026-09-07). Đã sửa thành v3. Chênh lệch chỉ ~1,5 điểm % nên khó là nguyên nhân chính, nhưng là bug rõ ràng cần sửa dù sao.
2. **Không lọc khung ổn định trước khi phân loại (nguyên nhân chính, chưa sửa)**: dòng 882-895 chạy CNN trên **MỌI khung hình** trong suốt lúc tay đang chuyển động ký hiệu động — không có bước chờ tay đứng yên/tin cậy như lúc test tĩnh. `v_norms` (độ ổn định chuyển động) và biến `hand_shape_conf` (độ tin cậy CNN) đã được tính mỗi khung nhưng **không được dùng để chọn khung tốt nhất** — mọi khung, kể cả khung mờ do tay đang di chuyển, đều được phân loại và ghi ngang hàng. Đây chính là cơ chế "bắt khung ổn định nhất" mà người dùng đã đề xuất trước đó — giờ có bằng chứng cụ thể nó cần thiết, không chỉ là ý tưởng lý thuyết.

## Hướng nâng cấp bộ handshape (đề xuất, ưu tiên theo chi phí tăng dần)

1. ~~Đổi `model_path` sang `best_model_v3.pth`~~ — **đã sửa** (2026-09-30).
2. ~~Áp dụng lọc khung ổn định~~ — **đã triển khai** (2026-09-30, xem mục dưới).
3. **Thu thêm ảnh C từ chính điều kiện webcam thật** (không phải Kaggle/studio) để giảm lệch miền — ưu tiên các chữ có ít mẫu nhất (C=64, V=54, L=53).
4. **Mở rộng thêm 2-3 chữ** (ưu tiên S cho YES, W cho WATER, 5 cho BYE/BAD/HELLO) chỉ SAU KHI đã xác nhận 5 chữ hiện có hoạt động ổn trong điều kiện thật — mở rộng trước khi chữ cũ còn lỗi là phí công.

## Triển khai lọc khung ổn định (2026-09-30)

Đã cài đặt cơ chế chống nhiễu khung chuyển động cho nhãn dáng tay, không cần train lại CNN:

- **`dynamic_dtw_engine.py`**: `GestureFrame` thêm 2 trường `hand_shape_conf: Optional[float]` và `v_norm: float` (mặc định `None`/`0.0`, không phá vỡ chỗ khởi tạo cũ theo thứ tự vị trí). `KineticPauseSegmenter.update()` nhận thêm tham số `hand_shape_conf`, và tự lưu `v_norm` (đã tính sẵn mỗi khung để dò swipe/pause) vào chính khung đó — trước đây giá trị này bị tính rồi bỏ, không gắn lại vào `GestureFrame`.
- **`collect_dynamic_templates.py`**: hàm mới `stabilize_hand_shapes(frames)` — với mỗi stroke vừa hoàn tất, chỉ tin nhãn CNN ở khung có `v_norm ≤ 0.04` (tay gần như đứng yên) VÀ `hand_shape_conf ≥ 0.55`; khung không đạt được gán lại bằng nhãn ổn định gần nhất trong cùng stroke (forward-fill, lan ngược cho các khung trước khung ổn định đầu tiên). Nếu cả stroke không có khung nào đạt ngưỡng, giữ nguyên hành vi cũ không đổi (an toàn khi CNN tắt). Gọi hàm này thay cho việc lấy `hand_shape` thô trước khi đưa vào `resample_9d_gesture_sequence`.
- Toàn bộ 94 test liên quan (`test_robustness_step`, `test_segmenter_fast_chain`, `test_collect_phrases`, `test_harness_utils`, `test_fast_dtw_parity`, `test_lead_trim`, `test_build_segment_features`) pass sau thay đổi.
- **Kết quả kiểm chứng của người dùng (2026-09-30, sau khi triển khai)**: thu lại dáng tay C bằng `--with-cnn` — chỉ một vài khung hiện đúng C, phần lớn hiện sai thành L hoặc B. Lọc khung ổn định không giải quyết được vấn đề → nghi ngờ lỗi nằm ở chính model, không chỉ ở tầng tích hợp.

## Xác nhận bằng số: lỗi thật sự nằm ở model (2026-09-30)

Chạy `best_model_v3.pth` qua TOÀN BỘ ảnh đã lưu trong `vsl_dataset/samples/` (511 ảnh, gồm cả ảnh dùng để train — tức đây là độ chính xác "tốt nhất có thể", chưa tính lệch miền webcam) để lấy ma trận nhầm lẫn đầy đủ theo từng lớp:

| Chữ | Acc (trên chính ảnh train) | Nhầm thành |
|---|---|---|
| A | 100,0% (135/135) | — |
| B | 99,4% (154/155) | 1 → C |
| **C** | **93,9% (108/115)** | **6 → L, 1 → B** |
| L | 98,1% (52/53) | 1 → V |
| **V** | **63,0% (34/54)** | **20 → L** |

**Kết luận**: ngay trên chính dữ liệu model đã học (chưa tính lệch miền webcam), **C đã bị nhầm với L** — khớp chính xác với quan sát thực tế của người dùng trên webcam (thấy C hiện ra L/B). Đây xác nhận người dùng đúng: vấn đề nằm ở ranh giới quyết định của bản thân model giữa nhóm chữ "khum/cong ngón" (C, V) và L, không chỉ là nhiễu tích hợp/lọc khung. Phát hiện thêm ngoài dự kiến: **V còn yếu hơn C nhiều (63%, gần 40% nhầm thành L)** — bị che khuất bởi con số tổng 92,3% (macro trung bình).

Nguyên nhân khả dĩ: C (64 mẫu) và V (54 mẫu) là 2 lớp ít mẫu nhất trong 5 chữ (so với B=155, A=125), và về mặt hình học, ranh giới C/V/L phụ thuộc độ cong/duỗi ngón — mờ hơn nhiều so với A (nắm đấm) hay B (bàn tay phẳng) mà ảnh DIP (viền + gradient) khó bắt đủ rõ.

**Việc tích hợp (model v2→v3, lọc khung ổn định) vẫn cần thiết và đã đúng hướng** — nhưng không đủ, vì đây là 2 tầng vấn đề độc lập: tầng tích hợp (đã sửa) và tầng model (chưa sửa, mới xác nhận). Hướng nâng cấp mục 3-4 ở trên (thu thêm ảnh C/V từ webcam thật, cân bằng lại số mẫu mỗi lớp trước khi train lại) giờ có cơ sở số liệu rõ ràng để ưu tiên — nên thêm V vào danh sách ưu tiên ngang C, không chỉ mình C.

## Class weight + dọn metadata (2026-09-30)

- **`train.py`**: thêm `--class_weight` (mặc định BẬT) — tính trọng số nghịch đảo tần suất mỗi lớp, truyền vào `CrossEntropyLoss(weight=...)`, chống thiên vị lớp nhiều mẫu (B=155 vs C=64/V=54). Đã smoke-test 1 epoch, log xác nhận đúng: `A=0.60, B=0.49, C=1.16, L=1.36, V=1.40`. Tắt bằng `--no_class_weight` nếu cần so sánh.
- **`clean_dataset_metadata.py`** (script mới): dọn các mục "samples" trong `vsl_dataset/metadata.json` trỏ tới file `.npy` đã mất trên đĩa. Chạy `--dry-run` trước để xem trước. Đã chạy: dọn 16 mục mồ côi (toàn bộ thuộc lớp A, không ảnh hưởng C/V), metadata giờ khớp đúng 512 file thật. Lưu ý: `train.py` tự quét thẳng thư mục `samples/` khi train (không đọc `metadata.json`), nên việc dọn này không ảnh hưởng kết quả train — chỉ để các script chẩn đoán khác (đối chiếu confusion matrix, v.v.) không lỗi `FileNotFoundError`.

## Ma trận nhầm lẫn THẬT của DTW trên 13 từ — quyết định chữ cái nào đáng train thêm (2026-09-30)

Trước khi mở rộng bảng chữ cái theo suy đoán "chữ nào nhìn giống từ nào", đo trực tiếp DTW đang thực sự nhầm cặp từ nào bằng quỹ đạo thuần (không dùng handshape). Công cụ mới: `eval_word_confusion.py` (LOSO, 10 lượt, 189 truy vấn). Kết quả: `results/baselines/word_confusion.json`.

| Từ | Acc (LOSO) | Nhầm với |
|---|---|---|
| WATER | 91,7% (11/12) | DRINK (1 lần) |
| 12 từ còn lại (gồm YES) | **100%** | không nhầm |

**Kết luận, thay cho suy đoán trước đó**:
- **DRINK↔WATER là cặp DUY NHẤT DTW thật sự nhầm** trong toàn bộ 13 từ (acc tổng 99,5%). Đây là bằng chứng trực tiếp để quyết định: **chỉ nên thêm chữ W** (khum tay 3 ngón của WATER) vào bộ handshape — vì nó giải quyết đúng cặp đang nhầm, khác hẳn C (khum 1 lòng bàn tay của DRINK) nên về lý thuyết dễ phân biệt bằng handshape hơn là bằng quỹ đạo thuần.
- **YES đã 100% chính xác bằng quỹ đạo (chuyển động nắm-đấm-gật đặc trưng)** — không có bằng chứng nào cho thấy cần thêm handshape S. Đề xuất trước đó ("YES gần giống A/S") là suy luận theo hình dạng chữ cái, sai hướng đầu tư — **bỏ hẳn kế hoạch train S**.
- Cảnh báo kỹ thuật (người dùng nêu, xác nhận hợp lý): W có nguy cơ nhầm với V/L vì cùng nhóm "ngón tay dựng vuông góc, khác nhau ở số ngón" — giống hệt kiểu lỗi C/V/L đã đo được ở model hiện tại (mục trên). Nếu train W, cần **nhiều mẫu hơn C/V** (có 2 "hàng xóm" dễ nhầm thay vì 1) và nên kiểm tra riêng độ phân giải ảnh DIP 128×128 có đủ bắt số ngón dựng hay không trước khi đầu tư nhiều công thu.

**Scope bộ chữ cái khuyến nghị (dựa trên bằng chứng, không suy đoán)**: A, B, C, L, V (đã có) + **W** (có lý do rõ ràng) — KHÔNG mở rộng toàn bộ 26 chữ, KHÔNG thêm S.

---

# Mô phỏng phiên xác nhận (confirm-enroll) trên Đăng & Vương (2026-09-30)

Công cụ: `eval_confirm.py` (mô phỏng một phiên `session_enroll.py`: mỗi mẫu của "người lạ" được nhận diện bằng bank hiện tại — bắt đầu là bank của Đạt — nếu không ACCEPTED thì hệ thống hỏi xác nhận, người dùng luôn trả lời đúng, mẫu được thêm vào bank cho các lần sau trong CÙNG phiên). Chạy: `py legacy/eval_confirm.py --targets Dang Vuong --repeats 3` (~1100s CPU). Kết quả: `results/baselines/confirm_dang_vuong.json`.

## Kết quả

| Người | mẫu/phiên | tĩnh (bank Đạt, không hỏi) | cap3 hỏi xác nhận (unc+rej) | P1→P4 (cap3) | hỏi | thêm |
|---|---|---|---|---|---|---|
| Mai (đối chứng cũ) | — | 66,0% | 87,5% | — | — | — |
| Đăng | 42 | 33,3% | 71,4% | 48%→83% | 12,0 | 12,0 |
| Vương | 52 | 17,3% | 66,0% | 38%→87% | 15,7 | 15,0 |

`cap2` cho số gần như y hệt `cap3` ở cả hai người → cap (số mẫu tối đa lưu mỗi từ) không phải nút thắt; `cap3 chỉ hỏi khi UNCERTAIN` (bỏ qua REJECTED) luôn thấp hơn hẳn `unc+rej` (Đăng 55,6% vs 71,4%; Vương 30,8% vs 66,0%) — nghĩa là phần lớn lợi ích đến từ việc hỏi CẢ những mẫu bị REJECTED, không chỉ UNCERTAIN.

## Phát hiện mới

1. **Đăng và Vương ở xa bank Đạt hơn Mai rất nhiều lúc chưa xác nhận gì** (tĩnh: 33,3% và 17,3% so với 66,0% của Mai) — khớp với phát hiện ở pilot 4 rằng bản thân Đạt là "người lạ khó" (target Dat luôn có baseline thấp nhất, ~60,9%) nhìn từ hướng ngược lại: bank của Đạt cũng khó với người khác tương xứng. Củng cố insight #2 đã ghi ở phần pilot 1-4: **độ khó không đối xứng và không đều giữa các cặp người** — không phải cứ "thêm người" là cải thiện đồng đều.
2. **Cơ chế hỏi-xác nhận hoạt động rất hiệu quả ngay trong 1 phiên** — chỉ với ~12-16 câu hỏi (trên 42-52 mẫu), độ chính xác tăng gấp đôi đến gấp bốn (Đăng 33%→71%, Vương 17%→66%), và đường cong theo 4 phần của phiên (P1→P4) cho thấy hệ thống "học" rất nhanh trong chính phiên đó (Vương: 38%→87% từ đầu đến cuối phiên). Đây là bằng chứng thực nghiệm trực tiếp cho tính "chống người lạ" bằng thích nghi tại chỗ — khác hẳn với việc chỉ đo độ chính xác tĩnh.
3. **Không có nét âm nào bị nhận nhầm thêm đáng kể** (Đăng 0/41, Vương chỉ 0,3/41 trung bình) sau khi bank được mở rộng bằng xác nhận trong phiên — nghĩa là việc thêm mẫu qua xác nhận không làm tăng rủi ro nhận nhầm cử chỉ ngẫu nhiên thành từ có nghĩa.
4. **Vương xuất phát khó nhất (17,3%) nhưng hội tụ ngang Đăng ở cuối phiên (87% vs 83%)** — cho thấy điểm xuất phát thấp không có nghĩa là người đó "khó dạy", chỉ là khoảng cách ban đầu với bank gốc lớn hơn.

## Ý nghĩa

Đây là một trục kết quả khác hẳn với các pilot học sâu/DTW tĩnh ở trên: thay vì hỏi "độ chính xác cố định là bao nhiêu", nó trả lời "hệ thống thích nghi nhanh đến đâu với MỘT người lạ hoàn toàn mới, chỉ trong một phiên duy nhất, không cần huấn luyện lại". Con số cụ thể (~12-16 câu hỏi để tăng gấp đôi độ chính xác, không tăng nhận nhầm) là một luận điểm mạnh cho tính ứng dụng thực tế — có thể đưa thẳng vào báo cáo như một "cold-start" guarantee, khác với hướng "cần bao nhiêu người tham chiếu" (pilot 4) vốn nói về pha CHUẨN BỊ bank trước, không phải pha SỬ DỤNG thực tế với người mới.

**Kết luận**: trước khi mở rộng số chữ cái, cần (1) bật `--with-cnn` thu thử một từ để có dữ liệu thật lần đầu tiên, (2) kiểm tra xem dùng khung hình ổn định (vận tốc thấp, `shape_conf` cao) thay vì lấy đa số cả 32 khung có giảm dao động nhãn không — việc này cần thu mới, chưa thực hiện.
