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
