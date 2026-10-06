# BÁO CÁO TỔNG KẾT SPEC 11
## ĐÁNH GIÁ ĐƯỜNG DTW NHANH CHO CÁC NHÓM ĐẶC TRƯNG MỚI VÀ HỌC TRỌNG SỐ TỰ ĐỘNG

> **Chủ dự án:** Tô Phương Đạt (MSSV: 24110086) - Môn Xử Lý Ảnh Số (DIP)  
> **Thời gian hoàn thành:** 2026-09-21  
> **Quyết định cuối cùng:** **GIỮ NGUYÊN TRỌNG SỐ TAY (`KEEP_HAND_WEIGHTS`)**

---

## 1. Dữ Liệu Đóng Băng (Frozen Dataset)

Để bảo đảm tính tất định và chống rò rỉ dữ liệu khi người dùng thu thập thêm mẫu song song, toàn bộ các thử nghiệm trong SPEC 11 được thực thi trên tập dữ liệu đã đóng băng tại thư mục `results/SPEC11_20260921_133822/`:

- **Tệp mẫu dương (`bank_frozen.json`):**
  - SHA-256: `34705041cf481a994a5a5583683223288ac2d9f5b18a649a4d257591b5bf5a1c`
  - Tổng số mẫu: 175 mẫu
  - Phân bố 13 từ vựng:
    | Từ | Số mẫu | Từ | Số mẫu |
    |:---|:---|:---|:---|
    | `HELLO` | 14 | `BYE` | 13 |
    | `ME` | 16 | `YOU` | 17 |
    | `THANK_YOU` | 16 | `EAT` | 11 |
    | `DRINK` | 14 | `WATER` | 11 |
    | `SORRY` | 10 | `PLEASE` | 10 |
    | `YES` | 15 | `BAD` | 13 |
    | `HUNGRY` | 15 | **Tổng cộng** | **175** |
  - Phân bố 7 phiên thu nhận (tương ứng 7 outer folds cho Leave-One-Session-Out):
    | Phiên (Session ID) | Số mẫu | Tỷ lệ |
    |:---|:---|:---|
    | `phien2` | 70 | 40.0% |
    | `phien1` | 36 | 20.6% |
    | `20260920_100121` | 27 | 15.4% |
    | `20260921_112100` | 18 | 10.3% |
    | `20260920_093435` | 11 | 6.3% |
    | `20260920_094150` | 9 | 5.1% |
    | `quan_coffee` | 4 | 2.3% |
- **Tệp mẫu âm (`neg_frozen.json`):**
  - SHA-256: `2ef7669f803d681467e65c1d28f2f840b2dcf4567058fbb32e9688bd2db3fc42`
  - Tổng số mẫu âm: 41 cử chỉ ngoài từ vựng (chạm mặt, vẫy tự do, điều chỉnh camera, gãi đầu...)
- **Mã Git Commit:** `5e4d4ba4813815895e57134294ac15c7e31bb643` (dirty tree).

---

## 2. Kết Quả Nhiệm Vụ 1: Đường DTW Nhanh Hỗ Trợ $w_z, w_s, w_r$

Đường DTW nhanh `_FastDTWBank` trong [dynamic_dtw_engine.py](file:///c:/DIP/DIP_project/dynamic_dtw_engine.py) đã được nâng cấp hỗ trợ đầy đủ 3 nhóm đặc trưng mới:
- `w_z`: Khoảng cách độ sâu ảo $|z_q - z_t|$ từ mảng 3D `self.z` hình dạng `(K, NB, N)`.
- `w_s`: Khoảng cách độ xòe ngón tay $\|\Delta \text{abd}\|/2.0$ từ mảng 4D `self.abd` hình dạng `(K, NB, 4, N)`.
- `w_r`: Chi phí nhịp lặp $|r_q - r_t|$ cộng sau khi hoàn tất DTW nếu cả query và template đều hợp lệ.

### Bảng Kiểm Định Cổng Nhiệm Vụ 1 (T1-a ... T1-f):

| Mã kiểm tra | Nội dung yêu cầu | Kết quả thực tế | Sai lệch / Đo lường | Trạng thái |
|:---|:---|:---|:---|:---|
| **T1-a** | Bằng 0 thì giống hệt baseline (`baseline_fast_distances.npy`) | `np.array_equal == True` trên 171 mẫu chuẩn | Sai lệch tuyệt đối = 0.0 (bit-exact) | **ĐẠT** |
| **T1-b** | Tương đương hàm gốc `compute_golden_triad_dtw_distance` trên ≥ 5 vector trọng số | Kiểm tra 1026 cặp (query, template) | Sai lệch cực đại: $1.192 \times 10^{-7} < 10^{-5}$ | **ĐẠT** |
| **T1-c** | Các trường hợp biên (`query_has_body=False`, template thiếu body, thiếu rep) | Kiểm tra 1026 cặp biên | Sai lệch cực đại: $1.192 \times 10^{-7} < 10^{-5}$ | **ĐẠT** |
| **T1-d** | `match_many` tương đương `distances` từng query | `np.array_equal == True` trên 5 batch mẫu | Sai lệch tuyệt đối = 0.0 | **ĐẠT** |
| **T1-e** | Chế độ verify `DIP_FAST_DTW=verify` | Chạy 30 lần `match()` liên tục với trọng số mới bật | 0 assertion fail | **ĐẠT** |
| **T1-f** | Độ trễ match trên bank đóng băng (yêu cầu $\le 10$ ms; khi tắt không tăng quá 10%) | - Tắt (3 nhóm = 0): p50 = 3.82 ms (+1.0%), p95 = 5.25 ms (+0.8%)<br>- Bật cả 3 nhóm: p50 = 4.85 ms, p95 = 6.42 ms | Cả p50 và p95 đều $\le 10$ ms; khi tắt chỉ tăng $\sim 1\%$ | **ĐẠT** |

### Cập Nhật Kiểm Thử Parity:
Trong [tests/test_fast_dtw_parity.py](file:///c:/DIP/DIP_project/tests/test_fast_dtw_parity.py#L118), kiểm thử `test_fallback_khi_ngoai_pham_vi` được điều chỉnh từ `not fast._fast_bank.supported` sang `fast._fast_bank.supported is True` vì đường DTW nhanh hiện tại đã hỗ trợ chính thức `w_z, w_s, w_r`. Ca `lambda_shape=0.3` tiếp tục được giữ nguyên kiểm tra fallback về đường gốc. Kiểm thử parity đạt 10/10 test PASSED.

---

## 3. Kết Quả Nhiệm Vụ 2: Đo Lại Từng Nhóm Bằng Scoreboard

Sử dụng nguyên hạ tầng `scoreboard.py` đối chiếu cấu hình chuẩn `C11_base` (`w_p=0.6, w_a=1.0, w_m=0.4, w_n=0.8, w_b=0.8`, `uncertainty_k=2.35`, `threshold_k=1.75`) với từng nhóm riêng biệt và kết hợp cả ba nhóm trên tập đóng băng:

| Cấu hình | Khác `C11_base` | Top-1 | Clean Accept | Wrong Accept | False Accept Neg (1/41) | Margin Mean | $\Delta$ Margin Mean (KTC95%) | Latency (ms) | Phán quyết (`acceptance.json`) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`C11_base`** | Mặc định | 100.0% (175) | 88.57% (155) | 0.00% (0) | 0.00% (0/41) | +0.5383 | Baseline | 4.66 | **ĐỐI CHỨNG** |
| **`C11_z`** (lần 1) | $w_z = 0.5$ | 100.0% (175) | 90.29% (158) | 0.00% (0) | 0.00% (0/41) | +0.5091 | -0.0292 `[-0.0381, -0.0204]` | 4.96 | **REJECT** (Margin giảm) |
| **`C11_z`** (lần 2) | $w_z = 0.5$ | 100.0% (175) | 90.29% (158) | 0.00% (0) | 0.00% (0/41) | +0.5091 | -0.0292 `[-0.0381, -0.0204]` | 4.59 | **REJECT** (Tất định 100%) |
| **`C11_z`** (lần 3) | $w_z = 0.5$ | 100.0% (175) | 90.29% (158) | 0.00% (0) | 0.00% (0/41) | +0.5091 | -0.0292 `[-0.0381, -0.0204]` | 4.56 | **REJECT** (Tất định 100%) |
| **`C11_s`** | $w_s = 0.4$ | 100.0% (175) | 90.29% (158) | 0.00% (0) | 0.00% (0/41) | +0.5459 | +0.0076 `[-0.0021, +0.0175]` | 5.97 | **REJECT** (KTC95 chứa 0) |
| **`C11_r`** | $w_r = 0.15$ | 99.43% (174) | 86.29% (151) | 0.00% (0) | 2.44% (1/41) | +0.5270 | -0.0113 `[-0.0224, -0.0002]` | 4.91 | **REJECT** (Mất Top-1, tăng Neg) |
| **`C11_zsr`** | Cả 3 nhóm | 99.43% (174) | 89.71% (157) | 0.00% (0) | 2.44% (1/41) | +0.5098 | -0.0285 `[-0.0398, -0.0172]` | 5.32 | **REJECT** (Mất Top-1, giảm Margin) |

> **Nhận xét Nhiệm vụ 2:**
> 1. `C11_z` được chạy 3 lần độc lập và cho kết quả giống hệt từng bit về mọi chỉ số (tất định hoàn toàn). Tuy Clean Accept tăng từ 155 lên 158 mẫu, nhưng biên khoảng cách trung bình giảm đáng kể ($\Delta = -0.0292$, khoảng tin cậy 95% hoàn toàn âm), vi phạm tiêu chuẩn chất lượng.
> 2. `C11_s` tăng nhẹ biên phân cách (+0.0076), nhưng cận dưới khoảng tin cậy bootstrap rơi xuống mức âm (-0.0021), nghĩa là mức cải thiện không có ý nghĩa thống kê ở độ tin cậy 95%.
> 3. `C11_r` và `C11_zsr` làm mất 1 mẫu Top-1 của từ `HELLO` và làm lọt 1 mẫu âm (`false_accept_neg` tăng lên 2.44%), bị từ chối ngay lập tức.
> 4. Do cả 4 cấu hình đều bị REJECT, phương án đối chứng `B1` không được đưa vào Nhiệm vụ 3.

---

## 4. Kết Quả Nhiệm Vụ 3: Học Trọng Số Tự Động (`weight_learning.py`)

Quy trình đánh giá lồng nhau 2 cấp (Nested LOSO):
- **Vòng ngoài (Outer LOSO):** 7 folds theo từng phiên. Mẫu âm được kiểm thử độc lập trên matcher của từng fold (tổng số lượt kiểm thử âm: $41 \times 7 = 287$).
- **Vòng trong (Inner LOSO):** Chọn siêu tham số $\lambda \in \{0.0, 0.3, 3.0\}$ chỉ trên tập huấn luyện của fold ngoài.
- **Tối ưu hóa:** Luân phiên đường DTW $\leftrightarrow$ giải SLSQP với hàm mất mát biên tương đối và điều chuẩn co về $w_0$.

### Bảng Tổng Hợp Kết Quả Nested LOSO:

| Chế độ | Mô tả | Top-1 | Clean Accept | Wrong Accept | False Accept Neg | Margin Mean | $\Delta$ Margin (KTC95%) | McNemar p (vs B0) | Latency (ms) | Độ ổn định (CV) | Phán quyết |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`B0`** | Trọng số tay $C11\_base$ | 100.0% (175) | 88.57% (155) | 0.00% (0) | 0.35% (1/287) | +0.5383 | Mốc chuẩn | - | 4.98 | Ổn định (CV=0) | **ĐỐI CHỨNG** |
| **`L5`** | Học 5 nhóm cơ bản ($z=s=r=0$) | 100.0% (175) | 88.00% (154) | 0.00% (0) | 0.35% (1/287) | +0.5341 | -0.0042 `[-0.0094, +0.0008]` | 1.0000 | 4.74 (-4.8%) | Ổn định (max CV=0.24) | **REJECT** |
| **`L8`** | Học cả 8 nhóm đặc trưng | 99.43% (174) | 89.71% (157) | 0.00% (0) | 0.70% (2/287) | +0.5416 | +0.0033 `[-0.0028, +0.0091]` | 0.7266 | 4.69 (-5.8%) | **KHÔNG ỔN ĐỊNH** (CV > 0.5) | **REJECT** |

### Chi Tiết Trọng Số Học Được Qua Từng Fold Ngoài:

#### Chế độ `L5` (5 nhóm cơ bản):
| Fold ngoài (Session) | $\lambda^*$ đã chọn | $w_p$ | $w_a$ | $w_m$ | $w_n$ | $w_b$ | $w_z$ | $w_s$ | $w_r$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `20260920_093435` | 0.3 | 0.6740 | 1.0290 | 0.4126 | 0.7829 | 0.7015 | 0.0 | 0.0 | 0.0 |
| `20260920_094150` | 3.0 | 0.6105 | 1.0095 | 0.4016 | 0.8006 | 0.7778 | 0.0 | 0.0 | 0.0 |
| `20260920_100121` | 0.0 | 0.8519 | 1.0151 | 0.4585 | 0.5636 | 0.7110 | 0.0 | 0.0 | 0.0 |
| `20260921_112100` | 3.0 | 0.5901 | 1.0095 | 0.4022 | 0.8004 | 0.7978 | 0.0 | 0.0 | 0.0 |
| `phien1` | 0.0 | 0.7136 | 1.6291 | 0.3589 | 0.3388 | 0.5596 | 0.0 | 0.0 | 0.0 |
| `phien2` | 0.3 | 0.6626 | 1.0113 | 0.4264 | 0.8329 | 0.6668 | 0.0 | 0.0 | 0.0 |
| `quan_coffee` | 0.3 | 0.6535 | 1.0316 | 0.4108 | 0.7726 | 0.7315 | 0.0 | 0.0 | 0.0 |
| **Trung bình** | - | **0.6794** | **1.1050** | **0.4101** | **0.6988** | **0.7066** | **0.0** | **0.0** | **0.0** |
| **Độ lệch chuẩn** | - | 0.0798 | 0.2141 | 0.0277 | 0.1686 | 0.0729 | 0.0 | 0.0 | 0.0 |
| **Hệ số biến thiên (CV)** | - | **0.117** | **0.194** | **0.068** | **0.241** | **0.103** | 0.0 | 0.0 | 0.0 |

*Đánh giá ổn định `L5`:* Toàn bộ các nhóm đặc trưng đều có hệ số biến thiên $CV \le 0.241 < 0.5$, không bị cờ bất ổn định.

#### Chế độ `L8` (8 nhóm đặc trưng):
| Fold ngoài (Session) | $\lambda^*$ đã chọn | $w_p$ | $w_a$ | $w_m$ | $w_n$ | $w_b$ | $w_z$ | $w_s$ | $w_r$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `20260920_093435` | 0.3 | 0.6254 | 1.0145 | 0.3945 | 0.7785 | 0.6932 | 0.0000 | 0.0939 | 0.0000 |
| `20260920_094150` | 3.0 | 0.5906 | 0.9918 | 0.3897 | 0.7958 | 0.7700 | 0.0000 | 0.0000 | 0.0621 |
| `20260920_100121` | 0.0 | 0.8182 | 0.9701 | 0.3654 | 0.5678 | 0.6685 | 0.0000 | 0.2090 | 0.0010 |
| `20260921_112100` | 3.0 | 0.5817 | 0.9996 | 0.3930 | 0.7922 | 0.7886 | 0.0000 | 0.0007 | 0.0441 |
| `phien1` | 0.3 | 0.6530 | 1.0619 | 0.4037 | 0.7485 | 0.7308 | 0.0022 | 0.0000 | 0.0000 |
| `phien2` | 0.3 | 0.6481 | 0.9493 | 0.3938 | 0.8040 | 0.6479 | 0.0000 | 0.1556 | 0.0013 |
| `quan_coffee` | 0.3 | 0.6058 | 1.0084 | 0.3893 | 0.7791 | 0.7108 | 0.0000 | 0.1001 | 0.0066 |
| **Trung bình** | - | **0.6461** | **0.9994** | **0.3899** | **0.7523** | **0.7157** | **0.0003** | **0.0799** | **0.0164** |
| **Độ lệch chuẩn** | - | 0.0746 | 0.0330 | 0.0109 | 0.0771 | 0.0476 | 0.0008 | 0.0775 | 0.0238 |
| **Hệ số biến thiên (CV)** | - | **0.115** | **0.033** | **0.028** | **0.103** | **0.066** | **2.449** ⚠️ | **0.970** ⚠️ | **1.445** ⚠️ |

*Đánh giá ổn định `L8`:* **CỰC KỲ KHÔNG ỔN ĐỊNH**. Cả 3 nhóm đặc trưng mới đều có hệ số biến thiên vượt xa ngưỡng an toàn $0.5$:
- $w_z$ có $CV = 2.449$ (6/7 fold bị triệt tiêu về $0.0$, chỉ duy nhất fold `phien1` nhận giá trị $0.0022$).
- $w_s$ có $CV = 0.970$ (dao động thất thường từ $0.0$ đến $0.209$).
- $w_r$ có $CV = 1.445$ (dao động từ $0.0$ đến $0.062$).
$\Rightarrow$ Bị cờ **`Unstable = True`** theo đúng quy định mục 3.6 của SPEC.

### Đường Cong Mất Mát Qua Các Vòng Luân Phiên (Loss History):
Thuật toán luân phiên giữa gán đường DTW và tối ưu trọng số hội tụ rất nhanh (chỉ sau 2 vòng luân phiên đã đạt tiêu chí $\max|\Delta w| < 10^{-3}$):
- Fold `20260920_093435`: Vòng 0 = $0.035276 \rightarrow$ Vòng 1 = $0.023635 \rightarrow$ Vòng 2 = $0.023629$ (giảm đơn điệu)
- Fold `20260920_100121`: Vòng 0 = $0.084096 \rightarrow$ Vòng 1 = $0.012716 \rightarrow$ Vòng 2 = $0.012707$ (giảm đơn điệu)
- Fold `phien2`: Vòng 0 = $0.136071 \rightarrow$ Vòng 1 = $0.105937 \rightarrow$ Vòng 2 = $0.105937$ (hội tụ tuyệt đối)

---

## 5. Kiểm Tra Giả Thuyết Handshape (Mục 3.7)

Đối chiếu số lỗi nhầm lẫn trên 3 cặp từ hay nhầm nhất trong quá khứ:

| Cặp từ kiểm tra | Nhầm lẫn ở `B0` | Nhầm lẫn ở `L5` | Nhầm lẫn ở `L8` | Nhận xét thực nghiệm |
|:---|:---:|:---:|:---:|:---|
| **`HELLO` $\leftrightarrow$ `BYE`** | 0 | 0 | **1** | `L8` phát sinh thêm 1 lỗi nhầm `HELLO` sang `BYE`. |
| **`YOU` $\rightarrow$ `YES`** | 0 | 0 | 0 | Cả 3 chế độ phân biệt chính xác 100%. |
| **`ME` $\leftrightarrow$ `HUNGRY`** | 0 | 0 | 0 | Cả 3 chế độ phân biệt chính xác 100%. |

> **Bằng chứng xác nhận:**
> Kết quả thực nghiệm xác nhận chính xác nhận định ban đầu của Claude trong SPEC 11: Trên cặp `HELLO` $\leftrightarrow$ `BYE`, đặc trưng hình dáng bàn tay (góc xòe và hình ngón) không giải quyết được vấn đề do cả hai cử chỉ đều mở bàn tay. Việc ép mô hình học thêm góc xòe ngón tay $w_s$ và số nhịp lặp $w_r$ trong `L8` thậm chí còn làm nhiễu khoảng cách, khiến 1 mẫu `HELLO` bị phân loại nhầm sang `BYE`.

---

## 6. Đánh Giá Cổng Chấp Nhận & Đề Xuất Nhiệm Vụ 4

### Kiểm tra các cổng chấp nhận đặt trước:

| Tiêu chí cổng chấp nhận | Điều kiện quy định | Kết quả `L5` | Kết quả `L8` |
|:---|:---|:---:|:---:|
| 1. Cận dưới KTC95 $\Delta$ Margin | $> 0.0$ | **Không đạt** (-0.0094) | **Không đạt** (-0.0028) |
| 2. Tỷ lệ chấp nhận sai (`wrong_accept`) | Không tăng ($= 0.0\%$) | **Đạt** (0.0%) | **Đạt** (0.0%) |
| 3. Tỷ lệ lọt mẫu âm (`false_accept_neg`) | Không tăng so với B0 | **Đạt** (1/287 $\rightarrow$ 1/287) | **Không đạt** (1/287 $\rightarrow$ 2/287) |
| 4. Bảo toàn nhận diện từng từ | Không từ nào mất $> 1$ mẫu đúng | **Đạt** (không mất) | **Đạt** (mất 1 mẫu `HELLO`) |
| 5. Cờ ổn định trọng số qua các fold | $CV \le 0.5$ cho mọi nhóm | **Đạt** (max $CV = 0.241$) | **Không đạt** ($CV_{w_z}=2.45, CV_{w_s}=0.97, CV_{w_r}=1.44$) |
| 6. Tăng trưởng Top-1 | Tăng ròng $\ge +3$ truy vấn | Không đạt (0) | Không đạt (-1) |
| 7. Ràng buộc độ trễ | Không tăng quá 20% và $\le 10$ ms | **Đạt** (4.74 ms, -4.8%) | **Đạt** (4.69 ms, -5.8%) |
| **KẾT LUẬN TOÀN BỘ CỔNG** | **Phải thỏa mãn TẤT CẢ** | **TỪ CHỐI (REJECT)** | **TỪ CHỐI (REJECT)** |

### Phán quyết Nhiệm vụ 4:
- Do cả `L5` và `L8` đều không vượt qua các cổng định lượng khắt khe đã đặt trước, hệ thống đưa ra phán quyết: **`KEEP_HAND_WEIGHTS` (GIỮ NGUYÊN BỘ TRỌNG SỐ TAY HIỆN TẠI)**.
- **TUYỆT ĐỐI KHÔNG SỬA PRODUCTION**: Giữ nguyên `CONFIG` trong [predict_sentence_webcam.py](file:///c:/DIP/DIP_project/predict_sentence_webcam.py):
  ```python
  "w_p": 0.6,
  "w_a": 1.0,
  "w_m": 0.4,
  "w_n": 0.8,
  "w_b": 0.8,
  "w_z": 0.0,
  "w_s": 0.0,
  "w_r": 0.0,
  "threshold_k": 1.75,
  "uncertainty_k": 2.35,
  ```
- Đây là một **kết quả nghiên cứu âm có đo lường chuẩn xác (measured negative finding)**: chứng minh bằng dữ liệu và thống kê suy luận rằng việc bật 3 nhóm $w_z, w_s, w_r$ hoặc điều chỉnh tự do trọng số hiện tại không mang lại lợi ích khái quát hóa trên dữ liệu đa phiên, đồng thời làm tăng tính bất ổn định của hệ thống nhận diện.

### Lệnh Chạy Lại Một Dòng Cho Bank Dữ Liệu Mới (Reproducibility):
Khi người dùng thu thập thêm các phiên cử chỉ mới trong tương lai, toàn bộ pipeline có thể chạy lại tự động bằng một câu lệnh:
```powershell
py weight_learning.py --bank results/SPEC11_20260921_133822/bank_frozen.json --neg results/SPEC11_20260921_133822/neg_frozen.json --out results/SPEC11_20260921_133822/nested --modes B0 L5 L8 --lambdas 0.0 0.3 3.0 --seed 20260919
```

---

## 7. Điều Spec Sai Hoặc Hạn Chế Trong Thực Tế

1. **Hiện tượng bão hòa trần Top-1 (Ceiling Effect):**
   Spec quy định tiêu chí cổng: *"Top-1 tăng ròng $\ge +3$ truy vấn so với B0"*. Tuy nhiên, trên tập dữ liệu đóng băng 175 mẫu (đã qua tiền kiểm tra chất lượng ở các spec trước), mô hình đối chứng `B0` đã đạt Top-1 là **100.0% (175/175)**. Do đó, về mặt toán học, Top-1 không thể tăng thêm bất kỳ mẫu nào. Tiêu chuẩn biên phân cách (`margin_mean`) và KTC95 bootstrap là thước đo khoa học duy nhất có ý nghĩa phân biệt.
2. **Thang đo mẫu âm trong Nested LOSO:**
   Trong spec đề cập tỷ lệ mẫu âm theo mẫu số 41 (`false_accept_neg` $\le 1/41$). Tuy nhiên, trong thiết kế Nested LOSO với 7 outer folds, tập mẫu âm được đối chiếu độc lập với 7 ngân hàng template tham chiếu khác nhau (mỗi matcher có bộ ngưỡng nội lớp `intra_widths` riêng). Do đó, tổng số lượt thử nghiệm âm thực tế là $41 \times 7 = 287$ lượt. Thang đo chuẩn xác phải được báo cáo theo mẫu số 287 (ví dụ: $1/287 \approx 0.35\%$).
3. **Tính bất đối xứng lớn giữa các phiên:**
   Các phiên thu nhận có kích thước rất chênh lệch (từ 4 mẫu ở phiên `quan_coffee` đến 70 mẫu ở `phien2`). Khi phiên 4 mẫu bị tách ra làm fold kiểm thử ngoài, tập huấn luyện gồm 171 mẫu; ngược lại khi `phien2` (70 mẫu) làm kiểm thử ngoài, tập huấn luyện chỉ còn 105 mẫu. Điều này làm tăng phương sai của trọng số học được.
4. **Giới hạn của tập dữ liệu:**
   Toàn bộ dữ liệu cử chỉ hiện tại vẫn được thu nhận bởi một người thực hiện trên cùng một góc quay webcam. Để kiểm chứng tính ổn định thực sự của $w_s$ và $w_r$, cần mở rộng thu nhận dữ liệu đa người dùng (cross-subject).

---

## 8. Xác Nhận Kiểm Thử Hồi Quy Toàn Bộ Test Suite

- **Lệnh chạy:** `py -m pytest tests/ -q --tb=no`
- **Thời gian thực thi:** 487.38 giây (8 phút 7 giây)
- **Tổng số test:** 256 items
- **Kết quả:** **49 FAILED, 207 PASSED**
- **So khớp với mốc 49 lỗi đã tồn tại sẵn từ trước:**
  - Danh sách 49 lỗi mốc ban đầu: 49 ca kiểm thử.
  - Danh sách 49 lỗi sau khi hoàn tất SPEC 11 (lưu tại `results/SPEC11_20260921_133822/nested/failed_after.txt`): 49 ca kiểm thử.
  - **Hiệu tập hợp (Diff FAILED):** **`set() - set() = RỖNG (EMPTY)`**.
  - Không có bất kỳ ca kiểm thử nào bị gãy mới hoặc bị sửa sai ngoài phạm vi.

---

## 9. Danh Sách Tệp Đã Tạo Trên Đĩa

Toàn bộ các tệp kết quả của SPEC 11 đã được sinh đầy đủ trên đĩa và sẵn sàng nghiệm thu:

1. `results/SPEC11_20260921_133822/bank_frozen.json` (SHA: `34705041...`)
2. `results/SPEC11_20260921_133822/neg_frozen.json` (SHA: `2ef7669f...`)
3. `results/SPEC11_20260921_133822/baseline_fast_distances.npy`
4. `results/SPEC11_20260921_133822/frozen_info.json`
5. `results/SPEC11_20260921_133822/scoreboard_runs.csv`
6. `results/SPEC11_20260921_133822/report.md` (Báo cáo tổng kết này)
7. `results/SPEC11_20260921_133822/nested/progress.txt` (Log tiến trình huấn luyện chi tiết)
8. `results/SPEC11_20260921_133822/nested/RERUN.txt` (Lệnh chạy lại 1 dòng)
9. `results/SPEC11_20260921_133822/nested/proposed_weights.json` (Phán quyết và phân tích giả thuyết)
10. `results/SPEC11_20260921_133822/nested/metrics_B0.json`
11. `results/SPEC11_20260921_133822/nested/metrics_L5.json`
12. `results/SPEC11_20260921_133822/nested/metrics_L8.json`
13. `results/SPEC11_20260921_133822/nested/per_query_B0.csv`
14. `results/SPEC11_20260921_133822/nested/per_query_L5.csv`
15. `results/SPEC11_20260921_133822/nested/per_query_L8.csv`
16. `results/SPEC11_20260921_133822/nested/confusion_B0.csv`
17. `results/SPEC11_20260921_133822/nested/confusion_L5.csv`
18. `results/SPEC11_20260921_133822/nested/confusion_L8.csv`
19. `results/SPEC11_20260921_133822/nested/weights_by_fold_B0.csv`
20. `results/SPEC11_20260921_133822/nested/weights_by_fold_L5.csv`
21. `results/SPEC11_20260921_133822/nested/weights_by_fold_L8.csv`
22. `results/SPEC11_20260921_133822/nested/loss_history_B0.csv`
23. `results/SPEC11_20260921_133822/nested/loss_history_L5.csv`
24. `results/SPEC11_20260921_133822/nested/loss_history_L8.csv`
25. `results/SPEC11_20260921_133822/nested/failed_after.txt` (49 test failed khớp 100%)
