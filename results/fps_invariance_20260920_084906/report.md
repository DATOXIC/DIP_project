# Báo cáo thí nghiệm bất biến FPS — 20260920_084906

- Bank: `webcam_templates.json` (13 từ, 107 mẫu). Stroke thô dùng: 171 (trong bank/LOO: 93, ngoài bank: 78). Seed=0, limit=0.
- Kiểm chứng (a) trên chính bank, không LOO: top-1 = 100.0%, d_đúng trung vị = 0.00008.
- Ngưỡng/khoảng cách: matcher như run_sentence_builder (threshold_mode=scale_k, k=1.6, unc_k=2.2, w_b=0.8).
- Mọi số ở bảng chính là trên stroke CÓ mẫu trong bank, bank đã loại đúng mẫu đó (leave-one-out).

## 0. Nhịp thời gian của kho thô (toàn bộ stroke thô, bỏ NEGATIVE)

| Phiên | Số stroke | Số cặp khung | Tốc độ đọc 1/mean(dt) | "FPS hiển thị" mean(1/dt) | Tỉ lệ khung TRÙNG | FPS khung MỚI | dt p05/p50/p95 (ms) | dt<15ms |
|---|---|---|---|---|---|---|---|---|
| phien1 | 51 | 1532 | 29.8 | 39.7 | 35.6% | 19.2 | 13/32/64 | 21% |
| phien2 | 115 | 3510 | 29.8 | 52.8 | 43.8% | 16.7 | 9/37/65 | 41% |
| thu_nghiem | 5 | 118 | 30.1 | 31.1 | 0.8% | 29.8 | 29/32/46 | 1% |

`FPS hiển thị` mô phỏng đại lượng mà EMA `fps = 0.9*fps + 0.1*(1/dt)` của predict hội tụ tới: khi các lần đọc dồn cục (dt ngắn xen dt dài) nó LỚN HƠN tốc độ đọc thật, và khi khung bị đọc trùng thì FPS khung MỚI còn nhỏ hơn nữa.

## 1. So khớp theo fps — Stroke trong bank (LOO) — BẢNG CHÍNH

### 1a. Chế độ `asis` (pipeline hiện tại, KHÔNG chỉnh gì)

| Nhóm | N | top-1 [CI95 Wilson] | d_đúng mean/med | margin (d_sai_min - d_đúng) mean/med | ACC/UNC/REJ (%) | ACCEPTED&đúng | không dựng được |
|---|---|---|---|---|---|---|---|
| a_goc | 93 | 100.0% [96-100] | 0.192 / 0.176 | +0.400 / +0.398 | 97 / 2 / 1 | 96.8% | 0 |
| b_60fps | 93 | 100.0% [96-100] | 0.188 / 0.167 | +0.395 / +0.396 | 96 / 3 / 1 | 95.7% | 0 |
| c_15fps | 93 | 100.0% [96-100] | 0.203 / 0.187 | +0.404 / +0.408 | 95 / 4 / 1 | 94.6% | 0 |
| d_trung | 93 | 100.0% [96-100] | 0.192 / 0.176 | +0.399 / +0.398 | 96 / 3 / 1 | 95.7% | 0 |
| e_45fps | 93 | 100.0% [96-100] | 0.188 / 0.170 | +0.397 / +0.397 | 96 / 3 / 1 | 95.7% | 0 |
| f_bo_trung | 93 | 100.0% [96-100] | 0.199 / 0.179 | +0.404 / +0.406 | 96 / 3 / 1 | 95.7% | 0 |
| g_palm_predict | 93 | 100.0% [96-100] | 0.210 / 0.197 | +0.408 / +0.391 | 94 / 5 / 1 | 93.5% | 0 |

### 1b. Ảnh hưởng theo nhóm cột (top-1 / d_đúng mean / ACCEPTED&đúng)

| Biến thể | asis | scale_vel | scale_dang | scale_both | orig_deriv |
|---|---|---|---|---|---|
| a_goc | 100.0% / 0.192 / 97% | 100.0% / 0.192 / 97% | 100.0% / 0.192 / 97% | 100.0% / 0.192 / 97% | 100.0% / 0.192 / 97% |
| b_60fps | 100.0% / 0.188 / 96% | 100.0% / 0.189 / 97% | 100.0% / 0.192 / 96% | 100.0% / 0.193 / 96% | 100.0% / 0.192 / 97% |
| c_15fps | 100.0% / 0.203 / 95% | 100.0% / 0.195 / 96% | 100.0% / 0.196 / 96% | 100.0% / 0.188 / 96% | 100.0% / 0.191 / 96% |
| d_trung | 100.0% / 0.192 / 96% | 100.0% / 0.192 / 96% | 100.0% / 0.192 / 96% | 100.0% / 0.192 / 96% | 100.0% / 0.192 / 96% |
| e_45fps | 100.0% / 0.188 / 96% | 100.0% / 0.189 / 97% | 100.0% / 0.191 / 96% | 100.0% / 0.192 / 96% | 100.0% / 0.192 / 96% |
| f_bo_trung | 100.0% / 0.199 / 96% | 100.0% / 0.193 / 96% | 100.0% / 0.194 / 97% | 100.0% / 0.188 / 96% | 100.0% / 0.191 / 96% |
| g_palm_predict | 100.0% / 0.210 / 94% | 100.0% / 0.210 / 94% | 100.0% / 0.210 / 94% | 100.0% / 0.210 / 95% | 100.0% / 0.209 / 94% |

Chú giải chế độ: `asis` = nguyên trạng (pipeline hiện tại); `scale_vel` = nhân dxs,dys x fps/30; `scale_dang` = nhân dangles x fps/30; `scale_both` = nhân cả hai nhóm x fps/30; `orig_deriv` = cận trên: thay đạo hàm bằng của bản gốc (a)

Hệ số nhân trung bình (fps_thực/30): a_goc=1.00, b_60fps=1.99, c_15fps=0.50, d_trung=1.00, e_45fps=1.50, f_bo_trung=0.59, g_palm_predict=1.00

### 1c. So cặp với (a), chế độ `asis`: số stroke đúng->sai / sai->đúng

| Biến thể | đúng->sai | sai->đúng |
|---|---|---|
| b_60fps | 0 | 0 |
| c_15fps | 0 | 0 |
| d_trung | 0 | 0 |
| e_45fps | 0 | 0 |
| f_bo_trung | 0 | 0 |
| g_palm_predict | 0 | 0 |

### 1d. Chênh lệch có cặp so với (a): Δd_đúng và Δmargin (mean, CI95 bootstrap 2000 lần, seed 0)

Top-1 thô (rời rạc) ít nhạy nên dùng khoảng cách làm thước đo. Δ>0 ở d_đúng = xấu đi; Δ<0 ở margin = xấu đi.

| Biến thể | Chế độ | Δd_đúng mean [CI95] | Δd_đúng % so với d(a) | Δmargin mean [CI95] |
|---|---|---|---|---|
| b_60fps | asis | -0.0037 [-0.0048, -0.0027] | -1.9% | -0.0044 [-0.0052, -0.0037] |
| b_60fps | scale_both | +0.0013 [+0.0009, +0.0017] | +0.7% | -0.0004 [-0.0009, +0.0001] |
| c_15fps | asis | +0.0114 [+0.0100, +0.0130] | +6.0% | +0.0038 [+0.0019, +0.0059] |
| c_15fps | scale_both | -0.0036 [-0.0049, -0.0024] | -1.9% | -0.0002 [-0.0016, +0.0013] |
| d_trung | asis | -0.0001 [-0.0014, +0.0011] | -0.0% | -0.0010 [-0.0027, +0.0010] |
| d_trung | scale_both | -0.0002 [-0.0015, +0.0011] | -0.1% | -0.0010 [-0.0028, +0.0010] |
| e_45fps | asis | -0.0036 [-0.0045, -0.0028] | -1.9% | -0.0027 [-0.0031, -0.0022] |
| e_45fps | scale_both | -0.0003 [-0.0007, -0.0000] | -0.2% | -0.0000 [-0.0003, +0.0003] |
| f_bo_trung | asis | +0.0070 [+0.0055, +0.0085] | +3.7% | +0.0042 [+0.0020, +0.0064] |
| f_bo_trung | scale_both | -0.0036 [-0.0047, -0.0024] | -1.9% | +0.0010 [-0.0007, +0.0027] |
| g_palm_predict | asis | +0.0179 [+0.0114, +0.0247] | +9.3% | +0.0082 [+0.0007, +0.0161] |
| g_palm_predict | scale_both | +0.0178 [+0.0115, +0.0246] | +9.3% | +0.0082 [+0.0004, +0.0157] |

## 1. So khớp theo fps — Stroke ngoài bank (bank đầy đủ, không cần LOO) — tham khảo

### 1a. Chế độ `asis` (pipeline hiện tại, KHÔNG chỉnh gì)

| Nhóm | N | top-1 [CI95 Wilson] | d_đúng mean/med | margin (d_sai_min - d_đúng) mean/med | ACC/UNC/REJ (%) | ACCEPTED&đúng | không dựng được |
|---|---|---|---|---|---|---|---|
| a_goc | 78 | 60.3% [49-70] | 0.470 / 0.313 | +0.056 / +0.271 | 64 / 10 / 26 | 47.4% | 0 |
| b_60fps | 78 | 60.3% [49-70] | 0.466 / 0.311 | +0.055 / +0.262 | 65 / 10 / 24 | 48.7% | 0 |
| c_15fps | 78 | 60.3% [49-70] | 0.483 / 0.321 | +0.058 / +0.270 | 60 / 13 / 27 | 43.6% | 0 |
| d_trung | 78 | 60.3% [49-70] | 0.469 / 0.315 | +0.057 / +0.270 | 64 / 12 / 24 | 47.4% | 0 |
| e_45fps | 78 | 60.3% [49-70] | 0.466 / 0.310 | +0.056 / +0.267 | 65 / 10 / 24 | 48.7% | 0 |
| f_bo_trung | 78 | 60.3% [49-70] | 0.481 / 0.319 | +0.056 / +0.267 | 62 / 10 / 28 | 44.9% | 0 |
| g_palm_predict | 78 | 61.5% [50-72] | 0.485 / 0.344 | +0.073 / +0.289 | 59 / 14 / 27 | 42.3% | 0 |

### 1b. Ảnh hưởng theo nhóm cột (top-1 / d_đúng mean / ACCEPTED&đúng)

| Biến thể | asis | scale_vel | scale_dang | scale_both | orig_deriv |
|---|---|---|---|---|---|
| a_goc | 60.3% / 0.470 / 47% | 60.3% / 0.470 / 47% | 60.3% / 0.470 / 47% | 60.3% / 0.470 / 47% | 60.3% / 0.470 / 47% |
| b_60fps | 60.3% / 0.466 / 49% | 60.3% / 0.468 / 47% | 60.3% / 0.470 / 49% | 60.3% / 0.472 / 47% | 60.3% / 0.471 / 47% |
| c_15fps | 60.3% / 0.483 / 44% | 60.3% / 0.475 / 47% | 60.3% / 0.477 / 44% | 60.3% / 0.469 / 50% | 60.3% / 0.470 / 50% |
| d_trung | 60.3% / 0.469 / 47% | 60.3% / 0.469 / 47% | 60.3% / 0.469 / 47% | 60.3% / 0.469 / 47% | 60.3% / 0.470 / 47% |
| e_45fps | 60.3% / 0.466 / 49% | 60.3% / 0.468 / 49% | 60.3% / 0.469 / 49% | 60.3% / 0.470 / 49% | 60.3% / 0.470 / 47% |
| f_bo_trung | 60.3% / 0.481 / 45% | 60.3% / 0.475 / 47% | 60.3% / 0.476 / 46% | 60.3% / 0.470 / 50% | 60.3% / 0.472 / 49% |
| g_palm_predict | 61.5% / 0.485 / 42% | 61.5% / 0.485 / 42% | 61.5% / 0.485 / 42% | 61.5% / 0.485 / 42% | 61.5% / 0.484 / 44% |

Chú giải chế độ: `asis` = nguyên trạng (pipeline hiện tại); `scale_vel` = nhân dxs,dys x fps/30; `scale_dang` = nhân dangles x fps/30; `scale_both` = nhân cả hai nhóm x fps/30; `orig_deriv` = cận trên: thay đạo hàm bằng của bản gốc (a)

Hệ số nhân trung bình (fps_thực/30): a_goc=0.99, b_60fps=1.98, c_15fps=0.50, d_trung=0.99, e_45fps=1.49, f_bo_trung=0.62, g_palm_predict=0.99

### 1c. So cặp với (a), chế độ `asis`: số stroke đúng->sai / sai->đúng

| Biến thể | đúng->sai | sai->đúng |
|---|---|---|
| b_60fps | 0 | 0 |
| c_15fps | 0 | 0 |
| d_trung | 0 | 0 |
| e_45fps | 0 | 0 |
| f_bo_trung | 0 | 0 |
| g_palm_predict | 0 | 1 |

### 1d. Chênh lệch có cặp so với (a): Δd_đúng và Δmargin (mean, CI95 bootstrap 2000 lần, seed 0)

Top-1 thô (rời rạc) ít nhạy nên dùng khoảng cách làm thước đo. Δ>0 ở d_đúng = xấu đi; Δ<0 ở margin = xấu đi.

| Biến thể | Chế độ | Δd_đúng mean [CI95] | Δd_đúng % so với d(a) | Δmargin mean [CI95] |
|---|---|---|---|---|
| b_60fps | asis | -0.0045 [-0.0058, -0.0032] | -0.9% | -0.0012 [-0.0028, +0.0003] |
| b_60fps | scale_both | +0.0012 [+0.0006, +0.0018] | +0.2% | -0.0003 [-0.0011, +0.0004] |
| c_15fps | asis | +0.0128 [+0.0107, +0.0152] | +2.7% | +0.0015 [-0.0011, +0.0041] |
| c_15fps | scale_both | -0.0017 [-0.0036, +0.0003] | -0.4% | +0.0006 [-0.0022, +0.0033] |
| d_trung | asis | -0.0011 [-0.0032, +0.0011] | -0.2% | +0.0011 [-0.0012, +0.0033] |
| d_trung | scale_both | -0.0013 [-0.0034, +0.0010] | -0.3% | +0.0010 [-0.0014, +0.0033] |
| e_45fps | asis | -0.0040 [-0.0051, -0.0029] | -0.8% | -0.0003 [-0.0014, +0.0009] |
| e_45fps | scale_both | -0.0003 [-0.0008, +0.0002] | -0.1% | +0.0002 [-0.0004, +0.0008] |
| f_bo_trung | asis | +0.0108 [+0.0082, +0.0132] | +2.3% | -0.0003 [-0.0028, +0.0022] |
| f_bo_trung | scale_both | -0.0001 [-0.0018, +0.0019] | -0.0% | -0.0006 [-0.0036, +0.0021] |
| g_palm_predict | asis | +0.0147 [+0.0084, +0.0213] | +3.1% | +0.0173 [+0.0098, +0.0250] |
| g_palm_predict | scale_both | +0.0145 [+0.0082, +0.0210] | +3.1% | +0.0172 [+0.0098, +0.0252] |

## 2. Phân đoạn (KineticPauseSegmenter, cấu hình CONFIG hiện hành)

Cấu hình: v_start=0.08, v_stop=0.04, pause_frames=9, min/max_stroke_frames=8/60, require_ready_pose=True, grace=0.15s, max_phase_merges=1 (từ make_segmenter). Luồng = đệm tĩnh 0.8s + stroke + đệm tĩnh 1.5s. Số liệu tính trên TẤT CẢ stroke (phân đoạn không phụ thuộc bank).

`asis` = segmenter y hệt production. `scaled` = THỬ NGHIỆM: v_start/v_stop chia r, pause_frames/ready_stable/max_stroke_frames/min_stroke_frames nhân r (sàn min=8), r = fps_thực/30 của biến thể; cho biết việc chuẩn hóa ngưỡng theo thời gian có cứu được phân đoạn không.

| Biến thể | Segmenter | N | đúng 1 stroke | cắt đôi (>=2) | nhỡ (0) | qua cổng production | swipe giả | thời lượng chốt (s) median | tỉ lệ so với (a) median | trễ bắt đầu (s) median | lệch kết thúc (s) median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a_goc | asis | 171 | 88.3% | 0.0% | 11.7% | 88.3% | 0.6% | 0.90 | 1.00 | +0.06 | +0.00 |
| a_goc | scaled | 171 | 88.3% | 0.0% | 11.7% | 88.3% | 0.6% | 0.88 | 1.00 | +0.06 | +0.00 |
| b_60fps | asis | 171 | 49.7% | 0.6% | 49.7% | 39.2% | 0.0% | 0.50 | 0.68 | +0.04 | -0.24 |
| b_60fps | scaled | 171 | 88.9% | 0.0% | 11.1% | 88.9% | 0.0% | 0.88 | 1.00 | +0.04 | -0.02 |
| c_15fps | asis | 171 | 90.6% | 0.0% | 9.4% | 88.3% | 2.9% | 1.33 | 1.46 | +0.00 | +0.39 |
| c_15fps | scaled | 171 | 73.7% | 0.0% | 26.3% | 70.8% | 2.9% | 0.85 | 1.03 | +0.00 | +0.00 |
| d_trung | asis | 171 | 93.0% | 0.0% | 7.0% | 91.2% | 0.0% | 0.91 | 1.00 | +0.07 | +0.00 |
| d_trung | scaled | 171 | 93.0% | 0.0% | 7.0% | 91.2% | 0.0% | 0.90 | 1.00 | +0.07 | +0.00 |
| e_45fps | asis | 171 | 67.8% | 0.0% | 32.2% | 63.7% | 0.0% | 0.63 | 0.81 | +0.05 | -0.15 |
| e_45fps | scaled | 171 | 88.9% | 0.0% | 11.1% | 88.9% | 0.0% | 0.87 | 1.00 | +0.03 | +0.00 |
| f_bo_trung | asis | 171 | 89.5% | 0.0% | 10.5% | 88.3% | 1.2% | 1.14 | 1.31 | +0.00 | +0.24 |
| f_bo_trung | scaled | 171 | 76.6% | 0.0% | 23.4% | 74.9% | 1.2% | 0.80 | 1.00 | +0.00 | +0.00 |
| g_palm_predict | asis | 171 | 92.4% | 0.0% | 7.6% | 91.8% | 0.6% | 0.92 | 1.00 | +0.06 | +0.00 |
| g_palm_predict | scaled | 171 | 91.8% | 0.0% | 8.2% | 91.2% | 0.6% | 0.91 | 1.00 | +0.06 | +0.00 |

Số stroke có số khung vượt trần max_stroke_frames=60 (segmenter asis) theo biến thể: a_goc=0, b_60fps=74, c_15fps=0, d_trung=0, e_45fps=42, f_bo_trung=0, g_palm_predict=0

## 3. Đầu-cuối: phân đoạn -> đặc trưng -> so khớp — stroke TRONG bank (LOO) — bảng chính

`thành công` = đúng MỘT từ được chốt (ACCEPTED) và đó là từ đúng; `từ sai` = số từ ACCEPTED khác từ đúng. `Đặc trưng scale_both` = nhân đạo hàm x fps/30 (fps đo từ chính stroke được chốt).

| Biến thể | Segmenter | Đặc trưng | N | thành công | tổng từ sai chốt | stroke có từ sai |
|---|---|---|---|---|---|---|
| a_goc | asis | asis | 93 | 88.2% | 0 | 0 |
| a_goc | asis | scale_both | 93 | 88.2% | 0 | 0 |
| a_goc | scaled | asis | 93 | 88.2% | 0 | 0 |
| a_goc | scaled | scale_both | 93 | 88.2% | 0 | 0 |
| b_60fps | asis | asis | 93 | 35.5% | 0 | 0 |
| b_60fps | asis | scale_both | 93 | 35.5% | 0 | 0 |
| b_60fps | scaled | asis | 93 | 91.4% | 0 | 0 |
| b_60fps | scaled | scale_both | 93 | 91.4% | 0 | 0 |
| c_15fps | asis | asis | 93 | 87.1% | 0 | 0 |
| c_15fps | asis | scale_both | 93 | 87.1% | 0 | 0 |
| c_15fps | scaled | asis | 93 | 68.8% | 0 | 0 |
| c_15fps | scaled | scale_both | 93 | 68.8% | 0 | 0 |
| d_trung | asis | asis | 93 | 84.9% | 0 | 0 |
| d_trung | asis | scale_both | 93 | 84.9% | 0 | 0 |
| d_trung | scaled | asis | 93 | 84.9% | 0 | 0 |
| d_trung | scaled | scale_both | 93 | 84.9% | 0 | 0 |
| e_45fps | asis | asis | 93 | 65.6% | 0 | 0 |
| e_45fps | asis | scale_both | 93 | 65.6% | 0 | 0 |
| e_45fps | scaled | asis | 93 | 91.4% | 0 | 0 |
| e_45fps | scaled | scale_both | 93 | 91.4% | 0 | 0 |
| f_bo_trung | asis | asis | 93 | 91.4% | 0 | 0 |
| f_bo_trung | asis | scale_both | 93 | 91.4% | 0 | 0 |
| f_bo_trung | scaled | asis | 93 | 75.3% | 0 | 0 |
| f_bo_trung | scaled | scale_both | 93 | 75.3% | 0 | 0 |
| g_palm_predict | asis | asis | 93 | 87.1% | 0 | 0 |
| g_palm_predict | asis | scale_both | 93 | 87.1% | 0 | 0 |
| g_palm_predict | scaled | asis | 93 | 87.1% | 0 | 0 |
| g_palm_predict | scaled | scale_both | 93 | 87.1% | 0 | 0 |

## 3. Đầu-cuối: phân đoạn -> đặc trưng -> so khớp — stroke NGOÀI bank (bank đầy đủ) — tham khảo

`thành công` = đúng MỘT từ được chốt (ACCEPTED) và đó là từ đúng; `từ sai` = số từ ACCEPTED khác từ đúng. `Đặc trưng scale_both` = nhân đạo hàm x fps/30 (fps đo từ chính stroke được chốt).

| Biến thể | Segmenter | Đặc trưng | N | thành công | tổng từ sai chốt | stroke có từ sai |
|---|---|---|---|---|---|---|
| a_goc | asis | asis | 78 | 37.2% | 7 | 7 |
| a_goc | asis | scale_both | 78 | 37.2% | 7 | 7 |
| a_goc | scaled | asis | 78 | 35.9% | 7 | 7 |
| a_goc | scaled | scale_both | 78 | 35.9% | 7 | 7 |
| b_60fps | asis | asis | 78 | 20.5% | 1 | 1 |
| b_60fps | asis | scale_both | 78 | 19.2% | 1 | 1 |
| b_60fps | scaled | asis | 78 | 41.0% | 7 | 7 |
| b_60fps | scaled | scale_both | 78 | 38.5% | 7 | 7 |
| c_15fps | asis | asis | 78 | 41.0% | 9 | 9 |
| c_15fps | asis | scale_both | 78 | 44.9% | 9 | 9 |
| c_15fps | scaled | asis | 78 | 30.8% | 2 | 2 |
| c_15fps | scaled | scale_both | 78 | 35.9% | 2 | 2 |
| d_trung | asis | asis | 78 | 38.5% | 9 | 9 |
| d_trung | asis | scale_both | 78 | 38.5% | 9 | 9 |
| d_trung | scaled | asis | 78 | 37.2% | 9 | 9 |
| d_trung | scaled | scale_both | 78 | 37.2% | 9 | 9 |
| e_45fps | asis | asis | 78 | 30.8% | 3 | 3 |
| e_45fps | asis | scale_both | 78 | 30.8% | 3 | 3 |
| e_45fps | scaled | asis | 78 | 42.3% | 7 | 7 |
| e_45fps | scaled | scale_both | 78 | 42.3% | 7 | 7 |
| f_bo_trung | asis | asis | 78 | 41.0% | 7 | 7 |
| f_bo_trung | asis | scale_both | 78 | 43.6% | 7 | 7 |
| f_bo_trung | scaled | asis | 78 | 28.2% | 3 | 3 |
| f_bo_trung | scaled | scale_both | 78 | 34.6% | 3 | 3 |
| g_palm_predict | asis | asis | 78 | 35.9% | 12 | 12 |
| g_palm_predict | asis | scale_both | 78 | 35.9% | 12 | 12 |
| g_palm_predict | scaled | asis | 78 | 35.9% | 11 | 11 |
| g_palm_predict | scaled | scale_both | 78 | 35.9% | 11 | 11 |

## 4. Top-1 theo từ (stroke trong bank, chế độ asis)

| Từ | n | a_goc | b_60fps | c_15fps | d_trung | e_45fps | f_bo_trung | g_palm_predict |
|---|---|---|---|---|---|---|---|---|
| BAD | 7 | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| BYE | 8 | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| DRINK | 4 | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| EAT | 8 | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| HELLO | 8 | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| HUNGRY | 8 | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| ME | 8 | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| PLEASE | 8 | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| SORRY | 6 | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| THANK_YOU | 8 | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| WATER | 4 | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| YES | 8 | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| YOU | 8 | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

## Tệp CSV
`match_results.csv`, `segmentation.csv`, `end_to_end.csv` (cùng thư mục).
