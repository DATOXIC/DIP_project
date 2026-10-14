# Cải tiến không cần thu thêm dữ liệu — số đo và cách dùng (2026-09-28)

Ba việc: (2) sửa bằng code cho FPS thật/khung trùng/Swipe, (3) tăng cường dữ liệu từ stroke thô, (4) từ chối từ lạ.
Mọi thay đổi vào code chạy thật đều **mặc định TẮT / giữ nguyên hành vi cũ** (đã kiểm: kết quả `scoreboard` của cấu hình hiện tại giống hệt từng truy vấn).
Bank dùng để đo: `webcam_templates.json` (175 mẫu, 13 từ, 7 phiên). Người lạ: `profiles/Mai.json` (51 mẫu, **chỉ để đo**).

## Tóm tắt một dòng cho từng việc

| Việc | Kết luận | Trạng thái mặc định |
|---|---|---|
| 3. Tăng cường dữ liệu | **Không nên đưa vào template runtime.** Tăng nhẹ độ chấp nhận trong cùng người ký, không giúp người lạ, tăng nhẹ nhận nhầm từ lạ | Chỉ là công cụ đo (`augment_templates.py`) |
| 4. Từ chối từ lạ | **Có cải thiện thật**: dùng một thang độ rộng chung (`width_blend=1`) thay vì độ rộng riêng từng từ | TẮT (`width_blend: 0.0`) — chờ bạn quyết |
| 2. Phân đoạn/Swipe/khung trùng | **Có cải thiện đo được** khi fps thật ≠ 30; chưa thử trên webcam thật | TẮT (`time_norm.enabled`, `dedupe_frames`) |

## Việc 3 — Tăng cường dữ liệu (SPEC 06A, chế độ A)

`augment_templates.py` sinh 7 biến thể/stroke ở mức stroke thô (co giãn thời gian, tỉ lệ, xoay ≤10°, lệch điểm đầu, rơi khung, nhiễu, kết hợp), rồi dựng lại toàn bộ đặc trưng 24D bằng đúng `rebuild_one_stroke`. Không lật gương; loại biến thể vi phạm cổng thời lượng/epenthesis (10–13/1050 bị loại).
Chỉ 150/175 mẫu bank tìm được stroke thô khớp (25 mẫu cũ không tái tạo được) → chỉ 150 mẫu sinh biến thể. Stroke thô của mẫu bạn đã hủy (`r`/`x`) **không** được dùng (raw_strokes có 327 stroke nhưng bank chỉ 175).

Phát hiện quan trọng trong lúc làm: `compute_intra_widths` tính độ rộng trên **mọi** mẫu, mà biến thể nằm sát mẫu gốc cùng stroke nên sẽ co độ rộng (và ngưỡng k×độ rộng) lại một cách giả. Đã sửa (cộng thêm): mẫu có `is_original=False` bị bỏ qua khi tính độ rộng (bank thường không có khóa này nên không đổi gì).

| Tham chiếu | LOSO chấp nhận đúng | Margin TB | Người lạ Mai: top-1 / chấp nhận | LOWO nhận nhầm từ lạ | Mẫu âm nhận nhầm | Độ trễ |
|---|---|---|---|---|---|---|
| chỉ mẫu gốc | 155/175 (88.6%) | 0.5383 | 47/51 · 31/51 | 3/175 | 0/41 | 4 ms |
| + biến thể, strength 0.5 | 158/175 (90.3%) | 0.5442 | 47/51 · 31/51 | 5/175 | 0/41 | 27 ms |
| + biến thể, strength 1.0 | 161/175 (92.0%) | 0.5478 | 46/51 · 32/51 | 5/175 | 0/41 | 26 ms |

Ghép cặp so với "chỉ gốc": strength 1.0 được thêm 6 truy vấn, mất 0 (McNemar chính xác p = 0.031); margin +0.0096, KTC95 [+0.0062, +0.0130]. Strength 0.5: +3/−0, p = 0.25.
**Quy tắc quyết định của SPEC 06A** ("nếu tăng nhận nhầm thì bỏ") → nhận nhầm từ lạ tăng 3→5 và người lạ gần như không được lợi (+1/51, top-1 giảm 1) ⇒ **khuyến nghị bỏ chế độ B**. Lý do dễ hiểu: bank đã bão hòa trong cùng người ký (top-1 100%), còn khác biệt giữa người ký là kiểu ký chứ không phải nhiễu nhỏ mà biến đổi mô phỏng được.

Tái lập: `py augment_templates.py --strength 1.0 --out results/aug_eval/bank_aug_s1.0.json` rồi `py scoreboard.py run --config configs/C11_base.json --bank <file> --neg negative_strokes.json` (dùng `--variants 0` để có bản gốc gắn `sample_id` cho so sánh ghép cặp).

## Việc 4 — Từ chối từ lạ

Hiện trạng: mẫu âm (`negative_strokes.json`) chỉ dùng khi đo, **không** dùng lúc chạy. Ngưỡng chấp nhận = k × độ rộng riêng của từ (SPEC 04B).
Tôi đo thêm hai thứ mà scoreboard chưa có (công cụ mới: `eval_open_set.py`):
- **LOWO** (bỏ hẳn một từ khỏi bank, coi các mẫu của nó là "từ lạ"): 175 truy vấn.
- **Người lạ Mai**: chấp nhận đúng trên mẫu của tình nguyện viên.

Chẩn đoán: độ rộng riêng của từ là ước lượng nhiễu (10–17 mẫu của một người) nên từ ký rất đều (BYE, độ rộng 0.156) bị siết ngưỡng quá chặt với người khác, trong khi từ khác vẫn nhận nhầm từ lạ. AUC tách "đúng từ" khỏi "từ lạ": điểm hiện tại `d1/độ rộng từ` = 0.970 (người lạ 0.898); dùng thang chung `d1/trung vị độ rộng` = **0.985** (người lạ **0.963**).

Cài đặt: `width_blend` (0 = như cũ, 1 = thang chung, giữa = trung bình nhân). Cộng thêm ở `GoldenTriadDTWMatcher` (tham số cuối, mặc định 0.0 ⇒ bit-giống hệt).

Toàn bộ bank (LOSO 175 · người lạ 51 · LOWO 175 · mẫu âm 41):

| Cấu hình | LOSO chấp nhận | Sai nặng | Người lạ chấp nhận | LOWO nhận nhầm | Mẫu âm nhận nhầm |
|---|---|---|---|---|---|
| hiện tại `scale_k:1.75` | 155/175 (88.6%) | 0 | 31/51 (61%) | 3/175 | 0/41 |
| `global_k:1.9` | 152/175 (86.9%) | 0 | 37/51 (73%) | 0/175 | 0/41 |
| `global_k:2.0` | 155/175 (88.6%) | 0 | 38/51 (75%) | 0/175 | 0/41 |

Người lạ Mai: +7 mẫu được chấp nhận, mất 0 (McNemar p = 0.016). **Nhưng phải đọc kèm phép thử độ nhạy** — 6 bank con ngẫu nhiên (70% số mẫu mỗi từ; 732 truy vấn/cấu hình):

| Cấu hình | LOSO chấp nhận | LOWO nhận nhầm |
|---|---|---|
| hiện tại | 87.4% | 23/732 (3.1%) |
| `global_k:1.9` | 86.1% | 12/732 (1.6%) |
| `global_k:2.0` | 89.9% | 20/732 (2.7%) |

⇒ Con số "0/175" ở bank đầy đủ là may; thực tế nhận nhầm từ lạ vẫn ~1.6–2.7%, **không** về 0. Cải thiện có thật nhưng vừa phải: `1.9` giảm một nửa nhận nhầm từ lạ với −1.3 điểm chấp nhận; `2.0` tăng chấp nhận +2.5 điểm và giảm nhẹ nhận nhầm. Ở bank đầy đủ có "vách": k=2.05 nhảy lên 4/175 nhận nhầm. **Khuyên chọn 1.9** nếu ưu tiên an toàn.
Hạn chế: chỉ một tình nguyện viên; BYE của Mai vẫn 0/8 được chấp nhận (khác biệt kiểu ký thật, thang chung không cứu được); ngưỡng chọn trên cùng bank dùng để đo (đã có phép thử bank con để giảm thiên lệch nhưng không thay được dữ liệu độc lập).

Bật thử: `py predict_sentence_webcam.py --width-blend 1.0 --threshold-k 1.9 --uncertainty-k 2.5` (**phải** đặt `--uncertainty-k` ≥ k+0.6, mặc định 2.35 sẽ làm vùng "không chắc" quá hẹp). Muốn đặt vĩnh viễn: sửa `width_blend`, `threshold_k`, `uncertainty_k` trong `CONFIG`. Scoreboard: `configs/G1p90.json`, `configs/G2p00.json`; đo lại: `py eval_open_set.py --config configs/C11_base.json --config configs/G1p90.json`.

## Việc 2 — Phân đoạn theo thời gian, khung trùng, Swipe-Left

Phần đã có sẵn (SPEC 08): `pause_sec`, `min_stroke_sec`, `ready_stable_sec` theo giây. Còn lại theo **khung**: ngưỡng vận tốc `v_start/v_stop`, `max_stroke_frames`, và bộ dò Swipe (6 mẫu, vận tốc mỗi khung). Đã thêm (tham số cuối, mặc định TẮT):
- `time_norm` (`KineticPauseSegmenter(time_normalized=True, …)`): quy vận tốc về "mỗi khung ở 30 fps" bằng dt ước lượng từ dấu thời gian; Swipe dò trong cửa sổ theo giây với ngưỡng theo giây (cửa sổ tự nới khi camera chậm để luôn có ≥3 mẫu). `v_scale_min=1.0` = chỉ bù khi fps **cao hơn** 30 (chuẩn hóa hai chiều làm xấu đi ở fps rất thấp — đã đo).
- `dedupe_frames`: bỏ khung trùng y hệt (dùng `CameraHealthMonitor.is_new`) trước khi dò tay, để dt/vận tốc phản ánh khung mới thật.

`py tools/eval_segmenter_fps.py` (150 stroke × 7 biến thể × 4 chế độ; số liệu = % stroke ra đúng 1 lần chốt / số Swipe nhầm trong 150 stroke từ vựng):

| Biến thể | production hiện tại | `time_norm` (v_scale_min=1) |
|---|---|---|
| gốc (~14 fps) | 90.0% · 0 | 90.0% · 0 |
| 60 fps (nội suy) | 61.3% · 0 (nhỡ 34.7%) | 78.7% · 0 (nhỡ 7.3%, nhưng cắt đôi 4%→14%) |
| 45 fps | 76.7% · 0 | 86.0% · 0 |
| ~7 fps (bỏ xen kẽ) | 91.3% · **9** | 97.3% · **0** |
| lặp khung (giữ khung trùng) | 87.3% · 0 | 88.0% · 0 |
| lặp khung **+ dedupe** | 91.3% · 9 | **97.3% · 0** |
| bỏ khung trùng sẵn có | 94.0% · 3 | 96.0% · 0 |

Swipe-Left giả lập (9 palm/s, 0.3 s): bản hiện tại bỏ sót ở 60 fps; `time_norm` bắt đúng 1 lần ở mọi fps 8–60 (kể cả khung lặp), và trôi chậm 2 palm/s không gây Swipe nhầm ở 8 fps (bản cũ nhầm 2 lần).
**Chưa kiểm chứng:** (a) trên webcam thật — mô phỏng dùng stroke đã thu; (b) giả định "driver lặp đúng bộ đệm" của `dedupe_frames` (xem `camera_health.py`): nếu webcam trả khung đã làm mịn thì `is_new` luôn True và cờ này vô tác dụng (không gây hại). Khuyên thử trực tiếp: `--time-norm --dedupe-frames`.
Ghi chú ngoài phạm vi: trên chính stroke gốc, production (ký nhanh mặc định) ra đúng 1 lần chốt 90.0% so với 95.3% của cấu hình cũ không ký nhanh — 5% bị cắt đôi (từ nhiều pha). Tôi không đụng tới vì bạn đã chốt sau khi thử trực tiếp.

## Chưa làm

- Cặp HELLO–BYE (bộ phân xử thứ hai): chưa làm. Số đo hiện có cho thấy vấn đề của BYE là khác biệt giữa người ký (Mai BYE 0/8), không phải nhầm với HELLO; cần thêm bằng chứng trước khi thêm cơ chế.
- Chưa bật mặc định bất kỳ cờ nào ở trên.

## File mới / đã sửa

Mới: `augment_templates.py`, `eval_open_set.py`, `eval_segmenter_fps.py`, `backup_utils.py`, `configs/G1p90.json`, `configs/G2p00.json`, `tests/test_robustness_step.py` (23 test).
Sửa (cộng thêm, mặc định giữ nguyên): `dynamic_dtw_engine.py` (`is_original` trong độ rộng nội lớp; `width_blend`/`effective_width`; tham số `time_normalized…` của `KineticPauseSegmenter`), `pipeline_common.py` (`make_segmenter` đọc `time_norm`), `scoreboard.py` (`global_k:<k>`, `blend_k:<b>:<k>`), `predict_sentence_webcam.py` (CONFIG `width_blend`/`time_norm`/`dedupe_frames`; cờ `--width-blend`, `--time-norm`, `--dedupe-frames`).
