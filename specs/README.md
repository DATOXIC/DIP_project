# Thư mục đặc tả — lộ trình cải tiến pipeline nhận diện thủ ngữ động

Mỗi file là một gói việc độc lập, viết để bàn giao cho một agent thực thi.

## Cách bàn giao

Với mỗi spec, gửi cho agent **2 file**: `_PREAMBLE.md` **và** file spec đó.
`_PREAMBLE.md` chứa môi trường, ràng buộc, số liệu mốc và cách báo cáo — agent nào
không đọc nó sẽ làm sai.

Câu lệnh bàn giao gợi ý:

> Đọc `specs/_PREAMBLE.md` rồi thực hiện `specs/SPEC_0X_....md`. Làm đúng mọi tiêu chí
> hoàn thành và báo cáo đầy đủ số liệu thật. Nếu có chỗ nào trong spec sai so với code
> thực tế, dừng lại và báo cáo thay vì đoán.

## Quyết định đã chốt (2026-09-19)

**Bank production = chỉ template webcam của người dùng.** ASL Citizen giữ lại làm tài liệu
tham khảo, không dùng làm template, không dùng để dò trọng số hay ngưỡng. Quy tắc thu theo
phương án A (dừng rồi ký).

## Thứ tự và phụ thuộc

```
[ĐÃ XONG]  Bước 0, SPEC 00, SPEC 01, SPEC 02, SPEC 03

SPEC 04A  Bảng điểm scoreboard.py (so sánh ghép cặp, sealed test)  [ĐÃ XONG, nhưng CÓ LỖI -> 04A.1]
SPEC 04A.1 Sửa lỗi lật gương hai lần trong scoreboard (bản vá nhỏ)   [AGENT, 1-2h, LÀM NGAY]
          KHÔNG cần dữ liệu mới, làm được NGAY, song song 05B  [AGENT, 4-6h]

SPEC 05A  Chuẩn bị collector: session_id, phím n, bảng phím    [AGENT, 1-2h]
        |
        v
SPEC 05B  Collector 30 FPS, đồng bộ collector<->predict,
          bộ lọc cổ tay dùng chung, YOU mới, lưu bank cũ       [AGENT, 4-6h]
        |
        v
SPEC 05C  Song song hóa MediaPipe (sửa FPS tụt khi có tay:
          mediapipe ~49 ms/frame, 3 bộ dò song song ~56 FPS)   [AGENT, 3-5h]
        |
        v
SPEC 05   Thu mẫu webcam ở 30 FPS: 2 phiên + mẫu âm            [NGƯỜI, ~55 phút]
        |
        v
SPEC 06A  (nên có) Augmentation, chế độ A: chỉ dùng để đánh giá [AGENT, 3-4h]
        |
        v
SPEC 04   Chọn cấu hình bằng bảng điểm: 2 mốc + QUY TẮC DỪNG,
          ablation nhóm đặc trưng, thử ASL-LEX có kiểm soát,
          ngưỡng + nhận nhầm, sealed test một lần              [AGENT, 3-4h]
        |
        v
SPEC 04B  Hiệu chuẩn ngưỡng theo thang từng từ + "độ khớp" thay
          "xác suất" + sửa cờ --global-weights (sửa: sai vẫn ra
          từ với xác suất cao). Cần 04A; nên có mẫu âm       [AGENT, 4-6h]
        |
        v
SPEC 06B  (TÙY CHỌN) DBA + bank mở rộng                        [AGENT, 3-4h]
```

**Quy tắc xuyên suốt:** không nhận một thay đổi nào chỉ vì "nhìn có vẻ tốt hơn". Mọi cấu hình đi qua
`scoreboard.py compare`; kết luận INCONCLUSIVE nghĩa là giữ cấu hình cũ.

## Cảnh báo xung đột file — ĐỪNG chạy song song sai

| Spec | File chính bị sửa |
|---|---|
| SPEC 00 | `dynamic_dtw_engine.py` |
| SPEC 01 | `dynamic_dtw_engine.py`, `predict_sentence_webcam.py`, `collect_dynamic_templates.py` |
| SPEC 02 | `predict_sentence_webcam.py`, `collect_dynamic_templates.py` |
| SPEC 03 | `dynamic_features.py` (mới), `dynamic_dtw_engine.py`, cả 4 đường trích xuất |
| SPEC 04A | `scoreboard.py` (mới), `configs/`, `results/` (mới); chỉ CỘNG THÊM nhỏ vào engine |
| SPEC 04 | `dynamic_dtw_engine.py` (thêm chế độ `modulate`, `flip_body_x`), `configs/`, `analysis/` |
| SPEC 05A | `collect_dynamic_templates.py`, `rebuild_templates_from_raw.py` |
| SPEC 05B | `pipeline_common.py` (mới), `collect_dynamic_templates.py`, `predict_sentence_webcam.py` |
| SPEC 05C | `pipeline_common.py`, `collect_dynamic_templates.py`, `predict_sentence_webcam.py` (chạy SAU 05B) |

> SPEC 05A và 05B đều sửa `collect_dynamic_templates.py` → **chạy tuần tự (05A xong rồi mới 05B)**.
| SPEC 06A, 06B | file mới, không sửa file cũ |

- **SPEC 00 và SPEC 01 KHÔNG được chạy song song** (cùng `dynamic_dtw_engine.py`).
- **SPEC 01 và SPEC 02 KHÔNG được chạy song song** (cùng `predict_sentence_webcam.py`).
- SPEC 02 có thể chạy **sau** SPEC 01 mà không cần đợi SPEC 03.
- Chỉ chạy một spec một lúc là an toàn nhất.

## Vì sao thứ tự này, không phải thứ tự khác

1. **SPEC 00 trước** vì `feature_dim=4` bị hardcode đang âm thầm hạ margin guard từ
   0.12 xuống 0.08 — mọi số liệu đo sau đó sẽ sai nếu không sửa trước.
2. **SPEC 01 trước khi thu mẫu** vì lỗi grace-window đang dán pha nhấc tay vào template.
   Bằng chứng: `duration` webcam 1.62s vs Citizen 0.43s, `pos_span` 1.28 vs 0.82.
3. **SPEC 03 trước SPEC 04** vì thêm 4 nhóm đặc trưng mà không dò lại trọng số thì các
   chiều mới gần như chắc chắn làm giảm độ chính xác.
4. **SPEC 05 sau SPEC 03/04** vì đặc trưng phải chốt xong mới thu — dù nhờ `raw_strokes/`
   thì đổi đặc trưng sau này cũng chỉ cần chạy lại `rebuild_templates_from_raw.py`.
5. **SPEC 06 cuối** vì augmentation cần `raw_strokes/` của bộ mẫu mới.

## Nguyên tắc chung xuyên suốt

- Mọi thay đổi phải **cộng thêm** và **tắt được**, mặc định giữ hành vi cũ.
- Mọi cải tiến phải có **số đo trước/sau**, không chấp nhận "cảm giác tốt hơn".
- Test suite đã hỏng sẵn (49 fail / 58 pass) — tiêu chí là **diff danh sách FAILED phải
  rỗng**, không phải "làm cho test xanh".
- 139 mẫu ASL Citizen là **hạ tầng đo lường**, không phải template production. Đừng
  trộn chúng vào bank khớp.

## Giai đoạn trải nghiệm (2026-09-20)

- **[ĐÃ XONG, không có file spec]** Giao diện demo + song ngữ (`ui_render.py`, `speech_service.py`) và
  DTW nhanh (`_FastDTWBank`, hết cú khựng ~0,7 s mỗi từ).
- **SPEC 09** `SPEC_09_noi_tu_lien_tuc.md`: nhận dạng nối từ (Boundary Detector + DTW quy hoạch động).
  Chế độ THỬ NGHIỆM có cổng kiểm tra, mặc định tắt. Cần người thu 2 phiên câu nối (S1 chỉnh, S2 niêm phong).
  Nhiệm vụ A, B, C song song được; D–E chạy được trước khi có dữ liệu thật nhờ dữ liệu tổng hợp.
- **SPEC 08** `SPEC_08_ky_nhanh.md`: ký nhanh (Cấp 1) — cửa sổ nối sau khi chốt, chốt sớm có kiểm soát, tham số phân đoạn tính bằng giây.
  Có cờ `--fast-chain`, mặc định TẮT; cổng bằng luồng mô phỏng + thử trực tiếp ~20 phút. Không cần dữ liệu mới.
- **[ĐÃ XONG]** Sửa `palm_len` (collector nhân nhầm bề rộng cho trục y): `compute_palm_len` dùng chung, bank `webcam_templates.json` dựng lại
  (97/107 mẫu; bản cũ `webcam_templates.backup_before_palmfix_20260920_091000.json`).
- **SPEC 08B** `SPEC_08B_cat_chuyen_tiep.md`: loại phần chuyển tiếp giữa hai từ trong `--fast-chain` (khoảng cấm cố định `--transition-lockout` và cắt dẫn đầu thích nghi `--lead-trim`, phím `k`). Không cần dữ liệu mới; đo bằng mô phỏng chuyển tiếp thực tế + thử trực tiếp. Là bản nhỏ của SPEC 09.
- **SPEC 10** `SPEC_10_sua_am_thanh.md`: từ thứ hai mất tiếng khi ký nhanh do `NonBlockingTTS` bỏ yêu cầu khi đang bận; sửa bằng `QueuedSpeaker` (hàng đợi, một luồng nền) trong `tts_queue.py`, kèm hai lỗi cùng kiểu ở `speech_service.py` (giọng Việt, WAV `SND_ASYNC`).
- **[BẢN CHUẨN từ 2026-09-21]** `--fast-chain` + `--lead-trim` là MẶC ĐỊNH (`CONFIG["fast_chain"]["enabled"]=True`, `lead_trim.enabled=True`). Quay về chế độ cũ: `--no-fast-chain` (hoặc phím `j`); tắt riêng cắt dẫn đầu: `--no-lead-trim` (phím `k`). `transition_lockout_sec` vẫn 0.0 (chưa thử trực tiếp). Cấu hình đã thử trực tiếp: `chain_window_sec=0.0`, `ready_stable_sec=0.1`, `lead_trim.max_variants=6`.
- **[ĐÃ XONG] SPEC 11** `SPEC_11_hoc_trong_so.md` (2026-09-21): đường nhanh đã hỗ trợ `w_z/w_s/w_r` (bit-exact, độ trễ ~5 ms ≤ 10 ms); đo lường từng nhóm và học trọng số lồng nhau Nested LOSO (L5/L8) không qua cổng đặt trước (CV > 0.5, delta margin <= 0); quyết định giữ trọng số tay KEEP_HAND_WEIGHTS; diff test suite rỗng so với mốc 49.
- **SPEC 10B** `SPEC_10B_sua_am_thanh_that.md` (2026-09-21): SPEC 10 chưa sửa hết. Đo bằng SAPI/pyttsx3 thật: `QueuedSpeaker` dùng lại một engine pyttsx3 nên các lời đọc từ thứ hai chỉ chạy ~0.1 s (không phát); thêm mỗi từ SAPI kèm ~0.7 s im lặng cuối
  (1.2–1.5 s/từ) nên ký nhanh bị tồn đọng và bỏ từ. Sửa: WAV dựng sẵn đã cắt im lặng (`audio_cache/en/`), dự phòng SAPI gọi trực tiếp đồng bộ, MỘT hàng đợi chung, nhật ký `DIP_TTS_DEBUG`, tự kiểm tra `tts_selftest.py`.
- **KẾ HOẠCH CHI TIẾT nhiều người ký** `KE_HOACH_CHI_TIET_NHIEU_NGUOI_KY.md` (2026-09-21): 19 mục từ dễ đến khó (E1–E5, M1–M7, H1a–H4, X1–X2) với cổng số đo, hộp thời gian, điểm quyết định DP1–DP4 và lịch 21/9–4/10. Đóng băng mặc định 28/9. Bổ sung cho `KE_HOACH_DEN_04_10.md`.
- **SPEC 12** `SPEC_12_bo_do_nguoi_la_va_do_ben_vung.md` (2026-09-21): bộ đo người lạ và độ bền vững, **không sửa production**. Bước 0 (làm trước khi thu tình nguyện viên): `--raw-dir` cho collector,
  `separate_volunteer_raw.py`, `check_volunteer_data.py`, `--exclude-session-prefix` cho rebuild (vì collector ghi chung `raw_strokes/` và rebuild không lọc phiên). Rồi 12C leave-signer-out (luật DP1 đặt trước),
  12A (ablation, đường cong theo số mẫu DTW/kNN/mạng nhỏ, mô phỏng tự thêm mẫu có 10% nhãn sai, phụ thuộc người theo nhóm đặc trưng), 12E (E3 thang chuẩn hóa), 12B (mô phỏng khung/s và điều kiện tối/chói/xa).
  **Quyết định 2026-09-21: bỏ hẳn ASL Citizen** (kể cả làm tập kiểm tra); nguồn đa dạng = tình nguyện viên + điều kiện của Đạt. Phiếu thu: `PHIEU_THU_TINH_NGUYEN_VIEN.md` (5 lần/từ).

## Giai đoạn mở rộng chức năng (2026-10-02)

Hai chức năng mới cho sentence builder (đánh số 14, 15), **chạy tuần tự 14 -> 15** (cùng sửa `predict_sentence_webcam.py` và `ui_side_panel.py`).
Nguyên tắc: không sửa `train.py`, `improved_dip_pipeline.py`, `best_model*.pth`; chỉ cộng thêm, tắt được, mặc định giữ hành vi cũ.
**Trước khi code mỗi spec, phải đối chiếu lại với code hiện tại** (đồng đội thường xuyên đẩy thay đổi mới, xem `git log`).

- **SPEC 14** `SPEC_14_gloss_thanh_cau.md`: chuỗi gloss -> câu hoàn chỉnh EN/VI bằng luật offline (không LLM),
  module thuần `sentence_composer.py`; gloss gốc luôn giữ, câu tự nhiên là dòng thêm, đọc câu khi chốt. Không cần dữ liệu mới.
- **SPEC 15** `SPEC_15_danh_van_chu_cai.md`: chế độ đánh vần (phím `s`) bằng CNN DIP mở rộng cho 12 chữ tĩnh + lớp NONE,
  model mới `fingerspell_v1.pth` (không đụng model 5 lớp), chia dữ liệu **theo phiên**. 3 giai đoạn: 15A (agent) ->
  Bước 15 (NGƯỜI thu ~1 giờ) -> 15B (agent: huấn luyện, cổng GO/NO-GO, đo thật).

| Spec | File chính bị sửa/tạo |
|---|---|
| SPEC 14 | `sentence_composer.py` (mới), `predict_sentence_webcam.py`, `ui_side_panel.py`, `tests/test_sentence_composer.py` |
| SPEC 15 | `fingerspell.py`, `collect_fingerspell.py`, `train_fingerspell.py`, `fingerspell_letters.json` (mới), `predict_sentence_webcam.py`, `ui_side_panel.py` |
