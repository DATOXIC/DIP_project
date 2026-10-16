# Kế hoạch tinh giản mã nguồn (đề xuất — CHƯA thực hiện)

Căn cứ: đồ thị `import` tính bằng AST trên 60 file `.py` ở thư mục gốc (+ 21 file trong `tools/`), đối chiếu với `dip.py`,
`tests/` và tài liệu. Không có file nào bị xóa hay di chuyển cho tới khi bạn đồng ý.

## 1. Lõi chạy thật (GIỮ ở thư mục gốc, 33 file)
Đây là tập file được `dip.py`, `predict_sentence_webcam.py`, `web_app.py`, `dip_app.py` import trực tiếp hoặc gián tiếp.
Chuyển chúng sẽ phải sửa import ở nhiều nơi, rủi ro cao mà không giảm được độ phức tạp thực.

`dip.py` `dip_app.py` `predict_sentence_webcam.py` `web_app.py` `dynamic_dtw_engine.py` `dynamic_features.py`
`pipeline_common.py` `improved_dip_pipeline.py` `train.py` `train_dynamic_dl.py` `dl_fusion.py` `handshape_fusion.py`
`handshape_gate.py` `hand_pose_features.py` `letter_landmark_classifier.py` `fingerspell.py` `sentence_composer.py`
`session_enroll.py` `spotter_dl.py` `spotter_features.py` `spotter_gate.py` `online_negatives.py` `people.py`
`speech_service.py` `tts_queue.py` `ui_render.py` `ui_side_panel.py` `camera_health.py` `lead_trim.py` `backup_utils.py`
`collect_continuous.py` `collect_phrases.py` `check_volunteer_data.py`

## 2. Không thuộc lõi chạy (27 file) — đề xuất xử lý

### 2a. Chuyển vào `tools/eval/` (đánh giá, chạy qua `py dip.py danh-gia ...`) — 13 file
`scoreboard.py` (được 10 file eval import — chuyển CÙNG nhóm) `eval_baselines.py` `eval_citizen.py` `eval_hybrid.py`
`eval_learning_curve.py` `eval_open_set.py` `eval_enroll.py` `eval_negative_session.py` `eval_replay_log.py`
`eval_rule_spotter.py` (do `train_spotter.py` import) `threshold_trial_report.py`
`eval_confirm.py` `eval_adapt.py` (dip.py gọi bằng tên ghép `f"eval_{kind}.py"` nên phân tích import không thấy; test
`test_moi_loai_danh_gia_tro_toi_script_ton_tai` sẽ bắt nếu thiếu)
→ cần: sửa đường dẫn trong `dip.py`, thêm `tools/eval` vào `sys.path` trong `tests/conftest.py`, sửa 3 test.

### 2b. Chuyển vào `tools/data/` (thu và xử lý dữ liệu) — 8 file
`collect_dynamic_templates.py` `camera_diag.py` `rebuild_templates_from_raw.py` `separate_volunteer_raw.py`
`freeze_volunteer_data.py` `augment_templates.py` `make_audio_cache.py` `train_spotter.py`
→ cần: sửa `dip.py` và 5 test.

### 2c. Chuyển vào `tools/train/` (huấn luyện) — 2 file
`weight_learning.py` `train_letter_landmarks.py` → cần sửa 2 test.

### 2d. Lưu trữ vào `legacy/` (thử nghiệm cũ, không ai gọi, không có test) — 4 file
`chain_replay.py` (1098 dòng) `chain_trial_report.py` `eval_dynamic_hybrid.py` `package_dynamic_colab.py`
→ chỉ nhắc tới trong tài liệu (`chain_replay` có 7 tài liệu nhắc); không xóa, chỉ chuyển.

## 3. Kết quả dự kiến
- Thư mục gốc: **60 → 33 file `.py`** (chỉ còn lõi chạy).
- Lệnh `py dip.py ...` giữ nguyên, người dùng không thấy khác biệt.
- `legacy/` đang có 17 file cũ từ trước: có thể gom thành một file nén sau khi bạn xác nhận không cần.

## 4. Việc KHÔNG nên làm lúc này
- Tách hoặc gộp các module lõi (ví dụ chia `predict_sentence_webcam.py` 3019 dòng, gộp 3 file `spotter_*`): gây
  rủi ro hỏng demo, không giúp báo cáo.
- Xóa hẳn file: các file ở 2d vẫn là bằng chứng quá trình thực nghiệm cho báo cáo.

## 5. Cách thực hiện an toàn
1. Làm trên một nhánh/thư mục tạm, KHÔNG commit cho tới khi bạn duyệt.
2. Di chuyển theo từng nhóm 2a → 2b → 2c → 2d; sau mỗi nhóm chạy toàn bộ test (`py -m pytest -q`, ~8 phút) và
   `py dip.py <lệnh> --help` cho mọi lệnh con.
3. Cuối cùng chạy thử `py dip.py nhan-dien` một lần trên camera.
