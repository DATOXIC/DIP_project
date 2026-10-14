# TRẠNG THÁI KIỂM THỬ

**Dự án**: Nhận diện thủ ngữ ASL thời gian thực và ghép câu (13 từ, chạy CPU)
**Cập nhật**: 2026-10-14 (thay bản 2026-09-12 chỉ có 89 test cho 5 từ)
**Môi trường chạy**: Python 3.12 hệ thống (`py`), `pytest`, `fastapi`, `httpx`. Thư mục `.venv` trong repo không dùng được (thiếu `cv2`, `pytest`).

---

## 1. Kết quả lần chạy gần nhất

Lệnh: `py -m pytest tests -q -p no:cacheprovider`

| Lần chạy | Kết quả |
|---|---|
| Toàn bộ trừ `test_web_app.py` (521 s) | 447 pass, 2 fail, 2 skip, 7 xfail |
| Sau khi sửa lỗi bên dưới: `test_spec12_step0.py` + `test_tts_real_engine.py` | 10 pass |
| `test_web_app.py` (sau khi cài `fastapi`, `httpx`) | 8 pass, 2 skip |

Chưa chạy lại toàn bộ bộ test sau bản sửa; con số tổng hợp ở trên là tổng của các lần chạy riêng, không phải một lần chạy duy nhất.

### Hai test từng lỗi
| Test | Nguyên nhân | Xử lý |
|---|---|---|
| `test_spec12_step0.py::test_t0_d_rebuild_exclude_session_prefix` | `os.path.relpath` ném `ValueError` khi file raw và thư mục chạy khác ổ đĩa Windows | Đã sửa ở `rebuild_templates_from_raw.py` (rơi về đường dẫn tuyệt đối) |
| `test_tts_real_engine.py::test_real_sapi_reads_every_queued_sentence` | Dùng engine SAPI thật của Windows, phụ thuộc thời gian và tải máy; chạy riêng thì pass | Không sửa code; coi là test dễ chập chờn |

### Test bị bỏ qua / kỳ vọng lỗi
- 7 `xfail` và 2+2 `skip` là các trường hợp đã biết (xem lý do bằng `pytest -rxs`).

---

## 2. Phạm vi kiểm thử

| Nhóm | File chính | Nội dung |
|---|---|---|
| Hợp đồng engine cũ (4D) | `test_tier1..4*.py`, `test_verification_steps.py`, `test_harness_utils.py` | Chạy trên `HybridDTWEngine` đã deprecated, giữ lại để không mất độ phủ |
| Engine thật (12D/24D) | `test_v51_golden_triad.py`, `test_fast_dtw_parity.py`, `test_fast_dtw_new_groups.py`, `test_match_many.py`, `test_palm_len_single_source.py` | Khớp số học giữa bản nhanh và bản chuẩn, các nhóm đặc trưng mới |
| Phân đoạn | `test_segmenter_fast_chain.py`, `test_fast_chain_config.py`, `test_lead_trim.py` | FSM `IDLE → RECORDING → PENDING_COMMIT`, chuỗi từ liền nhau |
| Đặc trưng | `test_build_segment_features.py`, `test_hand_pose_features.py`, `test_tier1_features.py` | Vector 24D, góc ngón |
| Dáng tay, chữ cái, spotter | `test_handshape_gate.py`, `test_letter_landmark_pipeline.py`, `test_spotter_gate.py` | Bộ lọc dáng tay, phân loại chữ cái bằng landmark |
| Ghép câu | `test_sentence_composer.py`, `test_sentence_composer_panel.py` | SPEC 14, chốt câu, ngữ pháp |
| Giao diện, TTS, web | `test_ui_side_panel_modes.py`, `test_tts_queue.py`, `test_tts_real_engine.py`, `test_web_app.py` | Panel, hàng đợi đọc, API web |
| Thu dữ liệu, công cụ | `test_collect_*.py`, `test_spec12_step0.py`, `test_dip_launcher.py`, `test_camera_health.py`, `test_weight_learning.py`, `test_threshold_trial_report.py` | Quy trình thu mẫu, rebuild bank, launcher |

---

## 3. Điều bộ test KHÔNG chứng minh

- Chất lượng nhận diện trên **camera thật với người thật**: test dùng dữ liệu tổng hợp hoặc phát lại. Việc này nằm ở `KIEM_TRA_TRUC_TIEP.md`.
- Giao diện có hiển thị đẹp, không tràn chữ.
- FPS thực tế trên máy khác. Các số FPS trong `results/` là đo trên máy phát triển, không do bộ test này đảm bảo.
- Chạy trên nhiều người ký: bank còn ít người (xem `docs/SO_SANH_BASELINE.md`, phần Giới hạn).

## 4. Cách chạy

```bash
py -m pip install pytest fastapi httpx
py -m pytest tests -q                              # toàn bộ (~9 phút)
py -m pytest tests/test_sentence_composer.py -q    # một nhóm
py tools/run_e2e_tests.py                                # bản tổng hợp theo tier (engine cũ)
```
