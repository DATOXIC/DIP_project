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
        |
        v
SPEC 05A  Chuẩn bị collector: bank mới, session_id, phím n (mẫu âm),
          bảng phím 10 từ                                     [AGENT, 1-2h]
        |
        v
SPEC 05   Thu mẫu webcam, 2 phiên + mẫu âm                    [NGƯỜI, ~55 phút]
        |
        v
SPEC 06A  Tăng cường dữ liệu từ stroke thô (chế độ A: chỉ để dò,
          KHÔNG làm template runtime)                         [AGENT, 3-4h]
        |
        v
SPEC 04   Dò trọng số + ngưỡng TRỰC TIẾP trên bank webcam
          (phía tham chiếu có thể gồm biến thể 06A),
          leave-one-session-out, đo nhận nhầm                 [AGENT, 4-6h]
        |
        v
SPEC 06B  (TÙY CHỌN) DBA + thử bank mở rộng                   [AGENT, 3-4h]
```

**Vì sao 06A đứng trước 04:** dò ~8 trọng số trên ~80 mẫu dễ overfit; augmentation làm phía
tham chiếu ổn định hơn. Augmentation chỉ cần stroke thô, không cần trọng số đã dò, nên
không phải dò hai lần. Nhưng biến thể của một stroke **không độc lập**, nên cross-validation
luôn phải tách theo `session_id` **và** `augment_of`.

## Cảnh báo xung đột file — ĐỪNG chạy song song sai

| Spec | File chính bị sửa |
|---|---|
| SPEC 00 | `dynamic_dtw_engine.py` |
| SPEC 01 | `dynamic_dtw_engine.py`, `predict_sentence_webcam.py`, `collect_dynamic_templates.py` |
| SPEC 02 | `predict_sentence_webcam.py`, `collect_dynamic_templates.py` |
| SPEC 03 | `dynamic_features.py` (mới), `dynamic_dtw_engine.py`, cả 4 đường trích xuất |
| SPEC 04 | `tune_weights.py` (mới), `dynamic_dtw_engine.py` |
| SPEC 05A | `collect_dynamic_templates.py`, `rebuild_templates_from_raw.py` |
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
