# PREAMBLE — ĐỌC TRƯỚC MỌI SPEC (bắt buộc)

Mọi spec trong thư mục này đều giả định bạn đã đọc file này. Nếu spec bạn nhận
không kèm file này, HÃY ĐỌC NÓ TRƯỚC KHI GÕ BẤT KỲ DÒNG CODE NÀO.

## Môi trường

- Thư mục dự án: `C:\DIP\DIP_project` (Windows, có cả PowerShell và Git Bash)
- Python: `C:\Users\To Phuong Dat\AppData\Local\Programs\Python\Python313\python.exe`
- **KHÔNG dùng lệnh `python` trần** — nó là alias Microsoft Store bị hỏng và sẽ báo
  "Python was not found". Dùng đường dẫn đầy đủ ở trên, hoặc `py`.
- File tạm/script nháp: ghi vào thư mục scratchpad của bạn, **KHÔNG ghi vào dự án**.
- **KHÔNG BAO GIỜ đặt tên file nháp là `inspect.py`** — nó che module chuẩn `inspect`
  và làm `import numpy` crash.

## Quy ước dự án

- Toàn bộ comment và chuỗi in ra cho người dùng viết bằng **tiếng Việt**.
  Chủ dự án: Tô Phương Đạt, MSSV 24110086, môn Xử Lý Ảnh Số (DIP).
- **Định vị code bằng CHUỖI TÌM KIẾM, không bằng số dòng.** Các spec này được viết
  sau nhiều lần sửa file; số dòng có thể đã lệch. Mỗi spec cho bạn một chuỗi duy nhất
  để `grep`. Nếu grep không ra, hoặc ra nhiều hơn 1 chỗ, **DỪNG LẠI VÀ BÁO CÁO**,
  đừng đoán.

## File TUYỆT ĐỐI KHÔNG ĐƯỢC SỬA

- `improved_dip_pipeline.py`
- `train.py`
- `vsl_dataset/models/best_model_v2.pth`

## Ràng buộc với `dynamic_dtw_engine.py`

- **Chỉ được sửa theo hướng CỘNG THÊM (additive).**
- Class `HybridDTWEngine` (engine 4D cũ) phải **giữ nguyên hành vi 100%**. Nó đang
  được `tests/conftest.py`, `tests/test_tier1_features.py`, `tests/test_tier2_boundaries.py`,
  `tests/test_tier3_interactions.py`, `tests/test_tier4_scenarios.py`,
  `tests/test_verification_steps.py`, `tests/test_harness_utils.py` gọi trực tiếp.
- Khi thêm tham số vào hàm/hàm khởi tạo đã tồn tại: **luôn thêm ở cuối và luôn có
  giá trị mặc định giữ nguyên hành vi cũ**. Không được đổi thứ tự tham số.

## Dữ liệu

- **Backup mọi file JSON trước khi ghi đè**, theo đúng kiểu đặt tên sẵn có trong repo:
  `<tên>.backup_<lý_do>_<YYYYMMDD_HHMMSS>.json`
- **KHÔNG xóa, KHÔNG ghi đè bất kỳ file `*.backup*.json` nào.** Đó là bản sao duy nhất
  của dữ liệu đã mất rất nhiều công thu.
- Không `git commit`, không `git push`, không tạo branch. Để nguyên thay đổi trong
  working tree cho chủ dự án xem diff.

## Tình trạng test suite (QUAN TRỌNG)

Test suite **đã hỏng sẵn từ trước**: `pytest tests/` cho **49 failed / 58 passed**,
`tools/run_e2e_tests.py` cho **42 passed / 47 failed**. Đây KHÔNG phải lỗi của bạn.

Cách kiểm tra hồi quy đúng: chạy test **trước** khi sửa, lưu danh sách `FAILED` đã
sort ra file; chạy lại **sau** khi sửa, sort, rồi `diff` hai danh sách. Yêu cầu:
**diff phải RỖNG**. Đừng cố sửa 49 test đang fail — ngoài phạm vi.

## Số liệu mốc (dùng để kiểm tra hồi quy)

Bộ dữ liệu, cắt về 12 chiều đầu, trọng số `w_p=0.6, w_a=1.0, w_m=0.4, w_n=0.8`,
`lambda_shape=0.0`, `lambda_shape_global=0.0`:

| Phép đo | Dữ liệu | Kết quả mốc |
|---|---|---|
| LOO nội miền | 34 mẫu webcam (các mẫu KHÔNG có khóa `source_video` trong `dynamic_templates.backup_before_12d_delete_20260918_212502.json`) | **32/34 = 94.1%**, d_correct 0.232, margin +0.395 |
| Chuyển miền, chưa canonical | 34 mẫu trên làm query → 5 từ tương ứng trong `asl_citizen_templates.backup_before_bye_delete_20260918_183236.json` | **12/34 = 35.3%**, margin **−0.157** |
| Chuyển miền, đã canonical (eps=0.05) | như trên | **21/34 = 61.8%**, margin +0.075 |
| Chuyển miền, đã canonical (eps=0.0) | như trên | **22/34 = 64.7%**, margin +0.087 |
| `tools/audit_sign_quality.py --all` | bank Citizen hiện tại | **129/129 = 100%** |

5 từ trùng nhau giữa hai bank: `HELLO, BYE, ME, YOU, THANK_YOU`.

## Cách báo cáo lại (bắt buộc, mọi spec)

1. Danh sách file đã sửa, kèm tên hàm/class đã chạm (không cần số dòng).
2. **Số liệu thật** của mọi tiêu chí hoàn thành trong spec — trước và sau.
3. Diff danh sách test FAILED (phải rỗng).
4. Bất cứ điều gì trong spec mà bạn thấy **SAI hoặc không khớp thực tế code** — nói
   thẳng ra. Spec được viết bởi người đọc code chứ không chạy được webcam, có thể sai.
5. Bất cứ chỗ nào bạn phải **tự quyết định** — nói rõ bạn chọn gì và vì sao.
6. Thứ gì bạn **KHÔNG kiểm chứng được** — nói rõ là không kiểm chứng được, đừng
   phát biểu như đã kiểm chứng.

## Nguyên tắc cuối

Nếu một yêu cầu trong spec mâu thuẫn với PREAMBLE này, PREAMBLE thắng.
Nếu bạn không chắc, **DỪNG VÀ BÁO CÁO** thay vì đoán rồi làm bừa.
