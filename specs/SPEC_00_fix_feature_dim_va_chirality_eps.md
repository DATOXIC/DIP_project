# SPEC 00 — Sửa 2 lỗi nhỏ đang bóp méo quyết định live

> Đọc `specs/_PREAMBLE.md` trước. Ước lượng: 1–2 giờ. Làm spec này TRƯỚC SPEC 01.

## Mục tiêu

Hai lỗi độc lập, nhỏ, nhưng cả hai đang làm sai lệch quyết định accept/reject lúc chạy thật.

---

## NHIỆM VỤ 1 — `feature_dim` bị hardcode = 4

### Vấn đề

Trong `dynamic_dtw_engine.py`, tìm chuỗi:

```
"feature_dim": 4,
```

Nó nằm trong `HybridDTWEngine.save_templates()`. Mọi phiên thu mẫu của
`collect_dynamic_templates.py` đều lưu qua hàm này, nên `dynamic_templates.json` bị
đóng dấu `feature_dim: 4` **dù vector thực tế là 17 chiều**.

Hệ quả: trong `GoldenTriadDTWMatcher.load_templates()`, tìm chuỗi:

```
self.is_legacy_4d = (self.feature_dim < 9)
```

→ `is_legacy_4d` thành `True` một cách sai lầm → trong `match()`, margin guard bị
siết/nới theo nhánh legacy: **0.12 bị hạ xuống 0.08**. Tìm chuỗi `margin_threshold`
trong `match()` để thấy nhánh đó.

Kiểm chứng nhanh trước khi sửa (để tự xác nhận vấn đề là thật):

```
py -c "import json;d=json.load(open(r'C:\DIP\DIP_project\dynamic_templates.json',encoding='utf-8'));print(d['metadata']['feature_dim'], len(d['vocabulary']['EAT']['samples'][0]['traj_features'][0]))"
```

Kỳ vọng in ra `4 17` — tức metadata nói 4 nhưng dữ liệu là 17.

### Việc cần làm

1. **Trước tiên kiểm tra xem có test nào assert `feature_dim == 4` không:**
   ```
   grep -rn "feature_dim" tests/ tools/run_e2e_tests.py
   ```
   - Nếu **CÓ** test assert giá trị 4 → **KHÔNG sửa `HybridDTWEngine.save_templates()`**.
     Thay vào đó, sau khi `save_templates()` ghi file, ghi đè lại đúng trường metadata
     ở phía gọi (trong `collect_dynamic_templates.py`). Báo cáo rõ bạn đã chọn đường này.
   - Nếu **KHÔNG** có → sửa trực tiếp trong `save_templates()`.

2. Cách sửa: **suy ra `feature_dim` từ dữ liệu thật** thay vì hardcode. Lấy độ dài
   của `traj_features[0]` của mẫu đầu tiên tìm được trong toàn bộ `vocabulary`.
   Nếu không có mẫu nào (từ vựng rỗng), giữ nguyên `4` như cũ để không đổi hành vi
   với file rỗng.

3. Sửa `dynamic_templates.json` hiện tại: viết một script **một lần** (đặt trong
   scratchpad, không để lại trong dự án) backup file rồi sửa trường
   `metadata.feature_dim` từ `4` thành `17`. Backup theo đúng quy ước PREAMBLE.

### Tiêu chí hoàn thành

- `tools/diagnose_chirality.py` hoặc lệnh kiểm chứng ở trên in ra `17 17`.
- Chạy `predict_sentence_webcam.py --help` (hoặc import module) không lỗi.
- **Báo cáo bắt buộc**: dòng cảnh báo `"mẫu 4D cũ"` (grep chuỗi `4D` trong
  `load_templates`) phải **biến mất** khi nạp `dynamic_templates.json`.
- **Báo cáo bắt buộc — phần quan trọng nhất**: margin guard giờ là 0.12 thay vì 0.08,
  tức **siết chặt hơn**. Hãy đo xem điều đó làm đổi bao nhiêu quyết định: chạy
  `tools/audit_sign_quality.py --all` trước và sau, đếm số mẫu chuyển từ `ACCEPTED` sang
  `UNCERTAIN`. In con số cụ thể. Nếu số đó lớn hơn 10% tổng số mẫu, **DỪNG và BÁO CÁO**
  thay vì tự ý nới `margin_threshold` để bù.

---

## NHIỆM VỤ 2 — Tinh chỉnh vùng mơ hồ của `canonicalize_chirality`

### Vấn đề

Trong `dynamic_dtw_engine.py` tìm chuỗi `CHIRALITY_AMBIGUITY_EPS`. Hàm
`canonicalize_chirality()` hiện dùng quy tắc: nếu không có nhãn `handedness` thì
lật khi `mean(nx) < 0`, nhưng **không lật** nếu `abs(mean(nx)) < 0.05` (vùng mơ hồ).

Đã đo: vùng mơ hồ này làm mất đúng 1 mẫu trong phép đo chuyển miền (61.8% thay vì
64.7%) và làm `intra_max` của từ `SORRY` **xấu đi** (0.411 → 0.433), vì SORRY có 2 mẫu
nằm trong vùng mơ hồ nên chúng không được lật trong khi các mẫu cùng từ thì đã lật.

Nguyên nhân: `mean(nx)` **triệt tiêu** khi lòng bàn tay quay qua phương thẳng đứng
giữa cử chỉ. Trung bình gần 0 KHÔNG có nghĩa là chirality mơ hồ — nó chỉ có nghĩa là
trung bình là một thống kê tệ cho việc này.

### Việc cần làm

Thay tiêu chí fallback (CHỈ nhánh fallback — nhánh dùng nhãn `handedness` giữ nguyên
tuyệt đối):

- Thay vì `mean(traj[:, 9])`, dùng **giá trị `nx` tại frame có `abs(nx)` LỚN NHẤT**.
  Đó là frame mà chirality bộc lộ rõ nhất.
- Lật khi giá trị đó `< 0`.
- Giữ lại cổng mơ hồ nhưng áp lên đại lượng mới: nếu `max(abs(nx))` trên toàn chuỗi
  `< 0.05` thì thực sự không quan sát được chirality → không lật.
- Giữ nguyên tên hằng số và tên tham số `ambiguity_eps` để không phá chữ ký hàm.
- Cập nhật docstring tiếng Việt để giải thích vì sao dùng frame cực đại thay vì trung bình.

### Tiêu chí hoàn thành (số liệu bắt buộc báo cáo)

1. Phép đo chuyển miền (xem bảng mốc trong PREAMBLE) phải đạt **>= 22/34 = 64.7%**,
   margin **>= +0.087**.
2. Chạy `tools/tune_thresholds.py` (KHÔNG có `--apply`) và báo cáo `intra_max` của `SORRY`.
   Yêu cầu: **không được lớn hơn 0.411** (giá trị trước khi có canonicalization).
3. LOO nội miền webcam phải giữ **>= 32/34 = 94.1%**.
4. `tools/audit_sign_quality.py --all` phải giữ **129/129**.
5. Chạy `tools/diagnose_chirality.py` và báo cáo số mẫu còn nằm trong vùng mơ hồ theo tiêu
   chí mới. Kỳ vọng: **ít hơn 6**.

---

## TUYỆT ĐỐI KHÔNG LÀM trong spec này

- Không đổi bất kỳ giá trị `lambda_shape` hay `lambda_shape_global` nào. Đã đo: mọi
  giá trị > 0 đều làm giảm độ chính xác (79.9% → 74.8% → 67.6% → 58.3%). Giữ 0.0.
- Không đổi `margin_threshold`, `uncertainty_ratio`, hay ngưỡng per-word nào.
- Không chạy `tools/tune_thresholds.py --apply`.
- Không sửa segmenter (đó là SPEC 01).
- Không thêm chiều đặc trưng nào (đó là SPEC 03).
