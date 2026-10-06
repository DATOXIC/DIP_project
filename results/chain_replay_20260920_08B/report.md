# SPEC 08B — BÁO CÁO KẾT QUẢ ĐO LƯỜNG

> Sinh viên: Tô Phương Đạt | MSSV: 24110086 | Môn: Xử Lý Ảnh Số (DIP)
> Thời gian: 20/09/2026

## Tóm tắt

| Nhiệm vụ | Trạng thái |
|-----------|-----------|
| A — Mô hình chuyển tiếp realistic | HOÀN THÀNH |
| B — transition_lockout_sec | HOÀN THÀNH |
| C — lead_trim.py | HOÀN THÀNH |
| D — Quét lưới, tập kiểm | HOÀN THÀNH |
| E — Docstring, trial report | HOÀN THÀNH |

## Hồi quy

- test_segmenter_fast_chain.py: 30/30 PASSED
- test_lead_trim.py: 6/6 PASSED
- test_fast_dtw_parity.py: 8/8 PASSED
- pytest -q tổng: 181 passed / 50 failed (50 failures là pre-existing, xem walkthrough)

## Tập chỉnh (seed A=20260920, 200 câu, 30 Hz, realistic, g=0.0)

| Cấu hình | Câu đúng | Từ đúng | Nhiễm dẫn đầu | Mất đầu từ | Hitch P95 |
|----------|----------|---------|-------------|-----------|----------|
| S0 fast baseline | 1.5% | 17.1% | 144 (29.3%) | 33 (6.7%) | 0 ms |
| S1 lockout=0.10 | 2.0% | 17.5% | 152 (31.0%) | 48 (9.8%) | 0 ms |
| S1 lockout=0.25 | 4.5% | 18.1% | 144 (29.3%) | 37 (7.5%) | 0 ms |
| S1 lockout=0.30 | 4.5% | 17.9% | 138 (28.1%) | 32 (6.5%) | 0 ms |
| S2 lead_trim (mg=0.10,p=0.30) | 9.0% | 27.3% | 107 (21.8%) | 54 (11.0%) | 33 ms |
| S3 lockout=0.10+lead_trim | 11.0% | 28.7% | 113 (23.0%) | 68 (13.8%) | 32 ms |
| S3 lockout=0.15+lead_trim | 10.5% | 27.7% | 113 (23.0%) | 68 (13.8%) | 37 ms |
| S4 production (no fast) | 0.5% | 3.3% | 107 (21.8%) | 3 (0.6%) | 0 ms |

## Tập kiểm (seed B=20260921, 200 câu, 30 Hz, realistic, g=0.0) — MỘT LẦN

| Cấu hình | Câu đúng | Từ đúng | Nhiễm dẫn đầu | Mất đầu từ | Hitch P95 |
|----------|----------|---------|-------------|-----------|----------|
| S0 fast baseline | 4.5% | 16.9% | 133 (27.7%) | 30 (6.2%) | 0 ms |
| S2 lead_trim (đóng băng) | 11.5% | 27.5% | 89 (18.5%) | 60 (12.5%) | 23.5 ms |
| S3 lockout=0.10+lead_trim | 14.5% | 29.2% | 94 (19.6%) | 72 (15.0%) | 25.4 ms |
| S4 production (no fast) | 0.5% | 1.7% | 103 (21.5%) | 0 (0.0%) | 0 ms |

## Tập kiểm 15 Hz (seed B=20260921)

| Cấu hình | Câu đúng | Nhiễm dẫn đầu | Mất đầu từ |
|----------|----------|-------------|-----------|
| S0 fast baseline | 1.5% | 238 (49.6%) | 3 (0.6%) |
| S2 lead_trim (đóng băng) | 3.0% | 219 (45.6%) | 18 (3.8%) |
| S3 lockout=0.10+lead_trim | 3.0% | 231 (48.1%) | 25 (5.2%) |
| S4 production | 0.0% | 244 (50.8%) | 0 (0.0%) |

## Hitch Latency (100 strokes, max_variants=3)

- Tất cả strokes: mean=7.77ms, p95=18.5ms
- Chỉ strokes có tìm kiếm (16/100): mean=15.8ms, p95=21.8ms
- Ngân sách spec: p95 <= 25ms → DAT

## LOO Từ đơn (n=203)

- Baseline: 203/203 top-1, 203/203 ACCEPTED-dung
- S2 (lead_trim mg=0.1, p=0.3): 200/203 top-1 (-3 hoi quy)
- Strokes co tau>0: 18/203 (8.9%)

## Cấu hình đóng băng: S2

```json
{
  "transition_lockout_sec": 0.0,
  "lead_trim": {
    "enabled": false,
    "max_sec": 0.45,
    "max_variants": 3,
    "good_r": 0.80,
    "min_gain": 0.10,
    "trim_penalty_per_sec": 0.30
  }
}
```

Kích hoạt: py predict_sentence_webcam.py --fast-chain --lead-trim

## CANH BAO TRUNG THUC

- So do la MO PHONG (synthetic strokes), khong phai chuyen dong tay that
- Ket luan cuoi can thu truc tiep (25 phut, xem Nhiem vu E)
- 15 Hz so do rat thap vi camera that tra ~6-14 khung moi/s
