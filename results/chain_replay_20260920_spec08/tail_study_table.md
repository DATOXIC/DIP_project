[chain_replay] bank=C:\Users\TOPHUO~1\AppData\Local\Temp\claude\C--DIP-DIP-project\7e077625-6739-41a5-b49d-32ee37f404e0\scratchpad\snap08\bank_frozen.json sha256=e47fdb55543d9d64 | từ=13 mẫu=106 | stroke thô=171 (ánh xạ bank: 96) | k=1.75, unc_k=2.35
[chain_replay] chế độ=PRODUCTION (mốc) | từ nhiều pha: ['EAT', 'WATER']

| biến thể | n | top-1 | ACCEPTED-đúng | wrong-accept | UNCERTAIN | REJECTED | loại bởi heuristic chuyển động chêm | d_correct TB | d_correct trung vị |
|---|---|---|---|---|---|---|---|---|---|
| native | 171 | 152 (88.9%) | 142 (83.0%) | 1 | 9 | 19 | 3 | 0.294 | 0.202 |
| trim0.45 | 171 | 152 (88.9%) | 142 (83.0%) | 1 | 9 | 19 | 5 | 0.295 | 0.203 |
| trim0.30 | 171 | 152 (88.9%) | 143 (83.6%) | 1 | 8 | 19 | 12 | 0.298 | 0.208 |
| trim0.20 | 171 | 152 (88.9%) | 139 (81.3%) | 1 | 11 | 20 | 17 | 0.305 | 0.216 |
| trim0.10 | 165 | 146 (88.5%) | 128 (77.6%) | 1 | 16 | 20 | 33 | 0.323 | 0.238 |
| trim0.30+pad | 171 | 152 (88.9%) | 141 (82.5%) | 1 | 10 | 19 | 12 | 0.296 | 0.203 |
| trim0.20+pad | 171 | 152 (88.9%) | 140 (81.9%) | 1 | 11 | 19 | 17 | 0.296 | 0.203 |
| trim0.10+pad | 171 | 152 (88.9%) | 140 (81.9%) | 1 | 11 | 19 | 33 | 0.296 | 0.201 |
| trim0.30+pad+jitter | 171 | 152 (88.9%) | 141 (82.5%) | 1 | 10 | 19 | 12 | 0.297 | 0.204 |
| trim0.20+pad+jitter | 171 | 152 (88.9%) | 140 (81.9%) | 1 | 11 | 19 | 2 | 0.298 | 0.204 |
| trim0.10+pad+jitter | 171 | 152 (88.9%) | 140 (81.9%) | 1 | 10 | 20 | 0 | 0.298 | 0.199 |
