# -*- coding: utf-8 -*-
"""
CHAIN TRIAL REPORT - SO SÁNH "KÝ NHANH" VỚI "KÝ THƯỜNG" TRÊN PHIÊN THỬ TRỰC TIẾP (SPEC 08 Nhiệm vụ E)
=====================================================================================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086 | Môn: Xử Lý Ảnh Số (DIP)

Đọc một hoặc nhiều file CSV phiên do `predict_sentence_webcam.py --log-session` ghi ra, cùng một file văn bản `truth.txt`
(mỗi dòng là MỘT câu đúng, các từ cách nhau bằng dấu cách, viết bằng KHÓA TỪ như trong bank: HELLO, THANK_YOU, ...,
theo đúng THỨ TỰ đã thử). In cho từng CSV:
    - tỉ lệ câu đúng (chuỗi từ ACCEPTED trùng khít với dòng truth tương ứng),
    - từ thiếu / từ thừa (căn theo dãy con chung dài nhất, đếm theo từ),
    - khoảng nghỉ giữa hai từ ACCEPTED liên tiếp trong cùng một câu = từ lúc CHỐT từ trước tới lúc BẮT ĐẦU stroke của từ
      sau (cột t_unix và t_start; không tính thời gian ký của từ sau): trung vị và p95 (giây),
    - số từ chốt sớm / chốt thường (cột commit_mode, chỉ có ở log của bản có SPEC 08).

CÁCH GHÉP CÂU THEO THỜI GIAN: các dòng log (mọi sự kiện: ACCEPTED, UNCERTAIN, REJECTED, OUT_OF_DURATION) được xếp theo
thời gian; nếu từ lúc chốt dòng trước tới lúc BẮT ĐẦU stroke của dòng sau > --sentence-gap (mặc định 1,8 s, đúng ngưỡng
"hạ tay chốt câu" của ứng dụng) thì tách sang câu mới. Dùng cột t_unix / t_start (mili giây) nếu có; log cũ chỉ có cột
timestamp (độ phân giải 1 giây) thì mọi con số thời gian thô (in cảnh báo). Sự kiện chốt câu KHÔNG có dòng log riêng,
nên chỉ dựa vào khoảng nghỉ.
Số nhóm câu tìm được nên bằng số dòng truth; nếu lệch, báo cáo nêu rõ và so theo thứ tự cho tới hết phần chung.

CÁCH CHẠY THỬ (khoảng 25 phút, ánh sáng tốt như lúc demo; mỗi câu ký xong hãy HẠ TAY khỏi khung > 2 giây rồi mới ký câu sau):
    1) Viết truth.txt: 15 câu hai từ + 5 câu ba từ, mỗi câu một dòng, ví dụ  "HELLO YOU"  /  "ME EAT WATER".
    2) Ký lần lượt CÁC CÂU ĐÓ trong 4 lượt (mỗi lượt một file CSV riêng):
       - Chế độ thường:
           py predict_sentence_webcam.py --no-fast-chain --log-session results\trial_thuong.csv
       - Chế độ nhanh hiện tại (S0):
           py predict_sentence_webcam.py --fast-chain --log-session results\trial_fast_s0.csv
       - Chế độ nhanh + khoảng cấm lockout:
           py predict_sentence_webcam.py --fast-chain --transition-lockout 0.15 --log-session results\trial_fast_lockout.csv
       - Chế độ nhanh + cắt phần dẫn đầu (lead_trim):
           py predict_sentence_webcam.py --fast-chain --lead-trim --log-session results\trial_fast_leadtrim.csv
       - Cả hai (khoảng cấm + cắt thích nghi):
           py predict_sentence_webcam.py --fast-chain --transition-lockout 0.15 --lead-trim --log-session results\trial_fast_both.csv
    3) So sánh:
           py chain_trial_report.py --truth truth.txt results\trial_thuong.csv results\trial_fast_s0.csv results\trial_fast_lockout.csv results\trial_fast_leadtrim.csv
    Ghi thêm cảm nhận "chuyển tiếp còn bị ghi không / có mất đầu từ không" (thang 1-5) cho từng chế độ.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def read_truth(path: str) -> List[List[str]]:
    """Mỗi dòng không rỗng = một câu; từ được chuẩn hóa: in hoa, bỏ khoảng trắng thừa."""
    out: List[List[str]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            toks = line.strip().upper().split()
            if toks:
                out.append(toks)
    return out


def _parse_ts(s: str) -> float:
    return time.mktime(time.strptime(s.strip(), "%Y-%m-%d %H:%M:%S"))


def read_log(path: str) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Đọc CSV phiên. Trả về (danh sách sự kiện có khóa t, event, word, mode), cờ has_ms (True nếu dùng t_unix mili giây).
    Dòng thiếu/hỏng bị bỏ qua. Log ghi nối nhiều lần (nhiều dòng tiêu đề) vẫn đọc được.
    """
    rows: List[Dict[str, Any]] = []
    has_ms = False
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header: Optional[List[str]] = None
        for r in reader:
            if not r:
                continue
            if r[0].strip() == "timestamp":       # dòng tiêu đề (có thể lặp lại nếu log được ghi nối)
                header = [c.strip() for c in r]
                continue
            if header is None:
                continue
            d = {header[i]: r[i] for i in range(min(len(header), len(r)))}
            try:
                if d.get("t_unix") not in (None, ""):
                    t = float(d["t_unix"])
                    has_ms = True
                else:
                    t = _parse_ts(d["timestamp"])
            except Exception:
                continue
            t_start = None
            try:
                if d.get("t_start") not in (None, ""):
                    t_start = float(d["t_start"])
                elif d.get("duration_sec") not in (None, ""):
                    t_start = t - float(d["duration_sec"])      # xấp xỉ (log cũ): thời lượng stroke gồm cả đuôi yên
            except Exception:
                t_start = None
            lead_trim_sec = None
            try:
                if d.get("lead_trim_sec") not in (None, ""):
                    lead_trim_sec = float(d["lead_trim_sec"])
            except Exception:
                lead_trim_sec = None
            lead_trim_tried = None
            try:
                if d.get("lead_trim_tried") not in (None, ""):
                    lead_trim_tried = int(d["lead_trim_tried"])
            except Exception:
                lead_trim_tried = None
            rows.append(dict(t=t, t_start=(t_start if t_start is not None else t), event=d.get("event", ""),
                             word=(d.get("predicted_word") or "").strip().upper(),
                             mode=d.get("commit_mode", ""), quality=d.get("match_quality", ""),
                             exact_start=(d.get("t_start") not in (None, "")),
                             lead_trim_sec=lead_trim_sec, lead_trim_tried=lead_trim_tried))
    rows.sort(key=lambda x: x["t"])
    return rows, has_ms


def split_sentences(rows: List[Dict[str, Any]], gap: float) -> List[List[Dict[str, Any]]]:
    groups: List[List[Dict[str, Any]]] = []
    for r in rows:
        # Khoảng nghỉ = từ lúc chốt dòng trước tới lúc BẮT ĐẦU stroke của dòng này (không tính thời gian ký của từ sau)
        if not groups or (r["t_start"] - groups[-1][-1]["t"]) > gap:
            groups.append([r])
        else:
            groups[-1].append(r)
    return groups


def analyse(rows: List[Dict[str, Any]], truth: List[List[str]], gap: float) -> Dict[str, Any]:
    groups = split_sentences(rows, gap)
    n_cmp = min(len(groups), len(truth))
    correct = lost = extra = 0
    gaps: List[float] = []
    early = normal = 0
    details = []
    for i in range(n_cmp):
        acc = [r for r in groups[i] if r["event"] == "ACCEPTED"]
        pred = [r["word"] for r in acc]
        tr = truth[i]
        sm = difflib.SequenceMatcher(a=tr, b=pred, autojunk=False)
        lcs = sum(b.size for b in sm.get_matching_blocks())
        lost += len(tr) - lcs
        extra += len(pred) - lcs
        ok = (pred == tr)
        correct += int(ok)
        for a, b in zip(acc[:-1], acc[1:]):
            gaps.append(max(0.0, b["t_start"] - a["t"]))
        for r in acc:
            if r["mode"] == "early":
                early += 1
            elif r["mode"] == "normal":
                normal += 1
        details.append((i + 1, " ".join(tr), " ".join(pred), ok))
    truth_words = sum(len(t) for t in truth[:n_cmp])
    return dict(
        n_truth=len(truth), n_groups=len(groups), n_cmp=n_cmp, correct=correct, lost=lost, extra=extra,
        acc=correct / n_cmp if n_cmp else float("nan"), truth_words=truth_words,
        gap_median=float(np.median(gaps)) if gaps else float("nan"), gap_p95=float(np.percentile(gaps, 95)) if gaps else float("nan"),
        n_gaps=len(gaps), early=early, normal=normal, details=details,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="Một hoặc nhiều file CSV phiên (--log-session).")
    ap.add_argument("--truth", required=True, help="File truth.txt: mỗi dòng một câu đúng (khóa từ cách nhau bằng dấu cách).")
    ap.add_argument("--sentence-gap", type=float, default=1.8, help="Khoảng nghỉ (giây) tách câu (mặc định 1.8).")
    ap.add_argument("--verbose", action="store_true", help="In từng câu (đúng/sai).")
    args = ap.parse_args()

    truth = read_truth(args.truth)
    if not truth:
        print("[!] truth.txt rỗng.")
        sys.exit(1)
    print(f"Truth: {len(truth)} câu, {sum(len(t) for t in truth)} từ (từ {args.truth}). Ngưỡng tách câu: {args.sentence_gap:.1f} s\n")
    results = []
    for path in args.csv:
        rows, has_ms = read_log(path)
        res = analyse(rows, truth, args.sentence_gap)
        res["path"], res["has_ms"], res["n_rows"], res["raw_rows"] = path, has_ms, len(rows), rows
        results.append(res)

    print("| phiên | dòng log | nhóm câu / truth | câu đúng | từ thiếu | từ thừa | khoảng nghỉ 2 từ trung vị (s) | p95 (s) | chốt sớm/thường |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        name = os.path.splitext(os.path.basename(r["path"]))[0]
        print(f"| {name} | {r['n_rows']} | {r['n_groups']} / {r['n_truth']} | {r['correct']}/{r['n_cmp']} ({100 * r['acc']:.1f}%) | "
              f"{r['lost']} | {r['extra']} | {r['gap_median']:.2f} | {r['gap_p95']:.2f} | {r['early']}/{r['normal']} |")
    for r in results:
        name = os.path.basename(r["path"])
        if r["n_groups"] != r["n_truth"]:
            print(f"\n[!] {name}: tìm được {r['n_groups']} nhóm câu nhưng truth có {r['n_truth']} dòng -> so theo thứ tự {r['n_cmp']} câu đầu. "
                  f"Kiểm tra lại: có câu bị bỏ, hai câu dính nhau (nghỉ < {args.sentence_gap:.1f} s) hoặc dòng log nhiễu.")
        if not r["has_ms"]:
            print(f"[!] {name}: log không có cột t_unix (mili giây) -> mọi con số thời gian chỉ chính xác tới ~1 giây.")
        if r["n_gaps"] == 0:
            print(f"[!] {name}: không có cặp từ liên tiếp nào để đo khoảng nghỉ.")
    if args.verbose:
        for r in results:
            print(f"\n--- {os.path.basename(r['path'])}")
            for i, tr, pr, ok in r["details"]:
                print(f"  {i:>2}. {'OK ' if ok else 'SAI'} truth: {tr:<28} | nhận: {pr}")
    # In thêm phân bố lead_trim_sec cho từng phiên nếu có
    has_lead_trim = any(any(r.get("lead_trim_sec") is not None for r in res.get("raw_rows", [])) for res in results)
    if has_lead_trim:
        print("\n| phiên | số stroke có lead_trim | τ trung vị (s) | τ p90 (s) | tỉ lệ τ > 0 | τ lớn nhất (s) |")
        print("|---|---|---|---|---|---|")
        for res in results:
            name = os.path.splitext(os.path.basename(res["path"]))[0]
            trims = [float(r["lead_trim_sec"]) for r in res.get("raw_rows", []) if r.get("lead_trim_sec") is not None]
            if trims:
                med = float(np.median(trims))
                p90 = float(np.percentile(trims, 90))
                frac_pos = float(np.mean([t > 0 for t in trims]))
                mx = float(np.max(trims))
                print(f"| {name} | {len(trims)} | {med:.3f} | {p90:.3f} | {100 * frac_pos:.1f}% | {mx:.3f} |")
            else:
                print(f"| {name} | 0 | - | - | - | - |")

    print("\nGhi chú: khoảng nghỉ = lúc bắt đầu stroke của từ sau trừ lúc chốt từ trước (không gồm thời gian ký từ sau). "
          "Chế độ nhanh chốt sớm hơn nên số đo có thể khác nhau dù người ký giữ nhịp như nhau; chỉ so hai chế độ trên CÙNG bộ câu.")


if __name__ == "__main__":
    main()
