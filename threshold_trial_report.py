# -*- coding: utf-8 -*-
"""
threshold_trial_report.py - Báo cáo lần thử trực tiếp bằng webcam để CHỌN ngưỡng threshold_k.

Đọc CSV do `predict_sentence_webcam.py --log-session` sinh ra, ghép các sự kiện với
danh sách "đáp án" (truth.txt) bằng căn chỉnh chỉnh sửa (edit-distance), rồi thống kê:
số lần nhận đúng, nhận sai, nghi vấn, từ chối, bỏ sót, nhận nhầm cử chỉ rác (NEG)
và "nhận ma" (sự kiện ACCEPTED thừa không ứng với mục nào), kèm khoảng tin cậy Wilson 95%.

Chỉ đọc file; không ghi và không sửa bất kỳ dữ liệu nào của dự án.
"""

import argparse
import csv
import math
import sys
from typing import Dict, List, Optional, Tuple

# Tên sự kiện thật do predict_sentence_webcam.py ghi (đã đối chiếu với code):
EV_ACCEPTED = "ACCEPTED"
EV_UNCERTAIN = "UNCERTAIN"
EV_REJECTED = "REJECTED"
EV_OUT_OF_DURATION = "OUT_OF_DURATION"
EVENTS_MATCHABLE = (EV_ACCEPTED, EV_UNCERTAIN, EV_REJECTED)

NEG_KEY = "NEG"

# Nhãn kết quả cho mục truth là từ thật
C_ACC_OK = "ACCEPTED_đúng"
C_ACC_BAD = "ACCEPTED_sai"
C_UNC_OK = "UNCERTAIN_top1_đúng"
C_UNC_BAD = "UNCERTAIN_top1_sai"
C_REJ = "REJECTED"
C_MISS = "BỎ_SÓT"
WORD_CLASSES = [C_ACC_OK, C_ACC_BAD, C_UNC_OK, C_UNC_BAD, C_REJ, C_MISS]

# Nhãn cho mục NEG
C_NEG_ACC = "NEG_bị_ACCEPTED"
C_NEG_OK = "ĐÚNG_KHI_KHÔNG_NHẬN"


# ---------------------------------------------------------------------------
# Thống kê
# ---------------------------------------------------------------------------

def wilson_interval(x: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    """Khoảng tin cậy Wilson (mặc định 95%) cho tỉ lệ x/n. n = 0 trả về (nan, nan)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = x / n
    z2 = z * z
    den = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_rate(x: int, n: int) -> str:
    """Chuỗi 'x/n = p% [lo; hi]' (Wilson 95%)."""
    if n <= 0:
        return f"{x}/0 = n/a"
    lo, hi = wilson_interval(x, n)
    return f"{x}/{n} = {100.0 * x / n:5.1f}% [{100.0 * lo:5.1f}; {100.0 * hi:5.1f}]"


# ---------------------------------------------------------------------------
# Đọc dữ liệu
# ---------------------------------------------------------------------------

def parse_truth(path: str) -> List[str]:
    """Đọc truth.txt: mỗi dòng một mục (khóa từ hoặc NEG); '#' bắt đầu chú thích."""
    items: List[str] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            items.append(line.upper())
    return items


def load_events(path: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Đọc CSV phiên. Trả về (sự kiện dùng để ghép, sự kiện OUT_OF_DURATION).
    Các sự kiện lạ (tên không nhận ra) bị bỏ qua có cảnh báo.
    """
    matchable: List[Dict[str, str]] = []
    out_of_dur: List[Dict[str, str]] = []
    unknown = 0
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "event" not in reader.fieldnames:
            raise ValueError(f"File '{path}' không có cột 'event' - không phải CSV của --log-session.")
        for row in reader:
            ev = (row.get("event") or "").strip().upper()
            row["event"] = ev
            row["predicted_word"] = (row.get("predicted_word") or "").strip().upper()
            if ev in EVENTS_MATCHABLE:
                matchable.append(row)
            elif ev == EV_OUT_OF_DURATION:
                out_of_dur.append(row)
            elif ev:
                unknown += 1
    if unknown:
        print(f"[Cảnh báo] {path}: bỏ qua {unknown} dòng có tên sự kiện không nhận ra.", file=sys.stderr)
    return matchable, out_of_dur


# ---------------------------------------------------------------------------
# Căn chỉnh chỉnh sửa (edit-distance)
# ---------------------------------------------------------------------------

# Chi phí căn chỉnh (số nguyên, thang x4 để có nửa đơn vị):
COST_MISMATCH = 4        # ghép mục truth với sự kiện có từ khác (thay thế)
COST_SKIP_WORD = 4       # mục truth là từ thật không có sự kiện (bỏ sót)
COST_SKIP_NEG = 2        # mục NEG không có sự kiện: rẻ hơn, vì cử chỉ rác thường không tách thành stroke
COST_SKIP_EVENT = 4      # sự kiện thừa không ứng với mục truth nào
COST_NEG_NONACCEPT = 3   # NEG ghép với UNCERTAIN/REJECTED: phản ứng đúng mong đợi nhưng không chắc chắn bằng khớp từ


def _pair_cost(truth_item: str, ev: Dict[str, str]) -> int:
    """
    Chi phí ghép mục truth với sự kiện:
      - truth là từ thật: 0 nếu từ dự đoán trùng khóa truth, ngược lại COST_MISMATCH;
      - truth là NEG: COST_NEG_NONACCEPT nếu sự kiện không phải ACCEPTED, COST_MISMATCH nếu ACCEPTED.
    (Không cho NEG ghép miễn phí với REJECTED, nếu không một REJECTED thật của từ kế tiếp
    có thể bị "cướp" làm phản ứng của một NEG không sinh sự kiện.)
    """
    if truth_item == NEG_KEY:
        return COST_NEG_NONACCEPT if ev["event"] != EV_ACCEPTED else COST_MISMATCH
    return 0 if ev["predicted_word"] == truth_item else COST_MISMATCH


def _skip_truth_cost(truth_item: str) -> int:
    return COST_SKIP_NEG if truth_item == NEG_KEY else COST_SKIP_WORD


def align(truth: List[str], events: List[Dict[str, str]]) -> List[Tuple[Optional[int], Optional[int]]]:
    """
    Căn chỉnh truth với dãy sự kiện. Trả về danh sách cặp (chỉ_số_truth, chỉ_số_sự_kiện):
      (i, j)    ghép; (i, None) bỏ sót; (None, j) sự kiện thừa.
    Khi hòa chi phí, ưu tiên ghép chéo, rồi bỏ sót truth, rồi sự kiện thừa (tất định).
    """
    n, m = len(truth), len(events)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + _skip_truth_cost(truth[i - 1])
    for j in range(1, m + 1):
        dp[0][j] = j * COST_SKIP_EVENT
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i - 1][j - 1] + _pair_cost(truth[i - 1], events[j - 1])
            skip_t = dp[i - 1][j] + _skip_truth_cost(truth[i - 1])
            skip_e = dp[i][j - 1] + COST_SKIP_EVENT
            dp[i][j] = min(diag, skip_t, skip_e)

    pairs: List[Tuple[Optional[int], Optional[int]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + _pair_cost(truth[i - 1], events[j - 1]):
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + _skip_truth_cost(truth[i - 1]):
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    pairs.reverse()
    return pairs


# ---------------------------------------------------------------------------
# Phân loại và tổng hợp
# ---------------------------------------------------------------------------

def classify_word(truth_item: str, ev: Optional[Dict[str, str]]) -> str:
    """Phân loại kết quả cho một mục truth là từ thật."""
    if ev is None:
        return C_MISS
    correct = (ev["predicted_word"] == truth_item)
    if ev["event"] == EV_ACCEPTED:
        return C_ACC_OK if correct else C_ACC_BAD
    if ev["event"] == EV_UNCERTAIN:
        return C_UNC_OK if correct else C_UNC_BAD
    return C_REJ


def classify_neg(ev: Optional[Dict[str, str]]) -> str:
    """Phân loại kết quả cho một mục NEG (cử chỉ không phải sign)."""
    if ev is not None and ev["event"] == EV_ACCEPTED:
        return C_NEG_ACC
    return C_NEG_OK


def summarize(truth: List[str], events: List[Dict[str, str]], out_of_dur: List[Dict[str, str]]) -> Dict:
    """Ghép, phân loại và đếm. Trả về từ điển kết quả (có cả danh sách chi tiết từng mục)."""
    pairs = align(truth, events)
    counts = {c: 0 for c in WORD_CLASSES}
    neg_acc = neg_ok = neg_no_event = 0
    ghost_acc = ghost_other = 0
    detail: List[Tuple[str, str, str]] = []  # (truth/‘-’, sự kiện, nhãn)

    for ti, ej in pairs:
        if ti is not None:
            t = truth[ti]
            ev = events[ej] if ej is not None else None
            if t == NEG_KEY:
                lab = classify_neg(ev)
                if lab == C_NEG_ACC:
                    neg_acc += 1
                else:
                    neg_ok += 1
                    if ev is None:
                        neg_no_event += 1
            else:
                lab = classify_word(t, ev)
                counts[lab] += 1
            ev_txt = "-" if ev is None else f"{ev['event']}:{ev['predicted_word'] or '?'}"
            detail.append((t, ev_txt, lab))
        else:
            ev = events[ej]
            if ev["event"] == EV_ACCEPTED:
                ghost_acc += 1
                lab = "CHÈN_THỪA_nhận_ma"
            else:
                ghost_other += 1
                lab = "CHÈN_THỪA_không_nhận"
            detail.append(("-", f"{ev['event']}:{ev['predicted_word'] or '?'}", lab))

    n_word = sum(counts.values())
    n_neg = neg_acc + neg_ok
    n_truth = len(truth)
    bad_accepts = counts[C_ACC_BAD] + neg_acc + ghost_acc
    return {
        "n_truth": n_truth, "n_word": n_word, "n_neg": n_neg,
        "counts": counts,
        "neg_acc": neg_acc, "neg_ok": neg_ok, "neg_no_event": neg_no_event,
        "ghost_acc": ghost_acc, "ghost_other": ghost_other,
        "out_of_duration": len(out_of_dur),
        "n_events": len(events),
        "bad_accepts": bad_accepts,
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# In báo cáo
# ---------------------------------------------------------------------------

def print_report(name: str, s: Dict, verbose: bool = False) -> None:
    """In bảng đếm và tỉ lệ (Wilson 95%) cho một file CSV."""
    print("=" * 78)
    print(f"  BÁO CÁO: {name}")
    print("=" * 78)
    print(f"Mục truth: {s['n_truth']} (từ thật: {s['n_word']}, NEG: {s['n_neg']}) | "
          f"sự kiện ghép được: {s['n_events']} | OUT_OF_DURATION (không ghép): {s['out_of_duration']}")
    print(f"\nA. Với {s['n_word']} lần ký từ thật (mẫu số = {s['n_word']}):")
    for c in WORD_CLASSES:
        print(f"   {c:<22}: {fmt_rate(s['counts'][c], s['n_word'])}")
    print(f"\nB. Với {s['n_neg']} cử chỉ NEG (mẫu số = {s['n_neg']}):")
    print(f"   {C_NEG_ACC:<22}: {fmt_rate(s['neg_acc'], s['n_neg'])}   <- nhận nhầm cử chỉ rác (tệ nhất)")
    print(f"   {C_NEG_OK:<22}: {fmt_rate(s['neg_ok'], s['n_neg'])}"
          f"   (trong đó không sinh sự kiện: {s['neg_no_event']})")
    print(f"\nC. Sự kiện chèn thừa (không ghép được với mục truth nào):")
    print(f"   nhận ma (ACCEPTED thừa) : {s['ghost_acc']}")
    print(f"   thừa nhưng không nhận   : {s['ghost_other']}")
    print(f"\nD. Chi phí = ACCEPTED_sai + NEG_bị_ACCEPTED + nhận ma "
          f"= {s['counts'][C_ACC_BAD]} + {s['neg_acc']} + {s['ghost_acc']} = {s['bad_accepts']}")
    print(f"   Chi phí / tổng lần thử : {fmt_rate(s['bad_accepts'], s['n_truth'])}")
    good = s["counts"][C_ACC_OK]
    print(f"   Lợi ích: ACCEPTED_đúng / từ thật: {fmt_rate(good, s['n_word'])}")
    if verbose:
        print("\nChi tiết ghép (truth | sự kiện | nhãn):")
        for t, e, lab in s["detail"]:
            print(f"   {t:<12} | {e:<24} | {lab}")
    print()


def print_compare(name_a: str, sa: Dict, name_b: str, sb: Dict) -> None:
    """In hai cấu hình cạnh nhau (chỉ so tỉ lệ, không kiểm định ghép cặp)."""
    print("=" * 78)
    print("  SO SÁNH HAI CẤU HÌNH (cạnh nhau)")
    print(f"  A = {name_a}")
    print(f"  B = {name_b}")
    print("=" * 78)
    rows = []
    for c in WORD_CLASSES:
        rows.append((f"{c} (/từ thật)", sa["counts"][c], sa["n_word"], sb["counts"][c], sb["n_word"]))
    rows.append((f"{C_NEG_ACC} (/NEG)", sa["neg_acc"], sa["n_neg"], sb["neg_acc"], sb["n_neg"]))
    rows.append(("nhận ma (/lần thử)", sa["ghost_acc"], sa["n_truth"], sb["ghost_acc"], sb["n_truth"]))
    rows.append(("CHI PHÍ (/lần thử)", sa["bad_accepts"], sa["n_truth"], sb["bad_accepts"], sb["n_truth"]))
    print(f"{'Chỉ số':<34}| {'A':<34}| {'B':<34}")
    print("-" * 104)
    for label, xa, na, xb, nb in rows:
        print(f"{label:<34}| {fmt_rate(xa, na):<34}| {fmt_rate(xb, nb):<34}")
    print(f"{'OUT_OF_DURATION (không ghép)':<34}| {sa['out_of_duration']:<34}| {sb['out_of_duration']:<34}")
    print()
    print("LƯU Ý: hai phiên thử KHÔNG ghép cặp theo từng lần ký (người ký mỗi lần một khác),")
    print("nên chỉ so sánh được TỈ LỆ; không dùng McNemar. Nếu hai khoảng tin cậy Wilson chồng")
    print("lên nhau thì chưa đủ bằng chứng là hai cấu hình khác nhau. Cỡ mẫu nhỏ (vài chục lần")
    print("thử) chỉ phân biệt được chênh lệch rất lớn; cần vài trăm lần để thấy chênh ~7 điểm %.")
    print()


# ---------------------------------------------------------------------------
# Dòng lệnh
# ---------------------------------------------------------------------------

EPILOG = """
ĐỊNH DẠNG truth.txt (mỗi dòng một mục, theo ĐÚNG THỨ TỰ bạn thử; dòng bắt đầu bằng # là chú thích):

    # Phiên thử 20/09, cấu hình k=1.6
    HELLO
    HELLO
    BYE
    NEG          # gãi mặt, KHÔNG phải sign
    THANK_YOU
    NEG          # đưa tay xuống bàn
    ME

  - Khóa từ là khóa trong bank (HELLO, BYE, ME, YOU, THANK_YOU, EAT, DRINK, WATER, SORRY, PLEASE, ...).
  - NEG là cử chỉ không phải sign: ACCEPTED = nhận nhầm rác (tệ nhất); còn lại là đúng.
  - Không cần ghi mục nào bị hệ thống bỏ sót: công cụ tự căn chỉnh. Nhưng KHÔNG được bỏ mục
    khỏi truth: mỗi lần bạn CỐ Ý thử phải có một dòng.

QUY TRÌNH:
  1) py predict_sentence_webcam.py --log-session results\\ab_k16.csv ...   (cấu hình 1)
  2) py predict_sentence_webcam.py --log-session results\\ab_k175.csv ...  (cấu hình 2)
     Mỗi cấu hình ký ĐÚNG cùng danh sách truth, cùng điều kiện ánh sáng. Dùng file CSV MỚI cho
     mỗi lần (log ghi nối tiếp; nếu dùng lại tên file, các lần chạy cũ sẽ lẫn vào).

VÍ DỤ:
  py threshold_trial_report.py results\\ab_k16.csv --truth truth.txt
  py threshold_trial_report.py results\\ab_k16.csv results\\ab_k16_lan2.csv --truth truth.txt
  py threshold_trial_report.py --compare results\\ab_k16.csv results\\ab_k175.csv --truth truth.txt
  py threshold_trial_report.py results\\ab_k16.csv --truth truth.txt --chi-tiet

CÁCH GHÉP: căn chỉnh chỉnh sửa (edit-distance) giữa danh sách truth và dãy sự kiện ACCEPTED /
UNCERTAIN / REJECTED theo thứ tự thời gian. Chi phí (thang x4): ghép đúng từ = 0; ghép sai từ = 4;
NEG ghép UNCERTAIN/REJECTED = 3, NEG ghép ACCEPTED = 4; bỏ sót từ = 4; bỏ sót NEG = 2 (cử chỉ rác
hay không tách thành stroke); sự kiện thừa = 4. Khi hòa, ưu tiên ghép chéo. Cảnh báo: khi một mục bị bỏ
sót và sự kiện thừa nằm sát nhau, cách ghép có thể nhập nhằng (bản chất của bài toán); dùng --chi-tiet
để kiểm tra bảng ghép. Sự kiện
OUT_OF_DURATION (cử chỉ ngoài 0.4-2.5s) KHÔNG tham gia ghép, chỉ được đếm riêng.
Kết quả UNCERTAIN/REJECTED không ghi từ đúng nên chỉ có nhãn top1 đúng/sai theo từ dự đoán.
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="threshold_trial_report.py",
        description=("Báo cáo lần thử trực tiếp bằng webcam: ghép CSV của `predict_sentence_webcam.py "
                     "--log-session` với truth.txt, thống kê đúng/sai/nghi vấn/từ chối/bỏ sót/nhận nhầm rác/nhận ma, "
                     "kèm khoảng tin cậy Wilson 95%. Dùng để so sánh các ngưỡng threshold_k."),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("csv", nargs="*", help="Một hoặc nhiều file CSV phiên (mỗi file báo cáo riêng).")
    p.add_argument("--truth", required=True, help="File truth.txt (mỗi dòng một mục: khóa từ hoặc NEG).")
    p.add_argument("--compare", nargs=2, metavar=("A.csv", "B.csv"),
                   help="So sánh hai cấu hình cạnh nhau (chỉ so tỉ lệ, không McNemar).")
    p.add_argument("--chi-tiet", action="store_true", help="In thêm bảng ghép từng mục (truth | sự kiện | nhãn).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    args = build_parser().parse_args(argv)
    if not args.csv and not args.compare:
        print("Lỗi: cần ít nhất một file CSV hoặc --compare A.csv B.csv.", file=sys.stderr)
        return 2
    try:
        truth = parse_truth(args.truth)
    except OSError as e:
        print(f"Lỗi đọc truth: {e}", file=sys.stderr)
        return 2
    if not truth:
        print("Lỗi: truth.txt không có mục nào.", file=sys.stderr)
        return 2

    if args.compare:
        results = []
        for path in args.compare:
            ev, ood = load_events(path)
            s = summarize(truth, ev, ood)
            results.append((path, s))
            print_report(path, s, verbose=args.chi_tiet)
        print_compare(results[0][0], results[0][1], results[1][0], results[1][1])
        return 0

    for path in args.csv:
        ev, ood = load_events(path)
        print_report(path, summarize(truth, ev, ood), verbose=args.chi_tiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
