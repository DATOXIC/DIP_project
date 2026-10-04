# -*- coding: utf-8 -*-
"""
FUSION SUY LUẬN SÂU (Bi-GRU + Attention, xem train_dynamic_dl.py) CHO DTW ĐỘNG — TẦNG
DỰ PHÒNG KHI DTW KHÔNG CHẮC, THAY THẾ BLEND ALPHA CỐ ĐỊNH ĐÃ SAI CỦA eval_dynamic_hybrid.py.

Bối cảnh: bản blend cũ (hybrid = 0.65*dl_probs + 0.35*dtw_scores, MỌI mẫu, không xét
result.status) đo được là ghi đè oan DTW đúng ở các từ DTW mạnh hơn DL (BYE 3/4 dtw vs
2/4 dl -> hybrid=2/4, EAT 4/4 vs 2/4 -> hybrid=2/4, DRINK 4/5 vs 2/5 -> hybrid=2/5 —
hybrid trùng khớp DL tuyệt đối, tức alpha=0.65 luôn thắng, không hề "cân đo" gì).
So sánh có kiểm soát thật (results/baselines/deep_pilot4.json, LOSO 5 người thật, quét
số người tham chiếu k=1..4) cho thấy DTW và GRU CẠNH TRANH NHAU tùy người/từ, không phải
một bên luôn thắng — nên quyết định "dùng kết quả nào" phải dựa vào ĐỘ TỰ TIN của từng
lần so khớp cụ thể, không phải một trọng số cố định áp cho mọi mẫu.

Khác handshape_fusion.py (chỉ giới hạn 2 từ DRINK/WATER và giới hạn từ đề xuất phải nằm
trong top-3 candidate của DTW, vì bộ phân loại dáng tay đó có SAI LỆCH HỆ THỐNG đã đo được
với người lạ ở các từ khác): DL KHÔNG bị giới hạn phải trùng 1 trong các candidate DTW đã
xếp hạng gần — giá trị lớn nhất của DL nằm chính ở các từ DTW SAI HẲN (không phải chỉ xếp
hạng thấp mà sai hẳn, ví dụ PLEASE/HUNGRY từng đo 0% với DTW trên người lạ Dang) nên nếu bắt
DL phải chọn trong tập candidate DTW đã đề xuất thì sẽ tự triệt tiêu đúng lợi ích đó.

LUẬT GATE (bắt buộc, không được nới lỏng — cùng tinh thần handshape_fusion.py):
  1. CHỈ can thiệp khi result.status != "ACCEPTED" — không bao giờ ghi đè quyết định mà
     DTW đang tự tin.
  2. CHỈ nhận đề xuất của DL nếu xác suất softmax lớn nhất của nó >= ngưỡng tin cậy đã
     hiệu chỉnh (mặc định DEFAULT_CONF_THRESHOLD, nên hiệu chỉnh lại bằng
     sweep_confidence_threshold() trên một tập VALIDATION riêng, không phải tập test cuối
     dùng để báo cáo số liệu, tránh rò rỉ).
  3. Từ DL chọn phải khớp một trong các candidate mà DTW đã liệt kê (mọi từ trong bank đều
     được liệt kê trong result.candidates, không chỉ top vài từ) — không bao giờ "bịa" ra
     một MatchCandidate mới nằm ngoài bank hiện tại.
"""
from __future__ import annotations

import dataclasses
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Do bang quet day du (khong chi lay "best" tu sweep_confidence_threshold - de bi trung diem tie-break
# xau) tren ca 4 nguoi la that (Dang/Mai/Phi/Vuong), buoc 0.02 tu 0.20 den 0.50 (xem lich su sua doi +
# calibrate_dl_threshold_nested.py). Duong cong accuracy trung binh co 2 mien on dinh: 0.20-0.26 va
# 0.34-0.42 (ca 2 deu 96.6%), nhung TUT xuong 96.0% dung o khoang 0.28-0.32 (1 mau EAT cua Dang co
# do tin cay DL nam dung trong "hom lom" nay va SAI). 0.32 (trung binh tho 3 ngưỡng single-best truoc
# do) vo tinh roi dung vao hom lom nay - da SUA lai bang cach chon giua mien on dinh 0.34-0.42, xa ca
# 2 mep, thay vi tin tuyet doi 1 con so "best" tu grid thua. Kiem lai bang nested CV (moi nguoi chon
# nguong tu 3 nguoi con lai, khong ro ri) cho ket qua trung binh 96.6% tai day, dung bang mien on dinh.
DEFAULT_CONF_THRESHOLD = 0.38

_MODEL_CACHE: Dict[str, Any] = {}


def _lazy_torch():
    import torch
    import torch.nn.functional as F
    return torch, F


def load_model(weights_path: Optional[str], device_str: str = "cpu") -> Optional[Dict[str, Any]]:
    """
    Nạp (và cache theo đường dẫn) LightweightGestureNet đã train từ train_dynamic_dl.py.
    Trả None nếu thiếu đường dẫn/file hoặc nạp lỗi — cho phép hệ thống chạy TIẾP KHÔNG có
    tầng DL (an toàn, cùng kiểu tự-tắt mà handshape_fusion dùng khi thiếu mẫu tham chiếu).
    """
    if not weights_path:
        return None
    if weights_path in _MODEL_CACHE:
        return _MODEL_CACHE[weights_path]
    if not os.path.exists(weights_path):
        _MODEL_CACHE[weights_path] = None
        return None
    try:
        torch, _ = _lazy_torch()
        from train_dynamic_dl import LightweightGestureNet, TARGET_WORDS
        device = torch.device(device_str)
        model = LightweightGestureNet(in_features=24, num_classes=len(TARGET_WORDS), hidden_dim=64).to(device)
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        words = list(checkpoint.get("target_words", TARGET_WORDS))
        bundle: Optional[Dict[str, Any]] = {"model": model, "device": device, "words": words}
    except Exception as e:
        print(f"[dl_fusion] Không nạp được mô hình DL tại {weights_path}: {e}")
        bundle = None
    _MODEL_CACHE[weights_path] = bundle
    return bundle


def predict(bundle: Dict[str, Any], query_traj_24d: np.ndarray) -> Optional[Tuple[str, float, np.ndarray]]:
    """Suy luận 1 mẫu, trả (từ_dự_đoán, độ_tin_cậy_softmax, toàn_bộ_phân_phối_xác_suất)
    hoặc None nếu query không đúng hình dạng (32, 24)."""
    traj = np.asarray(query_traj_24d, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[0] != 32:
        return None
    torch, F = _lazy_torch()
    model = bundle["model"]
    device = bundle["device"]
    words = bundle["words"]
    with torch.no_grad():
        x = torch.from_numpy(traj[None, ...]).to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    if idx >= len(words):
        return None
    return words[idx], float(probs[idx]), probs


def apply_dl_fallback(result: Any, query_traj_24d: Optional[np.ndarray], bundle: Optional[Dict[str, Any]],
                       conf_threshold: float = DEFAULT_CONF_THRESHOLD) -> Any:
    """
    Áp gate DL lên MatchResult của DTW. Xem LUẬT GATE ở docstring module — tuyệt đối không
    nới lỏng điều kiện 1 (chỉ can thiệp khi status != ACCEPTED).
    """
    if result.status == "ACCEPTED" or bundle is None or not result.candidates:
        return result

    pred = predict(bundle, query_traj_24d)
    if pred is None:
        return result
    word, conf, _probs = pred
    # Ghi lại đề xuất của DL ngay cả khi cuối cùng KHÔNG được nhận (dưới ngưỡng, hoặc trùng
    # top1 rồi) — panel demo "Nguồn DTW/DL" cần thấy DL đã được hỏi và trả lời gì, không chỉ
    # thấy im lặng khi nó bị từ chối.
    result = dataclasses.replace(result, dl_word=word, dl_confidence=conf)
    if conf < conf_threshold:
        return result

    cand_map = {c.word_key: c for c in result.candidates}
    chosen = cand_map.get(word)
    if chosen is None or chosen is result.candidates[0]:
        return result  # tu DL chon khong nam trong bank hien tai, hoac da la top1 roi

    new_candidates = [chosen] + [c for c in result.candidates if c is not chosen]
    note = f"dl_fallback:{word}({conf:.2f})<-{result.top_word}"
    return dataclasses.replace(
        result,
        status="ACCEPTED",
        top_word=chosen.word_key,
        top_display=chosen.display_name,
        best_distance=chosen.distance,
        best_confidence=chosen.confidence,
        candidates=new_candidates,
        match_quality=chosen.match_quality,
        r_ratio=chosen.r_ratio,
        fusion_note=note,
        result_source="DL",
    )


def gated_prediction(status: str, dtw_word: Optional[str], dl_word: Optional[str],
                      dl_conf: float, threshold: float) -> Optional[str]:
    """Bản rút gọn (không cần MatchResult đầy đủ) của cùng luật gate, dùng trong script
    đánh giá offline: DTW ACCEPTED hoặc DL không dự đoán được -> giữ DTW; ngược lại nhận
    DL nếu độ tin cậy vượt ngưỡng, không thì vẫn giữ nguyên top-1 hiện tại của DTW."""
    if status == "ACCEPTED" or dl_word is None:
        return dtw_word
    return dl_word if dl_conf >= threshold else dtw_word


def sweep_confidence_threshold(rows: List[Dict[str, Any]],
                                grid: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Quét ngưỡng tin cậy DL trên một tập đã có nhãn thật, để HIỆU CHỈNH conf_threshold —
    PHẢI chạy trên tập VALIDATION riêng, không phải tập test cuối cùng dùng để báo cáo số
    liệu (nếu không sẽ rò rỉ: chọn ngưỡng "đẹp nhất" cho đúng người/mẫu đang report).

    "best" KHÔNG đơn giản là điểm accuracy cao nhất đầu tiên gặp trong grid — đã đo được
    (xem lịch sử sửa DEFAULT_CONF_THRESHOLD) đường cong accuracy có thể có "hõm lõm" hẹp xen
    giữa 2 miền ổn định cao hơn (1 mẫu biên đổi kết quả trong khoảng hẹp). Chọn theo điểm cao
    nhất đầu tiên có thể vô tình rơi đúng cạnh hõm đó. Thay vào đó: tìm accuracy cao nhất, rồi
    trong các đoạn NGƯỠNG LIÊN TỤC đạt accuracy đó, chọn NGƯỠNG GIỮA của đoạn DÀI NHẤT — tức
    điểm xa 2 mép nhất trong miền ổn định rộng nhất, bền hơn khi test trên người/mẫu khác.

    rows: mỗi phần tử {"status", "true_word", "dtw_word", "dl_word", "dl_conf"}.
    Trả {"best": {"threshold", "accuracy"}, "table": [{"threshold", "accuracy"}, ...]}.
    """
    if grid is None:
        grid = [round(0.10 + 0.02 * i, 2) for i in range(41)]  # 0.10 .. 0.90
    table: List[Dict[str, float]] = []
    for thr in grid:
        correct = 0
        for r in rows:
            pred = gated_prediction(r["status"], r["dtw_word"], r["dl_word"], r["dl_conf"], thr)
            correct += int(pred == r["true_word"])
        table.append({"threshold": thr, "accuracy": correct / max(1, len(rows))})

    max_acc = max(t["accuracy"] for t in table)
    eps = 1e-9
    runs: List[List[int]] = []
    cur: List[int] = []
    for i, t in enumerate(table):
        if t["accuracy"] >= max_acc - eps:
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    longest = max(runs, key=len)
    mid_idx = longest[len(longest) // 2]
    best = {"threshold": table[mid_idx]["threshold"], "accuracy": table[mid_idx]["accuracy"]}
    return {"best": best, "table": table}
