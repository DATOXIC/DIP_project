# -*- coding: utf-8 -*-
"""
Dựng bảng 'từ -> nhóm dáng tay' cho handshape_gate từ bản ghi thô của CHỦ DỰ ÁN (raw_strokes/).
Chỉ dùng dữ liệu chủ dự án (không dùng người lạ) để phép đo trên người lạ không bị rò rỉ.

    py build_handshape_table.py [--model letter_landmark_model_image.pt] [--out handshape_gate_table.json]
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # chạy từ thư mục con: thêm gốc dự án để import module dùng chung
import argparse

from eval_handshape_gate import owner_evidence
from handshape_gate import learn_word_table, save_word_table
from letter_landmark_classifier import LetterLandmarkClassifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="letter_landmark_model_image.pt")
    ap.add_argument("--out", default="handshape_gate_table.json")
    ap.add_argument("--reliable-top", type=float, default=0.6)
    a = ap.parse_args()
    clf = LetterLandmarkClassifier(a.model, min_conf=0.0)
    table = learn_word_table(owner_evidence(clf), reliable_top=a.reliable_top)
    save_word_table(table, a.out)
    for w, v in sorted(table.items()):
        print(f"{w:<10} {'RÀNG BUỘC' if v.reliable else 'tự do    '} {sorted(v.allowed)}")
    print("Đã lưu", a.out)


if __name__ == "__main__":
    main()
