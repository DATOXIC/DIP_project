# -*- coding: utf-8 -*-
"""Smoke test đầu-cuối: dữ liệu tổng hợp (tay xoay 3D bất kỳ) -> train -> nạp model -> dự đoán."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hand_pose_features import _rotation_matrix, view_angle_deg  # noqa: E402
from letter_landmark_classifier import LetterLandmarkClassifier  # noqa: E402
from test_hand_pose_features import make_hand  # noqa: E402
from train_letter_landmarks import run  # noqa: E402

LETTERS = {"a": 0.95, "b": 0.0, "c": 0.5}


def _write_session(root, person, letter, k, rng):
    n = 40
    world, ang = [], []
    for _ in range(n):
        p = make_hand(curl=LETTERS[letter] + rng.normal(0, 0.03), seed=int(rng.integers(1e6)))
        rot = _rotation_matrix(*rng.uniform(-2.5, 2.5, 3))
        q = (p @ rot.T) * 0.09
        world.append(q); ang.append(view_angle_deg(q))
    os.makedirs(os.path.join(root, person), exist_ok=True)
    np.savez_compressed(os.path.join(root, person, f"{person}_{letter}_{k}.npz"),
                        world=np.stack(world).astype(np.float32), handed=np.array(["Right"] * n),
                        view_angle=np.array(ang, dtype=np.float32), letter=np.array(letter), person=np.array(person),
                        session_id=np.array(f"{person}_{letter}_{k}"))


def test_train_and_predict(tmp_path):
    rng = np.random.default_rng(0)
    root = str(tmp_path / "data")
    for person in ("P1", "P2"):
        for letter in LETTERS:
            for k in range(3):
                _write_session(root, person, letter, k, rng)
    out = str(tmp_path / "m.pt")
    res = run(root, out, holdout_person="P2", epochs=15, copies=2, no_aug=False, fit_all=False,
              results_path=str(tmp_path / "r.json"))
    assert res["test_real"]["acc"] > 0.9
    clf = LetterLandmarkClassifier(out, min_conf=0.3)
    p = make_hand(curl=0.95, seed=5) @ _rotation_matrix(1.0, -0.7, 2.0).T * 0.09
    label, conf, _ = clf.predict(p, "Right")
    assert label == "a" and conf > 0.5


def test_external_data_never_in_test(tmp_path):
    rng = np.random.default_rng(1)
    root = str(tmp_path / "data")
    for letter in LETTERS:
        for k in range(3):
            _write_session(root, "P1", letter, k, rng)
            _write_session(root, "EXT_kaggle", letter, k, rng)
    from train_letter_landmarks import is_ext, load_dataset, split_indices
    data = load_dataset(root)
    tr, te = split_indices(data, None, 0.25, 0)
    assert len(te) > 0 and not is_ext(data)[te].any()
    assert is_ext(data)[tr].any()
