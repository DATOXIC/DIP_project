# -*- coding: utf-8 -*-
"""
Test cho sentence_composer.py (SPEC 14): chuỗi gloss -> câu hoàn chỉnh EN/VI bằng luật offline.
Thuần Python, KHÔNG cần webcam / mediapipe / model.
"""
import json
import os
import re
import subprocess
import sys

import pytest

from sentence_composer import ComposeResult, compose, fallback_join

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Bảng cụm trọn chuỗi BẮT BUỘC (SPEC 14, mục 1.3): (khóa..., EN, VI)
TABLE = [
    ("HELLO", "Hello.", "Xin chào."),
    ("BYE", "Goodbye.", "Tạm biệt."),
    ("THANK_YOU", "Thank you.", "Cảm ơn."),
    ("SORRY", "I'm sorry.", "Tôi xin lỗi."),
    ("YES", "Yes.", "Vâng."),
    ("PLEASE", "Please.", "Làm ơn."),
    ("ME HUNGRY", "I am hungry.", "Tôi đói."),
    ("YOU HUNGRY", "You are hungry.", "Bạn đói."),
    ("ME BAD", "I feel bad.", "Tôi cảm thấy tệ."),
    ("YOU BAD", "You are bad.", "Bạn tệ."),
    ("ME EAT", "I eat.", "Tôi ăn."),
    ("ME DRINK", "I drink.", "Tôi uống."),
    ("ME DRINK WATER", "I drink water.", "Tôi uống nước."),
    ("DRINK WATER", "Drink water.", "Uống nước."),
    ("WATER PLEASE", "Water, please.", "Làm ơn cho tôi nước."),
    ("DRINK WATER PLEASE", "Could I have some water, please?", "Làm ơn cho tôi xin nước uống."),
    ("PLEASE WATER", "Water, please.", "Làm ơn cho tôi nước."),
    ("THANK_YOU BYE", "Thank you. Goodbye.", "Cảm ơn. Tạm biệt."),
    ("HELLO ME HUNGRY", "Hello. I am hungry.", "Xin chào. Tôi đói."),
    ("SORRY BYE", "I'm sorry. Goodbye.", "Tôi xin lỗi. Tạm biệt."),
    ("HELLO BYE", "Hello. Goodbye.", "Xin chào. Tạm biệt."),
    ("ME SORRY", "I'm sorry.", "Tôi xin lỗi."),
]


def _keys(s):
    return s.split()


# ------------------------------------------------------------------ bảng ----

@pytest.mark.parametrize("seq,en,vi", TABLE)
def test_bang_cum_tieng_anh(seq, en, vi):
    r = compose(_keys(seq), "en")
    assert r.text == en
    assert r.rule in ("table", "grammar")        # cả hai đều chấp nhận được nếu ra đúng câu


@pytest.mark.parametrize("seq,en,vi", TABLE)
def test_bang_cum_tieng_viet(seq, en, vi):
    assert compose(_keys(seq), "vi").text == vi


def test_cum_trong_bang_duoc_danh_dau_rule_table():
    assert compose(["DRINK", "WATER", "PLEASE"], "en").rule == "table"


def test_khoa_khong_phan_biet_hoa_thuong():
    assert compose(["me", "Hungry"], "en").text == "I am hungry."


def test_ngon_ngu_la_coi_la_tieng_anh():
    assert compose(["ME", "HUNGRY"], "fr").text == "I am hungry."


# --------------------------------------------------------------- ngữ pháp ----

def test_gioi_tu_lich_su_cuoi_cau():
    assert compose(["ME", "EAT", "PLEASE"], "en").text == "I eat, please."
    assert compose(["EAT", "PLEASE"], "en").text == "Eat, please."


def test_lich_su_dau_cau_cung_duoc():
    assert compose(["PLEASE", "EAT"], "en").text == "Eat, please."


def test_lich_su_tieng_viet():
    assert compose(["EAT", "PLEASE"], "vi").text == "Làm ơn ăn."


def test_dai_tu_dong_tu_tan_ngu():
    assert compose(["YOU", "DRINK", "WATER"], "en").text == "You drink water."
    assert compose(["YOU", "DRINK", "WATER"], "vi").text == "Bạn uống nước."


def test_menh_lenh_khong_dai_tu():
    assert compose(["EAT"], "en").text == "Eat."
    assert compose(["EAT"], "vi").text == "Ăn."


def test_danh_tu_mot_minh():
    assert compose(["WATER"], "en").text == "Water."
    assert compose(["WATER"], "vi").text == "Nước."


def test_tach_menh_de_voi_than_tu_giua_chuoi():
    r = compose(["YES", "ME", "HUNGRY"], "en")
    assert r.text == "Yes. I am hungry."
    assert r.rule == "grammar"
    assert compose(["HELLO", "YOU", "HUNGRY"], "en").text == "Hello. You are hungry."


def test_nhieu_than_tu_lien_nhau():
    assert compose(["YES", "THANK_YOU", "BYE"], "en").text == "Yes. Thank you. Goodbye."


# ------------------------------------------------------------ tên riêng ----

def test_ten_rieng_spell():
    assert compose(["ME", "SPELL:ANNA"], "en").text == "I am Anna."
    assert compose(["ME", "SPELL:ANNA"], "vi").text == "Tôi là Anna."
    assert compose(["HELLO", "SPELL:ANNA"], "en").text == "Hello, Anna."
    assert compose(["HELLO", "SPELL:ANNA"], "vi").text == "Xin chào, Anna."
    assert compose(["SPELL:ANNA"], "en").text == "Anna."


def test_dong_tu_voi_ten_roi_ve_fallback():
    r = compose(["EAT", "SPELL:ANNA"], "en")
    assert r.rule == "fallback"
    assert "Anna" in r.text and "Eat" in r.text


# ------------------------------------------------------------- an toàn ----

def test_rong():
    r = compose([], "en")
    assert (r.text, r.rule, r.changed) == ("", "empty", False)


def test_khoa_la_roi_ve_fallback_khong_nem_loi():
    r = compose(["ME", "XYZ"], "en")
    assert r.rule == "fallback"
    assert r.text.startswith("Me")
    assert r.text.endswith(".")


def test_khong_nem_loi_voi_dau_vao_xau():
    for bad in (None, [None], [""], ["  "], [123], ["ME", None, "EAT"]):
        r = compose(bad, "en")
        assert isinstance(r, ComposeResult)


def test_fallback_giu_du_va_dung_thu_tu_tu_noi_dung():
    keys = ["BAD", "WATER", "SORRY"]            # vô nghĩa -> không có luật
    r = compose(keys, "en")
    assert r.rule == "fallback"
    assert r.text == fallback_join(keys, "en")
    words = re.findall(r"[A-Za-z']+", r.text)
    assert words == ["Bad", "Water", "Sorry"]


def test_fallback_join_chuan_hoa_hoa_thuong_va_dau_cham():
    assert fallback_join(["ME", "HUNGRY"], "en") == "Me Hungry."
    assert fallback_join(["THANK_YOU"], "en") == "Thank You."
    assert fallback_join([], "en") == ""


def test_fallback_tieng_viet_giu_dau():
    assert fallback_join(["ME", "HUNGRY"], "vi") == "Tôi Đói."


def test_changed_dung_nghia():
    assert compose(["ME", "HUNGRY"], "en").changed is True
    r = compose(["BAD", "WATER", "SORRY"], "en")
    assert r.rule == "fallback" and r.changed is False


def test_tat_dinh():
    for seq, _, _ in TABLE:
        a = compose(_keys(seq), "en")
        b = compose(_keys(seq), "en")
        assert a == b


def test_phrase_fn_ben_ngoai_duoc_dung_cho_fallback():
    names = {"ME": "I", "XYZ": "Zed"}
    r = compose(["ME", "XYZ"], "en", phrase_fn=lambda k, lg: names.get(k, ""))
    assert r.text == "I Zed."


# ------------------------------------------------ phủ cụm trong kế hoạch ----

def _plan_phrases():
    with open(os.path.join(ROOT, "phrases_plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    out = []
    for sess in plan["sessions"].values():
        for clip in sess["clips"]:
            if clip.get("kind") == "phrase":
                out.append(list(clip["truth"]))
    return out


def test_moi_cum_trong_ke_hoach_ra_cau_hop_le_ca_hai_ngon_ngu():
    phrases = _plan_phrases()
    assert len(phrases) >= 50
    for truth in phrases:
        for lang in ("en", "vi"):
            r = compose(truth, lang)
            t = r.text
            assert t, (truth, lang)
            assert t[0].isupper(), (truth, lang, t)
            assert t[-1] in ".?!", (truth, lang, t)
            assert "_" not in t and "SPELL:" not in t, (truth, lang, t)
            assert not re.search(r"\b[A-Z]{3,}\b", t), (truth, lang, t)   # không lọt khóa thô viết hoa


# --------------------------------------------------------- phụ thuộc ----

def test_module_khong_keo_theo_cv2_mediapipe_torch():
    code = ("import sys, sentence_composer;"
            "bad=[m for m in ('cv2','mediapipe','torch','predict_sentence_webcam','ui_render','speech_service') if m in sys.modules];"
            "sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_me_sorry_trong_chuoi_dai_hon():
    # "ME SORRY" là cách ký "I'm sorry" nên cũng phải thành câu khi nằm giữa chuỗi dài hơn.
    r = compose(["HELLO", "ME", "SORRY"], "en")
    assert r.text == "Hello. I'm sorry."
    assert r.rule == "grammar"
    assert compose(["HELLO", "ME", "SORRY"], "vi").text == "Xin chào. Tôi xin lỗi."
    assert compose(["ME", "SORRY", "BYE"], "en").text == "I'm sorry. Goodbye."


def test_me_sorry_khong_lam_hong_cac_ca_khac():
    assert compose(["SORRY"], "en").text == "I'm sorry."
    assert compose(["YOU", "SORRY"], "en").rule == "fallback"     # chưa có luật cho YOU SORRY
    assert compose(["ME", "HUNGRY"], "en").text == "I am hungry."
