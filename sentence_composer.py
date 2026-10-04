# -*- coding: utf-8 -*-
"""
BỘ SINH CÂU TỪ CHUỖI GLOSS (SPEC 14)
====================================
Sinh viên: Tô Phương Đạt | MSSV: 24110086 | Môn: Xử Lý Ảnh Số (DIP)

Sentence builder chỉ NỐI các từ nhận được (ví dụ "Me Hungry ."). Module này biến chuỗi khóa từ vựng
(ví dụ ["ME", "HUNGRY"]) thành MỘT CÂU tự nhiên song ngữ ("I am hungry." / "Tôi đói.") bằng luật OFFLINE,
tất định, không dùng LLM hay mạng.

Thứ tự áp dụng (dừng ở bước đầu tiên áp được):
  1. empty    : không có từ nào.
  2. table    : khớp TRỌN chuỗi với bảng cụm (_TABLE).
  3. grammar  : tách mệnh đề rồi áp luật G1-G6 cho từng mệnh đề; nếu BẤT KỲ mệnh đề nào không áp được -> bước 4.
  4. fallback : cách nối thô như cũ (viết hoa chữ đầu, nối bằng dấu cách, kết thúc dấu chấm).

Tính chất an toàn: KHÔNG BAO GIỜ làm mất / đổi thứ tự từ nội dung ở nhánh fallback, KHÔNG ném ngoại lệ,
và chỉ ảnh hưởng hiển thị/giọng nói (không đụng tới nhận dạng).

Thuần Python: chỉ dùng thư viện chuẩn. KHÔNG import OpenCV / MediaPipe / PyTorch / giao diện / giọng nói.

Token tên riêng do chế độ đánh vần tạo ra có dạng "SPELL:<CHỮ HOA>" (ví dụ "SPELL:ANNA").
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ComposeResult:
    text: str        # câu cuối cùng (đã có dấu câu)
    rule: str        # "empty" | "table" | "grammar" | "fallback"
    changed: bool    # True nếu text KHÁC cách nối thô (fallback_join) của cùng đầu vào


# ----------------------------------------------------------------------------------------------
# Tên hiển thị nội bộ (dùng khi người gọi không truyền phrase_fn) — khớp ASL_VOCAB_MAP của 13 từ.
# ----------------------------------------------------------------------------------------------
_NAMES: Dict[str, Dict[str, str]] = {
    "en": {
        "HELLO": "Hello", "BYE": "Bye", "ME": "Me", "YOU": "You", "THANK_YOU": "Thank You",
        "EAT": "Eat", "DRINK": "Drink", "WATER": "Water", "SORRY": "Sorry", "PLEASE": "Please",
        "YES": "Yes", "BAD": "Bad", "HUNGRY": "Hungry",
    },
    "vi": {
        "HELLO": "Xin chào", "BYE": "Tạm biệt", "ME": "Tôi", "YOU": "Bạn", "THANK_YOU": "Cảm ơn",
        "EAT": "Ăn", "DRINK": "Uống", "WATER": "Nước", "SORRY": "Xin lỗi", "PLEASE": "Làm ơn",
        "YES": "Vâng", "BAD": "Tệ", "HUNGRY": "Đói",
    },
}

# Phân loại token (SPEC 14, mục 1.1)
_PRON = {"ME", "YOU"}
_VERB = {"EAT", "DRINK"}
_NOUN = {"WATER"}
_ADJ = {"HUNGRY", "BAD"}
_INTJ = {"HELLO", "BYE", "THANK_YOU", "SORRY", "YES"}
_POLITE = "PLEASE"

_SPELL_RE = re.compile(r"^SPELL:([A-Z]+)$")

# Từ dùng trong luật ngữ pháp: (EN, VI), chữ thường.
_PRON_TEXT = {"ME": ("I", "Tôi"), "YOU": ("You", "Bạn")}
_VERB_TEXT = {"EAT": ("eat", "ăn"), "DRINK": ("drink", "uống")}
_NOUN_TEXT = {"WATER": ("water", "nước")}
_ADJ_TEXT = {"HUNGRY": ("hungry", "đói"), "BAD": ("bad", "tệ")}

# Thán từ / lời đáp đứng riêng (EN, VI)
_INTJ_TEXT = {
    "HELLO": ("Hello.", "Xin chào."),
    "BYE": ("Goodbye.", "Tạm biệt."),
    "THANK_YOU": ("Thank you.", "Cảm ơn."),
    "SORRY": ("I'm sorry.", "Tôi xin lỗi."),
    "YES": ("Yes.", "Vâng."),
}

# Bảng cụm trọn chuỗi (SPEC 14, mục 1.3): khóa là tuple -> (EN, VI)
_TABLE: Dict[Tuple[str, ...], Tuple[str, str]] = {
    ("HELLO",): ("Hello.", "Xin chào."),
    ("BYE",): ("Goodbye.", "Tạm biệt."),
    ("THANK_YOU",): ("Thank you.", "Cảm ơn."),
    ("SORRY",): ("I'm sorry.", "Tôi xin lỗi."),
    ("YES",): ("Yes.", "Vâng."),
    ("PLEASE",): ("Please.", "Làm ơn."),
    ("ME", "HUNGRY"): ("I am hungry.", "Tôi đói."),
    ("YOU", "HUNGRY"): ("You are hungry.", "Bạn đói."),
    ("ME", "BAD"): ("I feel bad.", "Tôi cảm thấy tệ."),
    ("YOU", "BAD"): ("You are bad.", "Bạn tệ."),
    ("ME", "EAT"): ("I eat.", "Tôi ăn."),
    ("ME", "DRINK"): ("I drink.", "Tôi uống."),
    ("ME", "DRINK", "WATER"): ("I drink water.", "Tôi uống nước."),
    ("DRINK", "WATER"): ("Drink water.", "Uống nước."),
    ("WATER", "PLEASE"): ("Water, please.", "Làm ơn cho tôi nước."),
    ("DRINK", "WATER", "PLEASE"): ("Could I have some water, please?", "Làm ơn cho tôi xin nước uống."),
    ("PLEASE", "WATER"): ("Water, please.", "Làm ơn cho tôi nước."),
    ("THANK_YOU", "BYE"): ("Thank you. Goodbye.", "Cảm ơn. Tạm biệt."),
    ("HELLO", "ME", "HUNGRY"): ("Hello. I am hungry.", "Xin chào. Tôi đói."),
    ("SORRY", "BYE"): ("I'm sorry. Goodbye.", "Tôi xin lỗi. Tạm biệt."),
    ("HELLO", "BYE"): ("Hello. Goodbye.", "Xin chào. Tạm biệt."),
    ("ME", "SORRY"): ("I'm sorry.", "Tôi xin lỗi."),
}


# ----------------------------------------------------------------------------------------------
# Tiện ích
# ----------------------------------------------------------------------------------------------
def _lang_index(lang: str) -> int:
    """0 = tiếng Anh (mặc định cho mọi ngôn ngữ lạ), 1 = tiếng Việt."""
    return 1 if lang == "vi" else 0


def _normalize(keys) -> List[str]:
    """Chuẩn hóa đầu vào: bỏ phần tử không phải chuỗi / rỗng, cắt khoảng trắng, viết HOA. Không bao giờ ném lỗi."""
    try:
        items = list(keys) if keys is not None else []
    except TypeError:
        return []
    out: List[str] = []
    for k in items:
        if isinstance(k, str):
            k = k.strip().upper()
            if k:
                out.append(k)
    return out


def _spell_name(token: str) -> Optional[str]:
    """'SPELL:ANNA' -> 'Anna'; token khác -> None."""
    m = _SPELL_RE.match(token)
    return m.group(1).capitalize() if m else None


def _display(key: str, lang: str, phrase_fn: Optional[Callable[[str, str], str]]) -> str:
    """Tên hiển thị của một khóa cho nhánh fallback."""
    name = _spell_name(key)
    if name is not None:
        return name
    if phrase_fn is not None:
        try:
            p = phrase_fn(key, "vi" if lang == "vi" else "en")
        except Exception:
            p = ""
        if isinstance(p, str) and p.strip():
            return p.strip()
    internal = _NAMES["vi" if lang == "vi" else "en"].get(key)
    if internal:
        return internal
    return key.replace("_", " ").title()


def _end_sentence(text: str) -> str:
    text = text.strip()
    if text and text[-1] not in ".?!":
        text += "."
    return text


def fallback_join(keys: Sequence[str], lang: str = "en",
                  phrase_fn: Optional[Callable[[str, str], str]] = None) -> str:
    """Cách nối thô như cũ: tên hiển thị các từ nối bằng dấu cách, viết hoa chữ đầu, kết thúc dấu chấm."""
    tokens = _normalize(keys)
    if not tokens:
        return ""
    words = " ".join(_display(t, lang, phrase_fn) for t in tokens)
    words = words[0].upper() + words[1:]
    return _end_sentence(words)


# ----------------------------------------------------------------------------------------------
# Luật ngữ pháp cho một mệnh đề
# ----------------------------------------------------------------------------------------------
def _realize_core(body: List[str], lang: str) -> Optional[str]:
    """Luật G2-G6 cho một mệnh đề không có PLEASE. None nếu không áp được."""
    i = _lang_index(lang)
    n = len(body)

    if n == 1:
        t = body[0]
        name = _spell_name(t)
        if name is not None:
            return f"{name}."                                    # G5: tên riêng đứng một mình
        if t in _NOUN:
            return f"{_NOUN_TEXT[t][i].capitalize()}."           # G6
        if t in _VERB:
            return f"{_VERB_TEXT[t][i].capitalize()}."           # G4 (câu mệnh lệnh không tân ngữ)
        return None

    if n == 2:
        a, b = body
        if a in _PRON and b in _ADJ:                              # G2
            if a == "ME" and b == "BAD":
                return ("I feel bad.", "Tôi cảm thấy tệ.")[i]
            if i == 0:
                be = "am" if a == "ME" else "are"
                return f"{_PRON_TEXT[a][0]} {be} {_ADJ_TEXT[b][0]}."
            return f"{_PRON_TEXT[a][1]} {_ADJ_TEXT[b][1]}."
        if a in _PRON and b in _VERB:                             # G3
            return f"{_PRON_TEXT[a][i]} {_VERB_TEXT[b][i]}."
        if a == "ME" and _spell_name(b) is not None:              # G5: ME + tên
            return (f"I am {_spell_name(b)}.", f"Tôi là {_spell_name(b)}.")[i]
        if a in _VERB and b in _NOUN:                             # G4: động từ + danh từ
            return f"{_VERB_TEXT[a][i].capitalize()} {_NOUN_TEXT[b][i]}."
        return None

    if n == 3:
        a, b, c = body
        if a in _PRON and b in _VERB and c in _NOUN:              # G3 có tân ngữ
            return f"{_PRON_TEXT[a][i]} {_VERB_TEXT[b][i]} {_NOUN_TEXT[c][i]}."
        return None

    return None


def _realize_clause(tokens: List[str], lang: str) -> Optional[str]:
    """Sinh câu cho một mệnh đề (đã tách theo thán từ). None nếu không có luật phù hợp."""
    i = _lang_index(lang)
    if not tokens:
        return None

    if len(tokens) == 1 and tokens[0] in _INTJ:
        return _INTJ_TEXT[tokens[0]][i]

    if tokens == ["ME", "SORRY"]:                                  # "ME SORRY" là cách ký "I'm sorry"
        return _INTJ_TEXT["SORRY"][i]

    if len(tokens) == 2 and tokens[0] == "HELLO" and _spell_name(tokens[1]) is not None:   # G5
        name = _spell_name(tokens[1])
        return (f"Hello, {name}.", f"Xin chào, {name}.")[i]

    if tokens == [_POLITE]:
        return ("Please.", "Làm ơn.")[i]

    polite = False
    body = list(tokens)
    if len(body) > 1 and body[0] == _POLITE:                      # G1: PLEASE ở đầu
        body = body[1:]
        polite = True
    elif len(body) > 1 and body[-1] == _POLITE:                   # G1: PLEASE ở cuối
        body = body[:-1]
        polite = True

    base = _realize_core(body, lang)
    if base is None:
        return None
    if not polite:
        return base
    if i == 0:
        return base[:-1] + ", please."
    return "Làm ơn " + base[0].lower() + base[1:]


def _split_clauses(tokens: List[str]) -> List[List[str]]:
    """Mỗi thán từ đứng riêng một mệnh đề (trừ HELLO + tên); các token còn lại liền nhau tạo một mệnh đề."""
    clauses: List[List[str]] = []
    cur: List[str] = []
    idx = 0
    while idx < len(tokens):
        t = tokens[idx]
        if t in _INTJ:
            if cur:
                clauses.append(cur)
                cur = []
            if t == "HELLO" and idx + 1 < len(tokens) and _spell_name(tokens[idx + 1]) is not None:
                clauses.append([t, tokens[idx + 1]])
                idx += 2
                continue
            clauses.append([t])
        elif t == "ME" and idx + 1 < len(tokens) and tokens[idx + 1] == "SORRY":
            # "ME SORRY" gộp thành một mệnh đề (ngoại lệ của quy tắc "thán từ đứng riêng", như HELLO + tên).
            if cur:
                clauses.append(cur)
                cur = []
            clauses.append([t, tokens[idx + 1]])
            idx += 2
            continue
        else:
            cur.append(t)
        idx += 1
    if cur:
        clauses.append(cur)
    return clauses


# ----------------------------------------------------------------------------------------------
# Giao diện chính
# ----------------------------------------------------------------------------------------------
def compose(keys: Sequence[str], lang: str = "en",
            phrase_fn: Optional[Callable[[str, str], str]] = None) -> ComposeResult:
    """
    Sinh câu từ chuỗi khóa từ vựng. `phrase_fn(key, lang)` (tùy chọn) tra tên hiển thị bên ngoài cho nhánh fallback.
    Không bao giờ ném ngoại lệ; mọi trường hợp không áp được luật đều rơi về fallback.
    """
    tokens = _normalize(keys)
    if not tokens:
        return ComposeResult("", "empty", False)

    plain = fallback_join(tokens, lang, phrase_fn)
    i = _lang_index(lang)

    hit = _TABLE.get(tuple(tokens))
    if hit is not None:
        text = hit[i]
        return ComposeResult(text, "table", text != plain)

    sentences: List[str] = []
    for clause in _split_clauses(tokens):
        s = _realize_clause(clause, lang)
        if s is None:
            return ComposeResult(plain, "fallback", False)
        sentences.append(s)
    text = " ".join(sentences)
    return ComposeResult(text, "grammar", text != plain)
