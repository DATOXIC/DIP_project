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
# Tên hiển thị nội bộ (dùng khi người gọi không truyền phrase_fn) — khớp ASL_VOCAB_MAP (13 từ gốc + từ quy ước 1 tay).
# ----------------------------------------------------------------------------------------------
_NAMES: Dict[str, Dict[str, str]] = {
    "en": {
        "HELLO": "Hello", "BYE": "Bye", "ME": "Me", "YOU": "You", "THANK_YOU": "Thank You",
        "EAT": "Eat", "DRINK": "Drink", "WATER": "Water", "SORRY": "Sorry", "PLEASE": "Please",
        "YES": "Yes", "BAD": "Bad", "HUNGRY": "Hungry",
        "NO": "No", "LIKE": "Like", "WANT": "Want", "MY": "My", "NAME": "Name", "NICE": "Nice", "MEET": "Meet",
    },
    "vi": {
        "HELLO": "Xin chào", "BYE": "Tạm biệt", "ME": "Tôi", "YOU": "Bạn", "THANK_YOU": "Cảm ơn",
        "EAT": "Ăn", "DRINK": "Uống", "WATER": "Nước", "SORRY": "Xin lỗi", "PLEASE": "Làm ơn",
        "YES": "Vâng", "BAD": "Tệ", "HUNGRY": "Đói",
        "NO": "Không", "LIKE": "Thích", "WANT": "Muốn", "MY": "Của tôi", "NAME": "Tên", "NICE": "Vui", "MEET": "Gặp",
    },
}

# Phân loại token (SPEC 14, mục 1.1)
_PRON = {"ME", "YOU"}
_VERB = {"EAT", "DRINK"}
_NOUN = {"WATER"}
_ADJ = {"HUNGRY", "BAD"}
_INTJ = {"HELLO", "BYE", "THANK_YOU", "SORRY", "YES", "NO"}
_TRANS = {"LIKE", "WANT"}   # động từ cần tân ngữ: ME LIKE YOU, ME WANT WATER, ME WANT EAT
_POLITE = "PLEASE"

# Thán từ có thể đi kèm người được nhắc tới ngay sau nó ("HELLO YOU" = chào bạn, "SORRY YOU" = xin lỗi bạn): (EN, VI) theo đối tượng.
_ADDRESSED = {
    ("HELLO", "YOU"): ("Hello, you.", "Xin chào bạn."),
    ("HELLO", "ME"): ("Hello, it's me.", "Xin chào, tôi đây."),
    ("BYE", "YOU"): ("Goodbye, you.", "Tạm biệt bạn."),
    ("BYE", "ME"): ("Goodbye, from me.", "Tôi chào tạm biệt."),
    ("THANK_YOU", "YOU"): ("Thank you.", "Cảm ơn bạn."),
    ("THANK_YOU", "ME"): ("Thank you, from me.", "Tôi cảm ơn."),
    ("SORRY", "YOU"): ("I apologize to you.", "Xin lỗi bạn."),
    ("SORRY", "ME"): ("I'm sorry.", "Tôi xin lỗi."),
}

_SPELL_RE = re.compile(r"^SPELL:([A-Z]+)$")

# Từ dùng trong luật ngữ pháp: (EN, VI), chữ thường.
_PRON_TEXT = {"ME": ("I", "Tôi"), "YOU": ("You", "Bạn")}
_VERB_TEXT = {"EAT": ("eat", "ăn"), "DRINK": ("drink", "uống")}
_NOUN_TEXT = {"WATER": ("water", "nước")}
_ADJ_TEXT = {"HUNGRY": ("hungry", "đói"), "BAD": ("bad", "tệ")}
_TRANS_TEXT = {"LIKE": ("like", "thích"), "WANT": ("want", "muốn")}
_OBJ_PRON_TEXT = {"ME": ("me", "tôi"), "YOU": ("you", "bạn")}   # đại từ làm tân ngữ
_NEG_AUX = {"ME": ("I don't", "Tôi không"), "YOU": ("You don't", "Bạn không")}

# Thán từ / lời đáp đứng riêng (EN, VI)
_INTJ_TEXT = {
    "HELLO": ("Hello.", "Xin chào."),
    "BYE": ("Goodbye.", "Tạm biệt."),
    "THANK_YOU": ("Thank you.", "Cảm ơn."),
    "SORRY": ("I'm sorry.", "Tôi xin lỗi."),
    "YES": ("Yes.", "Vâng."),
    "NO": ("No.", "Không."),
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
    ("NO", "THANK_YOU"): ("No, thank you.", "Không, cảm ơn."),
    ("NO", "PLEASE"): ("No, please.", "Không, làm ơn."),
    ("YES", "THANK_YOU"): ("Yes, thank you.", "Vâng, cảm ơn."),
    ("NICE", "MEET", "YOU"): ("Nice to meet you.", "Rất vui được gặp bạn."),
    ("HELLO", "NICE", "MEET", "YOU"): ("Hello. Nice to meet you.", "Xin chào. Rất vui được gặp bạn."),
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
        if t in _ADJ:
            return f"{_ADJ_TEXT[t][i].capitalize()}."            # G7: tính từ đứng một mình ("Đói.")
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
        if a in ("MY", "ME") and b == "NAME" and _spell_name(c) is not None:   # G14: tên riêng
            return (f"My name is {_spell_name(c)}.", f"Tên tôi là {_spell_name(c)}.")[i]
        return _realize_transitive(body, lang)

    return _realize_transitive(body, lang)


def _realize_object(obj: List[str], lang: str) -> Optional[str]:
    """Tân ngữ của LIKE / WANT: đại từ (you), danh từ (water), động từ (to eat), động từ + danh từ (to drink water)."""
    i = _lang_index(lang)
    if len(obj) == 1:
        t = obj[0]
        if t in _OBJ_PRON_TEXT:
            return _OBJ_PRON_TEXT[t][i]
        if t in _NOUN:
            return _NOUN_TEXT[t][i]
        if t in _VERB:
            return ("to " + _VERB_TEXT[t][0], _VERB_TEXT[t][1])[i]
        return None
    if len(obj) == 2 and obj[0] in _VERB and obj[1] in _NOUN:
        return (f"to {_VERB_TEXT[obj[0]][0]} {_NOUN_TEXT[obj[1]][0]}", f"{_VERB_TEXT[obj[0]][1]} {_NOUN_TEXT[obj[1]][1]}")[i]
    return None


def _realize_transitive(body: List[str], lang: str) -> Optional[str]:
    """G15 / G16: ĐẠI TỪ + [NO] + LIKE|WANT + tân ngữ  ->  'I like you.' / "I don't want water." / 'Tôi muốn ăn.'"""
    i = _lang_index(lang)
    if len(body) < 3 or body[0] not in _PRON:
        return None
    subj, rest = body[0], body[1:]
    negated = rest[0] == "NO"
    if negated:
        rest = rest[1:]
    if len(rest) < 2 or rest[0] not in _TRANS:
        return None
    verb = _TRANS_TEXT[rest[0]][i]
    obj = _realize_object(rest[1:], lang)
    if obj is None:
        return None
    if negated:
        return f"{_NEG_AUX[subj][i]} {verb} {obj}."
    return f"{_PRON_TEXT[subj][i]} {verb} {obj}."


def _realize_clause(tokens: List[str], lang: str) -> Optional[str]:
    """Sinh câu cho một mệnh đề (đã tách theo thán từ). None nếu không có luật phù hợp."""
    i = _lang_index(lang)
    if not tokens:
        return None

    if len(tokens) == 1 and tokens[0] in _INTJ:
        return _INTJ_TEXT[tokens[0]][i]

    if tokens == ["NICE", "MEET", "YOU"]:                          # G17: "Rất vui được gặp bạn."
        return ("Nice to meet you.", "Rất vui được gặp bạn.")[i]

    if tokens == ["ME", "SORRY"]:                                  # "ME SORRY" là cách ký "I'm sorry"
        return _INTJ_TEXT["SORRY"][i]

    if tokens == ["YOU", "SORRY"]:                                 # G13: "YOU SORRY" = "Bạn xin lỗi."
        return ("You are sorry.", "Bạn xin lỗi.")[i]

    if len(tokens) == 2 and tokens[0] == "HELLO" and _spell_name(tokens[1]) is not None:   # G5
        name = _spell_name(tokens[1])
        return (f"Hello, {name}.", f"Xin chào, {name}.")[i]

    if len(tokens) == 2 and (tokens[0], tokens[1]) in _ADDRESSED:   # G8: thán từ + người được nhắc tới
        return _ADDRESSED[(tokens[0], tokens[1])][i]

    if tokens == ["YES", _POLITE]:                                 # G9: "Yes, please."
        return ("Yes, please.", "Vâng, làm ơn.")[i]

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

    if polite and body == ["WATER"]:                               # G10: "Water, please."
        return ("Water, please.", "Làm ơn cho tôi nước.")[i]

    base = _realize_core(body, lang)
    if base is None:
        return None
    if not polite:
        return base
    if i == 1 and tokens[0] == _POLITE and body[0] == "YOU" and len(body) >= 2 and body[1] in _VERB:
        return "Mời " + base[0].lower() + base[1:]                 # G11: "PLEASE YOU EAT" = "Mời bạn ăn."
    if i == 1 and tokens[0] != _POLITE and body[0] in _PRON:
        return base[:-1] + ", làm ơn."                             # G12: "ME HUNGRY PLEASE" = "Tôi đói, làm ơn."
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
        if tokens[idx:idx + 3] == ["NICE", "MEET", "YOU"]:
            if cur:
                clauses.append(cur)
                cur = []
            clauses.append(["NICE", "MEET", "YOU"])   # G17: lời chào gặp mặt đứng riêng một mệnh đề
            idx += 3
            continue
        if (t in ("MY", "ME") and idx + 2 < len(tokens) and tokens[idx + 1] == "NAME"
                and _spell_name(tokens[idx + 2]) is not None):
            if cur:
                clauses.append(cur)
                cur = []
            clauses.append(tokens[idx:idx + 3])       # G14: "MY NAME <tên>" đứng riêng một mệnh đề
            idx += 3
            continue
        if t == "NO" and cur and cur[-1] in _PRON and idx + 1 < len(tokens) and tokens[idx + 1] in _TRANS:
            cur.append(t)   # G16: "ME NO LIKE YOU" — NO là phủ định chứ không phải lời đáp "No."
            idx += 1
            continue
        if t in _INTJ:
            if cur:
                clauses.append(cur)
                cur = []
            if t == "HELLO" and idx + 1 < len(tokens) and _spell_name(tokens[idx + 1]) is not None:
                clauses.append([t, tokens[idx + 1]])
                idx += 2
                continue
            nxt = tokens[idx + 1] if idx + 1 < len(tokens) else None
            after = tokens[idx + 2] if idx + 2 < len(tokens) else None
            # G8: "HELLO YOU", "SORRY ME"... thành MỘT mệnh đề nếu đại từ KHÔNG mở đầu một vị ngữ ("HELLO YOU HUNGRY" vẫn tách).
            if (t, nxt) in _ADDRESSED and after not in _ADJ and after not in _VERB and after != "SORRY":
                clauses.append([t, nxt])
                idx += 2
                continue
            # G9: "YES PLEASE" thành một mệnh đề.
            if t == "YES" and nxt == _POLITE:
                clauses.append([t, nxt])
                idx += 2
                continue
            clauses.append([t])
        elif t in _PRON and idx + 1 < len(tokens) and tokens[idx + 1] == "SORRY":
            # "ME SORRY" / "YOU SORRY" gộp thành một mệnh đề (ngoại lệ của quy tắc "thán từ đứng riêng", như HELLO + tên).
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
