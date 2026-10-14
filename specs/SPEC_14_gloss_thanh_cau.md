# SPEC 14 — Từ chuỗi gloss sang câu hoàn chỉnh (song ngữ EN/VI)

> Đọc `specs/_PREAMBLE.md` trước. Dành cho agent. Ước lượng: 3–5 giờ.
> Không cần dữ liệu mới. **Chạy TRƯỚC SPEC 15** (cùng sửa `predict_sentence_webcam.py` và
> `ui_side_panel.py`). SPEC 13 đã bị bỏ (trùng `--confirm-enroll`).
>
> **Đối chiếu hiện trạng (2026-10-02, đã kiểm tra):** KHÔNG có sẵn bộ ghép câu nào. `predict_sentence_webcam.py`
> chỉ nối từ bằng `" ".join(...)` (docstring đầu file nói "ghép câu hoàn chỉnh" nhưng code chỉ nối). `web_app.py`
> trả `summary.sentence` cũng chỉ là danh sách tên từ (ngoài phạm vi spec này). Giao diện hiển thị mặc định là
> **`ui_side_panel.py`** (panel phải + panel dưới, vẽ bằng `cv2.putText`); `ui_render.DemoHUD` chỉ chạy ở chế độ
> `legacy_demo` mà code không còn cho chọn ⇒ **KHÔNG sửa `ui_render.py`**.

## Bối cảnh

Sentence builder hiện chỉ **nối từ**: `sentence_words` là danh sách cụm tiếng Anh, khi hạ tay
hơn 1.8 s thì câu được chốt thành `" ".join(sentence_words) + " ."` (ví dụ `Me Hungry .`).
Song song với nó luôn có `sentence_keys` (khóa từ vựng không phụ thuộc ngôn ngữ, ví dụ
`["ME", "HUNGRY"]`) — đây là đầu vào lý tưởng cho một bộ sinh câu.

**Quyết định của người dùng (đã chốt):** bộ sinh câu **dựa trên luật, chạy offline, KHÔNG dùng
LLM / mạng**. Chuỗi gloss gốc **luôn được giữ và hiển thị**, câu tự nhiên là dòng **thêm**.

Từ vựng production hiện có 13 khóa (`webcam_templates.json`):
`HELLO, BYE, ME, YOU, THANK_YOU, EAT, DRINK, WATER, SORRY, PLEASE, YES, BAD, HUNGRY`.
Bộ sinh câu phải xử lý thêm token đánh vần `SPELL:<CHỮ HOA>` (ví dụ `SPELL:ANNA`) do SPEC 15
tạo ra — nhưng phải **hoạt động đầy đủ khi SPEC 15 chưa tồn tại** (không có token đó).

> **Cập nhật 2026-10-14 (đã cài trong `sentence_composer.py`, 22 test mới ở `tests/test_sentence_composer.py`):**
> - Từ vựng mở rộng thêm `NO, LIKE, WANT, MY, NAME, NICE, MEET` (5 từ đầu có mẫu trong `profiles/Phi.json`; `NICE`/`MEET` chưa có mẫu).
>   Bảng tên hiển thị nội bộ `_NAMES` nay có 20 khóa (en/vi). Các phần "13 từ" bên dưới là mô tả bản gốc.
> - Luật mới: **G14** `MY|ME NAME SPELL:X` → "My name is X." / "Tên tôi là X." (đứng riêng một mệnh đề);
>   **G15** `ĐẠI TỪ LIKE|WANT <tân ngữ>` (tân ngữ: đại từ, danh từ, động từ, động từ + danh từ) → "I like you.", "I want water.",
>   "I want to drink water."; **G16** `ĐẠI TỪ NO LIKE|WANT ...` → phủ định "I don't like you.";
>   **G17** `NICE MEET YOU` → "Nice to meet you." / "Rất vui được gặp bạn." (đứng riêng một mệnh đề).
> - `NO` đứng riêng là lời đáp "No." (thán từ); đứng giữa đại từ và `LIKE`/`WANT` là phủ định; `NO THANK_YOU` → "No, thank you." (bảng cụm).
> - Chưa có luật cho `FROM`/`VIETNAM` (chưa có từ `FROM`), nên "I am from VietNam" chưa ra được.
> - Cách đọc gloss thiếu tân ngữ (`ME LIKE`, `MY NAME`, `WANT WATER`, `ME NO`) vẫn là `fallback`, không mất từ nào.

**Tính chất an toàn cần giữ:** bộ sinh câu **không bao giờ làm mất, đổi thứ tự, hay thêm từ
nội dung** khỏi gloss, và **không bao giờ ném ngoại lệ**. Mọi trường hợp không áp được luật ⇒
quay về cách nối cũ (`fallback`). Bộ sinh câu **không ảnh hưởng nhận dạng** (chỉ hiển thị/giọng nói).

## Định vị code (dùng chuỗi, không dùng số dòng)

| Việc | Chuỗi grep |
|---|---|
| Chốt câu khi hạ tay | `ĐÃ CHỐT CÂU HOÀN CHỈNH` (trong `predict_sentence_webcam.py`; khối `if len(sentence_words) > 0 and (curr_time - last_hand_seen_time) > 1.8 and not sentence_finalized:`) |
| Danh sách từ/khóa | `sentence_words: List[str] = []` và `sentence_keys: List[str] = []` |
| Hiển thị câu (panel dưới) | `def draw_bottom_panel(` và `sentence_display = f"CAU: ` trong `ui_side_panel.py`; dữ liệu qua `class SidePanelState:` (trường `sentence_words`, `sentence_finalized`) |
| Nơi dựng trạng thái panel | `panel_state = ui_side_panel.SidePanelState(` trong `predict_sentence_webcam.py` |
| Nhật ký sự kiện (panel phải) | `event_log.push(f"Chốt câu: ` trong `predict_sentence_webcam.py` |
| Bảng từ vựng | `ASL_VOCAB_MAP: Dict[str, Dict[str, str]] = {` và `def get_asl_phrase(` |
| Phát âm | `speech_service.speak_word(` và `def speak_word(key: str, lang: str, text: str)` trong `speech_service.py` |
| Đổi ngôn ngữ lúc chạy | `elif key in [ord('l'), ord('L')]:` |
| Xóa câu / Backspace | `elif key in [ord('c'), ord('C')]:` và `if is_swipe:` |
| Bỏ dấu tiếng Việt để vẽ bằng cv2 | `def strip_diacritics(` trong `ui_render.py` (chỉ IMPORT hàm này; không sửa file) |
| Cấu hình | `"camera_health": {` (thêm khối mới ngay trước/sau, trong `CONFIG`) |

Nếu một chuỗi không khớp, **DỪNG VÀ BÁO CÁO**.

---

## NHIỆM VỤ 1 — Module mới `sentence_composer.py` (thuần Python)

File **mới** ở thư mục gốc. Chỉ dùng thư viện chuẩn. **Không** import `cv2`, `mediapipe`,
`torch`, `predict_sentence_webcam`, `ui_render`, `speech_service`. Chú thích/chuỗi tiếng Việt.

```python
@dataclass(frozen=True)
class ComposeResult:
    text: str        # câu cuối cùng (đã có dấu câu)
    rule: str        # "empty" | "table" | "grammar" | "fallback"
    changed: bool    # True nếu text KHÁC cách nối thô (fallback_join) của cùng đầu vào

def compose(keys: Sequence[str], lang: str = "en",
            phrase_fn: Optional[Callable[[str, str], str]] = None) -> ComposeResult: ...

def fallback_join(keys: Sequence[str], lang: str = "en",
                  phrase_fn: Optional[Callable[[str, str], str]] = None) -> str: ...
```

- `phrase_fn(key, lang)` cho phép tra tên hiển thị từ `ASL_VOCAB_MAP` bên ngoài; nếu `None`
  thì dùng **bảng nhỏ nội bộ** (13 từ gốc ở bản đầu; nay 20 khóa, xem cập nhật ở mục Từ vựng) — để module chạy độc lập và test được.
- Ngôn ngữ không phải `"vi"` ⇒ coi là `"en"`.

### 1.1 Phân loại token

| Loại | Khóa |
|---|---|
| Đại từ (PRON) | `ME`, `YOU` |
| Động từ (VERB) | `EAT`, `DRINK` |
| Danh từ (NOUN) | `WATER` |
| Tính từ (ADJ) | `HUNGRY`, `BAD` |
| Thán từ/đáp (INTJ) | `HELLO`, `BYE`, `THANK_YOU`, `SORRY`, `YES` |
| Bổ ngữ lịch sự (POLITE) | `PLEASE` |
| Tên riêng (NAME) | token dạng `SPELL:<A-Z+>` |
| Khóa lạ | không thuộc loại nào ⇒ xử lý như `fallback` cho cả chuỗi |

### 1.2 Thứ tự áp dụng (dừng ở bước đầu tiên áp được)

1. **empty:** `keys` rỗng ⇒ `ComposeResult("", "empty", False)`.
2. **table (khớp trọn chuỗi, không phân biệt hoa/thường của khóa):** tra bảng dưới.
3. **grammar:** các luật G1–G6 dưới đây, áp **trên toàn chuỗi** (xem 1.3).
4. **fallback:** `fallback_join` — viết hoa chữ đầu, nối bằng dấu cách, kết thúc `"."`
   (đây chính là hành vi người dùng đang thấy, chỉ khác chuẩn hóa hoa/thường và dấu chấm).

### 1.3 Bảng cụm trọn chuỗi (BẮT BUỘC có đúng các dòng này)

| Chuỗi khóa | EN | VI |
|---|---|---|
| `HELLO` | `Hello.` | `Xin chào.` |
| `BYE` | `Goodbye.` | `Tạm biệt.` |
| `THANK_YOU` | `Thank you.` | `Cảm ơn.` |
| `SORRY` | `I'm sorry.` | `Tôi xin lỗi.` |
| `YES` | `Yes.` | `Vâng.` |
| `PLEASE` | `Please.` | `Làm ơn.` |
| `ME HUNGRY` | `I am hungry.` | `Tôi đói.` |
| `YOU HUNGRY` | `You are hungry.` | `Bạn đói.` |
| `ME BAD` | `I feel bad.` | `Tôi cảm thấy tệ.` |
| `YOU BAD` | `You are bad.` | `Bạn tệ.` |
| `ME EAT` | `I eat.` | `Tôi ăn.` |
| `ME DRINK` | `I drink.` | `Tôi uống.` |
| `ME DRINK WATER` | `I drink water.` | `Tôi uống nước.` |
| `DRINK WATER` | `Drink water.` | `Uống nước.` |
| `WATER PLEASE` | `Water, please.` | `Làm ơn cho tôi nước.` |
| `DRINK WATER PLEASE` | `Could I have some water, please?` | `Làm ơn cho tôi xin nước uống.` |
| `PLEASE WATER` | `Water, please.` | `Làm ơn cho tôi nước.` |
| `THANK_YOU BYE` | `Thank you. Goodbye.` | `Cảm ơn. Tạm biệt.` |
| `HELLO ME HUNGRY` | `Hello. I am hungry.` | `Xin chào. Tôi đói.` |
| `SORRY BYE` | `I'm sorry. Goodbye.` | `Tôi xin lỗi. Tạm biệt.` |
| `HELLO BYE` | `Hello. Goodbye.` | `Xin chào. Tạm biệt.` |

(Bạn có thể thêm dòng, **không được bớt hoặc đổi nội dung các dòng trên**. Mỗi dòng thêm phải có
EN và VI, và nằm trong phạm vi 13 từ.)

### 1.4 Luật ngữ pháp (khi bảng không khớp)

Chia chuỗi thành **các mệnh đề** theo quy tắc: mỗi INTJ đứng riêng là một mệnh đề; các token còn
lại liền nhau tạo một mệnh đề. Mỗi mệnh đề phải áp được **một** luật dưới đây, nếu **bất kỳ**
mệnh đề nào không áp được ⇒ cả chuỗi rơi về `fallback` (không trộn). Ghép các mệnh đề bằng dấu
cách; mỗi mệnh đề tự có dấu câu.

- **G1** (POLITE ở đầu hoặc cuối mệnh đề): bỏ `PLEASE` khỏi mệnh đề, sinh phần còn lại, rồi thêm
  `, please` trước dấu câu cuối (EN, ví dụ `Eat, please.`) hoặc `Làm ơn ` ở đầu (VI). Phần còn
  lại rỗng ⇒ `Please.` / `Làm ơn.`
- **G2** PRON + ADJ: `I am X.` / `You are X.` (VI: `Tôi X.` / `Bạn X.`).
- **G3** PRON + VERB [+ NOUN]: EN `I eat.` / `You drink water.` (động từ nguyên mẫu, không chia
  thêm); VI `Tôi ăn.` / `Bạn uống nước.`
- **G4** VERB [+ NOUN] không có PRON: câu mệnh lệnh: EN `Drink water.`; VI `Uống nước.`
- **G5** NAME: một mình ⇒ `Anna.`; `ME` + NAME ⇒ `I am Anna.` / `Tôi là Anna.`; INTJ `HELLO` +
  NAME ⇒ `Hello, Anna.` / `Xin chào, Anna.` (hai token này gộp vào **một** mệnh đề, ngoại lệ
  của quy tắc "INTJ đứng riêng"); VERB + NAME ⇒ `Eat Anna.` **KHÔNG áp dụng** ⇒ rơi `fallback`.
  Tên được viết `Title` (chữ đầu hoa, còn lại thường).
- **G6** NOUN một mình: `Water.` / `Nước.`

Chuỗi gồm các INTJ liền nhau hoặc xen mệnh đề (ví dụ `HELLO ME HUNGRY`) đi qua bảng trước; nếu
không có trong bảng thì tách mệnh đề rồi áp luật như trên.

### 1.5 Chuẩn hóa đầu ra

- Ký tự đầu của mỗi câu viết hoa; luôn kết thúc bằng một trong `. ? !`.
- Không để lọt khóa thô (`THANK_YOU`, `SPELL:…`, dấu `_`) vào `text` ở chế độ `table`/`grammar`.
- Tất định: cùng đầu vào ⇒ cùng đầu ra (không dùng ngẫu nhiên, không phụ thuộc thời gian).
- `changed = (text != fallback_join(...))`.

## NHIỆM VỤ 2 — Cấu hình, cờ dòng lệnh

1. Thêm vào `CONFIG` một khối mới (cuối khối, giữ phong cách chú thích hiện có):

   ```python
   "composer": {
       "enabled": True,          # sinh câu tự nhiên khi chốt câu (chỉ hiển thị/giọng nói, KHÔNG ảnh hưởng nhận dạng)
       "speak_sentence": True,   # chốt câu xong thì đọc câu tự nhiên nếu nó KHÁC cách nối thô
   },
   ```
2. Cờ `--composer / --no-composer` (`argparse.BooleanOptionalAction`, mặc định `None` = theo
   CONFIG); truyền vào `run_sentence_builder` bằng tham số **cuối, mặc định `None`**.
3. Tắt ⇒ hành vi y hệt hiện tại.

## NHIỆM VỤ 3 — Nối vào `predict_sentence_webcam.py` và `ui_side_panel.py`

1. Tại khối **chốt câu** (grep ở bảng trên), sau khi gán `final_sentence`: nếu composer bật, tính
   `composed = compose(sentence_keys, ui_lang, phrase_fn=get_asl_phrase_adapter)`
   (adapter bọc `get_asl_phrase(key, lang=lang, default=...)` thành `phrase_fn(key, lang)`).
   Lưu vào biến `composed_sentence` (kiểu `Optional[ComposeResult]`, khởi tạo `None` cạnh
   `sentence_finalized`).
2. **Giữ nguyên** dòng `[SentenceBuilder] ✓ ĐÃ CHỐT CÂU HOÀN CHỈNH: ...` và banner cũ; **thêm** một
   dòng in `[Composer] ...` và để `banner_subtitle` chứa câu tự nhiên khi `composed.changed`.
3. **Xóa `composed_sentence = None`** ở mọi chỗ làm câu thay đổi: nhận từ mới, Backspace/Swipe-Left,
   phím `c` (cả các nhánh xóa/đặt lại câu khác nếu có, ví dụ nhánh chốt từ qua `--confirm-enroll`). Quy tắc chung: bất cứ chỗ nào đặt `sentence_finalized = False`
   cũng phải đặt `composed_sentence = None`.
4. **Phím `l` (đổi ngôn ngữ)**: nếu câu đang ở trạng thái đã chốt, tính lại `composed_sentence`
   theo ngôn ngữ mới (không đọc lại bằng giọng nói).
5. **Giọng nói:** khi chốt câu, nếu `composer.speak_sentence` và `composed.changed` và
   `composed.rule != "fallback"`: gọi
   `speech_service.speak_word("__SENTENCE__", ui_lang, composed.text)` trong try/except như các chỗ
   khác. (Khóa `__SENTENCE__` không có file WAV nên rơi xuống TTS; **kiểm tra** trong
   `speech_service.speak_word` rằng đúng như vậy cho cả `en` và `vi` — nếu `vi` mà chưa có giọng,
   hàm trả `no_vi_voice`: chấp nhận, không báo lỗi.) Từng từ vẫn được đọc lúc ký như cũ.
6. **`SidePanelState` (`ui_side_panel.py`)**: thêm trường **cuối** `composed_sentence: Optional[str] = None`
   (mặc định `None`, các nơi dựng cũ không vỡ). `draw_bottom_panel`: khi `sentence_finalized` và
   `composed_sentence` khác `None`, vẽ **một dòng thứ hai** bên dưới dòng `CAU: "..."` (vị trí khoảng
   `y0 + 75`, nằm giữa dòng gloss ở `y0 + 40` và dòng phím tắt ở `y0 + h - 16`; `BOTTOM_H = 140`), chữ
   `cv2.FONT_HERSHEY_SIMPLEX` cỡ ~0.55, màu `COL_OK`, **không** thêm khung mới. Dòng gloss gốc **không bị
   đổi hay che**. Nếu chữ rộng hơn panel, thu cỡ chữ như đoạn hiện có (`font_scale` theo `max_w / tw`).
7. **Chữ có dấu:** panel vẽ bằng `cv2.putText` nên KHÔNG hiển thị được tiếng Việt có dấu (ra `???`). Khi
   `ui_lang == "vi"`, truyền sang panel **bản đã bỏ dấu** (`strip_diacritics(composed.text)`, như các chuỗi
   khác trong panel đã viết không dấu). Câu có dấu đầy đủ vẫn dùng cho giọng nói và nhật ký in terminal.
8. **Nhật ký sự kiện:** ngay sau dòng `event_log.push(f"Chốt câu: ...")` thêm
   `event_log.push(f"Cau tu nhien: {...}", ui_side_panel.COL_OK)` (cũng bản đã bỏ dấu khi `vi`), chỉ khi
   `composed.changed`.
9. Truyền `composed_sentence=<text hiển thị hoặc None>` vào nơi dựng `SidePanelState`.

## NHIỆM VỤ 4 — Test và số đo

1. **Test mới** `tests/test_sentence_composer.py` (thuần Python, không webcam):
   - Mọi dòng của bảng 1.3 cho cả `en` và `vi` (test dạng bảng, so khớp chuỗi chính xác).
   - Mọi `truth` có `kind == "phrase"` trong `phrases_plan.json` (đọc file thật bằng
     `encoding="utf-8"`): `compose(...)` trả `text` khác rỗng, bắt đầu bằng chữ hoa, kết thúc
     `. ? !`, **không chứa** `_`, `SPELL:` hay khóa thô viết hoa toàn bộ; với `lang="vi"` cũng vậy.
   - Tính tất định: gọi hai lần cho cùng kết quả.
   - `compose([])` ⇒ `rule == "empty"`, `text == ""`.
   - Khóa lạ (`["ME", "XYZ"]`) ⇒ `rule == "fallback"`, **không ném lỗi**.
   - `fallback` giữ **đủ và đúng thứ tự** mọi từ nội dung (không mất/đổi chỗ).
   - Token `SPELL:ANNA`: `ME SPELL:ANNA` ⇒ `I am Anna.`; `HELLO SPELL:ANNA` ⇒ `Hello, Anna.`;
     `SPELL:ANNA` ⇒ `Anna.`; `EAT SPELL:ANNA` ⇒ `fallback`.
   - `changed` đúng nghĩa (True cho `ME HUNGRY`, False khi `rule == "fallback"`).
   - Module không import được `cv2` (kiểm bằng cách import trong tiến trình mà `cv2` bị chặn, hoặc
     kiểm `sys.modules` sau import).
   - **Test panel (headless)** trong `tests/test_sentence_composer_panel.py`: dựng canvas numpy, gọi
     `ui_side_panel.draw_bottom_panel` với `SidePanelState(sentence_words=[...], sentence_finalized=True,
     composed_sentence="I am hungry.")` ⇒ ảnh khác ảnh khi `composed_sentence=None`; với
     `composed_sentence=None` ⇒ ảnh **giống hệt** bản cũ (bằng chứng không đổi hành vi); câu rất dài không
     ném lỗi; `strip_diacritics("Làm ơn cho tôi xin nước uống.")` không còn ký tự ngoài ASCII.
2. **Hồi quy** như PREAMBLE: danh sách FAILED trước/sau, diff rỗng.
3. **Số đo bắt buộc:** (a) thời gian `compose` cho chuỗi 6 từ: báo trung vị (kỳ vọng < 1 ms);
   (b) bảng: số cụm trong `phrases_plan.json` rơi vào `table` / `grammar` / `fallback` (để biết bộ
   luật phủ bao nhiêu % cụm đã lên kế hoạch thu); (c) với composer TẮT, chuỗi sự kiện phát lại
   (`chain_replay.py`/giả lập headless) **giống hệt** cũ.

## Tiêu chí hoàn thành

- [ ] `sentence_composer.py` thuần Python, đủ giao diện, tất định, không ném ngoại lệ.
- [ ] Mọi dòng bảng 1.3 đúng ở cả `en` và `vi`.
- [ ] 100% cụm `kind == "phrase"` trong `phrases_plan.json` cho câu hợp lệ (không rỗng, có dấu câu).
- [ ] `tests/test_sentence_composer.py` PASS; diff FAILED toàn suite **rỗng**.
- [ ] Gloss gốc luôn hiển thị; câu tự nhiên là dòng thêm; tắt composer ⇒ hành vi y hệt cũ.
- [ ] `SidePanelState` chỉ được CỘNG trường; **không sửa `ui_render.py`**.

## Ngoài phạm vi (KHÔNG làm)

- LLM, gọi mạng, thư viện NLP nặng; chia động từ ngôi/thì phức tạp; dịch tự do ngoài 13 từ.
- Nhận biết câu hỏi bằng nét mặt/cử chỉ; thêm từ vựng mới (mỗi từ mới cần thu mẫu — spec khác).
- Giọng đọc tiếng Việt (máy chưa có; chỉ hiển thị chữ).

## Cách thử trực tiếp (người dùng, ~5 phút)

1. Ký `ME` rồi `HUNGRY`, hạ tay 2 giây → thấy dòng gloss `Me Hungry` **và** dòng `I am hungry.`; nghe
   câu được đọc.
2. Ký `DRINK`, `WATER`, `PLEASE` → `Could I have some water, please?`
3. Nhấn `l` → câu đổi sang tiếng Việt (`Làm ơn cho tôi xin nước uống.`).
4. Ký thêm một từ sau khi chốt → dòng câu tự nhiên biến mất, chờ chốt lại.
5. `--no-composer` → như cũ.

---

## GHI CHÚ TRIỂN KHAI (2026-10-03)

**Đã làm:** `sentence_composer.py` (mới, thuần Python), `ui_side_panel.py` (thêm `SidePanelState.composed_sentence` + dòng câu tự nhiên
ở `draw_bottom_panel`), `predict_sentence_webcam.py` (cấu hình `CONFIG["composer"]`, cờ `--composer/--no-composer`, sinh câu khi chốt,
đọc câu, nhật ký sự kiện, truyền xuống panel), 3 file test mới (`tests/test_sentence_composer.py`, `tests/test_sentence_composer_panel.py`).
Không sửa `ui_render.py` (chỉ import `strip_diacritics`).

**Khác với spec ở trên (có chủ đích):**
1. **Không reset `composed_sentence` ở từng chỗ sửa câu.** Thay vào đó kết quả được tính lại khi `(chuỗi khóa, ngôn ngữ)` đổi
   (`composer_cache_key`), và chỉ hiển thị khi `sentence_finalized`. Cùng hành vi, ít điểm chạm hơn trong file hay bị đồng đội sửa,
   và phím `l` tự cập nhật câu mà không cần nhánh riêng. Giọng nói chỉ chạy đúng một lần lúc chốt câu (không đọc lại khi đổi ngôn ngữ).
2. **Chỉ hiện dòng câu tự nhiên khi `changed == True`** (khác cách nối thô). Câu rơi về `fallback` thì không hiện dòng thừa.
3. Tên hiển thị tiếng Việt cho nhánh fallback có bảng nội bộ dự phòng (13 từ) nếu `phrase_fn` không trả gì.

**Số đo (2026-10-03):**
- (Số đo lịch sử, chưa tính lại sau khi thêm luật G14–G17.) 70 cụm trong `phrases_plan.json` (sau khi thêm luật `ME SORRY`): **13 theo bảng + 33 theo luật = 46 (66%)** ra câu tự nhiên; 24 rơi về cách nối cũ.
  Cờ `natural` của kế hoạch (câu viết tay trong `CORE_2`/`CORE_3` của `collect_phrases.py`) cho thấy: trong 52 cụm `natural` có **40 (77%)** ra câu,
  **12 chưa có luật** (HELLO YOU, HELLO YOU THANK_YOU, HELLO ME THANK_YOU, THANK_YOU YOU BYE, THANK_YOU ME BYE, SORRY YOU, SORRY YOU BYE, BYE ME,
  BAD WATER, HUNGRY ME EAT, HUNGRY ME DRINK, HUNGRY YES); trong 18 cụm không `natural` (chỉ để phủ cặp từ, nhiều cụm vô nghĩa như `BAD EAT`) có 6 (33%) ra câu.
  **Đính chính:** bản ghi trước nói "phần fallback chủ yếu là tổ hợp vô nghĩa" là SAI một nửa (12/24 cụm fallback do kế hoạch đánh dấu là tự nhiên).
  **Quyết định của chủ dự án (2026-10-03):** chỉ cần ra câu cho các câu THỰC SỰ có nghĩa; KHÔNG ép các chuỗi đảo thứ tự / mơ hồ thành câu hoàn chỉnh.
  Vì vậy KHÔNG thêm luật cho 12 cụm chưa phủ ở trên; chúng giữ cách nối cũ (fallback) — an toàn, không bao giờ sai nghĩa. Chỉ thêm luật khi chủ dự án
  nêu một câu cụ thể có nghĩa mà đang bị bỏ sót (như `ME SORRY`).
- Thời gian `compose` chuỗi 6 từ: ~0.005 ms.
- Đã đối chiếu: KHÔNG có chức năng sinh câu nào có sẵn trong dự án (đã grep code/docs/web). Giao diện web chỉ ghép từng từ EN và VI.

**Chưa kiểm chứng được:** đường chạy thật trên camera (ký -> hạ tay 1.8 s -> thấy dòng câu tự nhiên + nghe đọc câu); môi trường agent không có webcam.
**Việc ngoài phạm vi, có thể làm sau:** (a) dùng cùng bộ sinh câu cho giao diện web (`web_app.py` `summary.sentence`, `web/index.html`);
(b) sửa chữ có dấu bị hỏng (`Nh???n t???`) ở nhật ký sự kiện hiện có; (c) dòng gloss đổi theo ngôn ngữ ở chế độ `vi`.
