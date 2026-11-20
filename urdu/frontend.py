"""Urdu front end for tiny-voca.

tiny-voca's 76 English phoneme symbols are kept (so the English checkpoint can
be fine-tuned) and NATIVE Urdu phones are appended: retroflex ʈ ɖ ɽ, aspirated
kʰ tʰ bʰ ..., q x ɣ ʋ, long/nasal vowels aː eː ẽ ã ... Every symbol is a label
of the wav2vec2 forced aligner, spelled exactly as in its vocab.json (nasal
vowels decomposed: base + U+0303). Mapping Urdu onto English symbols instead
(kʰ -> k h, ʈ -> t) made the aligner reject 998/1000 clips.

Input to ``phonemize``:
* ``"ipa:<phonemes>"``  the UrduTTS ``phonemes`` column (TRAINING)
* anything else         Roman Urdu, e.g. ``"aap kaise hain?"`` (INFERENCE)
  Capital T, D, R mark retroflex ʈ ɖ ɽ (``baRa`` = big, ``bura`` = bad).

Both routes go through ``_ipa_word``, so they produce one token language.
Stress marks are dropped (Roman Urdu has none to reproduce).
"""

import hashlib
import json
import re
import unicodedata

IPA_PREFIX = "ipa:"
VERSION = "urdu-2"
N = "̃"  # combining tilde (nasal)

# Appended to tiny-voca's SYMBOLS. APPEND-ONLY: reordering breaks checkpoints.
URDU_CONSONANTS = ("ʈ", "ɖ", "ɽ", "q", "x", "ɣ", "ʋ", "r", "ɟ", "c", "ʂ", "ʐ", "ɳ", "ɲ", "ɭ",
                   "kʰ", "pʰ", "tʰ", "ʈʰ", "bʰ", "dʰ", "ɖʰ", "ɡʰ", "ɟʰ", "cʰ")
URDU_VOWELS = ("a", "aː", "e", "eː", "o", "u", "ɛː", "ɪː", "ʊː",
               "a" + N, "e" + N, "e" + N + "ː", "o" + N, "i" + N, "u" + N, "ɛ" + N, "ɔ" + N, "ɐ" + N, "ɑ" + N)
URDU_PHONES = URDU_CONSONANTS + URDU_VOWELS
URDU_VOICELESS = ("ʈ", "q", "x", "c", "ʂ", "kʰ", "pʰ", "tʰ", "ʈʰ", "cʰ")

_IGNORED = set("ˈˌ.|‿͜͡")
# Last resort when no native symbol exists (rare in the data).
_APPROX = {"ɹ": "ɹ", "ɡ": "ɡ", "g": "ɡ", "ə": "ə", "ʌ": "ʌ", "ɐ": "ɐ", "ɑ": "ɑː", "ɔ": "ɔ", "ʊ": "ʊ", "ɪ": "ɪ", "i": "i", "ɛ": "ɛ"}
_NASAL_BASE = {"ə": "ɐ", "ʌ": "ɐ", "ɐ": "ɐ", "ɪ": "i", "ʊ": "u", "ɑ": "ɑ"}

_inventory = None   # set by install(): the full SYMBOLS phone set


def _resolve(base, long, aspirated, nasal, inventory):
    """One IPA segment -> list of inventory symbols, preferring the native one."""
    L, H = ("ː" if long else ""), ("ʰ" if aspirated else "")
    base = "ɡ" if base == "g" else base
    candidates = []
    if nasal:
        for b in (base, _NASAL_BASE.get(base, base)):
            candidates += [b + N + L, b + N]
    candidates += [base + H + L, base + H]
    if not H and not L:
        candidates.append(_APPROX.get(base, base))
    if long:
        candidates += [_APPROX.get(base, base) + "ː", {"ɑ": "ɑː"}.get(base, "")]
    for c in candidates:
        if c in inventory:
            return [c]
    if aspirated:                                  # e.g. ɽʰ -> ɽ h
        return _resolve(base, long, False, nasal, inventory) + ["h"]
    return [base]                                  # unknown -> <unk>, reported by the annotator


def _ipa_word(word, inventory=None):
    inventory = inventory or _inventory
    units = []
    for char in unicodedata.normalize("NFD", word):
        if char in _IGNORED:
            continue
        if char in "ː:" and units:
            units[-1][1] = True
        elif char == "ʰ" and units:
            units[-1][2] = True
        elif char == N and units:
            units[-1][3] = True
        elif unicodedata.combining(char):
            continue
        else:
            units.append([char, False, False, False])
    phones = []
    for base, long, aspirated, nasal in units:
        phones += _resolve(base, long, aspirated, nasal, inventory)
    return phones


# ---- Roman Urdu -> IPA string -> _ipa_word --------------------------------------------------

_LEXICON = {   # Roman spelling -> IPA as the dataset writes it
    "hai": "hɛ", "hay": "hɛ", "he": "hɛ", "hain": "hẽ", "hein": "hẽ", "hen": "hẽ",
    "mein": "mẽ", "main": "mẽ", "men": "mẽ", "me": "mẽ", "nahi": "nahiː", "nahin": "nahĩ", "nahee": "nahiː",
    "aur": "ɔr", "or": "ɔr", "ka": "kaː", "ki": "ki", "ke": "keː", "ko": "koː", "se": "seː", "ne": "neː",
    "par": "par", "pe": "peː", "yeh": "jeː", "ye": "jeː", "woh": "ʋoː", "wo": "ʋoː", "to": "toː", "bhi": "bʰiː",
    "hi": "hiː", "is": "ɪs", "us": "ʊs", "in": "ɪn", "un": "ʊn", "kya": "kjaː", "kyun": "kjũ", "kyon": "kjũ",
    "kuch": "kʊcʰ", "kuchh": "kʊcʰ", "tha": "tʰaː", "thi": "tʰiː", "the": "tʰeː", "thay": "tʰeː", "ho": "hoː",
    "hoon": "hũ", "hun": "hũ", "hum": "ham", "tum": "tʊm", "aap": "aːp", "ap": "aːp", "ek": "eːk", "do": "doː",
    "jo": "ɟoː", "jab": "ɟab", "tak": "tak", "allah": "allaːh", "mera": "meːraː", "meri": "meːriː",
    "mere": "meːreː", "apna": "apnaː", "apni": "apniː", "apne": "apneː", "sab": "sab", "ab": "ab",
}
# Extra spellings added after training. Kept OUT of _rules_digest so trained
# checkpoints still load (infer.py checks the frontend fingerprint). Training
# uses the ipa: path, so this only affects Roman input. Add words here freely.
_LEXICON_EXTRA = {
    "chai": "caːeː", "chaye": "caːeː", "chae": "caːeː", "chaaye": "caːeː", "chaey": "caːeː",
}
_RULES = (   # longest first; output is IPA
    ("chh", "cʰ"), ("Th", "ʈʰ"), ("Dh", "ɖʰ"), ("Rh", "ɽʰ"),
    ("kh", "kʰ"), ("gh", "ɡʰ"), ("ch", "c"), ("sh", "ʃ"), ("th", "tʰ"), ("dh", "dʰ"), ("ph", "pʰ"),
    ("bh", "bʰ"), ("jh", "ɟʰ"), ("zh", "ʒ"), ("ck", "k"),
    ("aa", "aː"), ("ee", "iː"), ("ii", "iː"), ("oo", "uː"), ("ai", "ɛ"), ("ay", "eː"), ("ey", "eː"),
    ("ei", "ɛ"), ("au", "ɔ"), ("ou", "ɔ"), ("ao", "aːoː"),
    ("T", "ʈ"), ("D", "ɖ"), ("R", "ɽ"),
)
_SINGLE = {"a": "a", "e": "eː", "i": "ɪ", "o": "oː", "u": "ʊ", "g": "ɡ", "j": "ɟ", "y": "j", "w": "ʋ",
           "v": "ʋ", "c": "k", "x": "ks", "r": "r"}
_VOWEL_LETTERS = set("aeiou")


def roman_to_ipa(word):
    if word.lower() in _LEXICON_EXTRA and not any(c in "TDR" for c in word[1:]):
        return _LEXICON_EXTRA[word.lower()]
    if word.lower() in _LEXICON and not any(c in "TDR" for c in word):
        return _LEXICON[word.lower()]
    # Capital T/D/R inside a word mark retroflex. A capital first letter is just
    # sentence case, and ALL CAPS words are read as plain lowercase.
    if word.isupper() and len(word) > 1:
        word = word.lower()
    word = word[:1].lower() + word[1:]
    if word.lower() in _LEXICON and not any(c in "TDR" for c in word):
        return _LEXICON[word.lower()]
    word = "".join(c if c in "TDR" else c.lower() for c in word.replace("'", ""))
    out, i, last = [], 0, ""
    while i < len(word):
        for letters, ipa in _RULES:
            if word.startswith(letters, i):
                out.append(ipa)
                i += len(letters)
                break
        else:
            ch, final = word[i], i == len(word) - 1
            if ch == "a" and final and len(word) > 1:
                ipa = "aː"
            elif ch == "i" and final and len(word) > 1:
                ipa = "iː"
            else:
                ipa = _SINGLE.get(ch, ch if ch.isalpha() else "")
            if ipa and not (ch not in _VOWEL_LETTERS and ch == last):   # "achha": say a doubled consonant once
                out.append(ipa)
            last = ch
            i += 1
            continue
        last = ""
    return "".join(out)


# ---- numbers -> spoken Urdu (Roman spelling, then the normal rules) ------------------------------
# Urdu has a distinct word for every number 0-99. NOT part of _rules_digest, so old checkpoints still load.
_W = ("sifar ek do teen chaar paanch chhe saat aath nau das "
      "gyarah barah terah chaudah pandrah solah satrah atharah unnees "
      "bees ikkees baaees teyees chaubees pachchees chhabbees sattaees atthaees untees "
      "tees iktees battees taitees chauntees paintees chhattees saintees artees untaalees "
      "chaalees iktaalees bayaalees taintaalees chauwaalees paintaalees chhiyaalees saintaalees artaalees unchaas "
      "pachaas ikyaavan baavan tirpan chauvan pachpan chhappan sattaavan atthaavan unsath "
      "saath iksath baasath tirsath chausath painsath chhiyaasath sarsath arsath unhattar "
      "sattar ikhattar bahattar tihattar chauhattar pachhattar chhihattar satattar athattar unaasi "
      "assi ikyaasi bayaasi tiraasi chauraasi pachaasi chhiyaasi sataasi athaasi navaasi "
      "nabbe ikyaanve baanve tiraanve chauraanve pachaanve chhiyaanve sattaanve atthaanve ninnaanve").split()
assert len(_W) == 100
_SCALES = ((10 ** 9, "arab"), (10 ** 7, "karoR"), (10 ** 5, "lakh"), (1000, "hazaar"), (100, "sau"))
_D = "[0-9\\u0660-\\u0669\\u06F0-\\u06F9\\u0966-\\u096F]"
# 1,250,000 or Pakistani grouping 10,00,000 or plain digits, each with an optional decimal part
_NUM = re.compile(rf"{_D}{{1,3}}(?:,{_D}{{3}})+(?:\.{_D}+)?|{_D}{{1,2}}(?:,{_D}{{2}})+,{_D}{{3}}(?:\.{_D}+)?|{_D}+(?:\.{_D}+)?")


def number_words(n):
    if n < 100:
        return _W[n]
    for size, name in _SCALES:
        if n >= size:
            q, r = divmod(n, size)
            return number_words(q) + " " + name + (" " + number_words(r) if r else "")


def _digit(c):
    o = ord(c)
    for zero in (0x30, 0x660, 0x6F0, 0x966):
        if zero <= o < zero + 10:
            return str(o - zero)
    return c


def _say_number(m):
    s = "".join(_digit(c) for c in m.group(0))
    whole, _, frac = s.partition(".")
    whole = whole.replace(",", "")
    if (len(whole) > 1 and whole[0] == "0") or len(whole) > 9:   # phone numbers, codes: digit by digit
        words = " ".join(_W[int(d)] for d in whole)
    else:
        words = number_words(int(whole))
    if frac:
        words += " poaint " + " ".join(_W[int(d)] for d in frac)
    return " " + words + " "


def expand_numbers(text):
    return _NUM.sub(_say_number, text)


_TOKEN = re.compile(r"[A-Za-z']+|[,.?!;:\"()—…]")


def pieces(text):
    if text.startswith(IPA_PREFIX):
        words = [w for w in (_ipa_word(w) for w in text[len(IPA_PREFIX):].split()) if w]
        return [(False, [[(p, 0) for p in w] for w in words])]
    from tiny_voca.text import normalize_text
    result, words = [], []
    for token in _TOKEN.findall(expand_numbers(normalize_text(text)).replace("-", " ")):
        if token[0].isalpha() or token[0] == "'":
            phones = _ipa_word(roman_to_ipa(token))
            if phones:
                words.append([(p, 0) for p in phones])
        else:
            if words:
                result.append((False, words))
                words = []
            result.append((True, token))
    if words:
        result.append((False, words))
    return result


def _rules_digest():
    blob = json.dumps([URDU_PHONES, _LEXICON, _RULES, _SINGLE, _APPROX, VERSION], ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---- tiny-voca integration -----------------------------------------------------------------

def install():
    """Extend tiny_voca's symbol inventory with Urdu phones and swap in this front
    end. Must run BEFORE tiny_voca.{data,align,annotate,train,infer} are imported."""
    global _inventory
    import dataclasses
    import importlib
    import tiny_voca
    from tiny_voca import config as config_module, text

    if getattr(text, "_urdu_installed", False):
        return
    english = text.SYMBOLS
    assert len(english) == 76 and not set(URDU_PHONES) & set(english)
    text.CONSONANTS = text.CONSONANTS + URDU_CONSONANTS
    text.VOWELS = text.VOWELS + URDU_VOWELS
    text.PHONES = text.PHONES + URDU_PHONES
    text.SYMBOLS = english + URDU_PHONES          # English ids unchanged; Urdu appended
    text.SYMBOL_TO_ID = {s: i for i, s in enumerate(text.SYMBOLS)}
    text.PHONE_SET = frozenset(text.PHONES)
    text.PHONE_IDS = frozenset(text.SYMBOL_TO_ID[p] for p in text.PHONES) | {text.SYMBOL_TO_ID[text.UNK]}
    text.VOICELESS = text.VOICELESS | frozenset(URDU_VOICELESS)
    text.VOWEL_SET = frozenset(text.VOWELS)
    _inventory = text.PHONE_SET

    def phonemize(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("text must not be empty")
        return text.tokenize(pieces(value))

    def frontend_identity():
        return {"frontend_version": VERSION, "phonemizer": "tiny-urdu", "rules": _rules_digest(), "symbols": list(text.SYMBOLS)}

    text.phonemize = phonemize
    text.frontend_identity = frontend_identity
    text.frontend_runtime = lambda: {"phonemizer": "tiny-urdu-" + _rules_digest()}

    # Config: vocab_size default follows the larger inventory.
    UrduConfig = dataclasses.make_dataclass(
        "Config", [("vocab_size", int, dataclasses.field(default=len(text.SYMBOLS)))],
        bases=(config_module.Config,), frozen=True)
    UrduConfig.__module__ = config_module.__name__
    config_module.Config = UrduConfig
    tiny_voca.Config = UrduConfig
    # model.py was imported by tiny_voca/__init__ with the old symbol sets: re-import it.
    model = importlib.reload(importlib.import_module("tiny_voca.model"))
    tiny_voca.TinyVoca = model.TinyVoca
    text._urdu_installed = True
