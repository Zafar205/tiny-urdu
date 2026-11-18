"""Urdu front end for tiny-voca.

tiny-voca has a fixed inventory of 76 English phoneme symbols. Instead of
growing it, this module squeezes Urdu into that inventory, so the English
checkpoint can be fine-tuned (embeddings and all) rather than trained from zero.

Two kinds of input are accepted by ``phonemize``:

* ``"ipa:<phonemes>"``  The UrduTTS ``phonemes`` column. Used for TRAINING:
  it is what was actually said in the audio. Stress marks are dropped
  (see below).
* anything else         Roman Urdu such as ``"aap kaise hain?"``. Used for
  INFERENCE, through the rule-based ``roman_to_phones`` G2P.

Both routes end in the same inventory phones, so the model sees one token
language. Known approximations (the inventory has no such sounds):

  retroflex ʈ ɖ ɽ -> t d ɾ     aspiration  kʰ -> k h    x ɣ q -> h ɡ k
  nasal vowel ẽ   -> e + n     ʋ -> v       ɟ c -> dʒ tʃ

Stress is dropped on purpose: the dataset's stress marks come from a rule
based phonemizer, and Roman Urdu text carries no stress to reproduce.

The UrduTTS ``roman`` column is NOT used. It is a vowel-less consonant
skeleton ("JS KY MSAL BNDGAN"), which cannot be spoken.
"""

import hashlib
import json
import re
import unicodedata

IPA_PREFIX = "ipa:"
VERSION = "urdu-1"

# ---- IPA (UrduTTS phonemes column) -> inventory --------------------------------------------

# (base, long) -> inventory phones. Anything missing falls through unchanged and
# becomes <unk>, which the annotator reports and rejects (so nothing hides).
_VOWELS = {
    ("a", False): ("ɐ",), ("a", True): ("ɑː",), ("ɑ", False): ("ɑː",), ("ɑ", True): ("ɑː",),
    ("ə", False): ("ə",), ("ə", True): ("ə",), ("ʌ", False): ("ʌ",), ("ʌ", True): ("ʌ",),
    ("ɛ", False): ("ɛ",), ("ɛ", True): ("ɛ",), ("e", False): ("eɪ",), ("e", True): ("eɪ",),
    ("i", False): ("i",), ("i", True): ("iː",), ("ɪ", False): ("ɪ",), ("ɪ", True): ("iː",),
    ("o", False): ("oː",), ("o", True): ("oː",), ("ɔ", False): ("ɔ",), ("ɔ", True): ("ɔː",),
    ("u", False): ("ʊ",), ("u", True): ("uː",), ("ʊ", False): ("ʊ",), ("ʊ", True): ("uː",),
    ("ɐ", False): ("ɐ",), ("ɐ", True): ("ɑː",),
}
_CONSONANTS = {
    "p": ("p",), "b": ("b",), "t": ("t",), "d": ("d",), "k": ("k",), "ɡ": ("ɡ",), "g": ("ɡ",),
    "f": ("f",), "s": ("s",), "z": ("z",), "ʃ": ("ʃ",), "ʒ": ("ʒ",), "h": ("h",),
    "m": ("m",), "n": ("n",), "l": ("l",), "j": ("j",), "w": ("w",), "v": ("v",),
    "r": ("ɾ",), "ɾ": ("ɾ",), "ɹ": ("ɹ",), "ʔ": ("ʔ",), "ð": ("ð",), "θ": ("θ",), "ŋ": ("ŋ",),
    "ɟ": ("dʒ",), "c": ("tʃ",), "ʂ": ("ʃ",), "ʐ": ("ʒ",), "ʋ": ("v",),
    "x": ("h",), "ɣ": ("ɡ",), "q": ("k",),
    "ʈ": ("t",), "ɖ": ("d",), "ɽ": ("ɾ",), "ɳ": ("n",), "ɭ": ("l",), "ɲ": ("n",), "ɫ": ("l",),
}
_IGNORED = set("ˈˌ.|‿͜͡")     # stress, syllable dots, tie bars
_COMBINING_NASAL = "̃"


def _ipa_word(word):
    """One IPA word -> inventory phones (list of str)."""
    word = unicodedata.normalize("NFD", word)
    units = []                           # [base, long, aspirated, nasal]
    for char in word:
        if char in _IGNORED:
            continue
        if char == "ː" and units:
            units[-1][1] = True
        elif char == "ʰ" and units:
            units[-1][2] = True
        elif char == _COMBINING_NASAL and units:
            units[-1][3] = True
        elif unicodedata.combining(char):
            continue
        else:
            units.append([char, False, False, False])
    phones = []
    for base, long, aspirated, nasal in units:
        if (base, long) in _VOWELS:
            phones.extend(_VOWELS[(base, long)])
            if nasal:
                phones.append("n")
        else:
            phones.extend(_CONSONANTS.get(base, (base,)))
            if aspirated:
                phones.append("h")
    return phones


def ipa_to_words(text):
    return [w for w in (_ipa_word(w) for w in text.split()) if w]


# ---- Roman Urdu -> inventory (rule based) ------------------------------------------------

# Very frequent words whose spelling varies or breaks the rules.
_LEXICON = {
    "hai": "h ɛ", "hay": "h ɛ", "he": "h ɛ", "hain": "h ɛ n", "hein": "h ɛ n", "hen": "h ɛ n",
    "mein": "m eɪ n", "main": "m eɪ n", "men": "m eɪ n", "me": "m eɪ n",
    "nahi": "n ɐ h iː", "nahin": "n ɐ h iː n", "nahee": "n ɐ h iː", "nai": "n ɐ h iː",
    "aur": "ɔ ɾ", "or": "ɔ ɾ", "ka": "k ɑː", "ki": "k iː", "ke": "k eɪ", "ko": "k oː",
    "se": "s eɪ", "ne": "n eɪ", "par": "p ɐ ɾ", "yeh": "j eɪ", "ye": "j eɪ", "woh": "v oː", "wo": "v oː",
    "to": "t oː", "bhi": "b iː", "hi": "h iː", "is": "ɪ s", "us": "ʊ s", "in": "ɪ n", "un": "ʊ n",
    "kya": "k j ɑː", "kyun": "k j ʊ n", "kyon": "k j ʊ n", "kuch": "k ʊ tʃ", "kuchh": "k ʊ tʃ",
    "tha": "t ɑː", "thi": "t iː", "the": "t eɪ", "thay": "t eɪ", "ho": "h oː", "hoon": "h uː n", "hun": "h uː n",
    "allah": "ɐ l l ɑː h", "mera": "m eɪ ɾ ɑː", "meri": "m eɪ ɾ iː", "mere": "m eɪ ɾ eɪ",
}

_DIGRAPHS = (
    ("chh", ("tʃ", "h")), ("kh", ("k", "h")), ("gh", ("ɡ", "h")), ("ch", ("tʃ",)), ("sh", ("ʃ",)),
    ("th", ("t", "h")), ("dh", ("d", "h")), ("ph", ("p", "h")), ("bh", ("b", "h")),
    ("jh", ("dʒ", "h")), ("zh", ("ʒ",)), ("ck", ("k",)),
    ("aa", ("ɑː",)), ("ee", ("iː",)), ("ii", ("iː",)), ("oo", ("uː",)), ("ai", ("ɛ",)), ("ay", ("eɪ",)),
    ("ey", ("eɪ",)), ("ei", ("ɛ",)), ("au", ("ɔ",)), ("ou", ("ɔ",)), ("ao", ("ɑː", "oː")),
)
_SINGLE = {
    "a": ("ɐ",), "e": ("eɪ",), "i": ("ɪ",), "o": ("oː",), "u": ("ʊ",),
    "b": ("b",), "p": ("p",), "t": ("t",), "d": ("d",), "k": ("k",), "g": ("ɡ",), "f": ("f",), "s": ("s",),
    "z": ("z",), "m": ("m",), "n": ("n",), "l": ("l",), "r": ("ɾ",), "h": ("h",), "j": ("dʒ",), "y": ("j",),
    "v": ("v",), "w": ("v",), "c": ("k",), "q": ("k",), "x": ("k", "s"),
}
_VOWEL_PHONES = {p for pair in _VOWELS.values() for p in pair}


def roman_to_phones(word):
    """One Roman Urdu word (letters only) -> inventory phones."""
    word = word.lower().replace("'", "")
    if word in _LEXICON:
        return _LEXICON[word].split()
    phones, i = [], 0
    while i < len(word):
        for letters, out in _DIGRAPHS:
            if word.startswith(letters, i):
                phones.extend(out)
                i += len(letters)
                break
        else:
            char, final = word[i], i == len(word) - 1
            if char == "a" and final and len(word) > 1:
                out = ("ɑː",)                      # kya, hamara, aaya: final a is long
            elif char == "i" and final and len(word) > 1:
                out = ("iː",)
            else:
                out = _SINGLE.get(char)
            if out:
                # geminate consonants ("achha") are written double; say them once
                if not (phones and out[0] == phones[-1] and out[0] not in _VOWEL_PHONES):
                    phones.extend(out)
            i += 1
    return phones


_TOKEN = re.compile(r"[A-Za-z']+|[,.?!;:\"()—…]")


def roman_pieces(text):
    from tiny_voca import text as base
    text = base.normalize_text(text).replace("-", " ")
    pieces, words = [], []
    for token in _TOKEN.findall(text):
        if token[0].isalpha() or token[0] == "'":
            phones = roman_to_phones(token)
            if phones:
                words.append(phones)
        else:
            if words:
                pieces.append((False, words))
                words = []
            pieces.append((True, token))
    if words:
        pieces.append((False, words))
    return pieces


# ---- tiny-voca integration -----------------------------------------------------------------

def _as_pieces(words):
    return [(False, [[(p, 0) for p in w] for w in words])]


def pieces(text):
    if text.startswith(IPA_PREFIX):
        return _as_pieces(ipa_to_words(text[len(IPA_PREFIX):]))
    return [(flag, value if flag else [[(p, 0) for p in w] for w in value]) for flag, value in roman_pieces(text)]


def _rules_digest():
    blob = json.dumps([_VOWELS_KEYS(), _CONSONANTS, _LEXICON, _DIGRAPHS, _SINGLE, VERSION],
                      sort_keys=True, ensure_ascii=False, default=list)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _VOWELS_KEYS():
    return {f"{k[0]}{int(k[1])}": v for k, v in _VOWELS.items()}


def install(text_module):
    """Replace tiny_voca.text's English front end with this one. Call BEFORE
    importing tiny_voca.annotate / train / infer, which bind ``phonemize`` by name."""
    def phonemize(text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must not be empty")
        return text_module.tokenize(pieces(text))

    def frontend_identity():
        return {"frontend_version": VERSION, "phonemizer": "tiny-urdu", "rules": _rules_digest(),
                "symbols": list(text_module.SYMBOLS)}

    def frontend_runtime():
        return {"phonemizer": "tiny-urdu-" + _rules_digest()}

    text_module.phonemize = phonemize
    text_module.frontend_identity = frontend_identity
    text_module.frontend_runtime = frontend_runtime
