import re

MN_CHARS = set("абвгдеёжзийклмноөпрстуүфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯ")

ER_EGSIG = set("аоуАОУ")

IY_SUFFIXES = {"ийн", "ийг", "ийд", "ийиг", "ийгээ", "ийнх"}

SOFT_SIGN_ENDINGS = set("л")
MN_SUFFIXES = [
    "байгаа", "болсон", "болно",
    "аарай", "ээрэй",
    "нууд",  "нүүд",
    "гаар",  "гээр",  "гоор",  "гөөр",
    "гийн",  "гын",
    "ийиг", "иар",                 
    "аас",   "ээс",   "оос",   "өөс",
    "аар",   "ээр",   "оор",   "өөр",
    "тай",   "тэй",   "той",   "төй",
    "ийн",   "ний",   "ын",
    "ийг",   "ыг",
    "лаа",   "лээ",   "лоо",   "лөө",
    "жээ",   "чээ",
    "сан",   "сэн",   "сон",   "сөн",
    "аад",   "ээд",   "оод",   "өөд",
    "хад",   "хэд",            
    "уд",    "үд",
    "нд",               
    "д",
     "ч",
]

PUNCT = re.compile(r'[.,!?;:"\'()\-—«»…\n\r]')

def is_mongolian(word: str) -> bool:
    if not word:
        return False
    mn_count = sum(1 for ch in word if ch in MN_CHARS)
    return mn_count > len(word) * 0.5

def has_er_egsig(word: str) -> bool:
    return any(ch in ER_EGSIG for ch in word)

def restore_soft_sign(root: str, suffix: str) -> str:
    if suffix not in IY_SUFFIXES:
        return root
    if not root:
        return root
    last_char = root[-1]
    if last_char not in SOFT_SIGN_ENDINGS:
        return root
    if not has_er_egsig(root):
        return root
    return root + "ь"

def strip_suffix(word: str) -> str:
    for suffix in MN_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            root = word[: -len(suffix)]
            root = restore_soft_sign(root, suffix)
            return root
    return word

def normalize_mongolian(text: str) -> str:
    cleaned = PUNCT.sub(" ", text)
    words = cleaned.split()
    result = []

    for word in words:
        word = word.strip()
        if not word or len(word) < 2:
            continue
        if is_mongolian(word):
            result.append(strip_suffix(word.lower()))
        else:
            result.append(word.lower())

    return " ".join(result)