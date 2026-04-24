"""
Монгол асуултын дүн шинжилгээ.
Асуултын төрлийг тодорхойлж, түлхүүр үгсийг олно.
"""
import re
from mongolian_utils import is_mongolian, strip_suffix

MN_STOP_WORDS = {
    "вэ", "бэ", "бол", "нь", "юм", "байна", "байгаа",
    "гэвэл", "гэх", "мэт", "болон", "дээр", "доор",
    "энэ", "тэр", "өөр", "бас", "ч", "л", "рүү", "руу",
    "болж", "байж", "гэж",
}

QUESTION_WORDS = {
    "хэн":    {"type": "person",    "hint": "хэн хүн нэр"},
    "хэзээ":  {"type": "date",      "hint": "хэзээ он жил огноо"},
    "хаана":  {"type": "location",  "hint": "хаана газар улс хот"},
    "юу":     {"type": "thing",     "hint": "юу тодорхойлолт утга"},
    "яаж":    {"type": "method",    "hint": "яаж арга алхам"},
    "хэрхэн": {"type": "method",    "hint": "яаж арга алхам"},
    "яагаад": {"type": "reason",    "hint": "яагаад учир шалтгаан"},
    "ямар":   {"type": "attribute", "hint": "ямар төрөл шинж"},
    "хэд":    {"type": "number",    "hint": "хэд тоо хэмжээ"},
    "хэдэн":  {"type": "number",    "hint": "хэд тоо хэмжээ"},
}

TYPE_INSTRUCTIONS = {
    "person":    "Хүний нэрийг тодорхой дурдаж хариул.",
    "date":      "Он сар өдрийг тодорхой дурдаж хариул.",
    "location":  "Газрын нэрийг тодорхой дурдаж хариул.",
    "thing":     "Тодорхойлолт өгч хариул.",
    "method":    "Алхам алхмаар тайлбарлаж хариул.",
    "reason":    "Шалтгааныг тайлбарлаж хариул.",
    "attribute": "Шинж чанарыг тодорхой дурдаж хариул.",
    "number":    "Тоог тодорхой дурдаж хариул.",
}

def analyze_question(text: str) -> dict:
    clean = re.sub(r'[?.!,;]', ' ', text.lower()).strip()
    words = clean.split()

    q_types = []
    q_hints = []
    for word in words:
        if word in QUESTION_WORDS:
            info = QUESTION_WORDS[word]
            if info["type"] not in q_types:
                q_types.append(info["type"])
                q_hints.append(info["hint"])

    nih_keywords = []
    for i, word in enumerate(words):
        if word == "нь" and i > 0:
            for j in range(i):
                w = words[j]
                if w in MN_STOP_WORDS or w in QUESTION_WORDS:
                    continue
                if w in ("энэ", "тэр", "энэхүү", "тэрхүү"):
                    continue
                if len(w) < 2:
                    continue
                root = strip_suffix(w) if is_mongolian(w) else w
                if root not in nih_keywords:
                    nih_keywords.append(root)
    gej_keywords = []
    for i, word in enumerate(words):
        if word == "гэж" and i > 0:
            prev = words[i - 1]
            root = strip_suffix(prev) if is_mongolian(prev) else prev
            if root not in nih_keywords:
                gej_keywords.append(root)

    q_word_set = set(QUESTION_WORDS.keys())
    skip_words = MN_STOP_WORDS | {"нь", "гэж", "энэ", "тэр", "энэхүү", "тэрхүү"}
    keywords = list(nih_keywords) + list(gej_keywords)

    for word in words:
        if word in skip_words:
            continue
        if word in q_word_set:
            continue
        if len(word) < 2:
            continue
        if not is_mongolian(word):
            if word not in keywords:
                keywords.append(word)
        else:
            root = strip_suffix(word)
            if root not in keywords:
                keywords.append(root)

    search_query = " ".join(keywords)

    instruction = " ".join(
        TYPE_INSTRUCTIONS.get(qt, "") for qt in q_types
    ) or "Хариултыг тодорхой, товч өгч хариул."

    return {
        "keywords":      keywords,
        "question_type": q_types,
        "search_query":  search_query,
        "hints":         q_hints,
        "instruction":   instruction,
        "original":      text,
    }