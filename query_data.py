import argparse
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function
from question_analyzer import analyze_question

CHROMA_PATH = "chroma"
SCORE_THRESHOLD = 1.0

NOT_FOUND_MSG = "Таны оруулсан файлд асуултын хариулт байхгүй байна. Материалын агуулгатай тохирох асуулт асууна уу!"


def detect_language(text: str) -> str:
    mn_chars = set("абвгдеёжзийклмноөпрстуүфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯ")
    count = sum(1 for ch in text if ch in mn_chars)
    return "mn" if count > len(text) * 0.2 else "en"


def build_prompt(context: str, question: str, instruction: str) -> str:
    return f"""Та туслах AI систем юм. Зөвхөн доорх контекстоос хариул.
ДҮРЭМ:
- Зөвхөн контекстэд байгаа мэдээллийг ашигла.
- Заавал Монгол хэлээр хариул.
- Хамгийн ихдээ 3 өгүүлбэр.
- {instruction}
- Контекстэд хариулт байхгүй бол яг ингэж хариул:
  "Таны оруулсан файлд асуултын хариулт байхгүй байна."
- Мэдэхгүй зүйлийг зохиож хариулах хэрэггүй.

КОНТЕКСТ:
{context}

---

АСУУЛТ: {question}"""


def query_rag(query_text: str, show_sources: bool = False) -> str:

    print(f"\nАсуулт: {query_text}")
    lang = detect_language(query_text)
    print(f"Хэл: {'Монгол' if lang == 'mn' else 'Англи'}")

    # 1. Асуултын дүн шинжилгээ
    analysis = analyze_question(query_text)
    print(f"Төрөл:   {analysis['question_type']}")
    print(f"Keyword: {analysis['keywords']}")
    print(f"Хайлт:   {analysis['search_query']}\n")

    # 2. ChromaDB хайлт
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function,
    )
    results = db.similarity_search_with_score(analysis["search_query"], k=5)

    if not results:
        _print_answer(NOT_FOUND_MSG)
        return NOT_FOUND_MSG

    # 3. Score шүүлт
    filtered = [(doc, score) for doc, score in results if score < SCORE_THRESHOLD]

    if not filtered:
        _print_answer(NOT_FOUND_MSG)
        return NOT_FOUND_MSG

    # 4. Хоёр шатлалтай keyword шалгалт
    keywords = analysis["keywords"]
    if keywords:
        from mongolian_utils import normalize_mongolian

        def find_best_chunks(candidates):
            """Хоёр keyword хоёулаа таарсан chunk-уудыг буцаана."""
            best = []
            for doc, score in candidates:
                norm = normalize_mongolian(doc.page_content)
                match_count = sum(1 for kw in keywords if kw in norm)
                if match_count == len(keywords):
                    best.append((doc, score))
            return best
        best_chunks = find_best_chunks(filtered)

        if not best_chunks:
            hint_query = analysis["search_query"] + " " + " ".join(
                analysis.get("hints", [])
            )
            results2 = db.similarity_search_with_score(hint_query, k=5)
            filtered2 = [(doc, score) for doc, score in results2 if score < SCORE_THRESHOLD]
            best_chunks = find_best_chunks(filtered2)

        if not best_chunks:
            scored = []
            for doc, score in filtered:
                norm = normalize_mongolian(doc.page_content)
                match_count = sum(1 for kw in keywords if kw in norm)
                if match_count > 0:
                    scored.append((doc, score, match_count))
            scored.sort(key=lambda x: x[2], reverse=True)
            best_chunks = [(doc, score) for doc, score, _ in scored]

        if not best_chunks:
            _print_answer(NOT_FOUND_MSG)
            return NOT_FOUND_MSG

        filtered = best_chunks
    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _ in filtered]
    )

    prompt = build_prompt(context_text, query_text, analysis["instruction"])
    model = OllamaLLM(
        model="qwen2.5:1.5b",
        temperature=0.0,
        num_predict=300,
    )

    print("Хариулт боловсруулж байна...")
    response_text = model.invoke(prompt)

    _print_answer(response_text.strip())
    if show_sources:
        print("-" * 60)
        print("ЭХ СУРВАЛЖ:")
        for i, (doc, score) in enumerate(filtered, 1):
            print(f"  {i}. {doc.metadata.get('id', '?')}  (score: {score:.4f})")
        print("=" * 60 + "\n")

    return response_text

def _print_answer(text: str):
    print("\n" + "=" * 60)
    print("ХАРИУЛТ:")
    print("=" * 60)
    print(text)
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="RAG дээр суурилсан оюутны туслах"
    )
    parser.add_argument("query_text", type=str, help="Асуулт бичнэ үү.")
    parser.add_argument(
        "--show-sources", action="store_true", help="Эх сурвалжийг харуулах."
    )
    args = parser.parse_args()
    query_text: str = args.query_text
    show_sources: bool = args.show_sources
    query_rag(query_text, show_sources=show_sources)


if __name__ == "__main__":
    main()