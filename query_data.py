import argparse
import re
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function
from question_analyzer import analyze_question
from mongolian_utils import normalize_mongolian

CHROMA_PATH = "chroma"
SCORE_THRESHOLD = 1.0
RERANK_THRESHOLD = 3.0
BATCH_SIZE = 5
MAX_BATCHES = 2

NOT_FOUND_MSG = "Таны оруулсан файлд асуултын хариулт байхгүй байна. Материалын агуулгатай тохирох асуулт асууна уу!"


def detect_language(text: str) -> str:
    mn_chars = set("абвгдеёжзийклмноөпрстуүфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯ")
    count = sum(1 for ch in text if ch in mn_chars)
    return "mn" if count > len(text) * 0.2 else "en"


def cross_encoder_rerank(candidates: list, question: str, keywords: list, model: OllamaLLM) -> list:
    """
    candidates: [(doc, emb_score, mc)]
    Буцаах:     [(doc, emb_score, mc, rerank_score)]
    """
    reranked = []

    for doc, emb_score, mc in candidates:
        if mc == 0:
            reranked.append((doc, emb_score, mc, 0.0))
            continue

        prompt = f"""Дараах текст нь асуултад хариулахад хэр хамааралтай байна?
0-10 оноо өг. Зөвхөн тоо бичнэ үү.

10 = Асуултад яг шууд хариулсан мэдээлэл агуулна
7-9 = Асуултад холбогдох мэдээлэл агуулна
4-6 = Хэсэгчлэн хамааралтай
1-3 = Бага хамааралтай
0   = Огт хамааралгүй

ТЕКСТ:
{doc.page_content[:500]}

АСУУЛТ: {question}

ОНОО (зөвхөн 0-10 тоо):"""

        try:
            scores = []
            for _ in range(2):
                result = model.invoke(prompt).strip()
                numbers = re.findall(r'\b([0-9]|10)\b', result)
                scores.append(float(numbers[0]) if numbers else 0.0)
            rerank_score = sum(scores) / len(scores)
        except Exception:
            rerank_score = 0.0

        reranked.append((doc, emb_score, mc, rerank_score))

    return reranked


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

    analysis = analyze_question(query_text)
    keywords = analysis["keywords"]

    model = OllamaLLM(
        model="qwen2.5:1.5b",
        temperature=0.0,
        num_predict=300,
    )

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )

    all_candidates = []

    for batch_num in range(MAX_BATCHES):
        k = BATCH_SIZE * (batch_num + 1)
        results = db.similarity_search_with_score(analysis["search_query"], k=k)
        batch = results[batch_num * BATCH_SIZE: (batch_num + 1) * BATCH_SIZE]
        filtered = [(doc, score) for doc, score in batch if score < SCORE_THRESHOLD]

       # print(f"\n--- Batch {batch_num + 1} (k={k}) ---")
        for doc, score in batch:
            norm = normalize_mongolian(doc.page_content)
            mc = sum(1 for kw in keywords if kw in norm)
            #(f"  score={score:.4f} | kw={mc}/{len(keywords)} | {doc.page_content[:60]}...")

        if not filtered:
            continue
        keyword_scored = []
        for doc, score in filtered:
            norm = normalize_mongolian(doc.page_content)
            mc = sum(1 for kw in keywords if kw in norm)
            if mc > 0:
                keyword_scored.append((doc, score, mc))

        if not keyword_scored:
         #   print("  → Keyword таарсан chunk олдсонгүй.")
            continue

       # print(f"  → Keyword таарсан: {len(keyword_scored)} chunk")
        reranked = cross_encoder_rerank(keyword_scored, query_text, keywords, model)
        top_chunks = []
        for doc, score, mc, rs in reranked:
            if rs < RERANK_THRESHOLD:
                continue
            kw_ratio = mc / len(keywords) if keywords else 0
            combined = (((1 - score)*1) + (kw_ratio)*1 + ((rs / 10)*0.5))/2.5
            top_chunks.append((doc, score, mc, rs, combined))

        if not top_chunks:
         #   print(f"  → Rerank threshold ({RERANK_THRESHOLD}) хэтрэхгүй chunk байсангүй.")
            continue

        top_chunks.sort(key=lambda x: x[4], reverse=True)
        best = top_chunks[0]
       # print(f"  → Хамгийн сайн: rerank={best[3]:.0f}/10, combined={best[4]:.3f}")
        all_candidates.append(best)

    if not all_candidates:
        _print_answer(NOT_FOUND_MSG)
        return NOT_FOUND_MSG

    all_candidates.sort(key=lambda x: x[4], reverse=True)
    best_doc, best_emb, best_mc, best_rs, best_combined = all_candidates[0]

   # print(f"\n→ Хамгийн сайн chunk: rerank={best_rs:.0f}/10, combined={best_combined:.3f}")
   # print(f"  {best_doc.page_content[:100]}...")

    context_text = best_doc.page_content
    prompt = build_prompt(context_text, query_text, analysis["instruction"])
    response_text = model.invoke(prompt)

    _print_answer(response_text.strip())

    if show_sources:
        print("-" * 60)
        print("ЭХ СУРВАЛЖ:")
        print(f"  {best_doc.metadata.get('id', '?')}  (score: {best_emb:.4f})")
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
        description="RAG тогтолцоо дээр суурилсан оюутны туслах"
    )
    parser.add_argument("query_text", type=str, help="Асуулт бичнэ үү.")
    parser.add_argument(
        "--show-sources", action="store_true", help="Эх сурвалжийг харуулах."
    )
    args = parser.parse_args()
    query_rag(args.query_text, show_sources=args.show_sources)


if __name__ == "__main__":
    main()