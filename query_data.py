import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

# Similarity score-ийн босго утга
# ChromaDB distance: 0 = яг таарсан, 2 = огт таарахгүй
# 1.0-ээс дээш бол материалд хамааны гүй гэж үзнэ
SCORE_THRESHOLD = 1.0

PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the context below.
STRICT RULES:
- Answer ONLY using facts from the context. Do NOT make up anything.
- If the question is in Mongolian, answer in Mongolian language only.
- If the question is in English, answer in English only.
- Keep the answer short and factual — 1 to 3 sentences maximum.
- Do NOT write stories, chapters, or fictional content.
- If the answer is not clearly stated in the context, respond with exactly: "Таны асуултад тохирох мэдээлэл материалаас олдсонгүй."

CONTEXT:
{context}

---

QUESTION: {question}

ANSWER (short and factual only):"""


def main():
    parser = argparse.ArgumentParser(
        description="RAG тогтолцоо дээр суурилсан оюутны туслах CLI"
    )
    parser.add_argument("query_text", type=str, help="Асуулт бичнэ үү.")
    parser.add_argument(
        "--show-sources", action="store_true", help="Эх сурвалжийг харуулах."
    )
    args = parser.parse_args()
    query_rag(args.query_text, show_sources=args.show_sources)


def detect_language(text: str) -> str:
    """Монгол кирилл тэмдэгт байвал 'mn', үгүй бол 'en' буцаана."""
    mongol_chars = set("абвгдеёжзийклмноөпрстуүфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯ")
    mongol_count = sum(1 for ch in text if ch in mongol_chars)
    return "mn" if mongol_count > len(text) * 0.2 else "en"


def query_rag(query_text: str, show_sources: bool = False) -> str:
    """RAG хайлт хийж хариулт гаргана."""

    print(f"\nАсуулт: {query_text}")
    lang = detect_language(query_text)
    print(f"Илэрсэн хэл: {'Монгол' if lang == 'mn' else 'Англи'}\n")

    # 1. ChromaDB-с хамгийн ойр 5 chunk авна
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)

    if not results:
        _print_answer("Мэдээллийн сангаас мэдээлэл олдсонгүй.")
        return ""

    # 2. Score шалгах — хэт холын, хамааны гүй chunk-уудыг шүүх
    filtered = [(doc, score) for doc, score in results if score < SCORE_THRESHOLD]

    if not filtered:
        msg = "Таны асуултад тохирох мэдээлэл материалаас олдсонгүй." if lang == "mn" \
              else "No relevant information found in the uploaded materials."
        _print_answer(msg)
        return msg

    # 3. Контекст бүрдүүлэх
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in filtered])

    # 4. Prompt үүсгэх
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # 5. Загвар ашиглан хариулт авах
    model = OllamaLLM(
        model="qwen2.5:1.5b",
        temperature=0.0,    # Тогтвортой хариулт
        num_predict=300,    # Богино хариулт
    )

    print("Хариулт боловсруулж байна...")
    response_text = model.invoke(prompt)

    # 6. Хариулт гаргах
    _print_answer(response_text.strip())

    if show_sources:
        print("-" * 60)
        print("ЭХ СУРВАЛЖ:")
        for i, (doc, score) in enumerate(filtered, 1):
            src = doc.metadata.get("id", "?")
            print(f"  {i}. {src}  (score: {score:.4f})")
        print("=" * 60 + "\n")

    return response_text


def _print_answer(text: str):
    print("\n" + "=" * 60)
    print("ХАРИУЛТ:")
    print("=" * 60)
    print(text)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()