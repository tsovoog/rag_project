import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

# ─────────────────────────────────────────────
#  Prompt template: Монгол болон Англи хэлийг
#  хоёуланг нь дэмжих, зөв өгүүлбэр зүйтэй
#  хариулт өгөхийг тодорхой зааж өгсөн.
# ─────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the context below.
STRICT RULES:
- Answer ONLY using facts from the context. Do NOT make up stories or add extra information.
- If the question is in Mongolian, answer in Mongolian language only.
- If the question is in English, answer in English only.
- Keep the answer short and factual — 1 to 3 sentences maximum.
- Do NOT write stories, chapters, or fictional content.
- If the answer is not in the context, say: "Контекстэд мэдээлэл байхгүй байна."

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
    """
    Асуултын хэлийг тодорхойлно.
    Монгол кирилл тэмдэгт байвал 'mn', үгүй бол 'en' буцаана.
    """
    mongol_chars = set("абвгдеёжзийклмноөпрстуүфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯ")
    mongol_count = sum(1 for ch in text if ch in mongol_chars)
    return "mn" if mongol_count > len(text) * 0.2 else "en"


def query_rag(query_text: str, show_sources: bool = False) -> str:
    """RAG хайлт хийж, TinyLlama ашиглан хариулт гаргана."""

    print(f"\nАсуулт: {query_text}")
    lang = detect_language(query_text)
    print(f"Илэрсэн хэл: {'Монгол' if lang == 'mn' else 'Англи'}\n")

    # 1. ChromaDB-с хамгийн ойр 5 chunk авна
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)

    if not results:
        msg = "Мэдээллийн сангаас холбогдох мэдээлэл олдсонгүй." if lang == "mn" \
              else "No relevant information found in the database."
        print(msg)
        return msg

    # 2. Контекст бүрдүүлэх
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # 3. Prompt үүсгэх
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # 4. TinyLlama загвар ашиглан хариулт авах
    #    TinyLlama нь хурдан, CPU дээр ажиллах боломжтой SLM загвар.
    #    Ollama-д: `ollama pull tinyllama` гэж татаж авна.
    model = OllamaLLM(
        model="qwen2.5:1.5b",
        temperature=0.0,        # Бүрэн тогтвортой, давтагдах хариулт
        num_predict=200,        # Богино хариулт — урт үлгэр зохиохоос сэргийлнэ
    )

    print("Хариулт боловсруулж байна...")
    response_text = model.invoke(prompt)

    # 5. Хариулт гаргах
    print("\n" + "=" * 60)
    print("ХАРИУЛТ:")
    print("=" * 60)
    print(response_text.strip())

    if show_sources:
        print("\n" + "-" * 60)
        print("ЭХ СУРВАЛЖ:")
        sources = [doc.metadata.get("id", "?") for doc, _score in results]
        for i, src in enumerate(sources, 1):
            score = results[i - 1][1]
            print(f"  {i}. {src}  (ойролцоо байдал: {score:.4f})")

    print("=" * 60 + "\n")
    return response_text


if __name__ == "__main__":
    main()