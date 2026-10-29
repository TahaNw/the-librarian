
import json
from pathlib import Path

path = Path("storage/bm25_corpus.json")

with path.open("r", encoding="utf-8") as file:
    documents = json.load(file)

terms = ["Moodle", "5%", "increase", "taxation policy"]

for term in terms:
    print("\n" + "=" * 70)
    print(f"SEARCH TERM: {term}")
    print("=" * 70)

    matches = [
        doc for doc in documents
        if term.casefold() in doc.get("text", "").casefold()
    ]

    print(f"Matching chunks: {len(matches)}")

    for doc in matches[:5]:
        metadata = doc.get("metadata", {})
        print("\nChunk:", doc.get("chunk_id", "unknown"))
        print("Title:", metadata.get("title", "Untitled"))
        print("URL:", metadata.get("source_url", "No URL"))
        text = doc.get("text", "")
        position = text.casefold().find(term.casefold())
        start = max(0, position - 250)
        end = min(len(text), position + 500)
        print("Excerpt:", text[start:end])
