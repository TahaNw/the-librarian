
"""Split collected university web pages into overlapping chunks."""

import json
import re
from pathlib import Path

# Input: pages collected by src/ingest_web.py
INPUT_PATH = Path("data/smu_public_pages.jsonl")

# Output: passages ready for embedding and keyword indexing
OUTPUT_PATH = Path("data/chunks.jsonl")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MIN_PAGE_LENGTH = 500


def normalise_for_deduplication(text: str) -> str:
    """Normalise whitespace for exact duplicate detection."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def split_text(text: str) -> list[str]:
    """Split text into chunks with a configurable overlap.

    Prefer ending a chunk at a sentence or word boundary
    when one is available near the target size.
    """
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        # Try to avoid cutting a sentence in half.
        if end < len(text):
            boundary = text.rfind(". ", start, end)

            # Use the sentence boundary if it is reasonably close
            # to the end of the proposed chunk.
            if boundary > start + CHUNK_SIZE // 2:
                end = boundary + 1
            else:
                # Otherwise, at least try to preserve whole words.
                space = text.rfind(" ", start, end)
                if space > start:
                    end = space

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        # Move forward while retaining some context from the
        # previous chunk.
        next_start = max(0, end - CHUNK_OVERLAP)

        # Avoid repeatedly selecting the same position.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def load_pages() -> list[dict]:
    """Read valid page records from the collected JSONL corpus."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found: {INPUT_PATH}. "
            "Run python -m src.ingest_web first."
        )

    pages = []

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {line_number}.")
                continue

            text = record.get("text", "").strip()
            metadata = record.get("metadata", {})

            if len(text) < MIN_PAGE_LENGTH:
                title = metadata.get("title", "Unknown page")
                print(f"Skipping short page: {title}")
                continue

            if not metadata.get("source_url"):
                print("Skipping page without a source URL.")
                continue

            pages.append(record)

    return pages


def build_chunks(pages: list[dict]) -> list[dict]:
    """Create chunks and retain their source metadata."""
    all_chunks = []
    seen_chunks = set()

    for page in pages:
        metadata = page["metadata"]
        page_chunks = split_text(page["text"])
        added_from_page = 0

        for index, chunk_text in enumerate(page_chunks):
            # Skip exact duplicate text, while preserving the
            # first occurrence and its source citation.
            normalised = normalise_for_deduplication(chunk_text)

            if normalised in seen_chunks:
                continue

            seen_chunks.add(normalised)

            all_chunks.append(
                {
                    "chunk_id": f"chunk_{len(all_chunks) + 1:06d}",
                    "text": chunk_text,
                    "metadata": {
                        "title": metadata.get("title", "Unknown"),
                        "source_url": metadata["source_url"],
                        "retrieved_at": metadata.get("retrieved_at"),
                        "chunk_index": index,
                    },
                }
            )

            added_from_page += 1

        print(
            f"{metadata.get('title', 'Unknown')}: "
            f"{added_from_page} unique chunks"
        )

    return all_chunks


def main():
    """Load pages, create chunks, and save them as JSONL."""
    pages = load_pages()
    chunks = build_chunks(pages)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nPages processed: {len(pages)}")
    print(f"Unique chunks: {len(chunks)}")
    print(f"Saved to: {OUTPUT_PATH}")

    if chunks:
        print("\nFirst chunk preview:")
        print(chunks[0]["metadata"])
        print(chunks[0]["text"][:300])


if __name__ == "__main__":
    main()
