
"""Build the local vector and keyword indexes for The Librarian."""

import json
import re
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load optional settings from the project's .env file.
load_dotenv()

# Input chunks created by src/chunking.py.
CHUNKS_PATH = Path("data/chunks.jsonl")

# Chroma stores vectors and their metadata in this directory.
CHROMA_PATH = Path("storage/chroma")

# BM25 uses this JSON file to reconstruct its keyword index.
BM25_PATH = Path("storage/bm25_corpus.json")

# A small, local embedding model suitable for a first experiment.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

COLLECTION_NAME = "smu_documents"
BATCH_SIZE = 32


def load_chunks() -> list[dict]:
    """Read and validate the chunk records."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks not found: {CHUNKS_PATH}. "
            "Run python -m src.chunking first."
        )

    chunks = []

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            if not chunk.get("chunk_id") or not chunk.get("text", "").strip():
                raise ValueError(
                    f"Missing chunk ID or text on line {line_number}."
                )

            if not isinstance(chunk.get("metadata"), dict):
                raise ValueError(
                    f"Missing metadata on line {line_number}."
                )

            chunks.append(chunk)

    if not chunks:
        raise ValueError("No valid chunks were found.")

    # IDs must be unique because they identify individual passages.
    ids = [chunk["chunk_id"] for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs found.")

    return chunks


def keyword_tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens while preserving codes.

    For example, 'CS496' remains one token, helping exact course-code
    searches. This simple tokenizer will be evaluated later.
    """
    return re.findall(r"[^\W_]+(?:[-./][^\W_]+)*", text.casefold())


def clean_metadata(metadata: dict) -> dict:
    """Keep only metadata values supported by Chroma."""
    cleaned = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value

    return cleaned


def build_vector_index(chunks: list[dict]) -> None:
    """Embed the chunks and save them in a persistent Chroma collection."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # The model downloads its weights on first use if not already cached.
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Rebuild from scratch so rerunning this script doesn't duplicate data.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        # The collection may not exist on the first run.
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "SMU public website chunks"},
    )

    total = len(chunks)

    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]

        texts = [item["text"] for item in batch]
        ids = [item["chunk_id"] for item in batch]
        metadatas = [
            clean_metadata(item["metadata"])
            for item in batch
        ]

        # Generate vectors locally; no paid embedding API is used.
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
        )

        print(f"Indexed vectors: {min(start + len(batch), total)}/{total}")

    print(f"Vector index saved to: {CHROMA_PATH}")
    print(f"Total vectors: {collection.count()}")


def build_keyword_corpus(chunks: list[dict]) -> None:
    """Save tokenized documents and metadata for BM25 retrieval.

    We store the tokenized corpus rather than a Python pickle. The BM25
    index itself can be reconstructed from this file at search time.
    """
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)

    records = []

    for chunk in chunks:
        records.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "tokens": keyword_tokenize(chunk["text"]),
                "metadata": chunk["metadata"],
            }
        )

    with BM25_PATH.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False)

    print(f"Keyword corpus saved to: {BM25_PATH}")
    print(f"Total keyword documents: {len(records)}")


def main():
    """Build both indexes from the same source chunks."""
    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks.")
    build_vector_index(chunks)
    build_keyword_corpus(chunks)

    print("\nBoth indexes were built successfully.")


if __name__ == "__main__":
    main()
