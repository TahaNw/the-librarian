
"""Hybrid retrieval using Chroma, BM25, RRF, and a cross-encoder reranker."""

import json
import os
import re
from functools import lru_cache
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()

BM25_PATH = Path("storage/bm25_corpus.json")
CHROMA_PATH = Path("storage/chroma")
COLLECTION_NAME = "smu_documents"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

RRF_K = 60
KEYWORD_BOOST = 0.02

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "does", "do", "did", "what", "which", "when",
    "where", "how", "who", "why", "can", "may",
    "of", "on", "in", "to", "for", "and", "or",
    "by", "from", "with", "about", "mention",
    "mentioned", "website", "say", "apply",
}


def tokenize(text: str) -> list[str]:
    """Split text into searchable words."""
    return re.findall(r"[^\W_]+(?:[-./][^\W_]+)*", text.casefold())


def expand_query(question: str) -> str:
    """Add useful synonyms when wording differs."""
    expanded = question

    synonyms = {
        r"\blearning management system\b":
            "learning management system LMS Moodle",
        r"\blearning platform\b":
            "learning platform LMS Moodle",
    }

    for pattern, replacement in synonyms.items():
        expanded = re.sub(
            pattern,
            replacement,
            expanded,
            flags=re.IGNORECASE,
        )

    return expanded


@lru_cache(maxsize=1)
def load_keyword_corpus() -> list[dict]:
    """Load and cache the indexed chunk records."""
    if not BM25_PATH.exists():
        raise FileNotFoundError(
            f"{BM25_PATH} not found. Run: python -m src.indexing"
        )

    with BM25_PATH.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    if not documents:
        raise ValueError("The BM25 corpus is empty.")

    return documents


@lru_cache(maxsize=1)
def get_bm25() -> BM25Okapi:
    """Build and cache the BM25 keyword index."""
    documents = load_keyword_corpus()
    tokenized_documents = [
        doc.get("tokens") or tokenize(doc["text"])
        for doc in documents
    ]
    return BM25Okapi(tokenized_documents)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the embedding model."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Load and cache the cross-encoder reranker."""
    print(f"Loading reranker: {RERANKER_MODEL}")
    return CrossEncoder(RERANKER_MODEL)


@lru_cache(maxsize=1)
def get_collection():
    """Connect to and cache the Chroma collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception as error:
        raise RuntimeError(
            f"Could not open Chroma collection '{COLLECTION_NAME}' "
            f"at '{CHROMA_PATH}'. Run: python -m src.indexing"
        ) from error


def retrieve(question: str, top_k: int = 5) -> list[dict]:
    """Retrieve candidates using BM25 and vectors, fuse, then rerank."""
    if not question or not question.strip():
        raise ValueError("Please provide a non-empty question.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    question = question.strip()
    search_query = expand_query(question)
    documents = load_keyword_corpus()

    # Stage 1: BM25 keyword search using the expanded question.
    bm25 = get_bm25()
    keyword_scores = bm25.get_scores(tokenize(search_query))

    keyword_order = sorted(
        range(len(documents)),
        key=lambda index: keyword_scores[index],
        reverse=True,
    )

    # Give BM25 its own candidate limit.
    keyword_candidate_count = min(100, len(documents))

    # Stage 2: Vector search using the expanded question.
    collection = get_collection()
    collection_count = collection.count()

    if collection_count == 0:
        raise ValueError(
            "The Chroma vector collection is empty. "
            "Run: python -m src.indexing"
        )

    model = get_embedding_model()
    question_embedding = model.encode(
        search_query,
        normalize_embeddings=True,
    ).tolist()

    vector_candidate_count = min(
        max(top_k * 10, 30),
        len(documents),
        collection_count,
    )

    vector_results = collection.query(
        query_embeddings=[question_embedding],
        n_results=vector_candidate_count,
        include=["documents", "metadatas", "distances"],
    )

    docs_by_id = {doc["chunk_id"]: doc for doc in documents}
    fused_scores: dict[str, float] = {}
    result_details: dict[str, dict] = {}

    # Stage 3: Reciprocal Rank Fusion (RRF).
    for rank, index in enumerate(
        keyword_order[:keyword_candidate_count], start=1
    ):
        doc = documents[index]
        chunk_id = doc["chunk_id"]

        fused_scores[chunk_id] = (
            fused_scores.get(chunk_id, 0.0)
            + 1.0 / (RRF_K + rank)
        )

        result_details[chunk_id] = {
            "chunk_id": chunk_id,
            "text": doc["text"],
            "metadata": doc.get("metadata", {}),
            "bm25_rank": rank,
            "vector_rank": None,
            "bm25_score": float(keyword_scores[index]),
            "vector_distance": None,
        }

    vector_ids = vector_results["ids"][0]
    vector_texts = vector_results["documents"][0]
    vector_metadata = vector_results["metadatas"][0]
    vector_distances = vector_results["distances"][0]

    for rank, chunk_id in enumerate(vector_ids, start=1):
        fused_scores[chunk_id] = (
            fused_scores.get(chunk_id, 0.0)
            + 1.0 / (RRF_K + rank)
        )

        doc = docs_by_id.get(chunk_id)
        text = doc["text"] if doc else vector_texts[rank - 1]
        metadata = (
            doc.get("metadata", {})
            if doc
            else (vector_metadata[rank - 1] or {})
        )

        if chunk_id not in result_details:
            result_details[chunk_id] = {
                "chunk_id": chunk_id,
                "text": text,
                "metadata": metadata,
                "bm25_rank": None,
                "vector_rank": rank,
                "bm25_score": None,
                "vector_distance": float(vector_distances[rank - 1]),
            }
        else:
            result_details[chunk_id]["vector_rank"] = rank
            result_details[chunk_id]["vector_distance"] = float(
                vector_distances[rank - 1]
            )

    # Stage 4: Small keyword-overlap boost.
    query_terms = set(tokenize(search_query)) - STOP_WORDS

    for chunk_id in fused_scores:
        passage_terms = set(tokenize(result_details[chunk_id]["text"]))
        overlap = query_terms & passage_terms

        if overlap:
            fused_scores[chunk_id] += KEYWORD_BOOST * len(overlap)

    # Keep a broad candidate pool for reranking.
    ranked_ids = sorted(
        fused_scores,
        key=lambda chunk_id: fused_scores[chunk_id],
        reverse=True,
    )

    rerank_ids = ranked_ids[
        :min(max(top_k * 10, 30), len(ranked_ids))
    ]

    # Stage 5: Cross-encoder reranking.
    # IMPORTANT: Use the expanded query here as well.
    rerank_question = expand_query(question)

    reranker = get_reranker()
    pairs = [
        (rerank_question, result_details[chunk_id]["text"])
        for chunk_id in rerank_ids
    ]

    rerank_scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    results = []

    for chunk_id, score in zip(rerank_ids, rerank_scores):
        item = result_details[chunk_id].copy()
        item["hybrid_score"] = float(fused_scores[chunk_id])
        item["rerank_score"] = float(score)

        passage_terms = set(tokenize(item["text"]))
        overlap = query_terms & passage_terms

        item["keyword_overlap"] = len(overlap)
        item["final_score"] = (
            item["rerank_score"] + KEYWORD_BOOST * len(overlap)
        )
        results.append(item)

    results.sort(
        key=lambda item: item["final_score"],
        reverse=True,
    )

    return results[:top_k]


def main() -> None:
    """Run retrieval tests on representative questions."""
    questions = [
        "What is South Mediterranean University?",
        "What information does SMU provide about admissions and fees?",
        "Which student platforms are listed on the SMU website?",
        "What annual tuition fee increase does SMU say it may apply?",
        "What learning management system is mentioned on the SMU website?",
    ]

    for question in questions:
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")

        results = retrieve(question, top_k=3)

        for number, result in enumerate(results, start=1):
            metadata = result["metadata"]
            print(f"\n{number}. Chunk: {result['chunk_id']}")
            print(f"   Final score: {result['final_score']:.4f}")
            print(f"   Rerank score: {result['rerank_score']:.4f}")
            print(f"   Hybrid score: {result['hybrid_score']:.5f}")
            print(f"   BM25 rank: {result['bm25_rank']}")
            print(f"   Vector rank: {result['vector_rank']}")
            print(f"   Title: {metadata.get('title', 'Unknown title')}")
            print(f"   Source: {metadata.get('source_url', 'Unknown source')}")
            print(f"   Passage: {result['text'][:500]}...")


if __name__ == "__main__":
    main()
