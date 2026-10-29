
from src.retrieval import load_keyword_corpus, get_bm25, tokenize

question = "What learning management system is mentioned on the SMU website?"
documents = load_keyword_corpus()
bm25 = get_bm25()

scores = bm25.get_scores(tokenize(question))

ranked = sorted(
    range(len(documents)),
    key=lambda i: scores[i],
    reverse=True,
)

for rank, i in enumerate(ranked[:15], start=1):
    doc = documents[i]
    print(f"\n{rank}. {doc['chunk_id']} | BM25 score: {scores[i]:.4f}")
    print(doc["text"][:350])

target = next(
    (i for i, doc in enumerate(documents)
     if doc["chunk_id"] == "chunk_000033"),
    None,
)

if target is not None:
    target_rank = ranked.index(target) + 1
    print(f"\nMoodle passage rank: {target_rank}")
else:
    print("\nchunk_000033 was not found in the BM25 corpus.")
