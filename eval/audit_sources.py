import json
import re
from pathlib import Path
from collections import Counter

CORPUS = Path("data/smu_public_pages.jsonl")
QUESTIONS = Path("eval/questions.jsonl")

STOP = set("""
the a an and or is are was were what when which how do does did to of in on
for at by with from that this it its under include including about as be has
have university universitys mention states
""".split())

def tokens(text):
    return {
        t for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) > 2 and t not in STOP
    }

pages = []
with CORPUS.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            m = r.get("metadata", {})
            pages.append({
                "title": m.get("title", "NO TITLE"),
                "url": m.get("source_url", "NO URL"),
                "text": r.get("text", ""),
                "tokens": tokens(r.get("text", "")),
            })

questions = [
    json.loads(line)
    for line in QUESTIONS.open(encoding="utf-8")
    if line.strip()
]

print(f"Pages: {len(pages)} | Questions: {len(questions)}")
print("Keyword overlap is a candidate finder, not proof that a source supports an answer.")

for q in questions:
    if not q.get("answerable"):
        continue

    target = tokens(q["question"] + " " + q["reference_answer"])
    ranked = []
    for p in pages:
        overlap = target & p["tokens"]
        score = len(overlap) / max(len(target), 1)
        ranked.append((score, len(overlap), p))

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)

    print(f"\n{q['id']}: {q['question']}")
    print("Expected:", ", ".join(q.get("supporting_urls", [])))
    for score, count, p in ranked[:3]:
        print(f"  score={score:.2f} matched_terms={count} | {p['title']} | {p['url']}")
