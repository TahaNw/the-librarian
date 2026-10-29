import json
import re

terms = {
    "q002 founding date": ["founded", "2002"],
    "q004 tuition fees": ["tuition", "fees"],
    "q005 VAT": ["VAT", "7%"],
    "q006 fee increase": ["increase", "5%"],
    "q007 payment methods": ["payment", "cheque", "transfer"],
    "q008 learning platform": ["Moodle", "learning management"],
}

pages = {}
with open("data/smu_public_pages.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            m = r.get("metadata", {})
            pages[m.get("source_url", "")] = r.get("text", "")

faq_url = "https://www.smu.tn/smu/faq"
story_url = "https://www.smu.tn/smu/about-smu/our-story"
homepage_url = "https://www.smu.tn/smu"

for label, keywords in terms.items():
    print(f"\n--- {label} ---")
    candidates = [faq_url]
    if label == "q002 founding date":
        candidates += [story_url, homepage_url]
    for url in candidates:
        text = pages.get(url, "")
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        matches = [
            s.strip() for s in sentences
            if any(k.lower() in s.lower() for k in keywords)
        ]
        print(f"\nSOURCE: {url}")
        if matches:
            for s in matches[:8]:
                print("-", s[:700])
        else:
            print("(No matching sentences found)")
