
"""Check the quality and contents of the collected SMU corpus."""

import json
from collections import Counter
from pathlib import Path

CORPUS_PATH = Path("data/smu_public_pages.jsonl")
MIN_TEXT_LENGTH = 500
PREVIEW_COUNT = 10


def main():
    if not CORPUS_PATH.exists():
        print(f"Corpus not found: {CORPUS_PATH}")
        return

    pages = []
    invalid_lines = 0
    empty_pages = 0
    short_pages = 0

    with CORPUS_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Invalid JSON on line {line_number}: {exc}")
                invalid_lines += 1
                continue

            text = record.get("text", "")
            metadata = record.get("metadata", {})

            if not isinstance(text, str) or not text.strip():
                print(f"Warning: line {line_number} has empty or invalid text.")
                empty_pages += 1
                continue

            if not isinstance(metadata, dict):
                print(f"Warning: line {line_number} has invalid metadata.")
                metadata = {}
                record["metadata"] = metadata

            if len(text.strip()) < MIN_TEXT_LENGTH:
                short_pages += 1

            pages.append(record)

    urls = [
        page.get("metadata", {}).get("source_url", "").strip()
        for page in pages
    ]
    url_counts = Counter(url for url in urls if url)
    duplicate_urls = {
        url: count for url, count in url_counts.items() if count > 1
    }

    content_keys = [
        " ".join(page["text"].casefold().split())
        for page in pages
    ]
    content_counts = Counter(content_keys)
    duplicate_content_groups = sum(
        1 for count in content_counts.values() if count > 1
    )

    missing_titles = sum(
        not page.get("metadata", {}).get("title", "").strip()
        for page in pages
    )
    missing_urls = sum(
        not page.get("metadata", {}).get("source_url", "").strip()
        for page in pages
    )

    total_chars = sum(len(page["text"]) for page in pages)

    print("\n=== SMU CORPUS QUALITY REPORT ===")
    print(f"Valid pages:              {len(pages)}")
    print(f"Total characters:         {total_chars:,}")
    print(f"Average characters/page:  {total_chars // len(pages) if pages else 0:,}")
    print(f"Invalid JSON lines:       {invalid_lines}")
    print(f"Empty pages skipped:      {empty_pages}")
    print(f"Short pages (<500 chars): {short_pages}")
    print(f"Duplicate URLs:           {len(duplicate_urls)}")
    print(f"Duplicate content groups: {duplicate_content_groups}")
    print(f"Missing titles:           {missing_titles}")
    print(f"Missing source URLs:      {missing_urls}")

    print(f"\n=== FIRST {min(PREVIEW_COUNT, len(pages))} PAGES ===")
    for index, page in enumerate(pages[:PREVIEW_COUNT], start=1):
        metadata = page.get("metadata", {})
        print(f"\n--- Page {index} ---")
        print(f"Title: {metadata.get('title') or 'Unknown'}")
        print(f"URL: {metadata.get('source_url') or 'Unknown'}")
        print(f"Characters: {len(page['text']):,}")
        print(f"Preview: {page['text'][:250]}")

    if duplicate_urls:
        print("\n=== DUPLICATE URLS ===")
        for url, count in duplicate_urls.items():
            print(f"{count} records: {url}")

    if duplicate_content_groups:
        print(
            "\nNote: identical-content groups are counted, "
            "not individual duplicate records."
        )


if __name__ == "__main__":
    main()
