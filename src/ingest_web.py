"""Crawl public SMU webpages and save readable content as JSONL."""

import json
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import (
    parse_qsl,
    urlencode,
    urljoin,
    urlparse,
    urlunparse,
)

import httpx
from bs4 import BeautifulSoup


ALLOWED_DOMAINS = {
    "www.smu.tn",
    "smu.tn",
    "programs.smu.tn",
}

START_URLS = [
    "https://www.smu.tn/smu",
    "https://www.smu.tn/smu/faq",
    "https://www.smu.tn/smu/student",
    "https://programs.smu.tn/",
]

OUTPUT_PATH = Path("data/smu_public_pages.jsonl")

# Conservative crawl limits.
MAX_PAGES = 200
MAX_DEPTH = 3
MIN_TEXT_LENGTH = 500
MAX_TEXT_LENGTH = 25000
REQUEST_DELAY_SECONDS = 0.8

USER_AGENT = "TheLibrarianStudentProject/1.0"

# Avoid pages likely to be login portals, private systems, or irrelevant.
BLOCKED_PATH_TERMS = {
    "login",
    "log-in",
    "signin",
    "sign-in",
    "logout",
    "log-out",
    "register",
    "my-info",
    "myinfo",
    "portal",
    "moodle",
    "outlook",
    "onedrive",
    "teams",
    "webmail",
    "account",
    "password",
    "auth",
    "sso",
    "dashboard",
    "faculty-list",
    "testimonials",
}

NON_HTML_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".ppt", ".pptx", ".zip", ".rar", ".png",
    ".jpg", ".jpeg", ".gif", ".svg", ".webp",
    ".mp4", ".mp3", ".css", ".js", ".xml",
}


def clean_text(text: str) -> str:
    """Normalize whitespace without removing meaningful punctuation."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_url(url: str) -> str | None:
    """Normalize a URL and reject URLs outside the approved public scope."""
    try:
        parsed = urlparse(url)

        if parsed.scheme.lower() != "https":
            return None

        hostname = (parsed.hostname or "").lower()

        if hostname not in ALLOWED_DOMAINS:
            return None

        if parsed.username or parsed.password:
            return None

        path = re.sub(r"/+", "/", parsed.path or "/")
        path_lower = path.lower()

        if any(term in path_lower for term in BLOCKED_PATH_TERMS):
            return None

        suffix = Path(path_lower).suffix
        if suffix in NON_HTML_EXTENSIONS:
            return None

        # Remove common tracking parameters.
        tracking_prefixes = ("utm_",)
        tracking_names = {
            "fbclid", "gclid", "mc_cid", "mc_eid",
        }

        query_pairs = [
            (key, value)
            for key, value in parse_qsl(
                parsed.query, keep_blank_values=True
            )
            if key.lower() not in tracking_names
            and not key.lower().startswith(tracking_prefixes)
        ]

        query = urlencode(query_pairs, doseq=True)

        # Fragments do not identify separate server-side pages.
        return urlunparse((
            "https",
            hostname,
            path,
            "",
            query,
            "",
        ))

    except (ValueError, TypeError):
        return None


def extract_page_text(soup: BeautifulSoup) -> tuple[str, str]:
    """Extract the page title and readable main content."""
    # Remove elements that normally do not contain useful evidence.
    for element in soup([
        "script", "style", "nav", "footer", "header",
        "noscript", "svg", "iframe", "form",
    ]):
        element.decompose()

    # Remove common interface elements and social/cookie notices.
    for element in soup.select(
        '[role="navigation"], [role="banner"], '
        '[role="contentinfo"], .cookie-banner, '
        '.cookie-consent, .social-share, .breadcrumb'
    ):
        element.decompose()

    title = (
        soup.title.get_text(" ", strip=True)
        if soup.title
        else "Untitled page"
    )

    # Prefer the main article/content area where available.
    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id="content")
        or soup.find(id="main-content")
        or soup.body
        or soup
    )

    text = clean_text(main_content.get_text(" ", strip=True))

    # Keep very large pages from dominating the corpus.
    text = text[:MAX_TEXT_LENGTH]

    return title, text


def fetch_page(client: httpx.Client, url: str) -> dict | None:
    """Fetch one approved public HTML page and extract readable content."""
    normalized_url = normalize_url(url)

    if not normalized_url:
        return None

    try:
        response = client.get(normalized_url)
        response.raise_for_status()

        final_url = normalize_url(str(response.url))

        # Check the final URL after redirects as well.
        if not final_url:
            print(f"  Skipping unapproved redirect: {response.url}")
            return None

        content_type = response.headers.get("content-type", "").lower()

        if "text/html" not in content_type:
            print(f"  Skipping non-HTML page: {final_url}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        title, text = extract_page_text(soup)

        if len(text) < MIN_TEXT_LENGTH:
            print(
                f"  Skipping short page ({len(text)} characters): "
                f"{final_url}"
            )
            return None

        return {
            "text": text,
            "metadata": {
                "title": title,
                "source_url": final_url,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            },
        }

    except httpx.HTTPError as exc:
        print(f"  Could not fetch {url}: {exc}")
        return None
    except Exception as exc:
        print(f"  Could not parse {url}: {exc}")
        return None


def extract_links(html: str, base_url: str) -> list[str]:
    """Find candidate links restricted to approved domains."""
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for anchor in soup.find_all("a", href=True):
        candidate = urljoin(base_url, anchor["href"])
        normalized = normalize_url(candidate)

        if normalized:
            links.add(normalized)

    return sorted(links)


def main() -> None:
    """Crawl public pages and write JSONL records."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    queue = deque()

    for url in START_URLS:
        normalized = normalize_url(url)
        if normalized:
            queue.append((normalized, 0))

    visited = set()
    records = []
    seen_content = set()
    headers = {"User-Agent": USER_AGENT}

    with httpx.Client(
        follow_redirects=True,
        timeout=20.0,
        headers=headers,
    ) as client:

        while queue and len(visited) < MAX_PAGES:
            url, depth = queue.popleft()

            if url in visited:
                continue

            print(
                f"[{len(visited) + 1}/{MAX_PAGES}] "
                f"Depth {depth}: {url}"
            )

            try:
                response = client.get(url)
                response.raise_for_status()

                final_url = normalize_url(str(response.url))
                content_type = response.headers.get(
                    "content-type", ""
                ).lower()

                if not final_url or "text/html" not in content_type:
                    print(
                        "  Skipping disallowed redirect or non-HTML page."
                    )
                    visited.add(url)
                    continue

                # Track both the requested URL and the final URL so
                # redirects are not repeatedly crawled.
                visited.add(url)

                if final_url in visited and final_url != url:
                    print("  Skipping already visited redirect target.")
                    continue

                visited.add(final_url)

                # Discover links before parsing the page content.
                if depth < MAX_DEPTH:
                    for link in extract_links(
                        response.text, final_url
                    ):
                        if link not in visited:
                            queue.append((link, depth + 1))

                soup = BeautifulSoup(response.text, "html.parser")
                title, text = extract_page_text(soup)

                # IMPORTANT: the character cap is applied in
                # extract_page_text(), which this crawler calls.
                if len(text) < MIN_TEXT_LENGTH:
                    print(f"  Skipping short page ({len(text)} chars).")
                    continue

                # Deduplicate pages with identical extracted text.
                content_key = re.sub(
                    r"\W+", " ", text.casefold()
                ).strip()

                if content_key in seen_content:
                    print("  Skipping duplicate page content.")
                    continue

                seen_content.add(content_key)

                records.append({
                    "text": text,
                    "metadata": {
                        "title": title,
                        "source_url": final_url,
                        "retrieved_at": datetime.now(
                            timezone.utc
                        ).isoformat(),
                    },
                })

                print(
                    f"  Saved candidate: {title} "
                    f"({len(text)} chars)"
                )

            except httpx.HTTPError as exc:
                visited.add(url)
                print(f"  Could not fetch {url}: {exc}")

            time.sleep(REQUEST_DELAY_SECONDS)

    # Write the newly crawled corpus to JSONL.
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print("\nCrawl complete.")
    print(f"Pages visited: {len(visited)}")
    print(f"Unique pages saved: {len(records)}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()