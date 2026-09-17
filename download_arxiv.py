import argparse
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ElementTree


API_URL = "https://export.arxiv.org/api/query"
SEARCH_QUERY = "(ti:LLM OR abs:LLM) AND cat:cs.*"
MAX_DOCUMENTS = 5
DEFAULT_PAGE_SIZE = 5
DEFAULT_DELAY = 3.0
ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"


def slugify(title: str) -> str:
    """Turn a paper title into a readable, filesystem-safe filename."""
    title = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
    return title[:120] or "arxiv_paper"


def fetch(url: str, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": "research-chatbot/1.0"})
            with urlopen(request, timeout=60) as response:
                return response.read()
        except (HTTPError, URLError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"Could not download {url}: {exc}") from exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not download {url}")


def search_page(query: str, start: int, page_size: int) -> list[tuple[str, str, str]]:
    params = urlencode(
        {
            "search_query": query,
            "start": start,
            "max_results": page_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    root = ElementTree.fromstring(fetch(f"{API_URL}?{params}"))
    papers = []
    for entry in root.findall(f"{{{ATOM_NAMESPACE}}}entry"):
        paper_id = entry.findtext(f"{{{ATOM_NAMESPACE}}}id", "").rstrip("/").split("/")[-1]
        title = " ".join(entry.findtext(f"{{{ATOM_NAMESPACE}}}title", "").split())
        pdf_url = next(
            (
                link.attrib["href"]
                for link in entry.findall(f"{{{ATOM_NAMESPACE}}}link")
                if link.attrib.get("title") == "pdf"
            ),
            f"https://arxiv.org/pdf/{paper_id}",
        )
        if paper_id and title:
            papers.append((paper_id, title, pdf_url))
    return papers


def download_papers(query: str, output_dir: Path, page_size: int, delay: float) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    start = 0

    while downloaded < MAX_DOCUMENTS:
        papers = search_page(query, start, page_size)
        if not papers:
            break

        for paper_id, title, pdf_url in papers:
            if downloaded >= MAX_DOCUMENTS:
                break
            destination = output_dir / f"{paper_id}_{slugify(title)}.pdf"
            if destination.exists():
                print(f"Skipping {paper_id}: already exists")
                continue

            if downloaded > 0:
                time.sleep(delay)
            temporary = destination.with_suffix(".part")
            print(f"Downloading {paper_id}: {title}")
            temporary.write_bytes(fetch(pdf_url))
            temporary.replace(destination)
            downloaded += 1

        start += len(papers)
        if len(papers) < page_size:
            break
        time.sleep(delay)

    return downloaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download recent arXiv papers mentioning LLM in the title or "
            "abstract within computer science categories."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./research_papers"),
        help="directory where PDFs are stored (default: research_papers)",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help=f"number of results per API request (default: {DEFAULT_PAGE_SIZE})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"seconds between downloads and API pages (default: {DEFAULT_DELAY})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.page_size < 1 or args.delay < 0:
        raise SystemExit("--page-size must be positive and --delay cannot be negative")
    count = download_papers(SEARCH_QUERY, args.output_dir, args.page_size, args.delay)
    print(f"Downloaded {count} paper(s).")


if __name__ == "__main__":
    main()