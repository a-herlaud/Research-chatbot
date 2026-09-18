# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pypdf",
#     "langchain-text-splitters",
#     "sentence-transformers",
#     "chromadb",
# ]
# ///
"""Chunk the downloaded arXiv PDFs and store embeddings in ChromaDB.

Run on the host with uv (no system pip required):

    uv run ingest_papers.py
    uv run ingest_papers.py --papers-dir ./research_papers --rebuild

The chunking and embedding logic lives in src/backend/rag.py so the running
backend queries the index with the identical method used to build it.
"""

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent / "src" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import rag  # noqa: E402
from pypdf import PdfReader  # noqa: E402


def paper_id_from_path(path: Path) -> str:
    # Filename is "<arxiv-id>_<slugified-title>.pdf"
    return path.stem.split("_", 1)[0]


def title_from_path(path: Path) -> str:
    parts = path.stem.split("_", 1)
    return parts[1].replace("_", " ") if len(parts) == 2 else path.stem


def extract_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def ingest(papers_dir: Path, rebuild: bool) -> int:
    if not papers_dir.is_dir():
        raise SystemExit(f"papers directory not found: {papers_dir}")

    pdfs = sorted(papers_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {papers_dir}. Run download_arxiv.py first.")
        return 0

    if rebuild:
        print("Rebuild requested: dropping existing collection.")
        client = rag.get_client()
        try:
            client.delete_collection(rag.COLLECTION_NAME)
        except Exception:
            pass  # nothing to drop yet

    collection = rag.get_collection(create=True)
    splitter = rag.build_splitter()
    total_chunks = 0

    for pdf in pdfs:
        paper_id = paper_id_from_path(pdf)
        title = title_from_path(pdf)
        text = extract_text(pdf)
        if not text.strip():
            print(f"Skipping {paper_id}: no extractable text")
            continue

        chunks = [c for c in splitter.split_text(text) if c.strip()]
        if not chunks:
            print(f"Skipping {paper_id}: empty after split")
            continue

        ids = [f"{paper_id}::{i}" for i in range(len(chunks))]
        metadatas = [
            {"paper_id": paper_id, "title": title, "source": pdf.name, "chunk": i}
            for i in range(len(chunks))
        ]
        collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
        total_chunks += len(chunks)
        print(f"Ingested {paper_id}: {len(chunks)} chunks")

    print(f"Collection '{rag.COLLECTION_NAME}' now holds {collection.count()} chunks.")
    return total_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk and embed research paper PDFs into ChromaDB."
    )
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=Path("./research_papers"),
        help="directory containing PDFs (default: research_papers)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="delete and recreate the collection before ingesting",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rag.get_embedding_function()  # warm up / fail fast if the model is missing
    count = ingest(args.papers_dir, args.rebuild)
    print(f"Ingested {count} chunk(s) across the corpus.")


if __name__ == "__main__":
    main()
