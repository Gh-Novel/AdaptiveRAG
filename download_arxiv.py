"""Bulk-download arXiv papers with metadata for large-scale ingestion.

Usage:
  python download_arxiv.py --total 200                    # pilot
  python download_arxiv.py --total 2000                   # full scale
  python download_arxiv.py --total 200 --categories cs.CL cs.LG

Writes:
  papers_arxiv/<id>.pdf          one PDF per paper
  papers_arxiv/metadata.json     {stem: {title, authors, abstract, ...}}

Resumable: already-downloaded PDFs are skipped, metadata is merged.
Respects arXiv rate limits (3s between API pages, 1s between PDFs).
"""
from __future__ import annotations

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

API_URL = "https://export.arxiv.org/api/query"
PDF_URL = "https://export.arxiv.org/pdf/{arxiv_id}"
NS = {"atom": "http://www.w3.org/2005/Atom"}

DEFAULT_CATEGORIES = ["cs.CL", "cs.LG", "cs.CV"]
PAGE_SIZE = 100
API_DELAY_S = 3.0   # arXiv asks for >= 3s between API requests
PDF_DELAY_S = 1.0
MIN_PDF_BYTES = 20_000  # smaller than this is an error page, not a paper


def _stem(arxiv_id: str) -> str:
    """'2401.12345v2' -> '2401_12345' (filesystem + doc_id safe)."""
    base = re.sub(r"v\d+$", "", arxiv_id)
    return base.replace(".", "_").replace("/", "_")


def fetch_page(category: str, start: int, n: int) -> list[dict]:
    """One API page of paper metadata for a category."""
    params = {
        "search_query": f"cat:{category}",
        "start": start,
        "max_results": n,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    r = requests.get(API_URL, params=params, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    papers = []
    for entry in root.findall("atom:entry", NS):
        raw_id = entry.findtext("atom:id", "", NS)  # http://arxiv.org/abs/2401.12345v2
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1]
        if not arxiv_id:
            continue
        title = " ".join((entry.findtext("atom:title", "", NS) or "").split())
        abstract = " ".join((entry.findtext("atom:summary", "", NS) or "").split())
        authors = [
            a.findtext("atom:name", "", NS)
            for a in entry.findall("atom:author", NS)
        ]
        cats = [
            c.get("term", "")
            for c in entry.findall("{http://www.w3.org/2005/Atom}category")
        ]
        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "authors": authors[:8],
                "categories": cats,
                "published": entry.findtext("atom:published", "", NS),
            }
        )
    return papers


def download_pdf(arxiv_id: str, dest: Path) -> bool:
    url = PDF_URL.format(arxiv_id=arxiv_id)
    try:
        r = requests.get(url, timeout=60, allow_redirects=True)
        r.raise_for_status()
    except Exception as exc:
        print(f"    ✗ {arxiv_id}: {exc}")
        return False
    if len(r.content) < MIN_PDF_BYTES or not r.content.startswith(b"%PDF"):
        print(f"    ✗ {arxiv_id}: not a valid PDF ({len(r.content)} bytes)")
        return False
    dest.write_bytes(r.content)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=200, help="Total papers to download")
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    ap.add_argument("--out", default="papers_arxiv")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.json"
    metadata: dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    per_cat = args.total // len(args.categories)
    print(f"Target: {args.total} papers ({per_cat}/category from {args.categories})")
    print(f"Output: {out_dir}/  ({len(metadata)} already in manifest)\n")

    downloaded = skipped = failed = 0
    t0 = time.time()

    for cat in args.categories:
        print(f"[{cat}] fetching metadata…")
        got_for_cat = 0
        start = 0
        while got_for_cat < per_cat:
            page = fetch_page(cat, start, min(PAGE_SIZE, per_cat * 2))
            time.sleep(API_DELAY_S)
            if not page:
                print(f"  (no more results for {cat})")
                break
            start += len(page)

            for paper in page:
                if got_for_cat >= per_cat:
                    break
                stem = _stem(paper["arxiv_id"])
                pdf_path = out_dir / f"{stem}.pdf"

                if pdf_path.exists():
                    metadata.setdefault(stem, paper)
                    got_for_cat += 1
                    skipped += 1
                    continue

                ok = download_pdf(paper["arxiv_id"], pdf_path)
                time.sleep(PDF_DELAY_S)
                if ok:
                    metadata[stem] = paper
                    got_for_cat += 1
                    downloaded += 1
                    if downloaded % 10 == 0:
                        elapsed = time.time() - t0
                        print(f"  {downloaded} downloaded "
                              f"({skipped} skipped, {failed} failed, {elapsed:.0f}s)")
                        meta_path.write_text(json.dumps(metadata, indent=1))
                else:
                    failed += 1

        print(f"[{cat}] done: {got_for_cat} papers\n")

    meta_path.write_text(json.dumps(metadata, indent=1))
    dt = time.time() - t0
    total_pdfs = len(list(out_dir.glob("*.pdf")))
    print(f"Finished in {dt/60:.1f} min — "
          f"{downloaded} new, {skipped} existing, {failed} failed. "
          f"{total_pdfs} PDFs on disk, {len(metadata)} in manifest.")


if __name__ == "__main__":
    main()
