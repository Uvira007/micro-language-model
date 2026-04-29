"""
Download Stack Exchange Q&A from Hugging Face (raw .jsonl.gz) and save as CSV for training.

Data lives as gzip-compressed JSON lines on the dataset repo
(https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl)

Uses the Flax dataset: flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl
(per Hugging Face). Has per-site splits (e.g. datascience, stats, ai). Cleansing uses
micro_lm.datasets.preprocessing so prep logic stays in one place.

Run from project root:
  python -m micro_lm.scripts.download_stackexchange --split datascience --output data/stackexchange_qa.csv
  python -m micro_lm.scripts.download_stackexchange --split stats --max-rows 50000
  python -m micro_lm.scripts.download_stackexchange --url "https://example.com/my.jsonl.gz"
  python -m micro_lm.scripts.download_stackexchange --file "path/to/datascience.stackexchange.com.jsonl.gz"
"""

from __future__ import annotations
import argparse
import json
import gzip
import io
import csv
from pathlib import Path
from typing import Iterator
import urllib.request
import urllib.error


from micro_lm.config import DATA_DIR
from micro_lm.datasets.preprocessing import strip_html, truncate_words

# Raw files on the hub
HF_DATASETS_REPO_FILES_BASE = (
    "https://huggingface.co/datasets/flax-sentence-embeddings/"
    "stackexchange_titlebody_best_voted_answer_jsonl/resolve/main/"
)

# short --split names -> actual filenames on the repo (most sites use *.stackexchange.com.jsonl.gz)
SPLIT_FILENAME_ALIASES: dict[str, str] = {
    "mathoverflow": "mathoverflow.net.jsonl.gz",
    "askubuntu": "askubuntu.com.jsonl.gz",
    "datascience": "datascience.stackexchange.com.jsonl.gz",
    "stats": "stats.stackexchange.com.jsonl.gz",
    "ai": "ai.stackexchange.com.jsonl.gz",
}

def split_to_hub_filename(split: str) -> str:
    """ Map CLI --split name to actual filenamesin the hub repo root"""
    if split.endswith(".jsonl.gz") or split.endswith(".jsonl"):
        return split
    if split in SPLIT_FILENAME_ALIASES:
        return SPLIT_FILENAME_ALIASES[split]
    return f"{split}.stackexchange.com.jsonl.gz"


def _open_text_line_iterator(path: Path) -> Iterator[str]:
    if path.suffix == ".gz" or str(path).endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            yield from f
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            yield from f

def _iter_jsonl_dict_from_url(url: str, timeout_s: int = 600) -> Iterator[dict]:
    """Stream and parse JSON lines from a URL (handles gzip if .gz extension)."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "micro-lm-stackexchange-download/0.1"},
        method="GET"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            is_gzip = url.endswith(".gz") or ".jsonl.gz" in url.lower()
            if is_gzip:
                stream = gzip.GzipFile(fileobj=resp)
                buf = b""
                while True:
                    chunk = stream.read(65536)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if not line.strip():
                            continue
                        try:
                            yield json.loads(line.decode("utf-8", errors="replace"))
                        except json.JSONDecodeError:
                            continue
                if buf.strip():
                    try:
                        yield json.loads(buf.decode("utf-8", errors="replace"))
                    except json.JSONDecodeError:
                        pass
            else:
                text_io = io.TextIOWrapper(resp, encoding="utf-8", errors="replace", newline = "")
                for line in text_io:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP error {e.code} when accessing {url}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"URL error when accessing {url}: {e.reason}") from e

def _iter_jsonl_dict_from_path(path: Path) -> Iterator[dict]:
    for line in _open_text_line_iterator(path):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue

def iter_records(
        *,
        url: str | None,
        file: Path | None,
        split: str,
) -> Iterator[dict]:
    if url:
        yield from _iter_jsonl_dict_from_url(url)
    elif file:
        if not file.is_file():
            raise ValueError(f"File not found: {file}")
        yield from _iter_jsonl_dict_from_path(file)
    else:
        filename = split_to_hub_filename(split)
        full_url = HF_DATASETS_REPO_FILES_BASE + filename
        yield from _iter_jsonl_dict_from_url(full_url)

def download_and_save(
    split: str = "datascience",
    output_path: str | Path | None = None,
    url: str | None = None,
    file: str | Path | None = None,
    max_rows: int | None = None,
    max_question_words: int = 80,
    max_answer_words: int = 150,
) -> Path:
    if url and file is not None:
        raise ValueError("Cannot specify both --url and --file")
    
    file_path = Path(file) if file is not None else None

    if output_path is None:
        output_path = DATA_DIR / f"stackexchange_{split}.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str]] = []
    for row in iter_records(url = url, file = file_path, split=split):
        if max_rows is not None and len(rows) >= max_rows:
            break
        if isinstance(row, dict):
            tb = row.get("title_body")
            uv = row.get("upvoted_answer")
            title_body = "" if tb is None else str(tb)
            upvoted = "" if uv is None else str(uv)
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            title_body = "" if row[0] is None else str(row[0])
            upvoted = "" if row[1] is None else str(row[1])
        else:
            continue
        title_body = strip_html(title_body)
        upvoted = strip_html(upvoted)
        if len(title_body) < 15 or len(upvoted) < 15:
            continue
        question = truncate_words(title_body, max_question_words)
        answer = truncate_words(upvoted, max_answer_words)
        rows.append((question, answer))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question", "answer"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    return output_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download Stack Exchange Q&A from Hugging Face (Flax dataset) raw .jsonl.gz files and save as CSV."
    )
    p.add_argument(
        "--split",
        type=str,
        default="datascience",
        help=("Hub file stem (default builds <split>.stackexchange.com.jsonl.gz). "
        "Examples: datascience, stats, ai, mathoverflow, askubuntu. "
        "Ignored if --url or --file is set"
        )
    )
    p.add_argument(
        "--url",
        type=str,
        default=None,
        help="Direct URL to .jsonl or .jsonl.gz file to download (overrides --split)",
    )
    p.add_argument(
        "--file",
        type=str,
        default=None,
        help="Local path to .jsonl or .jsonl.gz file to process (overrides --split)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: data/stackexchange_<split>.csv)",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Max number of Q&A pairs to save (default: all)",
    )
    p.add_argument(
        "--max-question-words",
        type=int,
        default=80,
        help="Truncate question to this many words",
    )
    p.add_argument(
        "--max-answer-words",
        type=int,
        default=150,
        help="Truncate answer to this many words",
    )
    args = p.parse_args()

    download_and_save(
        split=args.split,
        output_path=args.output,
        url=args.url,
        file=args.file,
        max_rows=args.max_rows,
        max_question_words=args.max_question_words,
        max_answer_words=args.max_answer_words,
    )


if __name__ == "__main__":
    main()
