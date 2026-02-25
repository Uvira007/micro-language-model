"""
Download Stack Exchange Q&A from Hugging Face and save as CSV for training.

Uses the Flax dataset: flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl
(per Hugging Face). Has per-site splits (e.g. datascience, stats, ai). Cleansing uses
micro_lm.datasets.preprocessing so prep logic stays in one place.

Run from project root:
  python -m micro_lm.scripts.download_stackexchange --split datascience --output data/stackexchange_qa.csv
  python -m micro_lm.scripts.download_stackexchange --split stats --max-rows 50000
"""
import argparse
import csv
from pathlib import Path

from micro_lm.config import DATA_DIR
from micro_lm.datasets.preprocessing import strip_html, truncate_words


def download_and_save(
    split: str = "datascience",
    output_path: str | Path | None = None,
    max_rows: int | None = None,
    max_question_words: int = 80,
    max_answer_words: int = 150,
) -> Path:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install the 'datasets' package to use Stack Exchange download: pip install datasets"
        )

    # Per Hugging Face: load_dataset("flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl")
    # Returns a DatasetDict with one key per community (datascience, stats, ai, ...)
    ds = load_dataset(
        "flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl",
        trust_remote_code=False,
    )
    if split in ds:
        dataset = ds[split]
    elif "train" in ds:
        dataset = ds["train"]
    else:
        dataset = next(iter(ds.values()))

    if output_path is None:
        output_path = DATA_DIR / f"stackexchange_{split}.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, row in enumerate(dataset):
        if max_rows is not None and len(rows) >= max_rows:
            break
        title_body = row.get("title_body") or ""
        upvoted = row.get("upvoted_answer") or ""
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
        description="Download Stack Exchange Q&A from Hugging Face (Flax dataset) and save as CSV."
    )
    p.add_argument(
        "--split",
        type=str,
        default="datascience",
        help="Dataset split (site name). Examples: datascience, stats, ai, mathoverflow, physics",
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
        max_rows=args.max_rows,
        max_question_words=args.max_question_words,
        max_answer_words=args.max_answer_words,
    )


if __name__ == "__main__":
    main()
