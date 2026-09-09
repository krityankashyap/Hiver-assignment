#!/usr/bin/env python3
"""
preprocess.py — turn the raw twcs.csv into a cleaned, analysis-ready table.

Input:  data/archive/twcs/twcs.csv   (raw, ~3M rows, not committed)
Output: data/twcs_clean.csv          (cleaned, not committed)

Raw columns:
    tweet_id, author_id, inbound, created_at, text,
    response_tweet_id, in_response_to_tweet_id

What this does (minimal, deterministic cleaning):
    - parse `created_at` to UTC datetime
    - coerce `inbound` to a real boolean
    - normalize whitespace in `text`
    - drop rows with empty text
    - sort by created_at

Usage:
    python data/preprocess.py
    python data/preprocess.py --input path/to/twcs.csv --output out.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = DATA_DIR / "archive" / "twcs" / "twcs.csv"
DEFAULT_OUTPUT = DATA_DIR / "twcs_clean.csv"

_WS = re.compile(r"\s+")


def _normalize_text(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .astype(str)
        .str.replace(_WS, " ", regex=True)
        .str.strip()
    )


def preprocess(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise SystemExit(
            f"error: {input_path} not found. Run `bash data/download.sh` first."
        )

    print(f"Reading {input_path} ...")
    df = pd.read_csv(input_path)

    # Twitter timestamps look like "Wed Oct 11 06:55:44 +0000 2017".
    df["created_at"] = pd.to_datetime(
        df["created_at"],
        format="%a %b %d %H:%M:%S %z %Y",
        errors="coerce",
        utc=True,
    )
    df["inbound"] = df["inbound"].astype(bool)
    df["text"] = _normalize_text(df["text"])

    before = len(df)
    df = df[df["text"].str.len() > 0]
    df = df.sort_values("created_at").reset_index(drop=True)
    print(f"Dropped {before - len(df)} empty-text rows; {len(df)} remain.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    preprocess(args.input, args.output)


if __name__ == "__main__":
    main()
