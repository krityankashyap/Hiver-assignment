#!/usr/bin/env python3
"""
threads.py — reconstruct (customer message -> brand reply) pairs from twcs.csv.

The raw dataset is a flat list of tweets. Conversations are encoded by two id
columns:
    in_response_to_tweet_id : the tweet THIS tweet is replying to (parent)
    response_tweet_id       : the tweet(s) that replied to THIS one (children)

Authorship convention in the data:
    - brands have a text author_id  (e.g. "AppleSupport")
    - customers have a numeric author_id (e.g. "105834")
    - inbound == True  -> customer -> brand
    - inbound == False -> brand -> customer

What this produces
------------------
For a chosen brand, every direct question/answer pair:
    customer_msg  : text of a customer tweet
    brand_reply   : the brand's reply to it
    thread_id     : id of the root tweet of the conversation (for grouping turns)
    + a few id/time columns for traceability.

This pair table is the foundation for BOTH:
    - grounding (retrieve over historical brand_reply texts), and
    - labelling the golden set (label the customer_msg).

Usage
-----
    # 1) See which brands have the most volume (pick one):
    python src/threads.py --rank

    # 2) Build pairs for a brand:
    python src/threads.py --brand AppleSupport
    #   -> writes data/threads/AppleSupport_pairs.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "archive" / "twcs" / "twcs.csv"
OUT_DIR = ROOT / "data" / "threads"

USECOLS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "in_response_to_tweet_id",
]

_MENTION = re.compile(r"@\w+")
_WS = re.compile(r"\s+")


def load(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise SystemExit(
            f"error: {input_path} not found. Run `bash data/download.sh` first."
        )
    print(f"Reading {input_path} ...")
    df = pd.read_csv(
        input_path,
        usecols=USECOLS,
        dtype={
            "tweet_id": "int64",
            "author_id": "string",
            "inbound": "boolean",
            "text": "string",
        },
    )
    # parent id is stored as float (has NaNs); keep as nullable int for lookups
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].astype("Int64")
    print(f"  {len(df):,} tweets loaded")
    return df


def rank_brands(df: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    """Brands ranked by number of replies they sent (a proxy for usable volume)."""
    brand_replies = df[~df["inbound"].fillna(False)]
    counts = (
        brand_replies["author_id"]
        .value_counts()
        .rename_axis("brand")
        .reset_index(name="brand_replies")
        .head(top)
    )
    return counts


def _clean(text: str) -> str:
    """Light normalization: drop @-mentions and collapse whitespace.

    We strip mentions (the brand handle and anonymized @numbers) so the message
    reads as plain content. Heavier cleaning lives in data/preprocess.py.
    """
    if text is None:
        return ""
    text = _MENTION.sub("", str(text))
    return _WS.sub(" ", text).strip()


