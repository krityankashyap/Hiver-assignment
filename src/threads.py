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


def _root_ids(df: pd.DataFrame) -> dict[int, int]:
    """Map every tweet_id -> the root tweet_id of its conversation.

    Follows in_response_to_tweet_id up to the tweet with no parent, memoizing
    along the way so the whole dataset is resolved in one near-linear pass.
    """
    parent = dict(
        zip(
            df["tweet_id"].tolist(),
            df["in_response_to_tweet_id"].tolist(),  # may contain <NA>
        )
    )
    root: dict[int, int] = {}
    for tid in parent:
        # walk up, collecting the chain, until we hit a known root / missing parent
        chain = []
        cur = tid
        while cur in root:  # already resolved
            break
        while True:
            if cur in root:
                r = root[cur]
                break
            chain.append(cur)
            p = parent.get(cur)
            if p is None or pd.isna(p) or p not in parent:
                r = cur  # no (in-dataset) parent -> this is the root
                break
            cur = int(p)
        for c in chain:
            root[c] = r
    return root


def build_pairs(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    # lookups by tweet_id
    idx = df.set_index("tweet_id")
    text_by_id = idx["text"]
    inbound_by_id = idx["inbound"]

    # every reply the brand sent
    brand_replies = df[(df["author_id"] == brand) & (~df["inbound"].fillna(False))]
    if brand_replies.empty:
        raise SystemExit(
            f"error: no brand replies found for '{brand}'. "
            f"Run `python src/threads.py --rank` to see valid brand names."
        )

    parent_id = brand_replies["in_response_to_tweet_id"]
    # keep replies whose parent exists and is a customer (inbound) message
    has_parent = parent_id.notna()
    br = brand_replies[has_parent].copy()
    pid = br["in_response_to_tweet_id"].astype("int64")

    parent_is_customer = pid.map(inbound_by_id).fillna(False).to_numpy()
    br = br[parent_is_customer]
    pid = pid[parent_is_customer]

    customer_msg = pid.map(text_by_id).map(_clean).to_numpy()
    brand_reply = br["text"].map(_clean).to_numpy()

    root = _root_ids(df)
    thread_id = [root.get(int(t), int(t)) for t in br["tweet_id"].to_numpy()]

    out = pd.DataFrame(
        {
            "thread_id": thread_id,
            "customer_tweet_id": pid.to_numpy(),
            "brand_tweet_id": br["tweet_id"].to_numpy(),
            "created_at": br["created_at"].to_numpy(),
            "customer_msg": customer_msg,
            "brand_reply": brand_reply,
        }
    )
    # drop pairs where either side is empty after cleaning, and exact dups
    out = out[(out["customer_msg"].str.len() > 0) & (out["brand_reply"].str.len() > 0)]
    out = out.drop_duplicates(subset=["customer_tweet_id", "brand_tweet_id"])
    out = out.reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--brand", type=str, help="brand author_id, e.g. AppleSupport")
    ap.add_argument("--rank", action="store_true", help="list top brands and exit")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    df = load(args.input)

    if args.rank or not args.brand:
        print("\nTop brands by reply volume:\n")
        print(rank_brands(df).to_string(index=False))
        if not args.brand:
            print("\nRe-run with --brand <name> to build pairs.")
            return

    pairs = build_pairs(df, args.brand)
    out_path = args.output or (OUT_DIR / f"{args.brand}_pairs.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out_path, index=False)

    n_threads = pairs["thread_id"].nunique()
    print(
        f"\n{args.brand}: {len(pairs):,} (customer -> brand) pairs "
        f"across {n_threads:,} threads"
    )
    print(f"Wrote {out_path}")
    print("\nSample:")
    with pd.option_context("display.max_colwidth", 70):
        print(pairs[["customer_msg", "brand_reply"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()

