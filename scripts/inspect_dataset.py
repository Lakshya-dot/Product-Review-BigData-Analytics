"""
STEP 1 - inspect_dataset.py
Looks at the Amazon Reviews files you already downloaded and reports:
  - which files exist and how big they are
  - the columns in each file + one example record
  - (with --full) total records, reviews per YEAR, ratings, categories

It reads files line by line, so even multi-GB files will not crash your laptop.

Usage (run from the Product_Review_BigData folder):
    python scripts/inspect_dataset.py                 # quick look (first 200,000 lines per file)
    python scripts/inspect_dataset.py --full          # scan every line (slower, exact counts)
"""

import argparse
import gzip
import json
import os
from collections import Counter
from datetime import datetime, timezone

RAW_DIR_DEFAULT = os.path.join("data", "raw", "amazon_original_dataset")
SUPPORTED = (".jsonl", ".jsonl.gz", ".json", ".json.gz", ".csv", ".csv.gz", ".parquet")


def human_size(n_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def iter_records(path):
    """Yield one record (dict) at a time, whatever the file type."""
    name = path.lower()
    if name.endswith(".parquet"):
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=50_000):
            for row in batch.to_pylist():
                yield row
    elif ".csv" in name:
        import pandas as pd
        for chunk in pd.read_csv(path, chunksize=50_000, low_memory=False):
            for row in chunk.to_dict("records"):
                yield row
    else:  # JSON Lines (Amazon Reviews 2023 format)
        opener = gzip.open if name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def to_year(ts):
    """Amazon 2023 timestamps are in milliseconds; older versions use seconds."""
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return None
    if ts > 1e11:  # milliseconds
        ts /= 1000
    return datetime.fromtimestamp(ts, tz=timezone.utc).year


def inspect_file(path, limit):
    print("\n" + "=" * 70)
    print(f"FILE: {os.path.basename(path)}   ({human_size(os.path.getsize(path))})")
    print("=" * 70)

    total = 0
    columns = Counter()
    years, ratings, main_cats, missing_text = Counter(), Counter(), Counter(), 0
    first = None

    for rec in iter_records(path):
        if first is None:
            first = rec
        total += 1
        columns.update(rec.keys())

        if "timestamp" in rec:
            y = to_year(rec["timestamp"])
            if y:
                years[y] += 1
        if "rating" in rec:
            ratings[rec["rating"]] += 1
        if "text" in rec and not str(rec.get("text") or "").strip():
            missing_text += 1
        if "main_category" in rec:
            main_cats[rec["main_category"]] += 1

        if limit and total >= limit:
            break

    if first is None:
        print("  (file is empty or unreadable)")
        return

    kind = "METADATA (product info)" if "main_category" in columns or "store" in columns else "REVIEWS"
    print(f"Type guess : {kind}")
    print(f"Records read: {total:,}" + ("  (stopped at limit - use --full for exact count)" if limit and total >= limit else ""))
    print(f"Columns    : {', '.join(columns.keys())}")

    print("\nExample record:")
    for k, v in first.items():
        v = str(v)
        print(f"  {k:<18}: {v[:90] + '...' if len(v) > 90 else v}")

    if years:
        print("\nReviews per year:")
        for y in sorted(years):
            print(f"  {y}: {years[y]:>12,}")
    if ratings:
        print("\nRatings:")
        for r in sorted(ratings, key=lambda x: float(x)):
            print(f"  {r} stars: {ratings[r]:>12,}")
    if "text" in columns:
        print(f"\nEmpty review texts: {missing_text:,}")
    if main_cats:
        print("\nTop main_category values:")
        for c, n in main_cats.most_common(10):
            print(f"  {str(c):<35} {n:>10,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default=RAW_DIR_DEFAULT)
    parser.add_argument("--full", action="store_true", help="scan every line")
    parser.add_argument("--limit", type=int, default=200_000)
    args = parser.parse_args()

    if not os.path.isdir(args.raw_dir):
        print(f"Folder not found: {args.raw_dir}\nPut your Amazon files there first.")
        return

    files = sorted(
        os.path.join(root, f)
        for root, _, names in os.walk(args.raw_dir)
        for f in names
        if f.lower().endswith(SUPPORTED)
    )
    print(f"Found {len(files)} data file(s) in {args.raw_dir}:")
    for p in files:
        print(f"  - {os.path.relpath(p, args.raw_dir):<45} {human_size(os.path.getsize(p))}")

    for p in files:
        inspect_file(p, None if args.full else args.limit)

    print("\nDone. Copy everything above and paste it back to Claude.")


if __name__ == "__main__":
    main()
