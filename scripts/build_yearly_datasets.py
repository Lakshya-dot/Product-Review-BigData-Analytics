"""
STEP 2 - build_yearly_datasets.py
Builds four yearly datasets (2023, 2024, 2025, 2026) from your Amazon review files.

How it works (plain English):
  1. Reads each category file line by line (never loads 29 GB into memory).
  2. Keeps only reviews whose REAL date is in 2020, 2021, 2022 or 2023.
  3. Picks a fair random sample from each year (reservoir sampling, fixed seed
     so you get the same result every time you run it).
  4. Shifts the date forward by 3 years for the prototype:
        2020 -> 2023, 2021 -> 2024, 2022 -> 2025, 2023 -> 2026
     The real date is kept in `original_timestamp` and `date_shifted = True`.
  5. Basic cleaning: drops empty text, missing ratings and duplicates.
  6. Saves Parquet files:
        data/<year>/<category>/reviews.parquet      (one per year + category)
        data/processed/reviews_<year>.parquet       (one per year, all 6 categories)
        results/dataset_summary.csv                 (counts for your report)

Every record is a GENUINE Amazon review (is_synthetic = False). Only the year is moved.

Usage (from the Product_Review_BigData folder):
    python scripts/build_yearly_datasets.py
"""

import json
import os
import random
import time
from datetime import datetime, timezone

import pandas as pd

try:
    import pyarrow  # noqa: F401  (needed to save Parquet files)
except ImportError:
    raise SystemExit("pyarrow is missing. Run:  pip install -r requirements.txt")

RAW_DIR = os.path.join("data", "raw", "amazon_original_dataset")
OUT_DIR = "data"
SEED = 42
YEAR_SHIFT = 3
BUFFER = 1.1  # sample 10% extra so cleaning losses and short categories can be covered

# file name (without .jsonl / .jsonl.gz) -> (folder name, display name)
CATEGORIES = {
    "Electronics":              ("electronics",      "Electronics"),
    "Home_and_Kitchen":         ("home_kitchen",     "Home and Kitchen"),
    "Health_and_Household":     ("health_household", "Health and Household"),
    "Beauty_and_Personal_Care": ("beauty",           "Beauty and Personal Care"),
    "Sports_and_Outdoors":      ("sports",           "Sports and Outdoors"),
    "Books":                    ("books",            "Books"),
}

# Reviews per CATEGORY per NEW year (about 100,000 each, with variation).
# Books 2026 is 55,000 because only ~55,700 real 2023 Books reviews exist.
CATEGORY_TARGETS = {
    2023: {"Electronics":  96_000, "Home_and_Kitchen": 100_000, "Health_and_Household":  92_000,
           "Beauty_and_Personal_Care":  95_000, "Sports_and_Outdoors":  88_000, "Books": 99_000},
    2024: {"Electronics": 100_000, "Home_and_Kitchen": 104_000, "Health_and_Household":  97_000,
           "Beauty_and_Personal_Care":  99_000, "Sports_and_Outdoors":  93_000, "Books": 97_000},
    2025: {"Electronics": 103_000, "Home_and_Kitchen": 108_000, "Health_and_Household": 103_000,
           "Beauty_and_Personal_Care": 104_000, "Sports_and_Outdoors":  96_000, "Books": 96_000},
    2026: {"Electronics": 106_000, "Home_and_Kitchen": 115_000, "Health_and_Household": 112_000,
           "Beauty_and_Personal_Care": 110_000, "Sports_and_Outdoors": 104_000, "Books": 55_000},
}
YEAR_TARGETS = {y: sum(c.values()) for y, c in CATEGORY_TARGETS.items()}
# 2023: 570,000   2024: 590,000   2025: 610,000   2026: 602,000


def year_bounds_ms(year):
    start = datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
    return start, end


SOURCE_YEARS = [y - YEAR_SHIFT for y in YEAR_TARGETS]          # 2020..2023
BOUNDS = {y: year_bounds_ms(y) for y in SOURCE_YEARS}
LOW, HIGH = BOUNDS[min(SOURCE_YEARS)][0], BOUNDS[max(SOURCE_YEARS)][1]


def fast_timestamp(line):
    """Read the timestamp without parsing the whole JSON line (much faster)."""
    i = line.find('"timestamp":')
    if i == -1:
        return None
    i += len('"timestamp":')
    j = i
    while j < len(line) and line[j] in " 0123456789":
        j += 1
    try:
        return int(line[i:j])
    except ValueError:
        return None


def source_year(ts):
    if ts is None or ts < LOW or ts >= HIGH:
        return None
    for y, (a, b) in BOUNDS.items():
        if a <= ts < b:
            return y
    return None


def shift_timestamp(ts_ms, years):
    d = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    try:
        d2 = d.replace(year=d.year + years)
    except ValueError:            # 29 Feb -> 28 Feb in a non-leap year
        d2 = d.replace(year=d.year + years, day=28)
    return int(d2.timestamp() * 1000)


def find_file(stem):
    for ext in (".jsonl", ".jsonl.gz"):
        p = os.path.join(RAW_DIR, stem + ext)
        if os.path.exists(p):
            return p
    return None


def open_text(path):
    if path.endswith(".gz"):
        import gzip
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def sample_category(path, stem):
    """Reservoir-sample each source year for one category. Returns {src_year: [lines]}."""
    caps = {}
    for new_year, total in YEAR_TARGETS.items():
        caps[new_year - YEAR_SHIFT] = int(CATEGORY_TARGETS[new_year][stem] * BUFFER)

    reservoirs = {y: [] for y in SOURCE_YEARS}
    seen = {y: 0 for y in SOURCE_YEARS}
    rng = random.Random(f"{SEED}-{stem}")

    t0 = time.time()
    with open_text(path) as f:
        for n, line in enumerate(f, 1):
            y = source_year(fast_timestamp(line))
            if y is not None:
                seen[y] += 1
                res, cap = reservoirs[y], caps[y]
                if len(res) < cap:
                    res.append(line)
                else:
                    k = rng.randrange(seen[y])
                    if k < cap:
                        res[k] = line
            if n % 5_000_000 == 0:
                print(f"    ...{n:,} lines read ({time.time() - t0:.0f}s)")
    print(f"    finished in {time.time() - t0:.0f}s")
    return reservoirs, seen


def lines_to_df(lines, stem, src_year):
    rows = []
    for line in lines:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append({
            "rating": r.get("rating"),
            "title": r.get("title"),
            "text": r.get("text"),
            "asin": r.get("asin"),
            "parent_asin": r.get("parent_asin"),
            "user_id": r.get("user_id"),
            "original_timestamp": r.get("timestamp"),
            "helpful_vote": r.get("helpful_vote", 0),
            "verified_purchase": r.get("verified_purchase"),
            "has_images": bool(r.get("images")),
        })
    df = pd.DataFrame(rows)
    before = len(df)

    # --- basic cleaning ---
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df = df[(df["text"] != "") & df["rating"].notna()]
    df = df.drop_duplicates(subset=["user_id", "asin", "original_timestamp", "text"])

    df["category_key"] = stem
    df["category"] = CATEGORIES[stem][1]
    df["original_year"] = src_year
    return df, before - len(df)


def main():
    os.makedirs(os.path.join(OUT_DIR, "processed"), exist_ok=True)
    os.makedirs("results", exist_ok=True)

    pools = {}          # (new_year, stem) -> cleaned, shuffled DataFrame (with buffer)
    cleaning_log = {}
    available = {}

    for stem in CATEGORIES:
        path = find_file(stem)
        if path is None:
            print(f"!! Missing file for {stem} - skipping")
            continue
        print(f"\n[{stem}] reading {os.path.basename(path)}")
        reservoirs, seen = sample_category(path, stem)
        for src_year, lines in reservoirs.items():
            new_year = src_year + YEAR_SHIFT
            df, dropped = lines_to_df(lines, stem, src_year)
            pools[(new_year, stem)] = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
            cleaning_log[(new_year, stem)] = dropped
            available[(new_year, stem)] = seen[src_year]
            print(f"    real {src_year} reviews in file: {seen[src_year]:,} -> sampled {len(df):,} (dropped {dropped})")
        del reservoirs

    summary = []
    for new_year, total in YEAR_TARGETS.items():
        print(f"\n=== Building {new_year} (target {total:,}) ===")
        wanted = dict(CATEGORY_TARGETS[new_year])

        # If a category is short, move its shortfall to categories with spare rows
        taken = {s: min(wanted[s], len(pools.get((new_year, s), []))) for s in CATEGORIES}
        shortfall = total - sum(taken.values())
        for s in sorted(CATEGORIES, key=lambda s: -(len(pools.get((new_year, s), [])) - taken[s])):
            if shortfall <= 0:
                break
            spare = len(pools.get((new_year, s), [])) - taken[s]
            extra = min(spare, shortfall)
            taken[s] += extra
            shortfall -= extra
        if shortfall > 0:
            print(f"!! {new_year} is short by {shortfall:,} rows - tell Claude (augmentation step needed)")

        year_parts = []
        for stem, (folder, _) in CATEGORIES.items():
            df = pools.get((new_year, stem))
            if df is None or taken[stem] == 0:
                continue
            df = df.head(taken[stem]).copy()
            df["timestamp"] = df["original_timestamp"].apply(lambda t: shift_timestamp(int(t), YEAR_SHIFT))
            df["review_date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
            df["year"] = new_year
            df["date_shifted"] = True
            df["is_synthetic"] = False
            df["review_id"] = [f"{new_year}_{folder}_{i:07d}" for i in range(len(df))]

            cols = ["review_id", "year", "review_date", "timestamp", "category", "category_key",
                    "rating", "title", "text", "asin", "parent_asin", "user_id",
                    "helpful_vote", "verified_purchase", "has_images",
                    "original_timestamp", "original_year", "date_shifted", "is_synthetic"]
            df = df[cols]

            cat_dir = os.path.join(OUT_DIR, str(new_year), folder)
            os.makedirs(cat_dir, exist_ok=True)
            df.to_parquet(os.path.join(cat_dir, "reviews.parquet"), index=False)
            year_parts.append(df)

            summary.append({
                "year": new_year, "category": CATEGORIES[stem][1],
                "original_year": new_year - YEAR_SHIFT,
                "real_reviews_available": available[(new_year, stem)],
                "records_saved": len(df), "genuine_records": len(df), "synthetic_records": 0,
                "dropped_in_cleaning": cleaning_log[(new_year, stem)],
            })
            print(f"  {CATEGORIES[stem][1]:<26} {len(df):>8,}")

        year_df = pd.concat(year_parts, ignore_index=True)
        year_df.to_parquet(os.path.join(OUT_DIR, "processed", f"reviews_{new_year}.parquet"), index=False)
        print(f"  {'TOTAL':<26} {len(year_df):>8,}  -> data/processed/reviews_{new_year}.parquet")

    pd.DataFrame(summary).to_csv(os.path.join("results", "dataset_summary.csv"), index=False)
    print("\nSaved results/dataset_summary.csv. Paste this whole output back to Claude.")


if __name__ == "__main__":
    main()
