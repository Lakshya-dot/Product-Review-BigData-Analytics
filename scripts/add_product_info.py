"""
STEP 3 - add_product_info.py
Adds product name, brand and other product details to your 4 yearly datasets,
using the Amazon meta_*.jsonl files.

How it works:
  1. Collects every parent_asin (product ID) that appears in your yearly datasets.
  2. Reads each meta file line by line and keeps ONLY those products
     (so memory stays small even though the meta files are several GB).
  3. Joins product info onto every review and saves the files again.
  4. Also saves a separate product catalogue: data/processed/products.parquet

New columns added to each review:
  product_title, brand, main_category, price, product_avg_rating,
  product_rating_count, product_features

Usage (from the Product_Review_BigData folder):
    python scripts/add_product_info.py
"""

import glob
import gzip
import json
import os
import time

import pandas as pd

RAW_DIR = os.path.join("data", "raw", "amazon_original_dataset")
PROCESSED = os.path.join("data", "processed")
YEARS = [2023, 2024, 2025, 2026]
FOLDERS = {
    "Electronics": "electronics", "Home_and_Kitchen": "home_kitchen",
    "Health_and_Household": "health_household", "Beauty_and_Personal_Care": "beauty",
    "Sports_and_Outdoors": "sports", "Books": "books",
}
PRODUCT_COLS = ["product_title", "brand", "main_category", "price",
                "product_avg_rating", "product_rating_count", "product_features"]


def open_text(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def fast_parent_asin(line):
    """Read parent_asin without parsing the whole line."""
    key = '"parent_asin": "'
    i = line.rfind(key)
    if i == -1:
        return None
    i += len(key)
    j = line.find('"', i)
    return line[i:j] if j != -1 else None


def to_price(v):
    try:
        return float(str(v).replace("$", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def get_brand(r):
    details = r.get("details") or {}
    for key in ("Brand", "Brand Name", "Manufacturer"):
        if isinstance(details, dict) and details.get(key):
            return str(details[key]).strip()
    store = r.get("store")
    return str(store).strip() if store else None


def read_meta(path, wanted):
    rows = []
    t0 = time.time()
    with open_text(path) as f:
        for n, line in enumerate(f, 1):
            pid = fast_parent_asin(line)
            if pid in wanted:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                feats = r.get("features") or []
                rows.append({
                    "parent_asin": r.get("parent_asin"),
                    "product_title": (r.get("title") or "").strip() or None,
                    "brand": get_brand(r),
                    "main_category": r.get("main_category"),
                    "price": to_price(r.get("price")),
                    "product_avg_rating": r.get("average_rating"),
                    "product_rating_count": r.get("rating_number"),
                    "product_features": " | ".join(str(x) for x in feats)[:1000] or None,
                })
            if n % 1_000_000 == 0:
                print(f"    ...{n:,} lines read ({time.time() - t0:.0f}s)")
    print(f"    finished in {time.time() - t0:.0f}s, matched {len(rows):,} products")
    return rows


def main():
    # 1. Load the yearly datasets and collect product IDs per category
    yearly = {}
    for y in YEARS:
        p = os.path.join(PROCESSED, f"reviews_{y}.parquet")
        if not os.path.exists(p):
            raise SystemExit(f"Missing {p}. Run Step 2 (build_yearly_datasets.py) first.")
        df = pd.read_parquet(p)
        yearly[y] = df.drop(columns=[c for c in PRODUCT_COLS if c in df.columns])
    all_reviews = pd.concat(yearly.values())
    wanted_by_cat = all_reviews.groupby("category_key")["parent_asin"].apply(set).to_dict()

    # 2. Read the meta files
    products = []
    for stem in FOLDERS:
        files = glob.glob(os.path.join(RAW_DIR, f"meta_{stem}.jsonl*"))
        if not files:
            print(f"!! meta_{stem} not found - its reviews will have no product name/brand")
            continue
        wanted = wanted_by_cat.get(stem, set())
        print(f"\n[meta_{stem}] looking for {len(wanted):,} products")
        for r in read_meta(files[0], wanted):
            r["category_key"] = stem
            products.append(r)

    prod_df = pd.DataFrame(products, columns=["parent_asin", "category_key"] + PRODUCT_COLS)
    prod_df = prod_df.drop_duplicates(subset=["parent_asin", "category_key"])
    prod_df["brand"] = prod_df["brand"].fillna("Unknown")
    prod_df.to_parquet(os.path.join(PROCESSED, "products.parquet"), index=False)
    print(f"\nSaved data/processed/products.parquet ({len(prod_df):,} products)")

    # 3. Join onto each year and save again
    print("\nYear   reviews   with product name   with real brand")
    for y, df in yearly.items():
        merged = df.merge(prod_df, on=["parent_asin", "category_key"], how="left")
        merged["brand"] = merged["brand"].fillna("Unknown")
        merged.to_parquet(os.path.join(PROCESSED, f"reviews_{y}.parquet"), index=False)
        for stem, folder in FOLDERS.items():
            part = merged[merged["category_key"] == stem]
            if len(part):
                part.to_parquet(os.path.join("data", str(y), folder, "reviews.parquet"), index=False)
        named = merged["product_title"].notna().mean() * 100
        branded = (merged["brand"] != "Unknown").mean() * 100
        print(f"{y}  {len(merged):>9,}   {named:>14.1f}%   {branded:>14.1f}%")

    print("\nDone. Paste this output back to Claude.")


if __name__ == "__main__":
    main()
