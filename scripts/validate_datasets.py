"""
STEP 4 - validate_datasets.py
Final check of the 4 yearly datasets before uploading to Databricks.
Prints PASS / FAIL for each rule from the project requirements, plus
a short summary you can put in your report.

Usage (from the Product_Review_BigData folder):
    python scripts/validate_datasets.py
"""

import os
import pandas as pd

PROCESSED = os.path.join("data", "processed")
YEARS = [2023, 2024, 2025, 2026]
REQUIRED = ["review_id", "year", "review_date", "category", "rating", "text",
            "parent_asin", "product_title", "brand", "is_synthetic", "date_shifted",
            "original_timestamp"]

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def main():
    sizes, frames = {}, {}
    for y in YEARS:
        p = os.path.join(PROCESSED, f"reviews_{y}.parquet")
        if not os.path.exists(p):
            raise SystemExit(f"Missing {p}")
        frames[y] = pd.read_parquet(p)
        sizes[y] = len(frames[y])

    print("\n=== Dataset rules ===")
    check("All 4 years have more than 200,000 records",
          all(n > 200_000 for n in sizes.values()), ", ".join(f"{y}: {n:,}" for y, n in sizes.items()))
    check("All 4 years have different sizes", len(set(sizes.values())) == 4)

    for y, df in frames.items():
        print(f"\n=== {y} ===")
        missing = [c for c in REQUIRED if c not in df.columns]
        check("All required columns present", not missing, f"missing: {missing}" if missing else "")
        check("Contains all 6 categories", df["category"].nunique() == 6, f"{df['category'].nunique()} found")
        check("All review dates are in this year",
              df["review_date"].str[:4].astype(int).eq(y).all())
        check("No duplicate review_id", not df["review_id"].duplicated().any())
        check("No empty review text", (df["text"].str.strip() != "").all())
        check("Ratings are between 1 and 5", df["rating"].between(1, 5).all())
        check("All records marked genuine (is_synthetic = False)", (~df["is_synthetic"]).all(),
              f"{int(df['is_synthetic'].sum())} synthetic")
        check("Product names attached", df["product_title"].notna().mean() > 0.9,
              f"{df['product_title'].notna().mean() * 100:.1f}%")

    other = pd.concat(frames.values(), ignore_index=True)
    print("\n=== Across all years ===")
    check("No review appears in two different years",
          not other.duplicated(subset=["category", "user_id", "asin", "original_timestamp"]).any())
    prod = os.path.join(PROCESSED, "products.parquet")
    check("Product catalogue exists", os.path.exists(prod))

    # ---- summary for the report ----
    print("\n=== Summary for your report ===")
    other["sentiment"] = pd.cut(other["rating"], bins=[0, 2, 3, 5],
                                labels=["Negative", "Neutral", "Positive"])
    table = other.pivot_table(index="category", columns="year", values="review_id",
                              aggfunc="count", observed=True)
    table.loc["TOTAL"] = table.sum()
    print("\nReviews per category and year:")
    print(table.to_string())

    print("\nSentiment share (rating-based labels):")
    sent = (pd.crosstab(other["year"], other["sentiment"], normalize="index") * 100).round(1)
    print(sent.to_string())

    print(f"\nUnique products: {other['parent_asin'].nunique():,}   "
          f"Unique brands: {other['brand'].nunique():,}   "
          f"Total reviews: {len(other):,}")

    os.makedirs("results", exist_ok=True)
    table.to_csv(os.path.join("results", "reviews_per_category_year.csv"))

    print("\n" + ("ALL CHECKS PASSED - safe to back up data/processed and delete raw files."
                  if all(results) else "SOME CHECKS FAILED - paste this output to Claude before deleting anything."))


if __name__ == "__main__":
    main()
