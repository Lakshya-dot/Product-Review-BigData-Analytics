"""
split_large_files.py
Splits every .jsonl file bigger than 4 GB into parts of about 3.5 GB,
so they fit on a FAT32 pen drive. Parts are written STRAIGHT to the
destination (your USB), so no extra laptop space is needed.

Each part is a normal .jsonl file (split only between lines, never in the
middle of a review), e.g.:
    Home_and_Kitchen.part01.jsonl, Home_and_Kitchen.part02.jsonl, ...

Files smaller than 4 GB are copied as they are (skipped if already on the USB).
Original files on the laptop are NOT deleted - you delete them yourself after checking.

Usage (from the Product_Review_BigData folder):
    python scripts/split_large_files.py --dest D:\\amazon_original_dataset
"""

import argparse
import glob
import os
import shutil
import time

SRC_DEFAULT = os.path.join("data", "raw", "amazon_original_dataset")
FAT32_LIMIT = 4 * 1024**3 - 1        # largest file FAT32 can hold
PART_SIZE = int(3.5 * 1024**3)       # target size of each part (safe margin)


def gb(n):
    return f"{n / 1024**3:.2f} GB"


def split_file(path, dest, part_size):
    base = os.path.basename(path)[: -len(".jsonl")]
    total = os.path.getsize(path)
    written, part_no, out = 0, 0, None
    part_bytes = 0
    t0 = time.time()
    with open(path, "rb") as f:
        for line in f:
            if out is None or part_bytes + len(line) > part_size:
                if out:
                    out.close()
                part_no += 1
                part_path = os.path.join(dest, f"{base}.part{part_no:02d}.jsonl")
                out = open(part_path, "wb", buffering=16 * 1024**2)
                part_bytes = 0
                print(f"    writing {os.path.basename(part_path)}")
            out.write(line)
            part_bytes += len(line)
            written += len(line)
    if out:
        out.close()

    # verify: all parts together must equal the original size
    parts = sorted(glob.glob(os.path.join(dest, f"{base}.part*.jsonl")))
    parts_total = sum(os.path.getsize(p) for p in parts)
    ok = parts_total == total
    print(f"    {len(parts)} parts, {gb(parts_total)} of {gb(total)} "
          f"-> {'OK' if ok else 'SIZE MISMATCH!'} ({time.time() - t0:.0f}s)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--dest", required=True, help=r"e.g. D:\amazon_original_dataset")
    ap.add_argument("--part_gb", type=float, default=3.5)
    args = ap.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    part_size = int(args.part_gb * 1024**3)
    files = sorted(glob.glob(os.path.join(args.src, "*.jsonl")))
    if not files:
        raise SystemExit(f"No .jsonl files found in {args.src}")

    need = sum(os.path.getsize(p) for p in files)
    free = shutil.disk_usage(args.dest).free
    print(f"Files to send: {len(files)} ({gb(need)})   Free on destination: {gb(free)}")
    if need > free:
        raise SystemExit("Not enough space on the destination drive.")

    results = {}
    for path in files:
        name, size = os.path.basename(path), os.path.getsize(path)
        print(f"\n{name} ({gb(size)})")
        if size > FAT32_LIMIT:
            results[name] = split_file(path, args.dest, part_size)
        else:
            target = os.path.join(args.dest, name)
            if os.path.exists(target) and os.path.getsize(target) == size:
                print("    already on destination - skipped")
                results[name] = True
            else:
                print("    copying as is...")
                shutil.copyfile(path, target)
                results[name] = os.path.getsize(target) == size
                print("    OK" if results[name] else "    SIZE MISMATCH!")

    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"  [{'OK' if ok else 'FAILED'}] {name}")
    if all(results.values()):
        print("\nAll files are safely on the destination. You can now delete the laptop copies.")
    else:
        print("\nSome files FAILED - do NOT delete anything. Paste this output to Claude.")


if __name__ == "__main__":
    main()
