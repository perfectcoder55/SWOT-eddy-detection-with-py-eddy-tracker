#!/usr/bin/env python3
"""合併 batch_detect_eddies.sh 產生的各 cycle_XXX/eddies_cycleXXX.csv，
加上 cycle 欄位後彙整成一份總表。"""
import argparse
import glob
import os
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="包含 cycle_XXX 子資料夾的根目錄")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pattern = os.path.join(args.root, "cycle_*", "eddies_cycle*.csv")
    files = sorted(glob.glob(pattern))
    print(f"找到 {len(files)} 個 CSV")

    all_rows = []
    header = None
    for f in files:
        m = re.search(r"cycle_(\d+)", os.path.basename(os.path.dirname(f)))
        cycle = m.group(1) if m else "?"
        with open(f) as fh:
            lines = fh.read().splitlines()
        if not lines:
            continue
        if header is None:
            header = lines[0]
        for line in lines[1:]:
            all_rows.append(f"{cycle},{line}")

    if not all_rows:
        print("沒有任何資料可合併。")
        return

    with open(args.out, "w") as fh:
        fh.write("cycle," + header + "\n")
        for row in all_rows:
            fh.write(row + "\n")
    print(f"已存: {args.out} ({len(all_rows)} 筆旋渦)")


if __name__ == "__main__":
    main()
