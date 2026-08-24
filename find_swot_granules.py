#!/usr/bin/env python3
"""
搜尋指定時間區間 + 空間範圍內可用的 SWOT L3 KaRIn Expert granule。

用法：
  # 用測站座標 + 半徑(度) 搜尋；--tz-offset 8 表示 --start/--end 是台灣當地時間
  python3 find_swot_granules.py \
      --start 2023-10-08T16:00:00 --end 2023-10-08T17:00:00 --tz-offset 8 \
      --lat 21.8063 --lon 123.7828 --radius 0.5

  # 用經緯度範圍框搜尋（輸入時間已是 UTC，不需要 --tz-offset）
  python3 find_swot_granules.py \
      --start 2023-10-08T08:00:00 --end 2023-10-08T09:00:00 \
      --lon-min 122.0 --lon-max 125.0 --lat-min 20.0 --lat-max 23.0

先用檔名裡的時間篩選，再對候選檔只讀 lat/lon 做空間比對，
最後印出符合條件的 granule 清單。

"""
import argparse
import glob
import os
import re
from datetime import datetime

import numpy as np
from netCDF4 import Dataset

FNAME_RE = re.compile(
    r"SWOT_L3_LR_SSH_Expert_(\d{3})_(\d{3})_(\d{8}T\d{6})_(\d{8}T\d{6})_v([\d.]+)\.nc$"
)


def parse_time(s):
    return datetime.strptime(s, "%Y%m%dT%H%M%S")


def norm360(x):
    """經度轉成 [0, 360)，SWOT 用東經 0~360 這個慣例。"""
    return x % 360.0


def find_time_candidates(data_dir, t_start, t_end):
    pattern = os.path.join(data_dir, "cycle_*", "SWOT_L3_LR_SSH_Expert_*.nc")
    candidates = []
    for f in glob.glob(pattern):
        m = FNAME_RE.search(os.path.basename(f))
        if not m:
            continue
        cycle, pas, dbeg, dend, version = m.groups()
        fbeg, fend = parse_time(dbeg), parse_time(dend)
        if fbeg <= t_end and fend >= t_start:
            candidates.append((f, cycle, pas, fbeg, fend, version))
    return candidates


def check_spatial(fpath, lon_min_n, lon_max_n, lat_min, lat_max):
    with Dataset(fpath) as ds:
        # netCDF4 預設就會套用 scale_factor/add_offset，這裡讀出來已經是
        # 還原後的實際經緯度，不用再手動乘一次。
        lat = np.ma.filled(ds.variables["latitude"][:].astype(float), np.nan)
        lon = np.ma.filled(ds.variables["longitude"][:].astype(float), np.nan)
        lon_n = norm360(lon)

        lat_mask = (lat >= lat_min) & (lat <= lat_max)
        if lon_min_n <= lon_max_n:
            lon_mask = (lon_n >= lon_min_n) & (lon_n <= lon_max_n)
        else:
            lon_mask = (lon_n >= lon_min_n) | (lon_n <= lon_max_n)

        mask = lat_mask & lon_mask
        return int(np.count_nonzero(mask)), mask.size


def main():
    ap = argparse.ArgumentParser(description="依時間+空間搜尋 SWOT L3 Expert granule")
    ap.add_argument("--start", required=True, help="開始時間 (UTC) YYYY-MM-DDTHH:MM:SS")
    ap.add_argument("--end", required=True, help="結束時間 (UTC) YYYY-MM-DDTHH:MM:SS")
    ap.add_argument("--lat", type=float, help="測站緯度 (與 --lon 搭配使用)")
    ap.add_argument("--lon", type=float, help="測站經度 (與 --lat 搭配使用，可用 -180~180 或 0~360)")
    ap.add_argument("--radius", type=float, default=0.5, help="測站搜尋半徑(度)，預設 0.5")
    ap.add_argument("--lon-min", type=float)
    ap.add_argument("--lon-max", type=float)
    ap.add_argument("--lat-min", type=float)
    ap.add_argument("--lat-max", type=float)
    ap.add_argument(
        "--data-dir",
        default="/home/donnee/SWOT/v300_expert",
        help="SWOT L3 資料根目錄 (預設 ML01 上的 v300_expert 路徑)",
    )
    ap.add_argument(
        "--tz-offset",
        type=float,
        default=0.0,
        help="輸入的 --start/--end 相對 UTC 的時區偏移(小時)。"
        "測站記錄若為台灣當地時間請填 8。SWOT 檔名時間本身是 UTC。預設 0。",
    )
    args = ap.parse_args()

    from datetime import timedelta

    tz_delta = timedelta(hours=args.tz_offset)
    t_start = datetime.fromisoformat(args.start) - tz_delta
    t_end = datetime.fromisoformat(args.end) - tz_delta
    if t_start >= t_end:
        ap.error(f"--start ({args.start}) 必須早於 --end ({args.end})")

    if args.lat is not None and args.lon is not None:
        lat_min, lat_max = args.lat - args.radius, args.lat + args.radius
        lon_min, lon_max = args.lon - args.radius, args.lon + args.radius
    elif None not in (args.lon_min, args.lon_max, args.lat_min, args.lat_max):
        lon_min, lon_max, lat_min, lat_max = (
            args.lon_min,
            args.lon_max,
            args.lat_min,
            args.lat_max,
        )
    else:
        ap.error("需提供 --lat/--lon (測站+半徑) 或 --lon-min/--lon-max/--lat-min/--lat-max (範圍框)")

    lon_min_n, lon_max_n = norm360(lon_min), norm360(lon_max)

    tz_note = f" (輸入視為 UTC{args.tz_offset:+g})" if args.tz_offset != 0 else ""
    print(f"搜尋區間: {t_start:%Y-%m-%d %H:%M:%S} ~ {t_end:%Y-%m-%d %H:%M:%S} UTC{tz_note}")
    print(f"搜尋範圍: lat {lat_min:.3f}~{lat_max:.3f}  lon {lon_min:.3f}~{lon_max:.3f}")

    candidates = find_time_candidates(args.data_dir, t_start, t_end)

    results = []
    for fpath, cycle, pas, fbeg, fend, version in candidates:
        try:
            n_match, n_total = check_spatial(fpath, lon_min_n, lon_max_n, lat_min, lat_max)
        except Exception as e:
            print(f"  [略過，讀取失敗] {fpath}: {e}")
            continue
        if n_match > 0:
            results.append((fpath, cycle, pas, fbeg, fend, version, n_match, n_total))

    if not results:
        print(f"\n找不到符合條件的檔案（時間重疊候選 {len(candidates)} 個，皆無空間命中）。")
        print("請確認測站座標/範圍是否正確，或放寬 --radius 與時間區間。")
        return

    print(f"\n找到 {len(results)} 個可用的 SWOT granule：\n")
    for fpath, cycle, pas, fbeg, fend, version, n_match, n_total in sorted(
        results, key=lambda r: r[3]
    ):
        print(
            f"cycle={cycle} pass={pas} v{version}  "
            f"{fbeg:%Y-%m-%d %H:%M:%S} ~ {fend:%H:%M:%S} UTC  "
            f"命中點數={n_match}/{n_total}"
        )
        print(f"  {fpath}")


if __name__ == "__main__":
    main()
