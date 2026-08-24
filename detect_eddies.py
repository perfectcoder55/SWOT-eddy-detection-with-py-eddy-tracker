#!/usr/bin/env python3
"""
在加瓜海脊固定區域框內，用單一 SWOT cycle 的資料組成網格化 ADT 場，
交給 py-eddy-tracker 做離散旋渦偵測（中心點、半徑、振幅）。

流程：收集 cycle 內所有 pass 的有效點、分箱取代表點、內插成網格、
門檻+陸地遮罩、高通+低通帶通濾波、py-eddy-tracker 偵測、輸出CSV+疊圖。

用法：
  conda activate pyeddy
  python3 detect_eddies.py --cycle 4 --out-dir ~/eddy_out

"""
import argparse
import glob
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import numpy.ma as ma
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from netCDF4 import Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from matplotlib.path import Path as MplPath

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from find_swot_granules import FNAME_RE, parse_time  # noqa: E402

from py_eddy_tracker.dataset.grid import RegularGridDataset


KM_PER_DEG_LAT = 111.0


def km_per_deg_lon(lat_deg):
    return 111.0 * np.cos(np.deg2rad(lat_deg))


def is_ascending(lat):
    """用該pass頭尾緯度判斷是 ascending(向北飛)還是 descending(向南飛)。"""
    row_has_data = np.isfinite(lat).any(axis=1)
    valid_rows = np.where(row_has_data)[0]
    if len(valid_rows) == 0:
        return None
    lat_start = np.nanmean(lat[valid_rows[0]])
    lat_end = np.nanmean(lat[valid_rows[-1]])
    return lat_end >= lat_start


def collect_cycle_points(data_dir, cycle, lon_min, lon_max, lat_min, lat_max, quality_max=3,
                          window_days=None, verbose=True, pass_direction="all"):
    """掃描指定 cycle 內所有 Expert granule，回傳落在區域框內且通過品質篩選的
    (lon, lat, adt, time_seconds_since_epoch)。

    window_days: 只保留「cycle 時間中點 ± window_days/2」內的 granule，用來
    測試縮短拼圖窗口對覆蓋率的影響。

    pass_direction: all / ascending / descending。
    """
    pattern = os.path.join(data_dir, f"cycle_{cycle:03d}", "SWOT_L3_LR_SSH_Expert_*.nc")
    files = sorted(glob.glob(pattern))
    if verbose:
        print(f"cycle_{cycle:03d} 下共有 {len(files)} 個 granule")

    if window_days is not None:
        all_times = []
        for f in files:
            m = FNAME_RE.search(os.path.basename(f))
            if not m:
                continue
            _, _, dbeg, dend, _ = m.groups()
            fbeg, fend = parse_time(dbeg), parse_time(dend)
            all_times.append(fbeg + (fend - fbeg) / 2)
        if not all_times:
            return None
        cycle_mid = min(all_times) + (max(all_times) - min(all_times)) / 2
        half = timedelta(days=window_days / 2)
        w_start, w_end = cycle_mid - half, cycle_mid + half

        kept = []
        for f in files:
            m = FNAME_RE.search(os.path.basename(f))
            if not m:
                continue
            _, _, dbeg, dend, _ = m.groups()
            fbeg, fend = parse_time(dbeg), parse_time(dend)
            fmid = fbeg + (fend - fbeg) / 2
            if w_start <= fmid <= w_end:
                kept.append(f)
        if verbose:
            print(f"  window_days={window_days}: {len(files)} -> {len(kept)} 個 granule 落在窗口內")
        files = kept

    lons, lats, adts, times = [], [], [], []
    for f in files:
        m = FNAME_RE.search(os.path.basename(f))
        if not m:
            continue
        _, pas, dbeg, dend, version = m.groups()
        fbeg, fend = parse_time(dbeg), parse_time(dend)
        fmid = fbeg + (fend - fbeg) / 2

        try:
            with Dataset(f) as ds:
                lat = ma.filled(ds.variables["latitude"][:].astype(float), np.nan)
                lon = ma.filled(ds.variables["longitude"][:].astype(float), np.nan)
                qflag = ma.filled(ds.variables["quality_flag"][:].astype(float), 255)
                ssha = ma.filled(ds.variables["ssha_filtered"][:].astype(float), np.nan)
                mdt = ma.filled(ds.variables["mdt"][:].astype(float), np.nan)
        except Exception as e:
            print(f"  [略過，讀取失敗] pass={pas}: {e}")
            continue

        if pass_direction != "all":
            asc = is_ascending(lat)
            if asc is None:
                continue
            if pass_direction == "ascending" and not asc:
                continue
            if pass_direction == "descending" and asc:
                continue

        lon_shift = np.where(lon > 180, lon - 360, lon)
        region = (
            (lat >= lat_min) & (lat <= lat_max) &
            (lon_shift >= lon_min) & (lon_shift <= lon_max)
        )
        good = region & (qflag <= quality_max) & np.isfinite(ssha) & np.isfinite(mdt)
        n_good = int(np.count_nonzero(good))
        if verbose:
            print(f"  pass={pas} {fbeg:%Y-%m-%d %H:%M} 區域內有效點={n_good}")
        if n_good == 0:
            continue

        adt = ssha + mdt
        lons.append(lon_shift[good])
        lats.append(lat[good])
        adts.append(adt[good])
        t_epoch = fmid.timestamp()
        times.append(np.full(n_good, t_epoch))

    if not lons:
        return None

    return (
        np.concatenate(lons),
        np.concatenate(lats),
        np.concatenate(adts),
        np.concatenate(times),
    )


def bin_to_representative_points(lon_pts, lat_pts, adt_pts, time_pts, lon_min, lat_min,
                                  resolution, t_mid_epoch):
    """分箱到目標網格解析度；同一格內多筆觀測時只留離 cycle 中點時間最近的
    那一筆，而不是取平均，避免把21天內真實發生的海洋演化平均抹平。"""
    ix = np.floor((lon_pts - lon_min) / resolution).astype(int)
    iy = np.floor((lat_pts - lat_min) / resolution).astype(int)
    keys = ix.astype(np.int64) * 100000 + iy.astype(np.int64)

    time_diff = np.abs(time_pts - t_mid_epoch)
    order = np.argsort(time_diff)
    seen = {}
    for idx in order:
        k = keys[idx]
        if k not in seen:
            seen[k] = idx
    rep_idx = np.array(list(seen.values()))
    return lon_pts[rep_idx], lat_pts[rep_idx], adt_pts[rep_idx]


def build_land_mask(lon_2d, lat_2d):
    """回傳跟 lon_2d/lat_2d 同形狀的布林陣列，True 表示落在陸地上。

    griddata 插值不知道陸地/海洋的差別，兩側海面都有資料時會直接跨過陸地
    生出數值，導致 py-eddy-tracker 在陸地上偵測出不存在的假渦旋。這裡
    用 Natural Earth 10m 陸地多邊形做 point-in-polygon 判斷來擋掉。
    """
    shp_path = shapereader.natural_earth(resolution="10m", category="physical", name="land")
    lon_min, lon_max = lon_2d.min(), lon_2d.max()
    lat_min, lat_max = lat_2d.min(), lat_2d.max()
    points = np.column_stack([lon_2d.ravel(), lat_2d.ravel()])
    land = np.zeros(points.shape[0], dtype=bool)
    for geom in shapereader.Reader(shp_path).geometries():
        minx, miny, maxx, maxy = geom.bounds
        if maxx < lon_min or minx > lon_max or maxy < lat_min or miny > lat_max:
            continue
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            path = MplPath(np.asarray(poly.exterior.coords))
            land |= path.contains_points(points)
    return land.reshape(lon_2d.shape)


def build_grid(rep_lon, rep_lat, rep_adt, lon_min, lon_max, lat_min, lat_max,
                resolution, max_gap_km):
    lon_1d = np.arange(lon_min, lon_max + resolution / 2, resolution)
    lat_1d = np.arange(lat_min, lat_max + resolution / 2, resolution)
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

    adt_interp = griddata(
        (rep_lon, rep_lat), rep_adt, (lon_2d, lat_2d), method="cubic"
    )
    # cubic 在資料稀疏處常回傳 NaN，用 nearest 補一版備用
    adt_nearest = griddata(
        (rep_lon, rep_lat), rep_adt, (lon_2d, lat_2d), method="nearest"
    )
    adt_2d = np.where(np.isfinite(adt_interp), adt_interp, adt_nearest)

    mean_lat = (lat_min + lat_max) / 2
    kx = km_per_deg_lon(mean_lat)
    ky = KM_PER_DEG_LAT
    tree = cKDTree(np.column_stack([rep_lon * kx, rep_lat * ky]))
    query_pts = np.column_stack([lon_2d.ravel() * kx, lat_2d.ravel() * ky])
    dist_km, _ = tree.query(query_pts)
    dist_km = dist_km.reshape(lon_2d.shape)

    land_mask = build_land_mask(lon_2d, lat_2d)
    mask = (dist_km > max_gap_km) | ~np.isfinite(adt_2d) | land_mask
    n_valid = int(np.count_nonzero(~mask))
    n_total = mask.size
    print(f"網格總點數={n_total}，陸地格點={int(land_mask.sum())}，"
          f"距觀測點 <= {max_gap_km}km 的有效格點={n_valid} ({100*n_valid/n_total:.1f}%)")

    adt_masked = ma.array(adt_2d, mask=mask)
    return lon_1d, lat_1d, adt_masked


def run_eddy_identification(lon_1d, lat_1d, adt_masked, date, step, pixel_min, pixel_max,
                             low_pass_km=None, high_pass_km=None):
    # with_array 要求維度順序是 (X, Y) = (lon, lat)，我們的陣列是 (lat, lon)，先轉置
    adt_xy = adt_masked.T
    grid = RegularGridDataset.with_array(
        coordinates=("longitude", "latitude"),
        datas={"longitude": lon_1d, "latitude": lat_1d, "adt": adt_xy},
    )
    if high_pass_km:
        # 去掉大尺度背景(MDT等)，只留中小尺度異常，比照 py-eddy-tracker 官方
        # 範例(pet_sla_and_adt)的做法。濾完後 ADT 會變成以0為中心的異常場。
        grid.bessel_high_filter("adt", high_pass_km)
    if low_pass_km:
        # 疊加低通去掉小尺度雜訊，跟高通合起來形成帶通
        grid.bessel_low_filter("adt", low_pass_km)
    grid.add_uv("adt")
    anticyclonic, cyclonic = grid.eddy_identification(
        "adt", "u", "v", date, step=step,
        pixel_limit=(pixel_min, pixel_max),
        force_height_unit="m", force_speed_unit="m/s",
    )
    # 抓濾波後(偵測實際用的)場轉回 (lat, lon)，讓背景圖跟偵測邏輯用同一份資料
    adt_filtered = ma.array(grid.vars["adt"].T, mask=adt_masked.mask)
    return grid, anticyclonic, cyclonic, adt_filtered


def summarize_eddies(obs, kind):
    rows = []
    if obs is None or len(obs) == 0:
        return rows
    dt = getattr(obs, "dtype", [])
    if hasattr(dt, "names") and dt.names is not None:
        names = dt.names
    else:
        # 這個環境的 py-eddy-tracker 版本 dtype 回傳的是 list of tuples
        names = [d[0] for d in dt]
    print(f"  [{kind}] 欄位: {names}")
    for i in range(len(obs)):
        row = {"kind": kind}
        for field in ("lon", "lat", "radius_e", "radius_s", "amplitude", "speed_average"):
            if field in names:
                row[field] = float(obs[field][i])
        rows.append(row)
    return rows


def plot_result(lon_1d, lat_1d, adt_field, anticyclonic, cyclonic, out_path, box, title_suffix="",
                 diverging=True):
    lon_min, lon_max, lat_min, lat_max = box
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=100)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=101)
    gl = ax.gridlines(draw_labels=True, linestyle=":", linewidth=0.5, color="gray")
    gl.top_labels = False
    gl.right_labels = False

    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    if diverging:
        # 高通濾波後的場以0為中心(正=暖丘/反氣旋，負=冷丘/氣旋)，色階對稱於0
        if adt_field.count():
            vmax = np.nanpercentile(np.abs(adt_field.compressed()), 95)
        else:
            vmax = 0.1
        norm = colors.Normalize(vmin=-vmax, vmax=vmax)
    else:
        if adt_field.count():
            vmin, vmax = np.nanpercentile(adt_field.compressed(), [5, 95])
        else:
            vmin, vmax = -0.3, 0.3
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    pc = ax.pcolormesh(
        lon_2d, lat_2d, adt_field, cmap="RdBu_r",
        norm=norm, transform=proj, shading="auto",
    )
    fig.colorbar(pc, ax=ax, label="ADT anomaly [m]" if diverging else "ADT [m]", shrink=0.8)

    ref_lon = (lon_min + lon_max) / 2
    for obs, color, label in ((anticyclonic, "red", "anticyclonic"), (cyclonic, "blue", "cyclonic")):
        if obs is None or len(obs) == 0:
            continue
        # 用官方 display() 畫出實際偵測到的等值線輪廓，而不是固定大小的圓圈
        obs.display(ax, lw=1.2, color=color, label=f"{label} ({{nb_obs}} eddies)", ref=ref_lon,
                    transform=proj)

    ax.legend(fontsize=8, loc="upper right")
    if diverging:
        title = f"SWOT Eddy Detection (ADT anomaly, high-pass filtered){title_suffix}"
    else:
        title = f"SWOT Eddy Detection (ADT, filtered){title_suffix}"
    ax.set_title(title, fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"已存: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="單一 cycle 的 SWOT 網格化 + 旋渦偵測")
    ap.add_argument("--cycle", type=int, required=True, help="cycle 編號")
    ap.add_argument("--lon-min", type=float, default=118.0)
    ap.add_argument("--lon-max", type=float, default=128.0)
    ap.add_argument("--lat-min", type=float, default=18.0)
    ap.add_argument("--lat-max", type=float, default=26.0)
    ap.add_argument("--resolution", type=float, default=0.07, help="網格解析度(度)")
    ap.add_argument("--max-gap-km", type=float, default=30.0, help="離最近觀測點的最大距離門檻(km)")
    ap.add_argument("--step", type=float, default=0.01, help="py-eddy-tracker 高度分層間距(m)")
    ap.add_argument("--pixel-min", type=int, default=4, help="旋渦最內層等值線最少像素數")
    ap.add_argument("--pixel-max", type=int, default=1000, help="旋渦最外層等值線最多像素數")
    ap.add_argument("--low-pass-km", type=float, default=20.0,
                     help="偵測前對 ADT 場做低通濾波的波長門檻(km)，0 表示不濾波")
    ap.add_argument("--high-pass-km", type=float, default=150.0,
                     help="偵測前對 ADT 場做高通濾波的波長門檻(km)，去掉大尺度背景(MDT等)，"
                          "0 表示不做高通濾波(維持原本絕對ADT背景圖)")
    ap.add_argument("--window-days", type=float, default=None,
                     help="只用 cycle 時間中點 ± window/2 天內的 granule 拼圖(預設用整個cycle)")
    ap.add_argument("--pass-direction", choices=["all", "ascending", "descending"], default="all",
                     help="只收集ascending或descending的pass來拼圖(預設all=兩者都用，建議維持預設)")
    ap.add_argument("--quiet", action="store_true", help="不印逐 pass 的細節，只留摘要")
    ap.add_argument("--data-dir", default="/home/donnee/SWOT/v300_expert")
    ap.add_argument("--out-dir", default="./eddy_out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    box = (args.lon_min, args.lon_max, args.lat_min, args.lat_max)

    result = collect_cycle_points(
        args.data_dir, args.cycle, args.lon_min, args.lon_max, args.lat_min, args.lat_max,
        window_days=args.window_days, verbose=not args.quiet, pass_direction=args.pass_direction,
    )
    if result is None:
        print("這個 cycle 在指定區域內沒有有效資料，中止。")
        return
    lon_pts, lat_pts, adt_pts, time_pts = result
    print(f"總共收集到 {len(lon_pts)} 個有效觀測點")

    t_mid_epoch = (time_pts.min() + time_pts.max()) / 2
    cycle_mid_date = datetime.utcfromtimestamp(t_mid_epoch)
    print(f"cycle 中點日期(UTC): {cycle_mid_date}")

    rep_lon, rep_lat, rep_adt = bin_to_representative_points(
        lon_pts, lat_pts, adt_pts, time_pts,
        args.lon_min, args.lat_min, args.resolution, t_mid_epoch,
    )
    print(f"分箱後代表點數: {len(rep_lon)}")

    lon_1d, lat_1d, adt_masked = build_grid(
        rep_lon, rep_lat, rep_adt,
        args.lon_min, args.lon_max, args.lat_min, args.lat_max,
        args.resolution, args.max_gap_km,
    )

    grid, anticyclonic, cyclonic, adt_filtered = run_eddy_identification(
        lon_1d, lat_1d, adt_masked, cycle_mid_date, args.step, args.pixel_min, args.pixel_max,
        low_pass_km=args.low_pass_km, high_pass_km=args.high_pass_km,
    )
    print(f"偵測到 anticyclonic={len(anticyclonic)}  cyclonic={len(cyclonic)}")
    plot_field = adt_filtered if args.high_pass_km else adt_masked

    suffix = "" if args.pass_direction == "all" else f"_{args.pass_direction}"

    rows = summarize_eddies(anticyclonic, "anticyclonic") + summarize_eddies(cyclonic, "cyclonic")
    csv_path = os.path.join(args.out_dir, f"eddies_cycle{args.cycle:03d}{suffix}.csv")
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
        print(f"已存: {csv_path}")
    else:
        print("沒有偵測到任何旋渦，不輸出 CSV。")

    plot_result(
        lon_1d, lat_1d, plot_field, anticyclonic, cyclonic,
        os.path.join(args.out_dir, f"eddies_cycle{args.cycle:03d}{suffix}.png"),
        box, title_suffix=f" — {args.pass_direction.capitalize()}" if args.pass_direction != "all" else "",
        diverging=bool(args.high_pass_km),
    )


if __name__ == "__main__":
    main()
