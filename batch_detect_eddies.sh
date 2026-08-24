#!/bin/bash
# 批次跑 detect_eddies.py，cycle_001 ~ cycle_050(Science Phase)
# 用法: bash batch_detect_eddies.sh [起始cycle] [結束cycle] [pass_direction: all|ascending|descending]
# 搭配 nohup 背景執行:
#   nohup bash batch_detect_eddies.sh 1 50 all > ~/eddy_out/batch.log 2>&1 &

set -e
START=${1:-1}
END=${2:-50}
DIRECTION=${3:-all}

if [ "$DIRECTION" = "all" ]; then
    OUT_ROOT=~/eddy_out
else
    OUT_ROOT=~/eddy_out_${DIRECTION}
fi

mkdir -p "$OUT_ROOT"

for CYCLE in $(seq -w "$START" "$END"); do
    CYCLE_NUM=$((10#$CYCLE))
    echo "===== cycle_${CYCLE} [${DIRECTION}] ($(date)) ====="
    python3 ~/detect_eddies.py \
        --cycle "$CYCLE_NUM" \
        --pass-direction "$DIRECTION" \
        --step 0.01 --pixel-min 4 --pixel-max 1000 --low-pass-km 20 \
        --quiet \
        --out-dir "$OUT_ROOT/cycle_${CYCLE}" \
        || echo "  [cycle_${CYCLE} 失敗，繼續下一個]"
done

echo "全部跑完，接著合併 CSV:"
python3 ~/combine_eddy_csvs.py --root "$OUT_ROOT" --out "$OUT_ROOT/all_cycles_eddies.csv"
