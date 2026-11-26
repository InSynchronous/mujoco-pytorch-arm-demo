#!/usr/bin/env bash
# Collect N randomized expert episodes across parallel shards.
#   ./scripts/collect_dataset.sh data 2000 6 1
set -euo pipefail
cd "$(dirname "$0")/.."

out=${1:-data}
episodes=${2:-2000}
shards=${3:-6}
seed=${4:-1}
width=${5:-128}
height=${6:-128}

mkdir -p "$out"
pids=()
for ((k = 0; k < shards; k++)); do
  ./build/collect_data --out "$out" --episodes "$episodes" --shards "$shards" \
    --shard "$k" --seed "$seed" --width "$width" --height "$height" \
    >"$out/collect_${k}.log" 2>&1 &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if ((status != 0)); then
  echo "a collector failed; see $out/collect_*.log" >&2
  exit 1
fi

python3 - "$out" <<'PY'
import sys
from pathlib import Path
import pandas as pd
base = Path(sys.argv[1])
total = ok = 0
for shard in sorted(base.glob("shard_*")):
    df = pd.read_csv(shard / "episodes.csv")
    total += len(df)
    ok += int(df["success"].sum())
print(f"collected {total} episodes, {ok} successful")
PY
