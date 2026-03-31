#!/usr/bin/env bash
# run_inference.sh — Run an Aurora forecast inside the PyTorch container
#
# Run this INSIDE the container launched by docker_run.sh.
#
# OPTIONAL EDITS:
#   DATE        — forecast start date (YYYYMMDD, default: yesterday)
#   TIME        — forecast start time (HHMM, default: 0000)
#   LEAD_TIME   — forecast horizon in hours (default: 24; rollout in 6h steps)
#   ASSETS_DIR  — where to cache downloaded model weights
#   OUTPUT_DIR  — where to write GRIB output

set -euo pipefail

MODEL_NAME="aurora"

DATE="${DATE:-$(date -u -d 'yesterday' '+%Y%m%d' 2>/dev/null || date -u -v-1d '+%Y%m%d')}"
TIME="${TIME:-0000}"
LEAD_TIME="${LEAD_TIME:-24}"
ASSETS_DIR="${ASSETS_DIR:-/predictions/assets/$MODEL_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-/predictions}"

mkdir -p "$ASSETS_DIR" "$OUTPUT_DIR/logs"

echo "=== Aurora Inference ==="
echo "  Date      : $DATE"
echo "  Time      : $TIME"
echo "  Lead time : ${LEAD_TIME}h (in 6h steps)"
echo "  Resolution: 0.1°"
echo "  Assets    : $ASSETS_DIR"
echo "  Output    : $OUTPUT_DIR/${MODEL_NAME}.grib"
echo ""
echo "Note: Aurora runs at 0.1° resolution — higher memory usage than"
echo "      PanguWeather or GenCast. Progress logged to logs/${MODEL_NAME}.log"
echo ""

ai-models \
    --download-assets \
    --assets "$ASSETS_DIR" \
    --input=cds \
    --date="$DATE" \
    --time="$TIME" \
    --lead-time="$LEAD_TIME" \
    --path="$OUTPUT_DIR/${MODEL_NAME}.grib" \
    "$MODEL_NAME" \
    > "$OUTPUT_DIR/logs/${MODEL_NAME}.log" 2>&1 &

PID=$!
echo "Running as background process (PID: $PID)"
echo "  tail -f $OUTPUT_DIR/logs/${MODEL_NAME}.log"
echo ""
tail -f "$OUTPUT_DIR/logs/${MODEL_NAME}.log" &
wait $PID

echo ""
echo "Output written to: $OUTPUT_DIR/${MODEL_NAME}.grib"
echo ""
echo "To visualize:"
echo "  python3 /recipe/grib_visualizer.py --input $OUTPUT_DIR/${MODEL_NAME}.grib"
