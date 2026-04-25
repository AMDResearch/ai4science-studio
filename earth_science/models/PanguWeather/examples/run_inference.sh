#!/usr/bin/env bash
# run_inference.sh — Run a PanguWeather forecast inside the JAX container
#
# Run this INSIDE the container launched by docker_run.sh.
#
# OPTIONAL EDITS:
#   DATE        — forecast start date (YYYYMMDD, default: yesterday)
#   TIME        — forecast start time (HHMM, default: 0000)
#   LEAD_TIME   — forecast horizon in hours (default: 24; max ~168)
#   ASSETS_DIR  — where to cache downloaded model weights (default: /predictions/assets)
#   OUTPUT_DIR  — where to write GRIB output (default: /predictions)

set -euo pipefail

MODEL_NAME="panguweather"

# Default to yesterday so CDS data is available
DATE="${DATE:-$(date -u -d 'yesterday' '+%Y%m%d' 2>/dev/null || date -u -v-1d '+%Y%m%d')}"
TIME="${TIME:-0000}"
LEAD_TIME="${LEAD_TIME:-24}"
ASSETS_DIR="${ASSETS_DIR:-/predictions/assets/$MODEL_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-/predictions}"

mkdir -p "$ASSETS_DIR" "$OUTPUT_DIR/logs"

echo "=== PanguWeather Inference ==="
echo "  Date      : $DATE"
echo "  Time      : $TIME"
echo "  Lead time : ${LEAD_TIME}h"
echo "  Assets    : $ASSETS_DIR"
echo "  Output    : $OUTPUT_DIR/${MODEL_NAME}.grib"
echo ""

ai-models \
    --download-assets \
    --assets "$ASSETS_DIR" \
    --input=cds \
    --date="$DATE" \
    --time="$TIME" \
    --lead-time="$LEAD_TIME" \
    --path="$OUTPUT_DIR/${MODEL_NAME}.grib" \
    "$MODEL_NAME"

echo ""
echo "Output written to: $OUTPUT_DIR/${MODEL_NAME}.grib"
echo ""
echo "To visualize (if grib_visualizer.py is on the recipe mount):"
echo "  python3 /recipe/grib_visualizer.py --input $OUTPUT_DIR/${MODEL_NAME}.grib"
