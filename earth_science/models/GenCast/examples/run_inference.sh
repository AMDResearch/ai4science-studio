#!/usr/bin/env bash
# run_inference.sh — Run a GenCast ensemble forecast inside the JAX container
#
# Run this INSIDE the container launched by docker_run.sh.
#
# OPTIONAL EDITS:
#   MODEL_VARIANT        — which GenCast checkpoint to use (default: gencast-0.25)
#   DATE                 — forecast start date (YYYYMMDD, default: yesterday)
#   TIME                 — forecast start time (HHMM, default: 0000)
#   LEAD_TIME            — forecast horizon in hours (default: 240 = 10 days)
#   NUM_ENSEMBLE_MEMBERS — number of ensemble members (must be multiple of GPU count)
#   ASSETS_DIR           — where to cache downloaded model weights
#   OUTPUT_DIR           — where to write GRIB output

set -euo pipefail

# --- Configuration ---
# Available variants: gencast-0.25, gencast-0.25-Oper, gencast-1.0, gencast-1.0-Mini
MODEL_VARIANT="${MODEL_VARIANT:-gencast-0.25}"
DATE="${DATE:-$(date -u -d 'yesterday' '+%Y%m%d' 2>/dev/null || date -u -v-1d '+%Y%m%d')}"
TIME="${TIME:-0000}"
LEAD_TIME="${LEAD_TIME:-240}"
NUM_ENSEMBLE_MEMBERS="${NUM_ENSEMBLE_MEMBERS:-1}"
ASSETS_DIR="${ASSETS_DIR:-/predictions/assets/$MODEL_VARIANT}"
OUTPUT_DIR="${OUTPUT_DIR:-/predictions}"

mkdir -p "$ASSETS_DIR" "$OUTPUT_DIR/logs"

# Optional: source XLA params for reduced peak memory usage
if [[ -f /recipe/set_XLA_params.sh ]]; then
    echo "Sourcing XLA memory parameters ..."
    source /recipe/set_XLA_params.sh
fi

echo "=== GenCast Inference ==="
echo "  Variant          : $MODEL_VARIANT"
echo "  Date             : $DATE"
echo "  Time             : $TIME"
echo "  Lead time        : ${LEAD_TIME}h"
echo "  Ensemble members : $NUM_ENSEMBLE_MEMBERS"
echo "  Assets           : $ASSETS_DIR"
echo "  Output           : $OUTPUT_DIR/${MODEL_VARIANT}.grib"
echo ""
echo "Note: GenCast is compute-intensive. A 10-day ensemble forecast may take"
echo "      several minutes. Progress is logged to $OUTPUT_DIR/logs/${MODEL_VARIANT}.log"
echo ""

ai-models \
    --download-assets \
    --assets "$ASSETS_DIR" \
    --input=cds \
    --date="$DATE" \
    --time="$TIME" \
    --lead-time="$LEAD_TIME" \
    --num-ensemble-members="$NUM_ENSEMBLE_MEMBERS" \
    --path="$OUTPUT_DIR/${MODEL_VARIANT}.grib" \
    "$MODEL_VARIANT" \
    > "$OUTPUT_DIR/logs/${MODEL_VARIANT}.log" 2>&1 &

PID=$!
echo "Running as background process (PID: $PID)"
echo "  tail -f $OUTPUT_DIR/logs/${MODEL_VARIANT}.log"
echo ""
tail -f "$OUTPUT_DIR/logs/${MODEL_VARIANT}.log" &
wait $PID

echo ""
echo "Output written to: $OUTPUT_DIR/${MODEL_VARIANT}.grib"
echo ""
echo "To visualize:"
echo "  python3 /recipe/grib_visualizer.py --input $OUTPUT_DIR/${MODEL_VARIANT}.grib"
