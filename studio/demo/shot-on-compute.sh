#!/bin/bash
# Screenshot a URL or file on a COMPUTE NODE with the right chrome library paths.
#
# Distilled from studio/demo/record_demo.slurm, which is the ONLY reliable way to
# run Playwright chromium on this cluster. Two hard-won tricks it encodes:
#   1. The login node cannot run chrome (memory cgroup kills the icu mmap) and the
#      ROCm container lacks chrome's GTK/X libs. So chrome runs on a compute node
#      with GTK/X libs staged via `apt-get download` + `dpkg-deb -x` (no root) onto
#      LD_LIBRARY_PATH.
#   2. NFS home cannot mmap chrome's icu data file, so chromium-1228 is copied to
#      local /scratch and PLAYWRIGHT_BROWSERS_PATH points there.
#
# Usage (attach to a running holder allocation):
#   SHOT_JOBID=<holder> SHOT_TARGET=<url|file:///...> SHOT_OUT=/path/out.png \
#     SHOT_SCRIPT=<node-script.js> bash shot-on-compute.sh
#
# SHOT_SCRIPT is a node script (run from studio/demo, where playwright resolves)
# that reads SHOT_TARGET / SHOT_OUT from env and does the capture.
set -uo pipefail

JOBID="${SHOT_JOBID:?set SHOT_JOBID to a running holder allocation}"
TARGET="${SHOT_TARGET:?set SHOT_TARGET to a url or file:// path}"
OUT="${SHOT_OUT:?set SHOT_OUT to the output png path}"
SCRIPT="${SHOT_SCRIPT:?set SHOT_SCRIPT to the capture node script}"
W="${SHOT_W:-1520}"; H="${SHOT_H:-940}"

DEMO="$HOME/Projects/ai4science-studio/studio/demo"
_U=(); while IFS= read -r v; do _U+=(-u "$v"); done \
  < <(env | grep -oE '^(PMIX_|PMI_|OMPI_)[A-Za-z0-9_]+')

srun --jobid="$JOBID" --overlap --ntasks=1 --cpu-bind=none env "${_U[@]}" bash -lc "
  set -e
  DEPS=/scratch/\$USER/shot_deps_\$\$
  CHROME_LOCAL=/scratch/\$USER/shot_chrome_\$\$
  mkdir -p \"\$DEPS\" \"\$CHROME_LOCAL\"

  echo '[shot] staging GTK/X libs...'
  cd /tmp
  apt-get download \
    libatk1.0-0t64 libatk-bridge2.0-0t64 libcups2t64 libasound2t64 \
    libxcomposite1 libxdamage1 libxrandr2 libatspi2.0-0t64 \
    libgbm1 libxkbcommon0 libxi6 2>&1 | grep -vE '^Get:|^Fetched|^Reading|^W:' || true
  for d in /tmp/*.deb; do [ -f \"\$d\" ] && dpkg-deb -x \"\$d\" \"\$DEPS/\"; done
  export LD_LIBRARY_PATH=\"\$DEPS/usr/lib/x86_64-linux-gnu:\$DEPS/usr/lib:\${LD_LIBRARY_PATH:-}\"

  echo '[shot] copying chromium to scratch (NFS mmap workaround)...'
  cp -r ~/.cache/ms-playwright/chromium-1228 \"\$CHROME_LOCAL/\"
  export PLAYWRIGHT_BROWSERS_PATH=\"\$CHROME_LOCAL\"

  export SHOT_TARGET='$TARGET' SHOT_OUT='$OUT' SHOT_W='$W' SHOT_H='$H'
  cd '$DEMO'
  node '$SCRIPT' 2>&1 | tail -5
"
