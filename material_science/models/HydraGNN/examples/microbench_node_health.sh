#!/usr/bin/env bash
# microbench_node_health.sh
#
# Per-node hardware/firmware/health survey for AMD MI355X (gfx950) nodes.
# Runs four micro-tests in ~30 wall-clock seconds:
#   1. host inventory                  - CPU, NUMA, kernel, NICs, PCIe link, mounts, /tmp dd
#   2. GPU + driver inventory          - rocminfo, rocm-smi (vbios/fw/topo), PCIe link state
#   3. CPU dual-NUMA STREAM            - COPY/SCALE/ADD/TRIAD, 128 threads, NUMA-interleaved
#   4. HIP kernel launch latency       - 100k empty-tensor kernels per GPU
#
# Designed to be:
#   - safe to run in a SLURM Prolog
#   - cheap enough to run on every job (~30s)
#   - detailed enough to spot the three failure modes we hit in May 2026:
#       a) NUMA bandwidth degradation (one DIMM/socket regression)
#       b) container bind-mount fault (NFS / overlay regression)
#       c) GPU dispatch-rate regression (driver/SMI hang)
#
# Usage:
#   sbatch --nodelist=<nodename> material_science/models/HydraGNN/examples/microbench_node_health.sh
#   srun   --nodelist=<nodename> material_science/models/HydraGNN/examples/microbench_node_health.sh
#
# Output: writes ${OUT_DIR}/<hostname>/ with one file per test,
#         and one summary line of pass/fail to stdout.
#
# Env vars (all optional, with defaults):
#   OUT_DIR            (default: $AI4S_SHARED_DIR/microbench-node-health)
#   HG_SIF             (default: $AI4S_SHARED_DIR/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif)
#   AI4S_SHARED_DIR    (default: /shared/$USER)
#   STREAM_THRESHOLD_COPY_GBPS    (default: 200)    - dual-NUMA COPY pass floor
#   STREAM_THRESHOLD_TRIAD_GBPS   (default: 200)    - dual-NUMA TRIAD pass floor
#   HIP_LAUNCH_MAX_US             (default: 6.0)    - per-GPU avg launch latency ceiling
#
# SBATCH directives:
#SBATCH --job-name=node-health
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:05:00
#SBATCH --output=microbench-node-health-%j.out
#
# NOTE: GPU tests are gated on whether SLURM allocated GPUs to the step. To
# get GPU coverage, sbatch with `--gres=gpu:amd_instinct_mi355_oam:8`.

set -u

H=$(hostname -s)
OUT_DIR="${OUT_DIR:-${AI4S_SHARED_DIR:-/tmp}/microbench-node-health}"
HOST_OUT="$OUT_DIR/$H"
mkdir -p "$HOST_OUT"

AI4S_SHARED_DIR="${AI4S_SHARED_DIR:-/tmp}"  # set AI4S_SHARED_DIR to your cluster's shared storage root
HG_SIF="${HG_SIF:-${AI4S_SHARED_DIR}/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif}"
STREAM_THRESHOLD_COPY_GBPS="${STREAM_THRESHOLD_COPY_GBPS:-200}"
STREAM_THRESHOLD_TRIAD_GBPS="${STREAM_THRESHOLD_TRIAD_GBPS:-200}"
HIP_LAUNCH_MAX_US="${HIP_LAUNCH_MAX_US:-6.0}"

START_TS=$(date +%s)
echo "[$H] microbench start at $(date -u +%FT%TZ)" | tee "$HOST_OUT/_start.txt"

# ---------- Test 1: host inventory ----------
{
  lscpu                                              > "$HOST_OUT/lscpu.txt"        2>&1
  numactl --hardware                                 > "$HOST_OUT/numactl.txt"      2>&1
  uname -a                                           > "$HOST_OUT/uname.txt"        2>&1
  cat /proc/cmdline                                  > "$HOST_OUT/cmdline.txt"      2>&1
  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor > "$HOST_OUT/cpu_governor.txt" 2>&1 || echo "no cpufreq" > "$HOST_OUT/cpu_governor.txt"
  grep -E '^Mem|HugePages|^DirectMap|^Swap' /proc/meminfo > "$HOST_OUT/meminfo.txt" 2>&1
  ip -br link                                        > "$HOST_OUT/ip_link.txt"      2>&1
  mount | grep -E " /home| ${AI4S_SHARED_DIR}|tmpfs|/tmp"  > "$HOST_OUT/mounts.txt"  2>&1
  df -h "$HOME" "$AI4S_SHARED_DIR" /tmp 2>/dev/null  > "$HOST_OUT/df.txt"           2>&1
}

# ---------- Test 2: container mount probe ----------
# Catches a6-class fault: container can't write to $HOME or $AI4S_SHARED_DIR
MOUNT_PROBE_PASS="yes"
for d in "$HOME" "$AI4S_SHARED_DIR"; do
  tmpf="$d/.health_probe_${SLURM_JOB_ID:-$$}"
  if ! ( echo ok > "$tmpf" && rm -f "$tmpf" ) 2>/dev/null; then
    MOUNT_PROBE_PASS="no($d)"
    break
  fi
done
echo "mount_probe: $MOUNT_PROBE_PASS" > "$HOST_OUT/mount_probe.txt"

# ---------- Test 3: dual-NUMA STREAM ----------
# Allocates ~6 GiB. Took 30-50s on a 2-socket Genoa MI355X node.
TMP=$(mktemp -d)
cat > "$TMP/stream.c" <<'C'
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define N (1L << 28)
#define NITER 8
double *a, *b, *c;
static double tsec(void){
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec*1e-9;
}
int main(void){
  a = aligned_alloc(64, N*sizeof(double));
  b = aligned_alloc(64, N*sizeof(double));
  c = aligned_alloc(64, N*sizeof(double));
  #pragma omp parallel for
  for(long i=0;i<N;i++){ a[i]=1.0; b[i]=2.0; c[i]=0.0; }
  printf("# nthreads=%d N=%ld arr_MiB=%ld\n", omp_get_max_threads(), N, N*8>>20);
  printf("# kernel  bytes  best_GBps  avg_GBps\n");
  for(int k=0;k<4;k++){
    double bw_best = -1e30, bw_sum = 0;
    for(int it=0;it<NITER;it++){
      double t = tsec();
      if(k==0)      { /* COPY  c=a */
        #pragma omp parallel for
        for(long i=0;i<N;i++) c[i]=a[i];
      } else if (k==1){ /* SCALE b=2c */
        #pragma omp parallel for
        for(long i=0;i<N;i++) b[i]=2.0*c[i];
      } else if (k==2){ /* ADD   c=a+b */
        #pragma omp parallel for
        for(long i=0;i<N;i++) c[i]=a[i]+b[i];
      } else {          /* TRIAD a=b+3c */
        #pragma omp parallel for
        for(long i=0;i<N;i++) a[i]=b[i]+3.0*c[i];
      }
      double dt = tsec()-t;
      double bytes = (k<2 ? 2.0 : 3.0) * N * 8;
      double bw = bytes/1e9/dt;
      if (bw>bw_best) bw_best = bw;
      bw_sum += bw;
    }
    const char *kn[] = {"COPY","SCALE","ADD","TRIAD"};
    long bytes_per_iter = (long)((k<2 ? 2.0 : 3.0) * N * 8);
    printf("%s    %ld  %.2f  %.2f\n", kn[k], bytes_per_iter, bw_best, bw_sum/NITER);
  }
  return 0;
}
C
gcc -O3 -fopenmp -march=native -o "$TMP/stream" "$TMP/stream.c" >/dev/null 2>&1
{
  echo "--- single-NUMA (node 0, 64 threads) ---"
  OMP_NUM_THREADS=64 OMP_PROC_BIND=true OMP_PLACES=cores numactl --cpunodebind=0 --membind=0 "$TMP/stream"
  echo ""
  echo "--- dual-NUMA (interleaved, 128 threads) ---"
  OMP_NUM_THREADS=128 OMP_PROC_BIND=true OMP_PLACES=cores numactl --interleave=all "$TMP/stream"
} > "$HOST_OUT/stream.txt" 2>&1
rm -rf "$TMP"

# Pass/fail extraction (dual-NUMA section)
STREAM_COPY=$(awk '/--- dual-NUMA/{flag=1; next} flag && /^COPY/{print $3; exit}'  "$HOST_OUT/stream.txt")
STREAM_TRIAD=$(awk '/--- dual-NUMA/{flag=1; next} flag && /^TRIAD/{print $3; exit}' "$HOST_OUT/stream.txt")
STREAM_PASS="yes"
[ -z "$STREAM_COPY" ]  && STREAM_PASS="no(no_output)"
[ -n "$STREAM_COPY" ]  && awk -v v="$STREAM_COPY" -v t="$STREAM_THRESHOLD_COPY_GBPS"  'BEGIN{exit !(v+0 < t+0)}' \
    && STREAM_PASS="no(COPY ${STREAM_COPY} < ${STREAM_THRESHOLD_COPY_GBPS})"
[ -n "$STREAM_TRIAD" ] && awk -v v="$STREAM_TRIAD" -v t="$STREAM_THRESHOLD_TRIAD_GBPS" 'BEGIN{exit !(v+0 < t+0)}' \
    && STREAM_PASS="no(TRIAD ${STREAM_TRIAD} < ${STREAM_THRESHOLD_TRIAD_GBPS})"

# ---------- Test 4: GPU + HIP launch latency (only if GPUs allocated) ----------
HIP_PASS="skipped(no_gpu_allocation)"
HIP_MEAN_US=""
if command -v rocm-smi >/dev/null 2>&1 || [ -e "$HG_SIF" ]; then
  if [ -e "$HG_SIF" ]; then
    apptainer exec --rocm --bind /opt/rocm-7.2.2:/opt/rocm-7.2.2 --bind "$AI4S_SHARED_DIR" \
      --env LD_LIBRARY_PATH=/opt/rocm-7.2.2/lib:/usr/lib/x86_64-linux-gnu \
      "$HG_SIF" bash -c "
PATH=/opt/rocm-7.2.2/bin:\$PATH rocminfo                                              > '$HOST_OUT/rocminfo.txt'           2>&1
PATH=/opt/rocm-7.2.2/bin:\$PATH rocm-smi --showvbios --showfwinfo --showdriverversion > '$HOST_OUT/rocm_smi_vbios_fw.txt' 2>&1
PATH=/opt/rocm-7.2.2/bin:\$PATH rocm-smi --showtopo                                   > '$HOST_OUT/rocm_smi_topo.txt'     2>&1
PATH=/opt/rocm-7.2.2/bin:\$PATH python3 - <<'PY'
import time, torch
N = torch.cuda.device_count()
NLAUNCH = 100000
lats = []
for d in range(N):
    torch.cuda.set_device(d)
    x = torch.zeros(1, device=f'cuda:{d}')
    for _ in range(1000): x.zero_()
    torch.cuda.synchronize(d)
    t0 = time.perf_counter()
    for _ in range(NLAUNCH): x.zero_()
    torch.cuda.synchronize(d)
    dt = time.perf_counter() - t0
    us = dt/NLAUNCH*1e6
    print(f'gpu{d}: {us:.2f} us/launch ({NLAUNCH/dt/1e6:.3f} M/s)')
    lats.append(us)
print(f'mean_us: {sum(lats)/len(lats):.2f}  max_us: {max(lats):.2f}  min_us: {min(lats):.2f}  n_gpu: {N}')
PY
" > "$HOST_OUT/hip_launch.txt" 2>&1
    HIP_MEAN_US=$(awk '/^mean_us:/{print $2}' "$HOST_OUT/hip_launch.txt")
    if [ -n "$HIP_MEAN_US" ]; then
      HIP_PASS="yes"
      awk -v v="$HIP_MEAN_US" -v t="$HIP_LAUNCH_MAX_US" 'BEGIN{exit !(v+0 > t+0)}' \
        && HIP_PASS="no(mean ${HIP_MEAN_US} us > ${HIP_LAUNCH_MAX_US})"
    else
      HIP_PASS="no(no_torch_output)"
    fi
  fi
fi

# /sys-based GPU PCIe link state (no container, no GPU allocation needed)
{
  for card in /sys/class/drm/card*/device; do
    n=$(basename $(dirname "$card"))
    cls=$(cat "$card/current_link_speed" 2>/dev/null || echo n/a)
    clw=$(cat "$card/current_link_width" 2>/dev/null || echo n/a)
    mls=$(cat "$card/max_link_speed"     2>/dev/null || echo n/a)
    mlw=$(cat "$card/max_link_width"     2>/dev/null || echo n/a)
    echo "$n  cur=$cls/$clw  max=$mls/$mlw"
  done
} > "$HOST_OUT/pcie_link_state.txt" 2>&1

# ---------- Summary line ----------
END_TS=$(date +%s)
SUMMARY="host=$H elapsed_s=$((END_TS - START_TS)) mount=$MOUNT_PROBE_PASS stream=$STREAM_PASS stream_copy_gbps=${STREAM_COPY:--} stream_triad_gbps=${STREAM_TRIAD:--} hip_launch=$HIP_PASS hip_mean_us=${HIP_MEAN_US:--}"
echo "$SUMMARY" > "$HOST_OUT/_summary.txt"
echo "$SUMMARY"

# Exit nonzero if any test failed (suitable for SLURM Prolog use)
case "$MOUNT_PROBE_PASS$STREAM_PASS$HIP_PASS" in
  yesyesyes|yesyesskipped*) exit 0 ;;
  *) exit 1 ;;
esac
