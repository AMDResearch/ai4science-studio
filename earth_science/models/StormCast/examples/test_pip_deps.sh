#!/usr/bin/env bash
# Interactive test: time each pip install step for StormCast inside the SIF.
# Run via: salloc --partition=lux --account=vultr_lux --nodes=1 --gres=gpu:1 --time=01:00:00
# Then on the node: bash test_pip_deps.sh
#
# Purpose: identify which packages are slow / require source builds so we can
# optimize the SLURM split-build approach.

set -euo pipefail

SC_SIF="${SC_SIF:-/shared/spannala/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif}"
TEST_DIR="${TEST_DIR:-/tmp/sc-pip-test-$$}"
ROCM_WHL_TAG="${ROCM_WHL_TAG:-rocm7.2}"

echo "=== StormCast pip dependency timing test ==="
echo "SIF      : $SC_SIF"
echo "Test dir : $TEST_DIR"
echo "Date     : $(date)"
echo ""

cat > /tmp/sc_test_install_$$.sh << 'INNEREOF'
#!/usr/bin/env bash
set -euo pipefail
source /opt/venv/bin/activate

TARGET="$1"
ROCM_WHL_TAG="$2"
mkdir -p "$TARGET"

time_step() {
    local label="$1"; shift
    echo ""
    echo "--- [STEP] $label ---"
    local t0=$SECONDS
    "$@"
    local elapsed=$(( SECONDS - t0 ))
    echo "  --> $label: ${elapsed}s"
}

time_step "earth2studio (--no-deps)" \
    pip install -q --no-cache-dir --no-deps --target "$TARGET" earth2studio 2>&1 | tail -3

time_step "nvidia-physicsnemo (--no-deps)" \
    pip install -q --no-cache-dir --no-deps --target "$TARGET" "nvidia-physicsnemo>=2.0" 2>&1 | tail -3

time_step "timm, warp-lang, tensordict (--no-deps)" \
    pip install -q --no-cache-dir --no-deps \
        --target "$TARGET" timm warp-lang tensordict 2>&1 | tail -3

# Split the big dep list into groups to isolate slowpokes
time_step "group-A: cftime gcsfs h5netcdf h5py huggingface-hub loguru nest-asyncio netcdf4 pandas pyarrow" \
    pip install -q --no-cache-dir --target "$TARGET" \
        cftime gcsfs h5netcdf h5py "huggingface-hub>=0.27.0" loguru nest-asyncio \
        "netcdf4<1.7.3,>=1.6.4" "pandas<3.0" pyarrow 2>&1 | tail -3

time_step "group-B: pygrib python-dotenv rich s3fs tqdm xarray zarr" \
    pip install -q --no-cache-dir --target "$TARGET" \
        pygrib python-dotenv rich s3fs "tqdm>=4.65.0" \
        "xarray[parallel]>=2023.1.0" "zarr>=3.1.0" 2>&1 | tail -3

time_step "group-C: einops scipy omegaconf pyproj cartopy" \
    pip install -q --no-cache-dir --target "$TARGET" \
        einops scipy omegaconf pyproj cartopy 2>&1 | tail -3

time_step "group-D: nvtx hydra-core gitpython importlib-metadata jaxtyping safetensors" \
    pip install -q --no-cache-dir --target "$TARGET" \
        nvtx hydra-core gitpython importlib-metadata jaxtyping safetensors 2>&1 | tail -3

echo ""
echo "=== Total staging size: $(du -sh "$TARGET" | cut -f1) ==="
echo "=== Test complete at $(date) ==="
INNEREOF

chmod +x /tmp/sc_test_install_$$.sh

apptainer exec \
    --rocm \
    --bind "$TEST_DIR":/sc-test \
    "$SC_SIF" \
    bash /tmp/sc_test_install_$$.sh /sc-test "$ROCM_WHL_TAG"

echo ""
echo "Cleaning up $TEST_DIR ..."
rm -rf "$TEST_DIR"
rm -f /tmp/sc_test_install_$$.sh
