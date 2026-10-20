#!/usr/bin/env bash
# ORNL Lux system OpenMPI module for Apptainer.
#
# MPI control traffic is TCP-only: OpenMPI ob1 + btl=tcp,self on bond0.
# UCX is intentionally not configured or supported here. RCCL remains a
# separate GPU data plane over the ionic HCAs.

HG_LUX_ROCM_MODULE="${HG_LUX_ROCM_MODULE:-rocm/7.2.4}"
HG_LUX_COMPILER_MODULE="${HG_LUX_COMPILER_MODULE:-amd-llvm/7.2.4}"
HG_LUX_OPENMPI_MODULE="${HG_LUX_OPENMPI_MODULE:-openmpi/5.0.10}"

lux_mpi_load_module() {
  command -v module >/dev/null 2>&1 || return 1
  # Lux uses a hierarchy: OpenMPI becomes visible only after ROCm + amd-llvm.
  module load "$HG_LUX_ROCM_MODULE" >/dev/null 2>&1 || return 1
  module load "$HG_LUX_COMPILER_MODULE" >/dev/null 2>&1 || return 1
  module load "$HG_LUX_OPENMPI_MODULE" >/dev/null 2>&1 || return 1
  command -v mpicc >/dev/null 2>&1
}

lux_mpi_available() {
  lux_mpi_load_module
}

# Compatibility no-op: the site module tree is already on every node.
lux_mpi_stage_local() {
  lux_mpi_available
}

# Longest directory prefix shared by the MPI install and its non-system libs.
# Spack layouts put hwloc/libevent/UCX next to OpenMPI, not inside its prefix.
# Override with HG_LUX_SW_ROOT when the discovered tree is too wide or too narrow.
lux_mpi_sw_root() {
  if [[ -n "${HG_LUX_SW_ROOT:-}" ]]; then
    printf '%s\n' "$HG_LUX_SW_ROOT"
    return 0
  fi
  local lib="${HG_OMPI_DIR}/lib/libmpi.so.40"
  [[ -e "$lib" ]] || lib="${HG_OMPI_DIR}/lib/libmpi.so"
  local prefix="$HG_OMPI_DIR"
  local line target candidate
  if [[ -e "$lib" ]]; then
    while IFS= read -r line; do
      [[ "$line" == *"=>"* ]] || continue
      target="${line##*=> }"
      target="${target%% *}"
      [[ "$target" == /* ]] || continue
      # Host ROCm and libc live outside the MPI software tree. Do not let them
      # collapse the bind root to /.
      case "$target" in
        /lib/*|/lib64/*|/usr/lib/*|/usr/lib64/*|/opt/rocm*|/opt/amdgpu*) continue ;;
      esac
      target="$(dirname "$target")"
      candidate="$prefix"
      while [[ "$target" != "$candidate"* && "$candidate" != "/" ]]; do
        candidate="$(dirname "$candidate")"
      done
      [[ "$candidate" == "/" ]] && continue
      prefix="$candidate"
    done < <(ldd "$lib" 2>/dev/null || true)
  fi
  [[ "$prefix" != "/" && -d "$prefix" ]] || return 1
  printf '%s\n' "$prefix"
}

# Load the site module and bind its software tree at the same path in Apptainer.
lux_mpi_setup_binds() {
  lux_mpi_available || return 1

  HG_OMPI_DIR="$(cd "$(dirname "$(dirname "$(command -v mpicc)")")" && pwd)"
  MPI4PY_LIBMPI="${HG_OMPI_DIR}/lib/libmpi.so.40"
  HG_MPI_EXTRA_LDPATH="${HG_OMPI_DIR}/lib"
  HG_LUX_OPAL_PREFIX="${HG_OMPI_DIR}"
  HG_LUX_SW_ROOT="$(lux_mpi_sw_root)" || return 1
  MPI_BINDS=( --bind "${HG_LUX_SW_ROOT}:${HG_LUX_SW_ROOT}:ro" )
  # This site build links a few host services directly. Bind only their SONAME
  # files into Ubuntu's loader path; never add all of /usr/lib64 to LD_LIBRARY_PATH.
  local _lib _src
  for _lib in \
    libpmix.so.2 libmunge.so.2 liblustreapi.so.1 liblnetconfig.so.4 \
    libyaml-0.so.2 libnl-genl-3.so.200 libnl-3.so.200; do
    _src=""
    [[ -e "/usr/lib64/${_lib}" ]] && _src="$(readlink -f "/usr/lib64/${_lib}")"
    [[ -z "$_src" && -e "/lib64/${_lib}" ]] && _src="$(readlink -f "/lib64/${_lib}")"
    [[ -n "$_src" ]] && MPI_BINDS+=( --bind "${_src}:/lib/x86_64-linux-gnu/${_lib}:ro" )
  done
  export HG_OMPI_DIR MPI4PY_LIBMPI HG_MPI_EXTRA_LDPATH HG_LUX_OPAL_PREFIX HG_LUX_SW_ROOT
}

lux_mpi_runtime_envs() {
  lux_mpi_available || return 1
  local _iface="${HG_LUX_MPI_TCP_IFACE:-${NCCL_SOCKET_IFNAME:-bond0}}"
  # Apptainer --env KEY=a,b truncates at the first comma (same trap as NCCL_IB_HCA).
  export APPTAINERENV_OMPI_MCA_btl="tcp,self"
  LUX_MPI_RUNTIME_ENVS=(
    --env "OPAL_PREFIX=${HG_OMPI_DIR}"
    --env "OMPI_MCA_pml=ob1"
    --env "OMPI_MCA_osc=^ucx"
    --env "OMPI_MCA_btl_tcp_if_include=${_iface}"
    --env "OMPI_MCA_coll=^ucc"
    --env "MPI4PY_RC_THREADS=false"
  )
}

lux_mpi_should_use() {
  lux_mpi_available
}

lux_mpi_setup() {
  lux_mpi_setup_binds
}

# MPI itself is TCP-only. These device views are for the independent RCCL
# ionic data plane used by multi-node PyTorch training.
lux_mpi_multinode_apptainer_binds() {
  LUX_MPI_MULTINODE_BINDS=()
  [[ "${NODES:-1}" -gt 1 ]] || return 0
  for _path in /sys/class/infiniband /dev/infiniband /sys/class/net; do
    [[ -e "$_path" ]] && LUX_MPI_MULTINODE_BINDS+=( --bind "${_path}:${_path}" )
  done
}

# RCCL env for PyTorch DDP across nodes on ORNL Lux Pensando ionic fabric.
# Native RCCL IB (not ANP) with bond0 OOB, ionic_0..7, and GID index 1.
# Leave NCCL_NET_PLUGIN unset.
#
# CRITICAL: Apptainer --env KEY=a,b,c truncates at the first comma. Comma-valued
# fabric vars MUST be exported as APPTAINERENV_* (Sidafa Primus container_run.sh).
lux_mpi_rccl_multinode_envs() {
  LUX_RCCL_MULTINODE_ENVS=()
  [[ "${NODES:-1}" -gt 1 ]] || return 1

  local _hca="${NCCL_IB_HCA:-ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7}"
  local _sock="${NCCL_SOCKET_IFNAME:-bond0}"

  # Comma-safe path into the container
  export APPTAINERENV_NCCL_IB_HCA="${_hca}"
  export APPTAINERENV_NCCL_SOCKET_IFNAME="${_sock}"
  export APPTAINERENV_GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${_sock}}"

  LUX_RCCL_MULTINODE_ENVS=(
    --env NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-1}"
    --env NCCL_GDR_FLUSH_DISABLE=1
    --env RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0
    --env NCCL_GDRCOPY_ENABLE=0
    --env NCCL_IB_PCI_RELAXED_ORDERING=1
    --env NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-1}"
    --env NCCL_IB_TC=104
    --env NCCL_IGNORE_CPU_AFFINITY=1
    --env NCCL_PXN_DISABLE=0
    --env NCCL_DMABUF_ENABLE=1
    --env NCCL_IB_USE_INLINE=1
    --env RCCL_LL128_FORCE_ENABLE=1
    --env IONIC_LOCKFREE=all
    --env RCCL_AINIC_ROCE=1
    --env USING_AINIC=1
    --env RCCL_USE_AMD_SMI_LIB="${RCCL_USE_AMD_SMI_LIB:-1}"
    --env NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  )
  return 0
}
