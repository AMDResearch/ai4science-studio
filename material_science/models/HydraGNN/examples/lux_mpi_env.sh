#!/usr/bin/env bash
# ORNL Lux MPI baseline — Shelby Lockhart config-bundle + from-source stack.
#
# Gold standard on Lux until OLCF ships a verbs-capable UCX/MPI module.
# See: /lustre/orion/ven136/proj-shared/slockhar/config-bundle/README.md
#
# The stack lives on Lustre but was built with prefix /ccs/home/slockhar/opt/...
# Apptainer cannot bind into another user's home — bind at /opt/lux-mpi/* and
# set OPAL_PREFIX=/opt/lux-mpi/openmpi inside the container.
#
# Usage (source from build_overlay_amd.sh / sbatch_train_amd.sh):
#   source "${SCRIPT_DIR}/lux_mpi_env.sh"
#   lux_mpi_setup_binds   # sets MPI_BINDS, HG_OMPI_DIR (container paths)

# Defaults — override before sourcing if your deployment differs.
HG_LUX_CONFIG_BUNDLE="${HG_LUX_CONFIG_BUNDLE:-/lustre/orion/ven136/proj-shared/slockhar/config-bundle}"
HG_LUX_MPI_OPT="${HG_LUX_MPI_OPT:-/lustre/orion/ven136/proj-shared/slockhar/mpi-build-tests/opt}"
HG_LUX_MPI_BUNDLE_MODE="${HG_LUX_MPI_BUNDLE_MODE:-native}"   # native | ucc
HG_LUX_MPI_RULES_FILE="${HG_LUX_MPI_RULES_FILE:-coll-tuned-dynamic-rules-mean.conf}"

# Lustre install trees (host)
HG_LUX_OMPI_LUSTRE="${HG_LUX_MPI_OPT}/openmpi/ompi5.0.10-ucc1.8.x-ucx1.21.x-gcc-gfortran"
HG_LUX_UCX_LUSTRE="${HG_LUX_MPI_OPT}/ucx/ucx1.21.x-gcc-gfortran"
HG_LUX_UCC_LUSTRE="${HG_LUX_MPI_OPT}/ucc/ucc1.8.x-ucx1.21.x-gcc-gfortran"

# Apptainer bind targets (container) — must match OPAL_PREFIX
HG_LUX_OMPI_CONTAINER="/opt/lux-mpi/openmpi"
HG_LUX_UCX_CONTAINER="/opt/lux-mpi/ucx"
HG_LUX_UCC_CONTAINER="/opt/lux-mpi/ucc"

# Host libfabric satisfies OpenMPI's libopen-pal (FABRIC_1.8). The Ubuntu SIF
# ships an older libfabric; bind the host library over the container path.
HG_LUX_LIBFABRIC_HOST="${HG_LUX_LIBFABRIC_HOST:-/lib64/libfabric.so.1}"
HG_LUX_LIBFABRIC_CONTAINER="${HG_LUX_LIBFABRIC_CONTAINER:-/lib/x86_64-linux-gnu/libfabric.so.1}"

lux_mpi_available() {
  [[ -x "${HG_LUX_OMPI_LUSTRE}/bin/mpicc" ]] \
    && [[ -d "${HG_LUX_UCX_LUSTRE}/lib" ]] \
    && [[ -d "${HG_LUX_UCC_LUSTRE}/lib" ]] \
    && [[ -f "${HG_LUX_CONFIG_BUNDLE}/ucx.conf" ]]
}

# Populate MPI_BINDS and container-side HG_OMPI_DIR / mpi4py paths.
lux_mpi_setup_binds() {
  if ! lux_mpi_available; then
    return 1
  fi

  HG_OMPI_DIR="${HG_LUX_OMPI_CONTAINER}"
  MPI4PY_LIBMPI="${HG_LUX_OMPI_CONTAINER}/lib/libmpi.so.40"
  HG_MPI_EXTRA_LDPATH="${HG_LUX_UCC_CONTAINER}/lib:${HG_LUX_UCX_CONTAINER}/lib"
  HG_LUX_OPAL_PREFIX="${HG_LUX_OMPI_CONTAINER}"

  MPI_BINDS=(
    --bind "${HG_LUX_OMPI_LUSTRE}:${HG_LUX_OMPI_CONTAINER}:ro"
    --bind "${HG_LUX_UCX_LUSTRE}:${HG_LUX_UCX_CONTAINER}:ro"
    --bind "${HG_LUX_UCC_LUSTRE}:${HG_LUX_UCC_CONTAINER}:ro"
    --bind "${HG_LUX_CONFIG_BUNDLE}:${HG_LUX_CONFIG_BUNDLE}:ro"
  )
  if [[ -f "${HG_LUX_LIBFABRIC_HOST}" ]]; then
    MPI_BINDS+=( --bind "${HG_LUX_LIBFABRIC_HOST}:${HG_LUX_LIBFABRIC_CONTAINER}:ro" )
    _fabric_real=$(readlink -f "${HG_LUX_LIBFABRIC_HOST}" 2>/dev/null || true)
    if [[ -n "${_fabric_real}" && -f "${_fabric_real}" ]]; then
      _fabric_base=$(basename "${_fabric_real}")
      MPI_BINDS+=( --bind "${_fabric_real}:/lib/x86_64-linux-gnu/${_fabric_base}:ro" )
    fi
  fi
  # Host libfabric + OpenMPI libopen-pal need RHEL libnl/EFA/RDMA at link and runtime.
  for _lib in \
    /lib64/liblnetconfig.so.4 \
    /lib64/liblustreapi.so.1 \
    /lib64/libnl-genl-3.so.200 \
    /lib64/libnl-3.so.200 \
    /lib64/libnl-route-3.so.200 \
    /lib64/libefa.so.1 \
    /lib64/librdmacm.so.1 \
    /lib64/libibverbs.so.1; do
    [[ -f "$_lib" ]] && MPI_BINDS+=( --bind "${_lib}:${_lib}:ro" )
  done
  # Ubuntu SIF linker searches /lib/x86_64-linux-gnu for libnl when linking libfabric.
  for _pair in \
    /lib64/libnl-3.so.200:/lib/x86_64-linux-gnu/libnl-3.so.200 \
    /lib64/libnl-route-3.so.200:/lib/x86_64-linux-gnu/libnl-route-3.so.200 \
    /lib64/libnl-genl-3.so.200:/lib/x86_64-linux-gnu/libnl-genl-3.so.200 \
    /usr/lib64/libefa.so.1:/lib/x86_64-linux-gnu/libefa.so.1 \
    /usr/lib64/libefa.so.1.4.61.0:/lib/x86_64-linux-gnu/libefa.so.1.4.61.0 \
    /usr/lib64/libibverbs.so.1:/lib/x86_64-linux-gnu/libibverbs.so.1 \
    /usr/lib64/libibverbs.so.1.15.61.0:/lib/x86_64-linux-gnu/libibverbs.so.1.15.61.0 \
    /lib64/librdmacm.so.1:/lib/x86_64-linux-gnu/librdmacm.so.1; do
    _src="${_pair%%:*}"
    _dst="${_pair##*:}"
    [[ -f "$_src" ]] && MPI_BINDS+=( --bind "${_src}:${_dst}:ro" )
  done
  return 0
}

# Apptainer --env flags for UCX + OpenMPI MCA tuning (runtime / multi-node).
lux_mpi_runtime_envs() {
  LUX_MPI_RUNTIME_ENVS=()
  if ! lux_mpi_available; then
    return 1
  fi

  # shellcheck disable=SC1090
  source "${HG_LUX_CONFIG_BUNDLE}/ucx.conf"

  local _mca="${HG_LUX_CONFIG_BUNDLE}/${HG_LUX_MPI_BUNDLE_MODE}/openmpi-mca-params.conf"
  local _rules="${HG_LUX_CONFIG_BUNDLE}/${HG_LUX_MPI_RULES_FILE}"

  LUX_MPI_RUNTIME_ENVS=(
    --env "OPAL_PREFIX=${HG_LUX_OMPI_CONTAINER}"
    --env "UCX_TLS=${UCX_TLS}"
    --env "UCX_ROCM_COPY_LAT=${UCX_ROCM_COPY_LAT}"
    --env "UCX_RNDV_THRESH=${UCX_RNDV_THRESH}"
    --env "UCX_RNDV_SCHEME=${UCX_RNDV_SCHEME}"
    --env "UCX_MAX_RNDV_RAILS=${UCX_MAX_RNDV_RAILS}"
    --env "UCX_IB_GPU_DIRECT_RDMA=${UCX_IB_GPU_DIRECT_RDMA}"
    --env "UCX_IB_ROCE_LOCAL_SUBNET=${UCX_IB_ROCE_LOCAL_SUBNET}"
    --env "UCX_IB_TRAFFIC_CLASS=${UCX_IB_TRAFFIC_CLASS}"
    --env "UCX_RC_MAX_NUM_EPS=${UCX_RC_MAX_NUM_EPS}"
    --env "UCX_RC_MAX_RD_ATOMIC=${UCX_RC_MAX_RD_ATOMIC}"
    --env "UCX_RC_RX_QUEUE_LEN=${UCX_RC_RX_QUEUE_LEN}"
    --env "UCX_RC_VERBS_RX_MATCH_BATCH=${UCX_RC_VERBS_RX_MATCH_BATCH}"
    --env "UCX_RC_TIMEOUT=${UCX_RC_TIMEOUT}"
    --env "UCX_RC_RNR_RETRY_COUNT=${UCX_RC_RNR_RETRY_COUNT}"
    --env "UCX_WARN_UNUSED_ENV_VARS=${UCX_WARN_UNUSED_ENV_VARS}"
    --env "OMPI_MCA_mca_base_param_files=${_mca}"
    --env "OMPI_MCA_coll_tuned_dynamic_rules_filename=${_rules}"
    --env "MPI4PY_RC_THREADS=false"
  )
  if [[ "${NODES:-1}" -gt 1 ]]; then
    if [[ "${HG_LUX_MPI_NIC_MODE:-tcp}" == "tcp" ]]; then
      # Apptainer cannot init ionic verbs RC inside the SIF — use ai* TCP netdevs.
      LUX_MPI_RUNTIME_ENVS+=( --env "UCX_TLS=${HG_LUX_UCX_TLS_TCP:-sm,self,rocm_copy,rocm_ipc,cma,tcp}" )
    else
      LUX_MPI_RUNTIME_ENVS+=( --env "UCX_NET_DEVICES=${UCX_NET_DEVICES:-^bnxt_re0}" )
    fi
  fi
  return 0
}

lux_mpi_should_use() {
  [[ "${HG_USE_LUX_MPI:-auto}" == "1" ]] && lux_mpi_available && return 0
  [[ "${HG_USE_LUX_MPI:-auto}" == "0" ]] && return 1
  [[ -x /opt/ompi/bin/mpicc ]] && return 1
  lux_mpi_available
}

lux_mpi_setup() {
  if lux_mpi_should_use && lux_mpi_setup_binds; then
    export HG_OMPI_DIR MPI4PY_LIBMPI HG_MPI_EXTRA_LDPATH HG_LUX_OPAL_PREFIX
    return 0
  fi
  return 1
}

# Apptainer binds so UCX/OpenMPI can see host RoCE/RDMA inside the container.
# Without these, multi-node jobs hang with "transport 'rc' is not available".
lux_mpi_multinode_apptainer_binds() {
  LUX_MPI_MULTINODE_BINDS=()
  [[ "${NODES:-1}" -gt 1 ]] || return 0

  for _path in /sys/class/infiniband /dev/infiniband /sys/class/net; do
    if [[ -d "$_path" ]]; then
      LUX_MPI_MULTINODE_BINDS+=( --bind "${_path}:${_path}" )
    fi
  done
  # Verbs provider libs only for RC/ionic mode — binding incomplete sets breaks TCP.
  if [[ "${HG_LUX_MPI_NIC_MODE:-tcp}" != "tcp" ]]; then
    if [[ -d /etc/libibverbs.d ]]; then
      LUX_MPI_MULTINODE_BINDS+=( --bind /etc/libibverbs.d:/etc/libibverbs.d )
    fi
    for _libdir in /usr/lib64/libibverbs /lib64/libibverbs; do
      if [[ -d "$_libdir" ]]; then
        LUX_MPI_MULTINODE_BINDS+=( --bind "${_libdir}:${_libdir}:ro" )
      fi
    done
  fi
  return 0
}

# RCCL env for PyTorch DDP across nodes on ORNL Lux Pensando ionic fabric.
# Sources (aligned 2026-09): mswu/rccl-tests-scripts/rccl-pensando-env.sh (native
# IB, not ANP — ANP blocked on ORNL rdma-core 61); sidconde Primus env.lux.sh
# (bond0 OOB, ionic_0..7, GID index 1). Leave NCCL_NET_PLUGIN unset.
lux_mpi_rccl_multinode_envs() {
  LUX_RCCL_MULTINODE_ENVS=()
  [[ "${NODES:-1}" -gt 1 ]] || return 0

  LUX_RCCL_MULTINODE_ENVS=(
    --env NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
    --env NCCL_IB_HCA="${NCCL_IB_HCA:-ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7}"
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
    --env NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  )
  return 0
}
