#!/usr/bin/env bash
# Apptainer binds for HydraGNN ADIOS datasets symlinked outside HG_DATA_DIR.
#
# ORNL Lux: weights/ often symlinks into cph161/lrn070 world-shared trees.
# Apptainer only sees explicitly bound paths — bind world-shared roots so
# symlink chains resolve inside the container (no multi‑tens‑of‑GiB copy).
#
# Usage (after DATASET_ARRAY and HG_DATA_DIR are set):
#   source "${SCRIPT_DIR}/hg_dataset_binds.sh"
#   hg_setup_dataset_binds
#   # append "${DATA_BINDS[@]}" to apptainer exec

hg_dataset_world_shared_root() {
  local path="$1"
  # Site mount root must be provided via AI4S_LUSTRE_ROOT (e.g. /lustre/orion on OLCF)
  # — kept as env var so the literal does not appear in active code.
  local _ws_root="${AI4S_LUSTRE_ROOT}/world-shared"
  case "$path" in
    "${_ws_root}"/*)
      echo "${_ws_root}"
      ;;
    "${AI4S_LUSTRE_ROOT}"/*/world-shared/*)
      echo "$path" | sed -n "s|^\(${AI4S_LUSTRE_ROOT}/[^/]*/world-shared\).*|\1|p"
      ;;
  esac
}

hg_add_dataset_bind_root() {
  local root="$1"
  [[ -n "$root" && -d "$root" ]] || return 0
  local seen
  for seen in "${DATA_BIND_SEEN[@]:-}"; do
    [[ "$seen" == "$root" ]] && return 0
  done
  DATA_BIND_SEEN+=("$root")
  DATA_BINDS+=( --bind "${root}:${root}:ro" )
  echo "  Data bind    : ${root}"
}

hg_setup_dataset_binds() {
  DATA_BINDS=()
  DATA_BIND_SEEN=()

  local ds link target resolved root paths=()
  for ds in "${DATASET_ARRAY[@]}"; do
    link="${HG_DATA_DIR}/${ds}-v2.bp"
    paths=()
    if [[ -L "$link" ]]; then
      target="$(readlink "$link")"
      [[ -n "$target" ]] && paths+=( "$target" )
    fi
    resolved="$(readlink -f "$link" 2>/dev/null || true)"
    [[ -n "$resolved" ]] && paths+=( "$resolved" )

    for path in "${paths[@]}"; do
      case "$path" in
        "${HG_DATA_DIR}"/*|"${HG_DATA_DIR}") continue ;;
      esac
      root="$(hg_dataset_world_shared_root "$path")"
      hg_add_dataset_bind_root "$root"
    done
  done
}
