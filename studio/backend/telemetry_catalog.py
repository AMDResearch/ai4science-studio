"""Catalog of Omnistat metrics selectable in the Studio telemetry panel.

Single source of truth for WHICH metrics can be plotted, their PromQL expression,
unit, friendly label, cross-GPU aggregation, and grouping. Both the on-disk harvest
(telemetry.harvest) and the live cross-node endpoint (telemetry.harvest_live) iterate
this catalog, so adding a metric here surfaces it everywhere (backend + the frontend
dropdown, which fetches /api/telemetry/catalog).

Metric names verified against a real gfx950 run (52 metrics, per-GPU `card` label).
All expressions AGGREGATE across the 8 GPUs (max/sum/avg) — no per-card fan-out.
Rate() windows are 30s for counters/interconnect (robust to the 0.5-1s sampling).

Fields per descriptor:
  key    stable id used in the payload + as the query param
  promql PromQL for the per-GPU series BEFORE cross-GPU aggregation ({jobid} unused)
  unit   display unit
  label  human-readable chart/tile label
  agg    "max" | "sum" | "avg" — which _range_* helper aggregates across GPUs
  group  dropdown section
  default  seed the panel (the interesting, non-flat metrics)
"""
from __future__ import annotations

# Convenience builders for the rocprofiler hardware counters.
def _rate(counter: str, mult: float, window: str = "30s") -> str:
    return f'rate(omnistat_hardware_counter{{{{name="{counter}"}}}}[{window}]) * {mult}'


CATALOG: list[dict] = [
    # ── GPU (AMD-SMI) ────────────────────────────────────────────────────────
    {"key": "gpu_util_pct", "promql": "rocm_utilization_percentage", "unit": "%",
     "label": "GPU Utilization", "agg": "max", "group": "GPU", "default": False},
    {"key": "power_w", "promql": "rocm_average_socket_power_watts", "unit": "W",
     "label": "GPU Power (socket)", "agg": "sum", "group": "GPU", "default": False},
    {"key": "temp_c", "promql": "rocm_temperature_celsius", "unit": "C",
     "label": "GPU Temperature", "agg": "max", "group": "GPU", "default": False},
    {"key": "temp_mem_c", "promql": "rocm_temperature_memory_celsius", "unit": "C",
     "label": "HBM Temperature", "agg": "max", "group": "GPU", "default": False},
    {"key": "vram_gb", "promql": "rocm_vram_used_percentage * rocm_vram_total_bytes / 100 / 1073741824",
     "unit": "GB", "label": "VRAM Used", "agg": "sum", "group": "GPU", "default": False},
    {"key": "vram_busy_pct", "promql": "rocm_vram_busy_percentage", "unit": "%",
     "label": "VRAM Busy", "agg": "max", "group": "GPU", "default": False},
    {"key": "sclk_mhz", "promql": "rocm_sclk_clock_mhz", "unit": "MHz",
     "label": "GPU Clock (sclk)", "agg": "avg", "group": "GPU", "default": False},
    {"key": "mclk_mhz", "promql": "rocm_mclk_clock_mhz", "unit": "MHz",
     "label": "Memory Clock (mclk)", "agg": "avg", "group": "GPU", "default": False},

    # ── Hardware performance counters (rocprofiler-sdk) ──────────────────────
    # fp64 GFLOP/s dominant term = MFMA_MOPS_F64 * 512. HydraGNN trains + does
    # force inference (dE/dx) in FP64 — this is the headline AI4S counter.
    {"key": "fp64_tflops", "promql": _rate("SQ_INSTS_VALU_MFMA_MOPS_F64", 512) + " / 1e12",
     "unit": "TFLOP/s", "label": "FP64 MFMA Throughput", "agg": "sum", "group": "Counters", "default": True},
    {"key": "fp64_full_tflops",
     "promql": ("(rate(omnistat_hardware_counter{{name=\"SQ_INSTS_VALU_MFMA_MOPS_F64\"}}[30s]) * 512"
                " + (rate(omnistat_hardware_counter{{name=\"SQ_INSTS_VALU_FMA_F64\"}}[30s]) * 2"
                " + rate(omnistat_hardware_counter{{name=\"SQ_INSTS_VALU_ADD_F64\"}}[30s])"
                " + rate(omnistat_hardware_counter{{name=\"SQ_INSTS_VALU_MUL_F64\"}}[30s])) * 64) / 1e12"),
     "unit": "TFLOP/s", "label": "FP64 Total (MFMA+VALU)", "agg": "sum", "group": "Counters", "default": False},
    {"key": "hbm_read_gbs", "promql": _rate("FETCH_SIZE", 1024) + " / 1e9",
     "unit": "GB/s", "label": "HBM Read Bandwidth", "agg": "sum", "group": "Counters", "default": True},

    # ── Interconnect (xGMI scale-up) ─────────────────────────────────────────
    {"key": "xgmi_read_gbs", "promql": "rate(rocm_xgmi_total_read_kilobytes[30s]) * 1024 / 1e9",
     "unit": "GB/s", "label": "xGMI Read Bandwidth", "agg": "sum", "group": "Interconnect", "default": True},
    {"key": "xgmi_write_gbs", "promql": "rate(rocm_xgmi_total_write_kilobytes[30s]) * 1024 / 1e9",
     "unit": "GB/s", "label": "xGMI Write Bandwidth", "agg": "sum", "group": "Interconnect", "default": False},

    # ── Network (scale-out) ──────────────────────────────────────────────────
    {"key": "net_rx_gbs", "promql": "rate(omnistat_network_rx_bytes[30s]) / 1e9",
     "unit": "GB/s", "label": "Network RX", "agg": "sum", "group": "Network", "default": False},
    {"key": "net_tx_gbs", "promql": "rate(omnistat_network_tx_bytes[30s]) / 1e9",
     "unit": "GB/s", "label": "Network TX", "agg": "sum", "group": "Network", "default": False},

    # ── Host I/O ─────────────────────────────────────────────────────────────
    {"key": "io_read_gbs", "promql": "rate(omnistat_host_io_read_total_bytes[30s]) / 1e9",
     "unit": "GB/s", "label": "Host I/O Read", "agg": "sum", "group": "Host I/O", "default": False},
    {"key": "io_write_gbs", "promql": "rate(omnistat_host_io_write_total_bytes[30s]) / 1e9",
     "unit": "GB/s", "label": "Host I/O Write", "agg": "sum", "group": "Host I/O", "default": False},

    # ── Host CPU / memory ────────────────────────────────────────────────────
    {"key": "cpu_util_cores", "promql": "omnistat_host_cpu_aggregate_core_utilization", "unit": "cores",
     "label": "CPU Cores Active", "agg": "max", "group": "Host CPU/Mem", "default": False},
    {"key": "cpu_load1", "promql": "omnistat_host_cpu_load1", "unit": "load",
     "label": "CPU Load (1m)", "agg": "max", "group": "Host CPU/Mem", "default": False},
    {"key": "host_mem_used_gb",
     "promql": "(omnistat_host_mem_total_bytes - omnistat_host_mem_available_bytes) / 1073741824",
     "unit": "GB", "label": "Host Memory Used", "agg": "max", "group": "Host CPU/Mem", "default": False},

    # ── RAS / ECC (correctable error counts; normally flat 0 — health signal) ─
    {"key": "ras_umc_correctable", "promql": "rocm_ras_umc_correctable_count", "unit": "count",
     "label": "RAS UMC Correctable", "agg": "sum", "group": "RAS/ECC", "default": False},
    {"key": "ras_gfx_correctable", "promql": "rocm_ras_gfx_correctable_count", "unit": "count",
     "label": "RAS GFX Correctable", "agg": "sum", "group": "RAS/ECC", "default": False},
    {"key": "ras_xgmi_correctable", "promql": "rocm_ras_xgmi_wafl_correctable_count", "unit": "count",
     "label": "RAS xGMI Correctable", "agg": "sum", "group": "RAS/ECC", "default": False},

    # ── Kernel trace (only present when OMNISTAT_KERNEL_TRACE=1) ──────────────
    {"key": "kernel_dispatch", "promql": "omnistat_kernel_dispatch_count", "unit": "count",
     "label": "Kernel Dispatches", "agg": "sum", "group": "Kernel Trace", "default": False},
    {"key": "kernel_dur_ms", "promql": "omnistat_kernel_total_duration_ns / 1e6", "unit": "ms",
     "label": "Kernel Total Duration", "agg": "sum", "group": "Kernel Trace", "default": False},
]

# Fast lookups.
BY_KEY: dict[str, dict] = {d["key"]: d for d in CATALOG}
DEFAULT_KEYS: list[str] = [d["key"] for d in CATALOG if d.get("default")]
# Tiles/energy: energy is derived in telemetry.py (not a raw series).


def as_json() -> dict:
    """Payload for GET /api/telemetry/catalog — descriptors + defaults, grouped."""
    return {
        "metrics": [
            {"key": d["key"], "label": d["label"], "unit": d["unit"],
             "group": d["group"], "default": bool(d.get("default"))}
            for d in CATALOG
        ],
        "defaults": DEFAULT_KEYS,
        "groups": list(dict.fromkeys(d["group"] for d in CATALOG)),
    }
