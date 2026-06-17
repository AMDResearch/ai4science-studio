from __future__ import annotations

import itertools
import json
import os
import signal
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass

from tabulate import tabulate
from tqdm import tqdm

import torch
import torch.utils.benchmark as benchmark
from torch._inductor.utils import do_bench_using_profiling

# PyTorch 2.10 / some ROCm builds omit FA3/FA4 helpers; keep --flash_test optional.
import torch.nn.attention as _torch_nn_attention

sdpa_kernel = _torch_nn_attention.sdpa_kernel
SDPBackend = _torch_nn_attention.SDPBackend
activate_flash_attention_impl = getattr(
    _torch_nn_attention, "activate_flash_attention_impl", None
)
restore_flash_attention_impl = getattr(
    _torch_nn_attention, "restore_flash_attention_impl", None
)
from torch.nn.functional import scaled_dot_product_attention


def benchmark_torch_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    # warmup
    for _ in range(5):
        func(*args, **kwargs)
    t0 = benchmark.Timer(
        stmt="func(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "func": func},
    )
    return t0.adaptive_autorange(min_run_time=0.1).median * 1e6


def benchmark_cuda_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    """Thin wrapper around do_bench_using_profiling (CUDA-oriented; may fail on some ROCm builds)."""

    def no_args():
        func(*args, **kwargs)

    try:
        time = do_bench_using_profiling(no_args)
        return time * 1e3
    except Exception:
        return benchmark_torch_function_in_microseconds(func, *args, **kwargs)


@dataclass(frozen=True)
class ExperimentConfig:
    batch_size: int
    num_heads: int
    q_seq_len: int
    kv_seq_len: int
    embed_dim: int
    is_causal: bool
    dtype: torch.dtype
    backend: SDPBackend | None
    flash_impl: str | None = None  # None/"FA2" for default, "FA3", or "FA4"
    device: torch.device = torch.device("cuda")

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    def asdict(self):
        dict_obj = asdict(self)
        dict_obj["head_dim"] = self.head_dim
        return dict_obj


@dataclass(frozen=True)
class ExperimentResults:
    forward_time: float  # microseconds
    backward_time: float  # microseconds
    forward_tflops: float
    backward_tflops: float

    def asdict(self):
        return asdict(self)


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    results: ExperimentResults

    def asdict(self):
        dict1 = self.config.asdict()
        dict2 = self.results.asdict()
        return {**dict1, **dict2}


def calculate_tflops(
    config: ExperimentConfig,
    time_us: float,
    is_backward: bool = False,
    sparsity: float = 0.0,
) -> float:
    """
    Calculate TFLOPS for scaled dot product attention.

    Parameters:
    - config: The experiment configuration
    - time_us: The execution time in microseconds
    - is_backward: Whether to calculate for backward pass (includes gradient computation)
    - sparsity: Sparsity factor between 0.0 and 1.0, where 0.0 means no sparsity and 1.0 means fully sparse

    Returns:
    - TFLOPS value
    """
    B = config.batch_size
    H = config.num_heads
    M = config.q_seq_len
    N = config.kv_seq_len
    D = config.head_dim

    # Calculate density factor (1.0 - sparsity)
    density = 1.0 - sparsity

    # Forward pass FLOPs
    qk_flops = (
        M * N * D * 2
    )  # Q*K^T matmul: (M,D) @ (D,N) with 2 FLOPs per multiply-add
    softmax_flops = M * N * 2  # Softmax operations (exp and div)
    av_flops = (
        M * N * D * 2
    )  # Attention @ V: (M,N) @ (N,D) with 2 FLOPs per multiply-add

    total_flops = B * H * (qk_flops + softmax_flops + av_flops)

    # Apply density factor to account for sparsity
    total_flops *= density

    # For backward pass flash uses 2.5x more flops will use this
    if is_backward:
        total_flops *= 2.5

    # Convert to TFLOPS: flops / (time_us * 1e-6) / 1e12
    tflops = total_flops / (time_us * 1e-6) / 1e12

    return tflops


def get_input(
    config: ExperimentConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(
        (config.batch_size, config.num_heads, config.q_seq_len, config.head_dim),
        dtype=config.dtype,
        device=config.device,
        requires_grad=True,
    )
    k = torch.randn(
        (config.batch_size, config.num_heads, config.kv_seq_len, config.head_dim),
        dtype=config.dtype,
        device=config.device,
        requires_grad=True,
    )
    v = torch.randn(
        (config.batch_size, config.num_heads, config.kv_seq_len, config.head_dim),
        dtype=config.dtype,
        device=config.device,
        requires_grad=True,
    )
    return q, k, v


def run_single_experiment(config: ExperimentConfig) -> ExperimentResults:
    q, k, v = get_input(config)
    is_causal = config.is_causal
    context = (
        sdpa_kernel(config.backend) if config.backend is not None else nullcontext()
    )

    # Activate flash attention implementation if specified (requires both helpers)
    if (
        activate_flash_attention_impl is not None
        and restore_flash_attention_impl is not None
        and config.backend is SDPBackend.FLASH_ATTENTION
        and config.flash_impl
        in (
            "FA3",
            "FA4",
        )
    ):
        try:
            activate_flash_attention_impl(config.flash_impl)
        except ImportError as e:
            raise RuntimeError(
                f"Failed to activate {config.flash_impl}: {e}\n"
                f"Please install the required flash attention library or run with default configuration (without --flash_test)."
            ) from e

    try:
        with context:
            forward_time = benchmark_cuda_function_in_microseconds(
                scaled_dot_product_attention,
                q,
                k,
                v,
                is_causal=is_causal,
                attn_mask=None,
            )
            out_torch = scaled_dot_product_attention(
                q, k, v, is_causal=is_causal, attn_mask=None
            )
            d_out = torch.randn_like(out_torch)
            backward_time = benchmark_cuda_function_in_microseconds(
                out_torch.backward, d_out, retain_graph=True
            )
    finally:
        if (
            restore_flash_attention_impl is not None
            and config.backend is SDPBackend.FLASH_ATTENTION
            and config.flash_impl
            in (
                "FA3",
                "FA4",
            )
        ):
            restore_flash_attention_impl()

    # Calculate TFLOPS for forward and backward passes
    sparsity = 0.5 if is_causal else 0.0
    forward_tflops = calculate_tflops(config, forward_time, sparsity=sparsity)
    backward_tflops = calculate_tflops(
        config, backward_time, is_backward=True, sparsity=sparsity
    )

    return ExperimentResults(
        forward_time=forward_time,
        backward_time=backward_time,
        forward_tflops=forward_tflops,
        backward_tflops=backward_tflops,
    )


def print_results(experiments: list[Experiment]):
    table_data = defaultdict(list)
    for experiment in experiments:
        for key, value in experiment.asdict().items():
            table_data[key].append(value)
    del table_data["device"]
    if table_data["backend"][0] is None:
        del table_data["backend"]
    if table_data["flash_impl"][0] is None:
        del table_data["flash_impl"]
    print(tabulate(table_data, headers="keys", tablefmt="pretty", floatfmt=".3f"))


def write_results_to_csv(
    experiments: list[Experiment], output_dir: str = "benchmark_results"
):
    """
    Write experiment results to a CSV file in the specified directory.
    The filename includes a timestamp for uniqueness.
    """
    import csv
    import os
    from datetime import datetime

    # Create output directory if it doesn't exist. Fall back to a writable temp dir
    # rather than aborting the whole sweep (e.g. read-only /shared inside Apptainer).
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        fallback = os.path.join(tempfile.gettempdir(), "orbit2_sdpa_benchmark_results")
        print(f"WARN: cannot write CSV to {output_dir!r} ({e}); using {fallback!r}", flush=True)
        output_dir = fallback
        os.makedirs(output_dir, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"benchmark_results_{timestamp}.csv")

    # Get all fields from the first experiment
    if not experiments:
        return

    fieldnames = list(experiments[0].asdict().keys())
    if "device" in fieldnames:
        fieldnames.remove("device")  # Remove device field as it's always cuda

    # Write results to CSV
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for experiment in experiments:
            row = experiment.asdict()
            if "device" in row:
                del row["device"]  # Remove device field
            writer.writerow(row)

    print(f"Results written to: {filename}")


def write_xformers_mea_results_to_csv(rows: list[dict], output_dir: str) -> None:
    """Append-style filename for xFormers subprocess sweep (separate from PyTorch SDPA CSV)."""
    import csv
    from datetime import datetime

    if not rows:
        return
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"xformers_mea_micro_{timestamp}.csv")
    fieldnames = list(rows[0].keys())
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"xFormers MEA results written to: {filename}", flush=True)


def generate_experiment_configs() -> list[ExperimentConfig]:
    batch_sizes = [1, 8, 16]
    num_heads = [16]
    q_kv_seq_lens = [(128, 128), (256, 256), (512, 512), (1024, 1024), (8192, 8192)]
    embed_dims = [2048]
    backends = [None]  # If set to None, all backends are enabled
    dtypes = [
        torch.bfloat16,
    ]
    is_causal = [True, False]
    all_configs = []
    for (
        bsz,
        heads,
        (q_seq_len, kv_seq_len),
        embed_dim,
        causal,
        dtype,
        backend,
    ) in itertools.product(
        batch_sizes, num_heads, q_kv_seq_lens, embed_dims, is_causal, dtypes, backends
    ):
        all_configs.append(
            ExperimentConfig(
                batch_size=bsz,
                num_heads=heads,
                q_seq_len=q_seq_len,
                kv_seq_len=kv_seq_len,
                embed_dim=embed_dim,
                is_causal=causal,
                dtype=dtype,
                backend=backend,
            )
        )

    return all_configs


def generate_experiment_configs_with_flash() -> list[ExperimentConfig]:
    batch_sizes = [1, 8, 16]
    num_heads = [16]
    q_kv_seq_lens = [(128, 128), (256, 256), (512, 512), (1024, 1024), (8192, 8192)]
    embed_dims = [2048]
    backends = [
        SDPBackend.FLASH_ATTENTION,
    ]
    dtypes = [
        torch.bfloat16,
    ]
    is_causal = [True, False]
    # FA2 (default), "FA3", "FA4" for the alternative implementations
    flash_impls = ["FA2", "FA3"]
    all_configs = []
    for (
        bsz,
        heads,
        (q_seq_len, kv_seq_len),
        embed_dim,
        causal,
        dtype,
        backend,
        flash_impl,
    ) in itertools.product(
        batch_sizes,
        num_heads,
        q_kv_seq_lens,
        embed_dims,
        is_causal,
        dtypes,
        backends,
        flash_impls,
    ):
        all_configs.append(
            ExperimentConfig(
                batch_size=bsz,
                num_heads=heads,
                q_seq_len=q_seq_len,
                kv_seq_len=kv_seq_len,
                embed_dim=embed_dim,
                is_causal=causal,
                dtype=dtype,
                backend=backend,
                flash_impl=flash_impl,
            )
        )

    return all_configs


def _parse_orbit_backend(name: str) -> SDPBackend | None:
    """Map CLI label → ``SDPBackend`` (ROCm: ``efficient`` is the closest public knob to a fused/CK-style path)."""
    n = name.strip().lower()
    if n in ("none", "default", "auto", ""):
        return None
    if n == "math":
        return SDPBackend.MATH
    if n in ("efficient", "efficient_attention"):
        return SDPBackend.EFFICIENT_ATTENTION
    if n == "ck":
        # Renamed: `ck` was a misleading alias for SDPBackend.EFFICIENT_ATTENTION, which on
        # ROCm is AOTriton (`attn_fwd`), NOT Composable Kernel. Real CK lives only in the
        # xFormers path (--orbit-include-xformers-ck / --orbit-xformers-modes ck).
        raise ValueError(
            "--orbit-sweep-backends label 'ck' was removed (it mapped to "
            "SDPBackend.EFFICIENT_ATTENTION, which is AOTriton on ROCm, not CK). "
            "Use 'efficient'. For actual Composable Kernel use --orbit-include-xformers-ck."
        )
    if n in ("flash", "flash_attention"):
        return SDPBackend.FLASH_ATTENTION
    if n in ("cudnn", "cudnn_attention"):
        return SDPBackend.CUDNN_ATTENTION
    raise ValueError(f"unknown --orbit-sweep-backends entry: {name!r}")


def generate_orbit_micro_configs(backend: SDPBackend | None) -> list[ExperimentConfig]:
    """Small bf16 configs (8 heads) including a 648×648 block similar to ORBIT-2 patch tokens."""
    dev = torch.device("cuda")
    return [
        ExperimentConfig(2, 8, 128, 128, 256, False, torch.bfloat16, backend, None, dev),
        ExperimentConfig(2, 8, 648, 648, 256, False, torch.bfloat16, backend, None, dev),
    ]


def generate_orbit_varagg_configs(
    backend: SDPBackend | None,
    *,
    batches: list[int],
    tokens: int = 648,
    n_i: int = 5,
    num_heads: int = 16,
    embed_dim: int = 256,
) -> list[ExperimentConfig]:
    """Replicate Bayes-CAST EDM ``var_agg``/``temporal_agg`` cross-attention shapes.

    ``aggregate_variables`` flattens ``(B, History, L)`` into the SDPA **batch** dim,
    so ``batch_size = batch * tokens`` (e.g. 256*648=165888), with ``q_seq_len = N_a = 1``
    and ``kv_seq_len = N_i`` (#input variables). This is the shape that makes ROCm Flash
    SDPA exceed the 65535 batch-grid cap (~batch>=128 at tokens=648) while EFFICIENT/MATH
    are fine. See recipes/perf-analysis/HANDOFF.md §RESOLVED.
    """
    dev = torch.device("cuda")
    return [
        ExperimentConfig(
            batch_size=b * tokens,
            num_heads=num_heads,
            q_seq_len=1,
            kv_seq_len=n_i,
            embed_dim=embed_dim,
            is_causal=False,
            dtype=torch.bfloat16,
            backend=backend,
            flash_impl=None,
            device=dev,
        )
        for b in batches
    ]


def generate_orbit_selfattn_configs(
    backend: SDPBackend | None,
    *,
    batches: list[int],
    seq_len: int = 648,
    num_heads: int = 8,
    embed_dim: int = 256,
) -> list[ExperimentConfig]:
    """Replicate Bayes-CAST EDM main self-attention shapes (Attention class).

    After variable/temporal aggregation the spatial transformer blocks run normal
    self-attention on ``(B=real_batch, L=seq_len, D)`` — here ``batch_size`` is the
    *real* per-rank batch (NOT inflated), ``q_seq_len == kv_seq_len == seq_len``
    (patch tokens; 108/6 * 216/6 = 648). This is where ROCm Flash is normally the
    fastest backend, so it answers "should we just force EFFICIENT everywhere?".
    """
    dev = torch.device("cuda")
    return [
        ExperimentConfig(
            batch_size=b,
            num_heads=num_heads,
            q_seq_len=seq_len,
            kv_seq_len=seq_len,
            embed_dim=embed_dim,
            is_causal=False,
            dtype=torch.bfloat16,
            backend=backend,
            flash_impl=None,
            device=dev,
        )
        for b in batches
    ]


def _orbit_xformers_micro_case_specs() -> list[dict]:
    """Named tensor specs for xFormers MEA (BHLD layout before transpose to xFormers BLHD)."""
    return [
        {
            "case_name": "micro_128_sym",
            "batch": 2,
            "num_heads": 8,
            "q_seq_len": 128,
            "kv_seq_len": 128,
            "embed_dim": 256,
        },
        {
            "case_name": "micro_648_sym",
            "batch": 2,
            "num_heads": 8,
            "q_seq_len": 648,
            "kv_seq_len": 648,
            "embed_dim": 256,
        },
        {
            "case_name": "asym_1_vs_6",
            "batch": 2,
            "num_heads": 8,
            "q_seq_len": 1,
            "kv_seq_len": 6,
            "embed_dim": 256,
        },
    ]


def _internal_xformers_worker_main(cfg_path: str) -> int:
    """Child entry: one xFormers MEA call (+ optional backward). Exits 0 on success."""
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    import torch
    import xformers.ops as xops

    torch.manual_seed(int(cfg.get("seed", 123)))
    device = torch.device("cuda")
    B, H, M, N = (
        int(cfg["batch"]),
        int(cfg["num_heads"]),
        int(cfg["q_seq_len"]),
        int(cfg["kv_seq_len"]),
    )
    E = int(cfg["embed_dim"])
    D = E // H
    dropout_p = float(cfg.get("dropout_p", 0.0))
    dtype = getattr(torch, str(cfg.get("dtype", "bfloat16")))
    mode = str(cfg.get("mode", "ck")).strip().lower()
    case_name = str(cfg["case_name"])

    q = torch.randn(B, H, M, D, device=device, dtype=dtype, requires_grad=True).contiguous()
    k = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True).contiguous()
    v = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True).contiguous()
    qx = q.transpose(1, 2).contiguous()
    kx = k.transpose(1, 2).contiguous()
    vx = v.transpose(1, 2).contiguous()

    def forward():
        if mode == "ck":
            return xops.memory_efficient_attention(
                qx, kx, vx, p=dropout_p, op=xops.MemoryEfficientAttentionCkOp
            )
        if mode == "dispatch":
            return xops.memory_efficient_attention(qx, kx, vx, p=dropout_p)
        raise ValueError(f"unknown xFormers mode {mode!r} (expected ck|dispatch)")

    try:
        forward_time = benchmark_cuda_function_in_microseconds(forward)
        out = forward()
        torch.cuda.synchronize()
        d_out = torch.randn_like(out)
        backward_time = benchmark_cuda_function_in_microseconds(
            out.backward, d_out, retain_graph=True
        )
    except Exception as e:
        print("__ORBIT_XF_ERROR__" + json.dumps({"case": case_name, "error": repr(e)}), flush=True)
        return 1

    row = {
        "case": case_name,
        "mode": mode,
        "dropout_p": dropout_p,
        "forward_us": forward_time,
        "backward_us": backward_time,
    }
    print("__ORBIT_XF_RESULT__" + json.dumps(row), flush=True)
    return 0


def _spawn_xformers_worker(cfg: dict, *, script_path: str, timeout_s: float) -> dict:
    """Run one xFormers case in a subprocess so SIGSEGV does not kill the SDPA parent."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tf:
        json.dump(cfg, tf)
        path = tf.name
    try:
        proc = subprocess.run(
            [sys.executable, script_path, "--internal-xformers-worker", path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        return {"outcome": "TIMEOUT", "returncode": None, "stderr": ""}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    out = proc.stdout or ""
    err = proc.stderr or ""
    rc = proc.returncode

    if rc == 0:
        for line in out.splitlines():
            if line.startswith("__ORBIT_XF_RESULT__"):
                payload = json.loads(line[len("__ORBIT_XF_RESULT__") :])
                return {"outcome": "OK", "returncode": rc, "row": payload, "stderr": err}
        for line in out.splitlines():
            if line.startswith("__ORBIT_XF_ERROR__"):
                payload = json.loads(line[len("__ORBIT_XF_ERROR__") :])
                return {
                    "outcome": "PYTHON_ERROR",
                    "returncode": rc,
                    "error": payload.get("error", payload),
                    "stderr": err,
                }
        return {
            "outcome": "NO_MARKER",
            "returncode": rc,
            "stderr": err,
            "stdout_tail": out[-2000:],
        }

    if rc is not None and rc < 0:
        sig = -rc
        try:
            sig_name = signal.Signals(sig).name
        except ValueError:
            sig_name = f"SIGNAL_{sig}"
        return {
            "outcome": "KILLED_BY_SIGNAL",
            "returncode": rc,
            "signal": sig_name,
            "stderr": err,
            "stdout_tail": out[-2000:],
        }

    return {
        "outcome": f"EXIT_{rc}",
        "returncode": rc,
        "stderr": err,
        "stdout_tail": out[-2000:],
    }


def run_orbit_xformers_mea_suite(
    *,
    script_path: str,
    modes: list[str],
    dropout_ps: list[float],
    timeout_s: float,
) -> list[dict]:
    """Each row is a dict suitable for tabulate (includes outcome, no ORBIT-specific code paths)."""
    rows: list[dict] = []
    for spec in _orbit_xformers_micro_case_specs():
        for mode in modes:
            for p in dropout_ps:
                cfg = {
                    **spec,
                    "mode": mode,
                    "dropout_p": p,
                    "dtype": "bfloat16",
                    "seed": 123,
                }
                res = _spawn_xformers_worker(cfg, script_path=script_path, timeout_s=timeout_s)
                base = {
                    "case": spec["case_name"],
                    "xformers_mode": mode,
                    "dropout_p": p,
                    "B": spec["batch"],
                    "H": spec["num_heads"],
                    "q_len": spec["q_seq_len"],
                    "kv_len": spec["kv_seq_len"],
                    "embed_dim": spec["embed_dim"],
                }
                if res["outcome"] == "OK" and "row" in res:
                    r = res["row"]
                    base["forward_us"] = r.get("forward_us")
                    base["backward_us"] = r.get("backward_us")
                    base["outcome"] = "OK"
                else:
                    base["forward_us"] = None
                    base["backward_us"] = None
                    base["outcome"] = res["outcome"]
                    if res.get("signal"):
                        base["signal"] = res["signal"]
                    if res.get("error"):
                        base["error"] = res["error"]
                rows.append(base)
    return rows


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SDPA benchmark runner")
    parser.add_argument(
        "--flash_test",
        action="store_true",
        help="Use flash attention configs (generate_experiment_configs_with_flash)",
    )
    parser.add_argument(
        "--orbit-micro",
        action="store_true",
        help="Studio/ROCm: tiny bf16 grid (see generate_orbit_micro_configs) instead of the full upstream sweep.",
    )
    parser.add_argument(
        "--orbit-sweep-backends",
        type=str,
        default=None,
        help="With --orbit-micro/--orbit-varagg: comma list (default: default,math,efficient,flash). "
        "Labels: default|math|efficient|flash|cudnn — maps to torch.nn.attention.SDPBackend. "
        "(NOTE: 'ck' was removed — efficient is AOTriton on ROCm, not Composable Kernel.)",
    )
    parser.add_argument(
        "--orbit-varagg",
        action="store_true",
        help="Studio/ROCm: replicate Bayes-CAST EDM var_agg cross-attention shapes "
        "(batch_size=batch*tokens, q_len=1, kv_len=N_i) to confirm the ROCm Flash 65535 "
        "batch-grid cap. Sweeps --orbit-varagg-batches across --orbit-sweep-backends.",
    )
    parser.add_argument(
        "--orbit-varagg-batches",
        type=str,
        default="32,64,128,256",
        help="Comma list of per-rank batch sizes for --orbit-varagg (B=batch*tokens).",
    )
    parser.add_argument(
        "--orbit-varagg-tokens",
        type=int,
        default=648,
        help="Tokens L per sample for --orbit-varagg (B=batch*tokens). Default 648.",
    )
    parser.add_argument(
        "--orbit-varagg-ni",
        type=int,
        default=5,
        help="N_i (#input variables, kv_seq_len) for --orbit-varagg. Default 5.",
    )
    parser.add_argument(
        "--orbit-selfattn",
        action="store_true",
        help="Studio/ROCm: time the main EDM self-attention shapes "
        "(batch_size=real batch, q_len=kv_len=seq) across --orbit-sweep-backends. "
        "Use to decide whether EFFICIENT is competitive with FLASH for normal self-attention.",
    )
    parser.add_argument(
        "--orbit-selfattn-batches",
        type=str,
        default="64,128,256,1024",
        help="Comma list of per-rank batch sizes for --orbit-selfattn.",
    )
    parser.add_argument(
        "--orbit-selfattn-seq",
        type=int,
        default=648,
        help="q_len=kv_len for --orbit-selfattn (patch tokens). Default 648.",
    )
    parser.add_argument(
        "--orbit-selfattn-heads",
        type=int,
        default=8,
        help="num_heads for --orbit-selfattn (EDM main blocks use 8). Default 8.",
    )
    parser.add_argument(
        "--orbit-include-xformers-ck",
        action="store_true",
        help="After the SDPA micro sweep: run xFormers memory_efficient_attention in **isolated subprocesses** "
        "(explicit MemoryEfficientAttentionCkOp and/or dispatcher). Confirms SIGSEGV on generic bf16 shapes "
        "without ORBIT-2 / Bayes-CAST code; parent process survives a child segfault.",
    )
    parser.add_argument(
        "--orbit-xformers-modes",
        type=str,
        default="ck",
        help="Comma list with --orbit-include-xformers-ck: ck|dispatch (default: ck). "
        "`ck` = MemoryEfficientAttentionCkOp; `dispatch` = no explicit op.",
    )
    parser.add_argument(
        "--orbit-xformers-dropout",
        type=str,
        default="0.0",
        help="Comma list of dropout_p for xFormers MEA (e.g. 0.0,0.1) matching probe training-like cases.",
    )
    parser.add_argument(
        "--orbit-xformers-timeout",
        type=float,
        default=180.0,
        help="Timeout seconds per isolated xFormers subprocess (default: 180).",
    )
    args = parser.parse_args()

    if args.flash_test and (
        activate_flash_attention_impl is None or restore_flash_attention_impl is None
    ):
        raise SystemExit(
            "ERROR: --flash_test requires activate_flash_attention_impl / "
            "restore_flash_attention_impl (missing in this PyTorch build). "
            "Omit --flash_test or use a newer PyTorch checkout."
        )

    seed = 123
    torch.manual_seed(seed)
    results: list[Experiment] = []

    if args.orbit_selfattn:
        sweep_raw = args.orbit_sweep_backends or "default,flash,efficient,math"
        labels = [s.strip() for s in sweep_raw.split(",") if s.strip()]
        batches = [int(s) for s in args.orbit_selfattn_batches.split(",") if s.strip()]
        out_dir = os.environ.get("ORBIT2_SDPA_CSV_DIR", "/tmp/orbit2_sdpa_benchmark_results")
        for lab in labels:
            bk = _parse_orbit_backend(lab)
            print(f"\n{'=' * 14} orbit self-attn sdpa_kernel={lab!r} ({bk!s}) "
                  f"seq={args.orbit_selfattn_seq} heads={args.orbit_selfattn_heads} {'=' * 14}\n",
                  flush=True)
            configs = generate_orbit_selfattn_configs(
                bk, batches=batches, seq_len=args.orbit_selfattn_seq,
                num_heads=args.orbit_selfattn_heads,
            )
            chunk: list[Experiment] = []
            for config in tqdm(configs, desc=f"selfattn[{lab}]"):
                try:
                    chunk.append(Experiment(config, run_single_experiment(config)))
                except Exception as e:  # noqa: BLE001
                    print(f"  [FAIL] batch={config.batch_size}: "
                          f"{type(e).__name__}: {str(e).splitlines()[0]}", flush=True)
            if chunk:
                print_results(chunk)
                write_results_to_csv(chunk, out_dir)
            results.extend(chunk)
        return

    if args.orbit_varagg:
        sweep_raw = args.orbit_sweep_backends or "default,math,efficient,flash"
        labels = [s.strip() for s in sweep_raw.split(",") if s.strip()]
        batches = [int(s) for s in args.orbit_varagg_batches.split(",") if s.strip()]
        out_dir = os.environ.get("ORBIT2_SDPA_CSV_DIR", "/tmp/orbit2_sdpa_benchmark_results")
        for lab in labels:
            bk = _parse_orbit_backend(lab)
            print(f"\n{'=' * 14} orbit var_agg sdpa_kernel={lab!r} ({bk!s}) "
                  f"tokens={args.orbit_varagg_tokens} N_i={args.orbit_varagg_ni} {'=' * 14}\n",
                  flush=True)
            configs = generate_orbit_varagg_configs(
                bk, batches=batches, tokens=args.orbit_varagg_tokens, n_i=args.orbit_varagg_ni
            )
            chunk: list[Experiment] = []
            for config in tqdm(configs, desc=f"varagg[{lab}]"):
                B_eff = config.batch_size
                try:
                    chunk.append(Experiment(config, run_single_experiment(config)))
                    print(f"  [PASS] batch={B_eff // args.orbit_varagg_tokens} "
                          f"(B={B_eff})", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"  [FAIL] batch={B_eff // args.orbit_varagg_tokens} "
                          f"(B={B_eff}): {type(e).__name__}: {str(e).splitlines()[0]}", flush=True)
            if chunk:
                print_results(chunk)
                write_results_to_csv(chunk, out_dir)
            results.extend(chunk)
        return

    if args.orbit_micro:
        sweep_raw = args.orbit_sweep_backends or "default,math,efficient,flash"
        labels = [s.strip() for s in sweep_raw.split(",") if s.strip()]
        out_dir = os.environ.get("ORBIT2_SDPA_CSV_DIR", "/tmp/orbit2_sdpa_benchmark_results")
        for lab in labels:
            bk = _parse_orbit_backend(lab)
            print(f"\n{'=' * 18} orbit sdpa_kernel={lab!r} ({bk!s}) {'=' * 18}\n", flush=True)
            configs = generate_orbit_micro_configs(bk)
            chunk: list[Experiment] = []
            for config in tqdm(configs, desc=f"orbit[{lab}]"):
                try:
                    chunk.append(Experiment(config, run_single_experiment(config)))
                except Exception as e:
                    print(f"SKIP {config.asdict()}: {type(e).__name__}: {e}", flush=True)
            if chunk:
                print_results(chunk)
                write_results_to_csv(chunk, out_dir)
            results.extend(chunk)

        if args.orbit_include_xformers_ck:
            modes = []
            for s in args.orbit_xformers_modes.split(","):
                m = s.strip().lower()
                if not m:
                    continue
                if m not in ("ck", "dispatch"):
                    raise SystemExit(
                        f"unknown --orbit-xformers-modes entry {m!r} (allowed: ck, dispatch)"
                    )
                modes.append(m)
            if not modes:
                raise SystemExit("--orbit-xformers-modes produced an empty list")
            dropout_ps: list[float] = []
            for s in args.orbit_xformers_dropout.split(","):
                s = s.strip()
                if not s:
                    continue
                dropout_ps.append(float(s))
            if not dropout_ps:
                raise SystemExit("--orbit-xformers-dropout produced an empty list")

            print(
                f"\n{'=' * 18} xFormers MEA (isolated subprocess per case) {'=' * 18}\n",
                flush=True,
            )
            script_path = os.path.abspath(__file__)
            xf_rows = run_orbit_xformers_mea_suite(
                script_path=script_path,
                modes=modes,
                dropout_ps=dropout_ps,
                timeout_s=float(args.orbit_xformers_timeout),
            )
            print(tabulate(xf_rows, headers="keys", tablefmt="pretty", floatfmt=".3f"), flush=True)
            write_xformers_mea_results_to_csv(xf_rows, out_dir)
        return

    if args.flash_test:
        configs = generate_experiment_configs_with_flash()
    else:
        configs = generate_experiment_configs()
    for config in tqdm(configs):
        results.append(Experiment(config, run_single_experiment(config)))

    print_results(results)
    write_results_to_csv(results, "../benchmark_results")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--internal-xformers-worker":
        raise SystemExit(_internal_xformers_worker_main(sys.argv[2]))
    main()
