"""Synthetic data generators for booth demo mode.

Each generator runs in ~5-15 seconds on a CPU and produces plausible-looking
output files that the Analyze step can display without a real GPU or dataset.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Callable


# ── Utility ───────────────────────────────────────────────────────────────────

def _progress(emit: Callable, run_id: str, steps: list[str], delay: float = 0.8):
    for step in steps:
        emit(run_id, f"[synthetic] {step}")
        time.sleep(delay)


# ── Earth Science ─────────────────────────────────────────────────────────────

def _earth_science(slug: str, task: str, prompt: str, params: dict,
                   out_dir: Path, emit: Callable, run_id: str) -> dict:
    try:
        import numpy as np
    except ImportError:
        emit(run_id, "[synthetic] numpy not available — generating minimal output")
        return {"type": "weather", "slug": slug, "status": "stub"}

    emit(run_id, f"[synthetic] Loading ERA5-format grid (721×1440)...")
    time.sleep(0.5)

    steps = [
        "Initializing model weights from checkpoint...",
        "Preprocessing atmospheric state (t2m, u10, v10, msl)...",
        "Running autoregressive forecast step 1/6...",
        "Running autoregressive forecast step 2/6...",
        "Running autoregressive forecast step 3/6...",
        "Running autoregressive forecast step 4/6...",
        "Running autoregressive forecast step 5/6...",
        "Running autoregressive forecast step 6/6...",
        "Writing output arrays to zarr...",
    ]
    _progress(emit, run_id, steps, delay=0.6)

    # Generate synthetic lat/lon temperature field
    lat = np.linspace(-90, 90, 73)
    lon = np.linspace(-180, 180, 144)
    LON, LAT = np.meshgrid(lon, lat)
    # Realistic temperature pattern: warmer tropics, cooler poles, zonal bands
    t2m = (295 - 40 * np.abs(np.sin(np.radians(LAT))) +
           5 * np.cos(2 * np.radians(LAT)) +
           3 * np.sin(np.radians(LON)) +
           np.random.default_rng(42).normal(0, 2, LAT.shape))

    # Save data as JSON for the frontend
    result = {
        "type": "weather_forecast",
        "slug": slug,
        "variable": "2m_temperature_K",
        "lat_min": -90, "lat_max": 90,
        "lon_min": -180, "lon_max": 180,
        "grid_shape": [73, 144],
        "forecast_steps": 6,
        "forecast_hours": [1, 6, 12, 18, 24, 48],
        "t2m_mean": float(np.mean(t2m)),
        "t2m_std": float(np.std(t2m)),
        "t2m_min": float(np.min(t2m)),
        "t2m_max": float(np.max(t2m)),
    }

    # Save field as flat list for visualization (downsampled)
    ds = t2m[::2, ::4]  # 37 x 36
    field_data = {
        "grid": ds.tolist(),
        "lat": lat[::2].tolist(),
        "lon": lon[::4].tolist(),
    }
    (out_dir / "forecast_field.json").write_text(json.dumps(field_data))
    (out_dir / "result_summary.json").write_text(json.dumps(result))

    emit(run_id, f"[synthetic] t2m mean={result['t2m_mean']:.1f} K, range=[{result['t2m_min']:.1f}, {result['t2m_max']:.1f}] K")
    emit(run_id, f"[synthetic] Output: forecast_field.json, result_summary.json")
    return result


# ── Materials ─────────────────────────────────────────────────────────────────

_ELEMENT_DATA = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
    "Fe": 55.845, "W": 183.84, "Li": 6.941, "Mn": 54.938,
    "Ti": 47.867, "Al": 26.982, "Co": 58.933, "Ni": 58.693,
    "Cu": 63.546, "Zn": 65.38,  "Ga": 69.723, "As": 74.922,
}

def _materials(slug: str, task: str, prompt: str, params: dict,
               out_dir: Path, emit: Callable, run_id: str) -> dict:
    rng = random.Random(42)
    if slug == "HydraGNN" and task == "train":
        # Replay REAL 1-GPU vs 8-GPU training results baked from the run logs.
        asset = Path(__file__).resolve().parent / "assets" / "hydragnn_training.json"
        if not asset.exists():
            emit(run_id, "[demo] training curves asset missing; run tools/bake_training_curves.py")
            return {"type": "training_convergence", "error": "asset_missing"}
        data = json.loads(asset.read_text())
        steps = [
            "Loading real HydraGNN training results (Alexandria DFT)...",
            "1-GPU run: 40k structures, 100 epochs...",
            "8-GPU DDP run: 268k structures, 100 epochs...",
            "Comparing convergence and final accuracy...",
            "Computing scaling speedup...",
        ]
        _progress(emit, run_id, steps, delay=0.6)
        (out_dir / "training_convergence.json").write_text(json.dumps(data, indent=2))
        m1 = data["runs"]["1gpu"]["metrics"]
        m8 = data["runs"]["8gpu"]["metrics"]
        emit(run_id, f"[demo] 1-GPU final: corr={m1.get('corr')} R2={m1.get('r2')} MAE={m1.get('mae')}")
        emit(run_id, f"[demo] 8-GPU final: corr={m8.get('corr')} R2={m8.get('r2')} MAE={m8.get('mae')}")
        result = dict(data)
        result["type"] = "training_convergence"
        result["slug"] = slug
        return result

    if slug == "HydraGNN":
        steps = [
            "Loading HydraGNN graph foundation model checkpoint...",
            "Building atomistic graph (nodes=atoms, edges=bonds)...",
            "Running multi-task GNN forward pass (energy, forces, stress)...",
            "Aggregating per-atom predictions...",
            "Computing bulk properties...",
        ]
        _progress(emit, run_id, steps, delay=0.7)

        n_atoms = rng.randint(12, 24)
        atoms = [{"id": i, "element": rng.choice(["Fe", "W", "C", "Ni", "Co"]),
                  "x": rng.uniform(-5, 5), "y": rng.uniform(-5, 5), "z": rng.uniform(-5, 5),
                  "fx": rng.gauss(0, 0.1), "fy": rng.gauss(0, 0.1), "fz": rng.gauss(0, 0.1)}
                 for i in range(n_atoms)]
        result = {
            "type": "atomistic_properties",
            "slug": slug,
            "n_atoms": n_atoms,
            "formation_energy_eV_per_atom": round(rng.gauss(-1.2, 0.4), 4),
            "bulk_modulus_GPa": round(rng.gauss(180, 30), 1),
            "shear_modulus_GPa": round(rng.gauss(65, 15), 1),
            "band_gap_eV": round(max(0, rng.gauss(0.5, 0.8)), 3),
            "magnetic_moment_muB": round(rng.gauss(2.2, 1.0), 3),
            "atoms": atoms,
        }
        (out_dir / "graph_properties.json").write_text(json.dumps(result, indent=2))
        emit(run_id, f"[synthetic] Formation energy: {result['formation_energy_eV_per_atom']} eV/atom")
        emit(run_id, f"[synthetic] Bulk modulus: {result['bulk_modulus_GPa']} GPa")
        return result

    elif slug == "MatterGen":
        steps = [
            "Initializing diffusion model (MatterGen)...",
            "Forward pass: noising crystal structure...",
            "Denoising step 1/20 (T=1.0 → T=0.95)...",
            "Denoising step 5/20 (T=0.75)...",
            "Denoising step 10/20 (T=0.50)...",
            "Denoising step 15/20 (T=0.25)...",
            "Denoising step 20/20 (T=0.0) — structure converged",
            "Running energy relaxation (DFT-lite approximation)...",
            "Checking thermodynamic stability...",
        ]
        _progress(emit, run_id, steps, delay=0.6)

        n_structures = 5
        structures = []
        for i in range(n_structures):
            a = round(rng.uniform(3.0, 6.5), 4)
            b = round(a * rng.uniform(0.85, 1.15), 4)
            c = round(a * rng.uniform(0.85, 1.40), 4)
            structures.append({
                "id": i,
                "formula": rng.choice(["WRe", "W2Re", "TiAl", "NiAl", "CoAl", "Fe3C"]),
                "space_group": rng.choice(["Pm-3m", "Im-3m", "Fm-3m", "P4/mmm", "P63/mmc"]),
                "a_A": a, "b_A": b, "c_A": c,
                "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
                "formation_energy_eV_per_atom": round(rng.gauss(-0.8, 0.5), 4),
                "is_stable": rng.random() > 0.3,
            })
        result = {"type": "crystal_generation", "slug": slug, "structures": structures, "n_generated": n_structures}
        (out_dir / "crystal_structures.json").write_text(json.dumps(result, indent=2))
        stable = sum(1 for s in structures if s["is_stable"])
        emit(run_id, f"[synthetic] Generated {n_structures} structures, {stable} thermodynamically stable")
        return result

    return {"type": "materials", "slug": slug, "status": "stub"}


# ── Healthcare ────────────────────────────────────────────────────────────────

# Valid SMILES for common scaffolds (hardcoded, no RDKit needed)
_SMILES_POOL = [
    "CC(=O)Oc1ccccc1C(=O)O",                     # aspirin
    "c1ccc(cc1)C(=O)Nc2ccccc2",                   # benzanilide
    "CC(C)(C)c1ccc(cc1)O",                         # BHT-like
    "O=C(O)c1ccccc1O",                             # salicylic acid
    "c1ccc2c(c1)cccc2",                            # naphthalene
    "CC1=CC=C(C=C1)S(=O)(=O)N",                   # sulfonamide
    "C1=CC=C(C=C1)OCC(=O)O",                      # phenoxyacetic acid
    "CC(=O)c1ccc(cc1)N",                           # aminoacetophenone
    "N1CCNCC1",                                    # piperazine
    "C1CCCCC1N",                                   # cyclohexylamine
    "c1ccc(cc1)Cl",                                # chlorobenzene
    "CC1=CN=CC=C1",                                # 2-picoline
    "c1ccncc1",                                    # pyridine
    "C1=CC=CN=C1",                                 # pyridine (alt)
    "CC(=O)NCC1=CC=CO1",                          # furamide
    "O=C1NCCO1",                                   # morpholin-3-one
    "c1ccc2[nH]cccc2c1",                          # indole
    "CN1CCCC1=O",                                  # pyrrolidone
    "CC(=O)c1ccco1",                               # acetylfuran
    "O=C(N)c1cccs1",                               # thioamide
]

def _healthcare(slug: str, task: str, prompt: str, params: dict,
                out_dir: Path, emit: Callable, run_id: str) -> dict:
    rng = random.Random(42)

    if slug == "GP-MoLFormer":
        steps = [
            "Loading GP-MoLFormer tokenizer and model weights...",
            "Encoding scaffold SMILES...",
            "Sampling from latent space (temperature=0.8)...",
            "Decoding token sequences to SMILES...",
            "Validating generated SMILES...",
            "Computing molecular properties (MW, LogP, TPSA)...",
        ]
        _progress(emit, run_id, steps, delay=0.7)

        n = 20
        molecules = []
        for i in range(n):
            smiles = _SMILES_POOL[i % len(_SMILES_POOL)]
            # Fake property estimates
            mw = round(rng.uniform(150, 500), 1)
            logp = round(rng.gauss(2.5, 1.5), 2)
            tpsa = round(rng.uniform(30, 140), 1)
            molecules.append({
                "id": i + 1,
                "smiles": smiles,
                "MW": mw,
                "LogP": logp,
                "TPSA": tpsa,
                "Lipinski_pass": mw <= 500 and logp <= 5 and tpsa <= 140,
            })
        result = {
            "type": "molecule_generation",
            "slug": slug,
            "n_generated": n,
            "n_valid": n,
            "lipinski_pass_rate": sum(1 for m in molecules if m["Lipinski_pass"]) / n,
            "molecules": molecules,
        }
        (out_dir / "generated_molecules.json").write_text(json.dumps(result, indent=2))
        emit(run_id, f"[synthetic] Generated {n} valid SMILES, Lipinski pass rate: {result['lipinski_pass_rate']:.0%}")
        return result

    elif slug == "SwinUNETR":
        steps = [
            "Loading SwinUNETR model (Swin Transformer backbone)...",
            "Loading 3D volume (128×128×64 voxels)...",
            "Patchifying volume into 3D tokens (patch_size=2)...",
            "Encoding with shifted-window attention (depth=4)...",
            "Decoding with skip connections...",
            "Applying softmax to get class probabilities...",
            "Thresholding segmentation map (confidence > 0.5)...",
        ]
        _progress(emit, run_id, steps, delay=0.6)
        result = {
            "type": "segmentation",
            "slug": slug,
            "volume_shape": [128, 128, 64],
            "n_classes": 4,
            "class_names": ["background", "tumor_core", "enhancing_tumor", "edema"],
            "dice_scores": {
                "tumor_core": round(rng.uniform(0.82, 0.92), 4),
                "enhancing_tumor": round(rng.uniform(0.78, 0.89), 4),
                "edema": round(rng.uniform(0.75, 0.88), 4),
            },
            "tumor_volume_mL": round(rng.uniform(5, 45), 1),
        }
        (out_dir / "segmentation_result.json").write_text(json.dumps(result, indent=2))
        emit(run_id, f"[synthetic] Tumor volume: {result['tumor_volume_mL']} mL, Dice (core): {result['dice_scores']['tumor_core']}")
        return result

    return {"type": "healthcare", "slug": slug, "status": "stub"}


# ── Physics Simulation ────────────────────────────────────────────────────────

def _physics(slug: str, task: str, prompt: str, params: dict,
             out_dir: Path, emit: Callable, run_id: str) -> dict:
    try:
        import numpy as np
    except ImportError:
        return {"type": "physics", "slug": slug, "status": "stub"}

    rng = np.random.default_rng(42)
    n_steps = 50
    nx = 64

    steps = [
        f"Initializing {slug} surrogate model...",
        "Loading pretrained weights...",
        "Setting initial condition (random Fourier modes)...",
        f"Running {n_steps}-step autoregressive rollout...",
        "Computing statistics (energy spectrum, RMSE)...",
        "Writing field snapshots...",
    ]
    _progress(emit, run_id, steps, delay=0.5)

    # 1D wave-like rollout
    x = np.linspace(0, 2 * np.pi, nx)
    traj = []
    u = np.sin(x) + 0.3 * np.sin(3 * x) + 0.1 * rng.standard_normal(nx)
    for i in range(n_steps):
        u = 0.95 * u + 0.05 * np.roll(u, 1) + 0.01 * rng.standard_normal(nx)
        if i % 10 == 0:
            traj.append({"step": i, "field": u.tolist(), "energy": float(np.sum(u**2))})

    result = {
        "type": "physics_rollout",
        "slug": slug,
        "n_steps": n_steps,
        "nx": nx,
        "trajectory": traj,
        "final_energy": float(np.sum(u**2)),
        "rmse_vs_step0": float(np.sqrt(np.mean((u - traj[0]["field"])**2))),
    }
    (out_dir / "rollout.json").write_text(json.dumps(result))
    emit(run_id, f"[synthetic] Rollout complete. Final energy: {result['final_energy']:.4f}")
    return result


# ── Protein Folding ───────────────────────────────────────────────────────────

def _protein(slug: str, task: str, prompt: str, params: dict,
             out_dir: Path, emit: Callable, run_id: str) -> dict:
    rng = random.Random(42)
    steps = [
        "Loading model weights...",
        "Encoding amino acid sequence...",
        "Running structure prediction (AlphaFold-style)...",
        "Relaxing predicted structure...",
    ]
    _progress(emit, run_id, steps, delay=1.0)
    n_res = rng.randint(80, 200)
    result = {
        "type": "protein_structure",
        "slug": slug,
        "n_residues": n_res,
        "plddt_mean": round(rng.uniform(72, 92), 2),
        "plddt_min": round(rng.uniform(45, 65), 2),
        "iptm": round(rng.uniform(0.6, 0.9), 3),
        "ptm": round(rng.uniform(0.7, 0.95), 3),
    }
    (out_dir / "structure_scores.json").write_text(json.dumps(result, indent=2))
    emit(run_id, f"[synthetic] pLDDT mean: {result['plddt_mean']}, iPTM: {result['iptm']}")
    return result


# ── Dispatcher ────────────────────────────────────────────────────────────────

def generate(slug: str, domain: str, task: str, prompt: str, params: dict,
             out_dir: Path, emit: Callable, run_id: str) -> dict:
    if domain == "earth_science":
        return _earth_science(slug, task, prompt, params, out_dir, emit, run_id)
    elif domain == "material_science":
        return _materials(slug, task, prompt, params, out_dir, emit, run_id)
    elif domain == "healthcare":
        return _healthcare(slug, task, prompt, params, out_dir, emit, run_id)
    elif domain == "physics_simulation":
        return _physics(slug, task, prompt, params, out_dir, emit, run_id)
    elif domain == "protein_folding":
        return _protein(slug, task, prompt, params, out_dir, emit, run_id)
    else:
        emit(run_id, f"[synthetic] Unknown domain: {domain}")
        return {"type": "unknown", "slug": slug, "domain": domain}
