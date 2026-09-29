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

def _progress(emit: Callable, run_id: str, steps: list[str], delay: float = 0.8,
              tag: str = "synthetic"):
    for step in steps:
        emit(run_id, f"[{tag}] {step}")
        time.sleep(delay)


# ── Earth Science ─────────────────────────────────────────────────────────────

def _earth_science(slug: str, task: str, prompt: str, params: dict,
                   out_dir: Path, emit: Callable, run_id: str) -> dict:
    # ORBIT-2 robustness + finetuning story — REPLAY of real ORBIT-2 GPU runs:
    # PRISM-trained model applied to INDEPENDENT Open-Meteo DC heatwave data (true
    # out-of-distribution): pretrained -> OOD gap -> broad finetune -> DC finetune.
    # Numbers/maps are baked from actual MI355X runs (orbit2_ood_dc harness), measured.
    if slug == "ORBIT-2" and task == "story":
        # Per-event story asset (July 16 2024 default; July 4 2026 selectable).
        adir = Path(__file__).resolve().parent / "assets"
        ev = params.get("dc_event", "july16_2024")
        per_event = adir / f"orbit2_story_{ev}.json"
        asset = per_event if per_event.exists() else adir / "orbit2_story.json"
        if asset.exists():
            story = json.loads(asset.read_text())
            m = story.get("metrics", {})
            dc = m.get("dc_tmax_mae_c", {})
            core = m.get("dc_core_tmax_mae_c", {})
            steps = [
                "Replaying real ORBIT-2 runs on AMD MI355X (measured, not synthetic)...",
                "Loading independent Open-Meteo ERA5 DC heatwave data (never seen in training)...",
                f"Act 1 — pretrained ORBIT-2 8M applied out-of-distribution: DC tmax MAE {dc.get('pretrained','?')}°C...",
                f"Act 2 — out-of-distribution gap: bilinear baseline {dc.get('bilinear','?')}°C beats the pretrained model...",
                "Act 3 — a ~10k-param physics-residual head corrects the bilinear field with real "
                "GHSL urban density (EU JRC) + a coastal distance-to-water field (ocean sentinel cleaned first)...",
                f"Act 3 — urban-core MAE {core.get('bilinear','?')}°C (bilinear) -> {core.get('urban','?')}°C, "
                f"and whole-land {dc.get('bilinear','?')}°C -> {dc.get('urban','?')}°C — beats bilinear everywhere...",
                f"Act 4 — explicit UHI + coastal equations add nothing: the head already internalized the physics ({core.get('physics','?')}°C core)...",
                "Rendering DC temperature maps (°C): coarse, pretrained, physics-residual head, +analytic UHI, Open-Meteo truth...",
            ]
            _progress(emit, run_id, steps, delay=0.6)
            story["slug"] = slug
            (out_dir / "orbit2_story.json").write_text(json.dumps(story, indent=2))
            emit(run_id, f"[demo] Real ORBIT-2 OOD replay — urban-core MAE {core.get('bilinear','?')}°C (bilinear) -> "
                         f"{core.get('urban','?')}°C via the physics-residual head")
            return story

    # ORBIT-2 DC temperature downscaling demo — uses REAL ERA5 data baked from Open-Meteo.
    if slug == "ORBIT-2":
        asset = Path(__file__).resolve().parent / "assets" / "dc_temperature.json"
        if asset.exists():
            data = json.loads(asset.read_text())
            # Determine which event the user asked for (check params and prompt keywords)
            event_key = params.get("dc_event", "july16_2024")
            if event_key not in data.get("events", {}):
                event_key = data.get("default_event", "july16_2024")
            ev = data["events"][event_key]
            steps = [
                f"Loading ERA5 reanalysis: {ev.get('label', event_key)}...",
                "Extracting DC-area patch (38–42°N, 74–80°W)...",
                f"Coarse grid ({ev['coarse']['resolution_deg']}° = {ev['coarse']['model']}): {len(ev['coarse']['lat'])}×{len(ev['coarse']['lat'])} cells...",
                "Running ORBIT-2 super-resolution (4× downscaling)...",
                f"Fine grid ({ev['fine']['resolution_deg']}° = {ev['fine']['model']}): {len(ev['fine']['lat'])}×{len(ev['fine']['lat'])} cells...",
                "Computing temperature gradient metrics...",
                f"Peak temperature: {ev['peak_temp_c']} °C ({ev['peak_temp_f']} °F) — {ev['date']}",
            ]
            _progress(emit, run_id, steps, delay=0.55)
            result = {
                "type": "dc_downscaling",
                "slug": slug,
                "event_key": event_key,
                "label": ev.get("label", event_key),
                "date": ev["date"],
                "peak_temp_c": ev["peak_temp_c"],
                "peak_temp_f": ev["peak_temp_f"],
                "dc": ev["dc"],
                "coarse": ev["coarse"],
                "fine": ev["fine"],
                "time_series": ev.get("time_series", {}),
                "available_events": {k: v.get("label", k) for k, v in data.get("events", {}).items()},
                "note": ev.get("note", ""),
                "source": data.get("source", "Open-Meteo ERA5 reanalysis"),
            }
            (out_dir / "dc_temperature.json").write_text(json.dumps({
                "coarse": ev["coarse"], "fine": ev["fine"],
                "dc": ev["dc"], "date": ev["date"],
            }, indent=2))
            emit(run_id, f"[demo] Real ERA5 data — peak {ev['peak_temp_c']}°C / {ev['peak_temp_f']}°F")
            emit(run_id, f"[demo] Coarse grid {len(ev['coarse']['lat'])}×{len(ev['coarse']['lat'])} | Fine grid {len(ev['fine']['lat'])}×{len(ev['fine']['lat'])}")
            return result

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


def _build_crystal(kind: str, a: float, elements: tuple, reps: int = 2):
    """Build a real periodic crystal supercell (BCC, FCC, or rocksalt).
    Returns (atoms_list, formula_str). Coordinates are physically sensible
    lattice positions, not random — so the 3D viewer shows a real structure.
    """
    # Fractional basis positions per lattice type
    bases = {
        "bcc":      [(0, 0, 0), (0.5, 0.5, 0.5)],
        "fcc":      [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
        "rocksalt": [(0, 0, 0), (0.5, 0.5, 0.5)],  # two interpenetrating FCC
    }
    basis = bases.get(kind, bases["bcc"])
    atoms, counts = [], {}
    aid = 0
    for i in range(reps):
        for j in range(reps):
            for k in range(reps):
                for bi, (fx, fy, fz) in enumerate(basis):
                    # Assign element: alternate for binary compounds
                    el = elements[bi % len(elements)]
                    x = (i + fx) * a
                    y = (j + fy) * a
                    z = (k + fz) * a
                    atoms.append({"id": aid, "element": el,
                                  "x": round(x, 3), "y": round(y, 3), "z": round(z, 3)})
                    counts[el] = counts.get(el, 0) + 1
                    aid += 1
    # Build reduced formula
    import math as _m
    g = 0
    for v in counts.values():
        g = _m.gcd(g, v)
    g = g or 1
    formula = "".join(f"{el}{(counts[el]//g) if counts[el]//g > 1 else ''}"
                      for el in sorted(counts))
    return atoms, formula, counts

def _materials(slug: str, task: str, prompt: str, params: dict,
               out_dir: Path, emit: Callable, run_id: str) -> dict:
    rng = random.Random(42)
    if slug == "HydraGNN" and task == "train" and ("telemetry" in prompt.lower() or "omnistat" in prompt.lower()):
        # Replay REAL 8-GPU training GPU telemetry (baked from an Omnistat run).
        asset = Path(__file__).resolve().parent / "assets" / "hydragnn_telemetry_8gpu.json"
        if not asset.exists():
            emit(run_id, "[demo] telemetry asset missing; run studio/telemetry/bake_telemetry_asset.py")
            return {"type": "training_telemetry", "error": "asset_missing"}
        data = json.loads(asset.read_text())
        steps = [
            "Submitting 8-GPU HydraGNN training to AMD MI355X...",
            "Starting Omnistat user-mode telemetry (interval=1s)...",
            f"Training {data.get('epochs','?')} epochs on Alexandria DFT (8 GPUs)...",
            "Sampling GPU util / power / temp / FP64 / HBM via rocprofiler...",
            "Querying VictoriaMetrics for peak + time-series metrics...",
        ]
        _progress(emit, run_id, steps, delay=0.6, tag="demo")
        _pk = (data.get("telemetry") or {}).get("peaks", {})
        emit(run_id, f"[demo] peak power={_pk.get('power_w')}W util={_pk.get('gpu_util_pct')}% "
                     f"fp64={_pk.get('fp64_tflops')}TFLOP/s energy={_pk.get('energy_kj')}kJ")
        result = dict(data)
        result["type"] = "training_telemetry"
        result["slug"] = slug
        return result

    if slug == "HydraGNN" and task == "train":
        # Replay REAL 1-GPU vs 8-GPU training results baked from the run logs.
        asset = Path(__file__).resolve().parent / "assets" / "hydragnn_training.json"
        if not asset.exists():
            emit(run_id, "[demo] training curves asset missing; run tools/bake_training_curves.py")
            return {"type": "training_convergence", "error": "asset_missing"}
        data = json.loads(asset.read_text())
        _n1 = data["runs"]["1gpu"]["n_samples"]
        _n8 = data["runs"]["8gpu"]["n_samples"]
        _e1 = len(data["runs"]["1gpu"]["epochs"])
        _e8 = len(data["runs"]["8gpu"]["epochs"])
        steps = [
            "Loading real HydraGNN training results (Alexandria DFT)...",
            f"1-GPU run: {_n1//1000}k structures, {_e1} epochs...",
            f"8-GPU DDP run: {_n8//1000}k structures, {_e8} epochs...",
            "Comparing convergence and final accuracy (same PBC-edge pipeline)...",
            "Computing scaling speedup...",
        ]
        _progress(emit, run_id, steps, delay=0.6, tag="demo")
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
        # Replay a REAL prediction from our trained model on a real held-out
        # Alexandria structure (baked by inference/bake_predictions.py). Demo mode
        # mirrors live mode: real atoms, real predicted-vs-DFT energy — not random.
        variant = str(params.get("model_variant", "8gpu")).lower()
        asset = Path(__file__).resolve().parent / "assets" / f"predictions_{variant}.json"
        if not asset.exists():
            asset = Path(__file__).resolve().parent / "assets" / "predictions_8gpu.json"
        if asset.exists():
            data = json.loads(asset.read_text())
            preds = data.get("predictions", [])
            if preds:
                steps = [
                    f"Loading trained HydraGNN model ({data.get('model_variant','8gpu')}, "
                    f"val corr={data.get('val_corr')})...",
                    "Reading held-out Alexandria DFT structure...",
                    "Building atomistic graph (stored PBC edges)...",
                    "Running PNA forward pass on GPU...",
                    "Comparing predicted energy vs DFT reference...",
                ]
                _progress(emit, run_id, steps, delay=0.6, tag="demo")
                # Pick a structure. An explicit struct_index (from params or the
                # HG_DEMO_STRUCT_INDEX env) selects one baked prediction
                # deterministically — used by the screenshot/demo tooling to step
                # through every curated material. Otherwise pick from the fixed seed.
                _si = params.get("struct_index")
                if _si is None:
                    _si = os.environ.get("HG_DEMO_STRUCT_INDEX")
                # A control file lets external tooling (screenshot capture) rotate
                # the demo material between UI-driven runs without a backend restart.
                if _si is None:
                    _sf = os.environ.get("HG_DEMO_STRUCT_INDEX_FILE")
                    if _sf and Path(_sf).exists():
                        _si = Path(_sf).read_text().strip()
                if _si is not None and str(_si).strip() != "":
                    choice = preds[int(_si) % len(preds)]
                else:
                    choice = preds[rng.randrange(len(preds))]
                result = {
                    "type": "atomistic_energy",
                    "slug": slug,
                    "model": data.get("model_label", "HydraGNN (trained on Alexandria DFT)"),
                    "model_variant": data.get("model_variant"),
                    "formula": choice["formula"],
                    "material_name": choice.get("material_name"),
                    "application": choice.get("application"),
                    "n_atoms": choice["n_atoms"],
                    "atoms": choice["atoms"],
                    # Periodic unit cell (3x3 lattice, A) + PBC flags so the viewer
                    # can bond atoms across periodic boundaries (minimum image).
                    "cell": choice.get("cell"),
                    "pbc": choice.get("pbc"),
                    "predicted_energy_ev_per_atom": choice["predicted_energy_ev_per_atom"],
                    "dft_reference_ev_per_atom": choice["dft_reference_ev_per_atom"],
                    "abs_error_ev_per_atom": choice["abs_error_ev_per_atom"],
                    "model_val_corr": data.get("val_corr"),
                    "model_val_mae": data.get("val_mae"),
                    "model_val_r2": data.get("val_r2"),
                    "note": ("Real prediction from our trained HydraGNN model on a held-out "
                             "Alexandria DFT structure. Demo replays a pre-computed real "
                             "inference; Live mode runs the model on the cluster in real time."),
                }
                (out_dir / "prediction.json").write_text(json.dumps(result, indent=2))
                emit(run_id, f"[demo] {choice['formula']} ({choice['n_atoms']} atoms): "
                             f"pred={choice['predicted_energy_ev_per_atom']} vs "
                             f"DFT={choice['dft_reference_ev_per_atom']} eV/atom")
                return result

        # Fallback if predictions asset not yet baked: emit a clear message.
        emit(run_id, "[demo] predictions asset missing — run inference/bake_predictions.py")
        emit(run_id, "[demo] Live mode runs the real model directly; demo replays baked predictions.")
        return {"type": "atomistic_energy", "slug": slug, "error": "predictions_not_baked",
                "note": "Run inference/bake_predictions.py to generate real prediction replays."}

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

    if slug == "GP-MoLFormer" and task == "finetune":
        # GP-MoLFormer pair-tuning demo — replays REAL baked pair-tuning results.
        asset = Path(__file__).resolve().parent / "assets" / "gpmolformer_finetune.json"
        if asset.exists():
            data = json.loads(asset.read_text())
            prop = params.get("pairtune_prop", data.get("property", "qed")).upper()
            steps = [
                "Loading GP-MoLFormer pretrained checkpoint (IBM Research)...",
                f"Building {prop}-steered molecule pairs (1000 pairs)...",
                "Initializing soft-prompt tokens (pair-tuning PEFT)...",
                f"Pair-tuning epoch 1/{data.get('num_epochs',10)} — backbone frozen, prompts training...",
                f"Pair-tuning epoch 5/{data.get('num_epochs',10)} — {prop} improving...",
                f"Pair-tuning epoch {data.get('num_epochs',10)}/{data.get('num_epochs',10)} — converged",
                f"Evaluating: generating 20 molecules (before vs after)...",
                "Computing molecular property shift...",
            ]
            _progress(emit, run_id, steps, delay=0.55)
            result = dict(data)
            result["type"] = "molecule_finetune"
            result["slug"] = slug
            b, a = data.get("before", {}), data.get("after", {})
            emit(run_id, f"[demo] Before tuning: QED mean={b.get('qed_mean','?')}, Lipinski={b.get('lipinski_pass_rate','?'):.0%}" if isinstance(b.get('lipinski_pass_rate'), float) else f"[demo] Before: {b}")
            emit(run_id, f"[demo] After tuning:  QED mean={a.get('qed_mean','?')}, Lipinski={a.get('lipinski_pass_rate','?'):.0%}" if isinstance(a.get('lipinski_pass_rate'), float) else f"[demo] After: {a}")
            (out_dir / "finetune_result.json").write_text(json.dumps(result, indent=2))
            return result
        # Fallback: synthesize plausible pair-tuning results if asset not yet baked.
        emit(run_id, "[demo] Pair-tuning asset not found — generating plausible synthetic results...")
        emit(run_id, "[demo] Run tools/bake_gpmolformer_finetune.py to bake real results.")
        before_mols, after_mols = [], []
        for i in range(10):
            smiles = _SMILES_POOL[i % len(_SMILES_POOL)]
            mw = round(rng.uniform(200, 450), 1)
            logp_b = round(rng.gauss(3.0, 1.5), 2)
            logp_a = round(rng.gauss(2.3, 1.0), 2)   # lower logP after QED tuning
            qed_b = round(rng.uniform(0.3, 0.6), 3)
            qed_a = round(rng.uniform(0.55, 0.85), 3)  # higher QED after tuning
            before_mols.append({"smiles": smiles, "qed": qed_b, "logp": logp_b,
                                 "mw": mw, "lipinski": int(mw <= 500 and logp_b <= 5)})
            after_mols.append({"smiles": smiles, "qed": qed_a, "logp": logp_a,
                                "mw": mw, "lipinski": int(mw <= 500 and logp_a <= 5)})
        def means(lst): return {k: round(sum(m[k] for m in lst)/len(lst), 4) for k in ("qed","logp") if lst}
        def lrate(lst): return round(sum(m["lipinski"] for m in lst)/len(lst), 4) if lst else 0
        result = {
            "type": "molecule_finetune", "slug": slug, "property": "qed", "num_epochs": 10,
            "epochs": [{"ep": i, "loss": round(1.2 - 0.08*i + rng.gauss(0, 0.03), 4)} for i in range(10)],
            "before": {**means(before_mols), "lipinski_pass_rate": lrate(before_mols), "molecules": before_mols},
            "after":  {**means(after_mols),  "lipinski_pass_rate": lrate(after_mols),  "molecules": after_mols},
        }
        (out_dir / "finetune_result.json").write_text(json.dumps(result, indent=2))
        return result

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
