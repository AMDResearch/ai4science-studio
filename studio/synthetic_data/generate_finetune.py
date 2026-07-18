#!/usr/bin/env python3
"""Generate instruction-following JSONL fine-tuning data for each domain.

Output: synthetic_data/<domain>_finetune.jsonl
Format: {"prompt": "...", "completion": "..."}

These are booth-showable as demo fine-tuning datasets.
Usage:
    python generate_finetune.py [--out-dir ./]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# ── Domain templates ──────────────────────────────────────────────────────────

EARTH_SCIENCE = [
    ("Run a 24-hour forecast for a Category 3 hurricane over the Gulf of Mexico starting {date} at 3 km resolution.",
     "Initializing StormCast with ERA5+HRRR boundary conditions... running 24 autoregressive steps... "
     "Track: 25.3°N, 90.1°W at t=0 → 28.7°N, 87.4°W at t=24h. Max wind: 125 kt. Landfall probability: 72%."),
    ("Downscale ERA5 reanalysis at 0.25° to 4 km over California for July {year}.",
     "Loading ORBIT-2 checkpoint... encoding 0.25° ERA5 fields... applying learned downscaling operator... "
     "Output: 4 km temperature, wind, and precipitation. Coastal cooling captured. RMSE vs PRISM: 0.8 K."),
    ("Produce a 10-member ensemble forecast for the Pacific Northwest for {date}.",
     "GenCast: initializing 10 ensemble members with perturbed ICs... 5-day global rollout... "
     "Mean precipitation: 18 mm. 90th percentile: 45 mm. Snow level ensemble spread: 1800-3200 m."),
]

MATERIAL_SCIENCE = [
    ("Predict the formation energy and bulk modulus of {formula} using HydraGNN.",
     "Building atomistic graph (N={n} atoms)... HydraGNN forward pass (energy + forces)... "
     "Formation energy: {ef:.3f} eV/atom. Bulk modulus: {bm:.1f} GPa. Shear modulus: {sm:.1f} GPa."),
    ("Generate 10 stable crystal structures for lithium cathode materials with MatterGen.",
     "MatterGen diffusion: 20 denoising steps... generating 10 structures... "
     "Stable: 7/10. Best: Li2MnO4 (E=-1.21 eV/atom, Fm-3m, a=4.02 Å). "
     "Predicted voltage: 4.1 V. Download: crystal_structures.json"),
    ("Screen {n} tungsten alloy compositions for radiation resistance.",
     "HydraGNN graph prediction on {n} W-X compositions... "
     "Top candidate: W-5%Re (displacement threshold: 98 eV, 312 K lattice stability). "
     "Predicted neutron activation: 0.12 Bq/g after 10y. Recommended for fusion blanket testing."),
]

HEALTHCARE = [
    ("Generate 20 drug-like molecules with scaffold {scaffold} for kinase inhibition.",
     "GP-MoLFormer sampling (T=0.8, scaffold-constrained)... validating 20 SMILES... "
     "All valid. Lipinski pass rate: 85%. MW range: 320-480 Da. Top candidate: IC50(EGFR) pred = 12 nM."),
    ("Segment tumor core and enhancing tumor in a 3D brain MRI volume.",
     "SwinUNETR forward pass (128³ input)... applying shifted-window attention (4 layers)... "
     "Tumor core Dice: 0.89. Enhancing tumor Dice: 0.84. Edema Dice: 0.82. "
     "Total tumor volume: 23.4 mL. Report: segmentation_result.json"),
    ("Run REINVENT4 lead optimization for EGFR inhibition, 200 steps.",
     "REINVENT4 RL loop... prior: MoLFormer-XL... scoring: EGFR docking + SA + QED... "
     "Step 200: mean reward=0.72. Best molecule: c1ccc(NC(=O)c2ccncc2)cc1 (pred IC50=8 nM, QED=0.81)."),
]

PHYSICS = [
    ("Run MATEY surrogate for turbulent channel flow at Re={re} for {n} time steps.",
     "MATEY loading pretrained weights... encoding initial condition (Fourier modes)... "
     "Rollout: {n} steps at dt=0.01. Turbulent kinetic energy: 2.34→2.41 (stable). "
     "Compared to DNS: RMSE=0.031. Speedup vs full simulation: 450×."),
    ("Apply Walrus-1.3B to a 2D Navier-Stokes problem for {n} autoregressive steps.",
     "Walrus encoder: tokenizing 2D field (64×64)... autoregressive rollout... "
     "Energy spectrum: E(k) ∝ k^-5/3 (Kolmogorov scaling preserved). "
     "Runtime: 2.1s on AMD MI300X. Equivalent PDE solver: 18 min."),
]

PROTEIN = [
    ("Predict the 3D structure of a {n}-residue protein with sequence beginning {prefix}.",
     "AlphaFold3 MSA search (UniRef90)... pair representation... structure module (48 blocks)... "
     "pLDDT mean: 86.2 (high confidence). ipTM: 0.84. pTM: 0.91. "
     "Output: predicted_structure.pdb. Predicted domain: TIM barrel fold."),
]

ALL_TEMPLATES = {
    "earth_science": EARTH_SCIENCE,
    "material_science": MATERIAL_SCIENCE,
    "healthcare": HEALTHCARE,
    "physics_simulation": PHYSICS,
    "protein_folding": PROTEIN,
}

_DATES = ["2025-07-15T06:00", "2024-08-20T12:00", "2025-01-10T00:00", "2024-06-01T18:00"]
_YEARS = [2023, 2024, 2025]
_FORMULAS = ["Fe3C", "W2Re", "TiAl", "NiAl3", "LiMn2O4", "Co3O4"]
_SCAFFOLDS = ["c1ccccc1", "c1ccncc1", "C1CCCNC1", "c1ccc2ncccc2c1"]
_RE = [180, 395, 590, 1000, 2000, 5200]
_PREFIXES = ["MKTLL", "GSSSS", "MATVK", "QVKLT"]


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        date=rng.choice(_DATES),
        year=rng.choice(_YEARS),
        formula=rng.choice(_FORMULAS),
        n=rng.randint(10, 50),
        ef=rng.gauss(-0.9, 0.4),
        bm=rng.gauss(180, 40),
        sm=rng.gauss(65, 20),
        scaffold=rng.choice(_SCAFFOLDS),
        re=rng.choice(_RE),
        prefix=rng.choice(_PREFIXES),
    )


def generate(domain: str, n: int, seed: int = 42) -> list[dict]:
    templates = ALL_TEMPLATES.get(domain, [])
    if not templates:
        return []
    rng = random.Random(seed)
    examples = []
    for i in range(n):
        prompt_tmpl, comp_tmpl = rng.choice(templates)
        try:
            p = _fill(prompt_tmpl, rng)
            c = _fill(comp_tmpl, rng)
        except (KeyError, ValueError):
            continue
        examples.append({"prompt": p, "completion": c, "domain": domain})
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--n-per-domain", type=int, default=100)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    for domain in ALL_TEMPLATES:
        examples = generate(domain, args.n_per_domain)
        outfile = out / f"{domain}_finetune.jsonl"
        with open(outfile, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"[generate_finetune] {domain}: {len(examples)} examples → {outfile}")
        total += len(examples)
    print(f"[generate_finetune] Total: {total} examples across {len(ALL_TEMPLATES)} domains")


if __name__ == "__main__":
    main()
