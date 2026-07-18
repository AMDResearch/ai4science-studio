"""Curated prompts per model + rule-based prompt validation."""
from __future__ import annotations

import re
import yaml
from pathlib import Path
from registry import REPO_ROOT

# Built-in curated prompts keyed by model slug
_CURATED: dict[str, list[dict]] = {
    "StormCast": [
        {"label": "Hurricane track forecast", "text": "Forecast a 48-hour track for a Gulf of Mexico tropical system starting 2025-08-15T06:00, output every 3 hours at 3 km resolution.", "task": "inference"},
        {"label": "Severe convection — Tornado Alley", "text": "Run a 24-hour convection-allowing forecast over Kansas and Oklahoma for 2024-05-20T12:00 during a severe weather outbreak.", "task": "inference"},
        {"label": "Rapid intensification ensemble", "text": "Generate a 10-member ensemble forecast of a Category 2 hurricane for 2025-09-10T00:00 over the Atlantic, 6 forecast steps.", "task": "ensemble"},
    ],
    "ORBIT-2": [
        {"label": "Western US climate downscaling", "text": "Downscale ERA5 reanalysis to 4 km over the Western US (California, Nevada, Arizona) for July 2024, focusing on temperature and precipitation.", "task": "inference"},
        {"label": "Pacific Northwest precipitation", "text": "Produce high-resolution precipitation maps for the Pacific Northwest for winter 2023-2024 from coarse GCM output.", "task": "inference"},
        {"label": "Heat dome event downscaling", "text": "Downscale the June 2021 Pacific Northwest heat dome event from ERA5 at hourly resolution over Washington and Oregon.", "task": "inference"},
    ],
    "ArchesWeather": [
        {"label": "Global 7-day forecast", "text": "Run a 7-day global weather forecast at 1.5 degree resolution starting 2025-01-15T00:00, outputting temperature, wind, and geopotential at 500 hPa.", "task": "inference"},
        {"label": "European winter storm", "text": "Forecast a 5-day trajectory of a North Atlantic winter storm impacting Western Europe, starting 2024-12-01T12:00.", "task": "inference"},
        {"label": "Training run (24-hour sample)", "text": "Run one training epoch of ArchesWeather on ERA5 data for 2020, with 24-hour rollout windows and a learning rate of 1e-4.", "task": "train"},
    ],
    "Aurora": [
        {"label": "10-day global forecast", "text": "Run a 10-day global Earth system forecast at 0.1 degree resolution from 2025-03-01T00:00, including atmospheric and surface variables.", "task": "inference"},
        {"label": "Monsoon season prediction", "text": "Forecast the 2024 South Asian monsoon onset over India, 15-day horizon from June 1 2024, at high resolution.", "task": "inference"},
    ],
    "GenCast": [
        {"label": "Ensemble weather forecast", "text": "Generate a 50-member probabilistic ensemble forecast for the contiguous US for 2025-04-10T00:00, 10-day horizon.", "task": "inference"},
        {"label": "Tropical storm ensemble", "text": "Run a 20-member ensemble for a developing tropical system in the Gulf of Mexico on 2025-09-01T06:00, 5-day horizon.", "task": "inference"},
    ],
    "NeuralGCM": [
        {"label": "Hybrid physics-ML forecast", "text": "Run a 30-day hybrid physics+ML atmospheric forecast starting 2025-01-01T00:00, comparing ML and physical parameterization outputs.", "task": "inference"},
    ],
    "PanguWeather": [
        {"label": "5-day deterministic forecast", "text": "Produce a 5-day deterministic global forecast at 0.25 degree resolution from 2025-02-15T00:00.", "task": "inference"},
        {"label": "Extreme cold air outbreak", "text": "Forecast the February 2021 Texas cold wave: 7-day horizon from 2021-02-07T00:00 focusing on 850 hPa temperature.", "task": "inference"},
    ],
    "HydraGNN": [
        {"label": "Iron-carbon alloy properties", "text": "Predict formation energy, atomic forces, and bulk modulus for an iron-carbon alloy with 5 atomic percent carbon using HydraGNN.", "task": "inference"},
        {"label": "Lithium-ion cathode screening", "text": "Screen 50 lithium manganese oxide compositions for voltage and capacity using the HydraGNN graph foundation model.", "task": "inference"},
        {"label": "Fusion blanket material", "text": "Predict radiation damage tolerance and thermal conductivity for tungsten-rhenium alloys relevant to fusion blanket design.", "task": "inference"},
        {"label": "Fine-tune on new dataset", "text": "Fine-tune HydraGNN on 1000 new DFT calculations of perovskite oxides for band gap prediction.", "task": "train"},
        {"label": "GPU scaling: 1 vs 8 GPUs", "text": "Compare HydraGNN energy-model training on 1 GPU versus 8 GPUs on Alexandria DFT data: show loss convergence, throughput speedup, and final accuracy.", "task": "train"},
    ],
    "MatterGen": [
        {"label": "Generate stable Li-ion cathode", "text": "Generate 20 novel stable crystal structures for lithium-ion cathode materials with target volumetric energy density > 800 Wh/L.", "task": "inference"},
        {"label": "High-temperature superconductor candidates", "text": "Generate 10 copper-oxide perovskite crystal structures with predicted Tc > 77 K.", "task": "inference"},
        {"label": "Fusion-grade tungsten alloy", "text": "Generate 15 binary tungsten alloy crystal structures optimized for high melting point and low neutron activation.", "task": "inference"},
        {"label": "Fine-tune on DFT dataset", "text": "Fine-tune MatterGen on a dataset of 5000 DFT-computed formation energies for transition metal nitrides.", "task": "train"},
    ],
    "GP-MoLFormer": [
        {"label": "Benzene-scaffold drug molecules", "text": "Generate 20 drug-like molecules with a benzene scaffold (SMILES: c1ccccc1) optimized for oral bioavailability.", "task": "inference"},
        {"label": "Kinase inhibitor generation", "text": "Generate 50 ATP-competitive kinase inhibitor candidates with molecular weight < 500 Da and LogP < 5.", "task": "inference"},
        {"label": "Antibiotic scaffold expansion", "text": "Generate 30 novel beta-lactam antibiotic analogs from the penicillin core scaffold for gram-negative bacteria.", "task": "inference"},
        {"label": "Fine-tune on bioactivity data", "text": "Fine-tune GP-MoLFormer on 2000 SMILES with IC50 measurements against EGFR kinase.", "task": "finetune"},
    ],
    "SwinUNETR": [
        {"label": "Brain tumor segmentation", "text": "Segment tumor core, enhancing tumor, and peritumoral edema in a 3D brain MRI using SwinUNETR trained on BraTS.", "task": "inference"},
        {"label": "Liver and tumor segmentation", "text": "Perform multi-organ segmentation of liver and hepatic tumors from contrast-enhanced CT volumes.", "task": "inference"},
        {"label": "Train on pancreas CT dataset", "text": "Fine-tune SwinUNETR for pancreas segmentation on 281 abdominal CT scans with cross-validation.", "task": "train"},
    ],
    "SemlaFlow": [
        {"label": "Drug-like 3D molecule generation", "text": "Generate 100 drug-like 3D molecular structures with 20-40 heavy atoms, targeting ADME-friendly property space.", "task": "inference"},
    ],
    "REINVENT4": [
        {"label": "Lead optimization for EGFR", "text": "Optimize a lead compound library for EGFR inhibition using REINVENT4 reinforcement learning, 200 optimization steps.", "task": "inference"},
    ],
    "MATEY": [
        {"label": "Turbulent channel flow surrogate", "text": "Run MATEY surrogate inference for turbulent channel flow at Re=5200, 100 time steps from a random initial condition.", "task": "inference"},
        {"label": "Plasma disruption prediction", "text": "Run MATEY on a tokamak plasma disruption scenario with Te=1 keV, ne=5e19 m^-3, predicting 50 ms evolution.", "task": "inference"},
    ],
    "Walrus": [
        {"label": "Fluid dynamics rollout", "text": "Run Walrus (1.3B) for 200 autoregressive steps on a 2D incompressible Navier-Stokes test case.", "task": "inference"},
        {"label": "Cross-domain physics surrogate", "text": "Apply Walrus to a coupled heat-transfer and fluid-flow problem with Re=100, Pr=0.71, 50 time steps.", "task": "inference"},
    ],
}

# Per-domain validation keywords
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "earth_science": ["date", "forecast", "hours", "days", "region", "resolution", "km", "degree",
                      "temperature", "pressure", "wind", "precipitation", "hurricane", "storm", "climate",
                      "era5", "gfs", "hrrr", "ensemble", "downscal"],
    "material_science": ["element", "alloy", "composition", "crystal", "energy", "force", "structure",
                         "bandgap", "lattice", "dft", "material", "perovskite", "oxide", "tungsten",
                         "lithium", "cathode", "percent", "atomic"],
    "healthcare": ["smiles", "molecule", "drug", "protein", "segmentation", "mri", "ct", "tumor",
                   "scaffold", "bioactivity", "kinase", "cancer", "organ", "3d", "inhibitor", "ic50"],
    "physics_simulation": ["reynolds", "mach", "turbulent", "flow", "navier", "plasma", "time step",
                           "channel", "surrogate", "rollout", "initial condition", "autoregressive"],
    "protein_folding": ["protein", "sequence", "structure", "folding", "amino acid", "chain", "pdb"],
}


def get_curated_prompts(slug: str, domain: str | None = None) -> list[dict]:
    # First check built-in
    if slug in _CURATED:
        return _CURATED[slug]
    # Then check model.yaml curated_prompts field
    model_yaml_path = REPO_ROOT / (domain or "") / "models" / slug / "model.yaml"
    if model_yaml_path.exists():
        with open(model_yaml_path) as f:
            data = yaml.safe_load(f) or {}
        if "curated_prompts" in data:
            return data["curated_prompts"]
    return []


def validate_prompt(slug: str, domain: str, prompt: str) -> dict:
    """Return {ok: bool, message: str}."""
    text = prompt.strip()
    if len(text) < 20:
        return {"ok": False, "message": "Prompt is too short. Add more detail about what you want to run."}
    if len(text) > 2000:
        return {"ok": False, "message": "Prompt is too long (max 2000 characters)."}

    keywords = _DOMAIN_KEYWORDS.get(domain, [])
    lower = text.lower()
    matched = [k for k in keywords if k in lower]
    if keywords and len(matched) == 0:
        hint_map = {
            "earth_science": "Try including a date, region, forecast duration, or variable name (temperature, wind, precipitation).",
            "material_science": "Try including a material composition, element, or property (formation energy, band gap, crystal structure).",
            "healthcare": "Try including a molecule scaffold (SMILES), organ, imaging modality, or biological target.",
            "physics_simulation": "Try including a flow condition (Reynolds number), time steps, or simulation type.",
            "protein_folding": "Try including an amino acid sequence or protein name.",
        }
        hint = hint_map.get(domain, "Add domain-specific details to your prompt.")
        return {"ok": False, "message": f"Prompt doesn't seem related to {domain.replace('_',' ')}. {hint}"}

    return {"ok": True, "message": "Prompt looks good."}
