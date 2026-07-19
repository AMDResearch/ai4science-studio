# Studio Backend Assets

Pre-baked data assets for the Studio demo mode. Files are excluded from git
(see .gitignore) because they are large and regenerable. Run the corresponding
script to recreate each asset.

## Assets

### `hydragnn_training.json` (~50 KB)
Real HydraGNN 1-GPU vs 8-GPU training results on Alexandria DFT data.

**Regenerate:**
```bash
cd studio/backend
python3 tools/bake_training_curves.py
```
*Requires: completed training runs in `/shared/spannala/models/HydraGNN/train_work/`*

---

### `dc_temperature.json` (~92 KB)
Real ERA5 center-point temperature for DC July 2024 heatwave (July 16) and
July 4 2024 Independence Day heat. Spatial field extended via physics model
(terrain lapse rate, urban heat island, Chesapeake Bay cooling).

**Regenerate (rate-limit immune, 8 API calls total):**
```bash
cd studio/backend
python3 tools/generate_dc_synthetic.py
```

**Regenerate (full bulk grid, ~150 real API calls per event — may rate-limit):**
```bash
cd studio/backend
python3 tools/fetch_dc_era5.py     # requires Open-Meteo rate limit to clear
```
*Source: Open-Meteo ERA5 reanalysis — free, no auth required*

---

### `gpmolformer_finetune.json` (pending)
GP-MoLFormer pair-tuning results: per-epoch loss, before/after QED/logP/Lipinski metrics,
sample molecules.

**Regenerate (requires SLURM + GP-MoLFormer container):**
```bash
cd studio/backend
python3 tools/bake_gpmolformer_finetune.py
# or submit: sbatch healthcare/models/GP-MoLFormer/examples/sbatch_pairtune_amd.sh
```
*Requires: IBM/gp-molformer cloned and RDKit available in the container*
