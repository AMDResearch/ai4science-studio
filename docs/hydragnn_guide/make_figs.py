"""Generate publication-quality figures for the HydraGNN guide from the CURRENT
consistent data (v3 8-GPU + 1-GPU-PBC runs, curated predictions, MACE sanity).
Uses the backend venv matplotlib. Pure read of committed assets + result JSONs.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = "/home/spannala/Projects/ai4science-studio/studio/backend/assets/"
R = "/shared/spannala/models/HydraGNN/train_work/results/"
OUT = "/home/spannala/Projects/ai4science-studio/docs/hydragnn_guide/figs/"
os.makedirs(OUT, exist_ok=True)

AMD_RED = "#ED1C24"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

tr = json.load(open(A + "hydragnn_training.json"))
g1, g8 = tr["runs"]["1gpu"], tr["runs"]["8gpu"]

# ── Fig 1: validation loss convergence ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
e1 = g1["epochs"]; e8 = g8["epochs"]
ax.plot([e["ep"] for e in e1], [e["val"] for e in e1], color="#f5a524", lw=1.6, label="1-GPU (40k structures)")
ax.plot([e["ep"] for e in e8], [e["val"] for e in e8], color=AMD_RED, lw=1.6, label="8-GPU (600k structures)")
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation loss (MSE)")
ax.set_title("HydraGNN validation-loss convergence on Alexandria DFT")
ax.set_ylim(0.1, 0.6); ax.legend()
fig.savefig(OUT + "fig_convergence.png"); plt.close(fig)
print("wrote fig_convergence.png")

# ── Fig 2: parity plots (predicted vs DFT), side by side ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
for ax, key, title, col in [(axes[0], "1gpu", f"1-GPU  (corr {g1['metrics']['corr']}, MAE {g1['metrics']['mae']})", "#f5a524"),
                            (axes[1], "8gpu", f"8-GPU  (corr {g8['metrics']['corr']}, MAE {g8['metrics']['mae']})", AMD_RED)]:
    p = tr["parity"][key]
    ax.scatter(p["true"], p["pred"], s=8, alpha=0.35, color=col, edgecolors="none")
    lo = min(min(p["true"]), min(p["pred"])); hi = max(max(p["true"]), max(p["pred"]))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("DFT reference (eV/atom)"); ax.set_ylabel("Predicted (eV/atom)")
    ax.set_title(title); ax.set_aspect("equal", "box")
fig.suptitle("Predicted vs DFT energy on held-out Alexandria structures")
fig.savefig(OUT + "fig_parity.png"); plt.close(fig)
print("wrote fig_parity.png")

# ── Fig 3: scaling bars (data size + accuracy) ──────────────────────────────
fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4))
axL.bar(["1-GPU", "8-GPU"], [g1["n_samples"], g8["n_samples"]], color=["#a1a1aa", AMD_RED])
axL.set_ylabel("Training structures"); axL.set_title("Training-set size")
for i, v in enumerate([g1["n_samples"], g8["n_samples"]]):
    axL.text(i, v, f"{v//1000}k", ha="center", va="bottom", fontweight="bold")
mets = ["corr", "r2", "mae"]; labels = ["Corr", "R²", "MAE"]
x = range(len(mets)); w = 0.35
axR.bar([i - w/2 for i in x], [g1["metrics"][m] for m in mets], w, label="1-GPU", color="#a1a1aa")
axR.bar([i + w/2 for i in x], [g8["metrics"][m] for m in mets], w, label="8-GPU", color=AMD_RED)
axR.set_xticks(list(x)); axR.set_xticklabels(labels); axR.set_title("Test accuracy"); axR.legend()
for i, m in enumerate(mets):
    axR.text(i - w/2, g1["metrics"][m], f"{g1['metrics'][m]:.2f}", ha="center", va="bottom", fontsize=9)
    axR.text(i + w/2, g8["metrics"][m], f"{g8['metrics'][m]:.2f}", ha="center", va="bottom", fontsize=9)
fig.suptitle("1-GPU vs 8-GPU scaling (identical PBC-edge pipeline; only data scale differs)")
fig.savefig(OUT + "fig_scaling.png"); plt.close(fig)
print("wrote fig_scaling.png")

# ── Fig 4: per-material predicted vs DFT (the 4 curated demo materials) ──────
p8 = json.load(open(A + "predictions_8gpu.json"))["predictions"]
p1 = {p["struct_index"]: p for p in json.load(open(A + "predictions_1gpu.json"))["predictions"]}
names = [p.get("material_name", p["formula"]).split(",")[0] for p in p8]
fig, ax = plt.subplots(figsize=(9, 4.6))
x = range(len(p8)); w = 0.26
ax.bar([i - w for i in x], [p1[p["struct_index"]]["predicted_energy_ev_per_atom"] for p in p8], w, label="1-GPU pred", color="#a1a1aa")
ax.bar([i for i in x],     [p["predicted_energy_ev_per_atom"] for p in p8], w, label="8-GPU pred", color=AMD_RED)
ax.bar([i + w for i in x], [p["dft_reference_ev_per_atom"] for p in p8], w, label="DFT reference", color="#38bdf8")
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(list(x)); ax.set_xticklabels([f"{n}\n({p['formula']})" for n, p in zip(names, p8)], fontsize=9)
ax.set_ylabel("Energy (eV/atom)"); ax.legend()
ax.set_title("Predicted vs DFT energy for the four curated demo materials")
fig.savefig(OUT + "fig_materials.png"); plt.close(fig)
print("wrote fig_materials.png")

print("all figures written to", OUT)
