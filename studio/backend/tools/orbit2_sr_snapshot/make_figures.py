"""Generate figures for the ORBIT-2 DC OOD physics writeup."""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
OUT = os.environ.get("FIG_DIR", os.path.join(_REPO, "docs", "images"))
os.makedirs(OUT, exist_ok=True)
# Result JSON written by orbit2_ood_dc.py (OUT_JSON there).
_RESULT = os.environ.get("OOD_RESULT_JSON") or (
    os.environ.get("ORBIT2_SR_DIR") and os.path.join(os.environ["ORBIT2_SR_DIR"], "orbit2_ood_dc.json"))
if not _RESULT:
    raise SystemExit("set OOD_RESULT_JSON or ORBIT2_SR_DIR")
S = json.load(open(_RESULT))
maps = S["maps"]
m = S["metrics"]

DC = (38.89, -77.04)


def grid(mp):
    return (np.array([[np.nan if v is None else v for v in row] for row in mp["temp_c"]], float),
            np.array(mp["lat"]), np.array(mp["lon"]))


def dc_marker(ax, lat, lon):
    ax.plot(DC[1], DC[0], 'o', mfc='white', mec='#ED1C24', mew=1.6, ms=6)
    ax.annotate("DC", (DC[1], DC[0]), color='white', fontsize=7, fontweight='bold',
                xytext=(3, 3), textcoords='offset points')


# ---- Figure 1: 6-panel maps (shared color scale on land) ----
panels = [("coarse", "Coarse input (40 arcmin)"),
          ("pretrained", "Pretrained (OOD)"),
          ("heatwave", "Heatwave finetune"),
          ("urban", "Urban-conditioned"),
          ("physics", "+ UHI physics"),
          ("truth", "Open-Meteo truth")]
allv = []
for k, _ in panels:
    g, _, _ = grid(maps[k]); allv.append(g[np.isfinite(g)])
vmin = float(np.percentile(np.concatenate(allv), 2))
vmax = float(np.percentile(np.concatenate(allv), 98))

fig, axes = plt.subplots(2, 3, figsize=(11, 7.2))
for ax, (k, title) in zip(axes.ravel(), panels):
    g, lat, lon = grid(maps[k])
    ext = [lon.min(), lon.max(), lat.min(), lat.max()]
    im = ax.imshow(g, origin="lower" if lat[1] > lat[0] else "upper",
                   extent=ext, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10, fontweight="bold")
    dc_marker(ax, lat, lon)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("ORBIT-2 4x downscaling — DC 2m max temperature, July 16 2024 (Open-Meteo OOD)",
             fontsize=12, fontweight="bold")
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
cbar.set_label("2m max temperature (°C)")
fig.savefig(f"{OUT}/orbit2_dc_maps.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote orbit2_dc_maps.png")


# ---- Figure 2: MAE bar chart (all-window + urban-core) ----
dc = m["dc_tmax_mae_c"]; core = m["dc_core_tmax_mae_c"]
stages = ["pretrained", "bilinear", "heatwave", "urban", "physics"]
labels = ["Pretrained\n(OOD)", "Bilinear\nbaseline", "Heatwave\nfinetune",
          "Urban\nconditioning", "+UHI\nphysics"]
x = np.arange(len(stages)); w = 0.38
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - w/2, [dc[s] for s in stages], w, label="Whole window", color="#9ca3af")
b2 = ax.bar(x + w/2, [core[s] for s in stages], w, label="Urban core", color="#ED1C24")
ax.axhline(dc["bilinear"], ls="--", c="#6b7280", lw=1, alpha=0.7)
ax.axhline(core["bilinear"], ls="--", c="#ED1C24", lw=1, alpha=0.5)
ax.set_ylabel("2m max-temperature MAE (°C)")
ax.set_title("True out-of-distribution error: finetuning plateaus, physics conditioning breaks it",
             fontsize=11, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.legend(); ax.bar_label(b1, fmt="%.2f", fontsize=8, padding=2)
ax.bar_label(b2, fmt="%.2f", fontsize=8, padding=2)
ax.grid(axis="y", alpha=0.25)
fig.savefig(f"{OUT}/orbit2_dc_mae.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote orbit2_dc_mae.png")


# ---- Figure 3: GHSL urban density field ----
u = json.load(open(os.path.join(_REPO, "studio", "backend", "assets", "dc_urban_density.json")))
uf = np.array(u["built_up_fraction"]); ulat = np.array(u["lat"]); ulon = np.array(u["lon"])
fig, ax = plt.subplots(figsize=(6, 5.2))
ext = [ulon.min(), ulon.max(), ulat.min(), ulat.max()]
im = ax.imshow(uf, origin="lower" if ulat[1] > ulat[0] else "upper", extent=ext,
               aspect="auto", cmap="inferno")
dc_marker(ax, ulat, ulon)
ax.set_title("GHSL built-up surface fraction (EU JRC R2023A)\nurban-heat-island conditioning field",
             fontsize=10, fontweight="bold")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
cbar = fig.colorbar(im, ax=ax, shrink=0.85); cbar.set_label("built-up fraction")
fig.savefig(f"{OUT}/orbit2_dc_urban.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote orbit2_dc_urban.png")
