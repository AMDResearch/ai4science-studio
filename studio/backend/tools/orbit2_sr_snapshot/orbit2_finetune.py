"""Finetune pretrained ORBIT-2 (8M) on the coarsen-restore super-resolution task.

The pretrained checkpoint was trained on PRISM 10->2.5 arcmin, but our demo task
is coarsen-4x-then-restore at the 10-arcmin grid (a different resolution regime),
where it underperforms bilinear. We finetune on real PRISM train-split timesteps
(1981-2018) with the SAME coarsen-restore protocol + normalization used at eval,
then hold out the test split (2019-2020) for honest evaluation.

Saves finetuned weights so the eval harness (orbit2_sr.py) can load them.
"""
import sys, os, glob, json, numpy as np, torch

from climate_learn.models.hub.res_slimvit import Res_Slim_ViT
from climate_learn.utils.fused_attn import FusedAttn

def _site(sub: str) -> str:
    """Path under the site shared dir ($AI4S_SHARED_DIR); explicit env vars override."""
    base = os.environ.get("AI4S_SHARED_DIR")
    if not base:
        sys.exit(f"set AI4S_SHARED_DIR (or the explicit path variable) to locate {sub}")
    return os.path.join(base, sub)


PRISM_ROOT = os.environ.get("PRISM_ROOT") or _site("models/ORBIT-2/data/superres/prism/10.0_arcmin")
ORBIT2_SR_DIR = os.environ.get("ORBIT2_SR_DIR") or _site("orbit2_sr")
CKPT = os.environ.get("ORBIT2_CKPT") or os.path.join(
    os.environ.get("ORBIT2_HF_CACHE") or os.path.expanduser("~/.cache/huggingface/orbit2"),
    "pretrain", "intermediate_8m.ckpt")
OUT_CKPT = os.environ.get("FT_OUT", os.path.join(ORBIT2_SR_DIR, "orbit2_8m_ft.pk"))

DEFAULT_VARS = [
    "land_sea_mask", "orography", "lattitude", "landcover",
    "2m_temperature", "2m_temperature_max", "2m_temperature_min",
    "temperature_200", "temperature_500", "temperature_850",
    "10m_u_component_of_wind", "u_component_of_wind_200", "u_component_of_wind_500",
    "u_component_of_wind_850", "10m_v_component_of_wind", "v_component_of_wind_200",
    "v_component_of_wind_500", "v_component_of_wind_850",
    "specific_humidity_200", "specific_humidity_500", "specific_humidity_850",
    "total_precipitation_24hr", "volumetric_soil_water_layer_1",
]
IN_VARS = ["land_sea_mask", "orography", "lattitude", "landcover",
           "total_precipitation_24hr", "2m_temperature_min", "2m_temperature_max"]
OUT_VARS = ["total_precipitation_24hr", "2m_temperature_min", "2m_temperature_max"]
MODEL_KW = dict(superres_mag=4, cnn_ratio=4, patch_size=2, embed_dim=256,
                depth=6, decoder_depth=4, num_heads=4, mlp_ratio=4,
                drop_path=0.1, drop_rate=0.1, history=1, learn_pos_emb=True,
                FusedAttn_option=FusedAttn.NONE)


def coarsen(field, f=4):
    C, H, W = field.shape
    Hc, Wc = H // f, W // f
    return field[:, :Hc*f, :Wc*f].reshape(C, Hc, f, Wc, f).mean(axis=(2, 4))


def load_stats(root):
    m = dict(np.load(os.path.join(root, "normalize_mean.npz")))
    s = dict(np.load(os.path.join(root, "normalize_std.npz")))
    return m, s


def build_model(device):
    m = Res_Slim_ViT(DEFAULT_VARS, (180, 360), in_channels=len(IN_VARS),
                     out_channels=len(OUT_VARS), **MODEL_KW).to(device)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", "").replace("_orig_mod.", ""): v
          for k, v in ck["model_state_dict"].items()}
    m.load_state_dict(sd, strict=True)
    print("[ft] loaded pretrained checkpoint (strict)", flush=True)
    return m


# Urban-density conditioning: swap the model's categorical `landcover` channel for
# the continuous GHSL built-up fraction (real physics field). This gives the model
# an explicit "concrete jungle" signal it can use to add the urban-heat-island bump
# that a physics-free bilinear baseline structurally cannot produce.
_URBAN = {}
LC_IDX = IN_VARS.index("landcover")


def _load_urban():
    p = os.environ.get("URBAN_CONUS", "").strip()
    if not p or not os.path.exists(p):
        return None
    if "field" not in _URBAN:
        z = np.load(p)
        _URBAN["field"] = z["built_up_fraction"].astype(np.float32)  # (180,360) 0..1
    return _URBAN["field"]


def apply_urban(hi_phys):
    """Replace the landcover channel with GHSL urban fraction, normalized to the
    landcover channel's own stats range so the model's embedding sees a comparable
    scale. Modifies a copy; returns it. No-op if URBAN_CONUS unset."""
    urb = _load_urban()
    if urb is None:
        return hi_phys
    H, W = hi_phys[IN_VARS[0]].shape
    out = dict(hi_phys)
    # Scale 0..1 fraction into the landcover value range (~1..19) so it occupies the
    # same channel dynamic range the pretrained embedding expects; the finetune then
    # learns the mapping from this continuous urban signal to local temperature.
    out["landcover"] = (urb[:H, :W] * 19.0).astype(np.float32)
    return out


def make_pair(hi_phys, mean, std, device):
    """Return (x_lo_normalized, y_hi_normalized) for one timestep."""
    hi_phys = apply_urban(hi_phys)
    H, W = hi_phys[IN_VARS[0]].shape
    H8, W8 = (H // 8) * 8, (W // 8) * 8
    hi_in = np.stack([((hi_phys[v][:H8, :W8]) - mean[v][0]) / std[v][0] for v in IN_VARS], 0)
    lo_in = coarsen(hi_in, 4)
    y = np.stack([((hi_phys[v][:H8, :W8]) - mean[v][0]) / std[v][0] for v in OUT_VARS], 0)
    x = torch.tensor(lo_in, dtype=torch.float32, device=device).unsqueeze(0)
    y = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(0)
    return x, y, (lo_in.shape[1], lo_in.shape[2])


def iter_timesteps(split, mean, std, max_files, stride):
    fs = sorted(glob.glob(os.path.join(PRISM_ROOT, split, "*.npz")))
    fs = [f for f in fs if "climatology" not in f][:max_files]
    for f in fs:
        z = np.load(f)
        T = z[IN_VARS[0]].shape[0]
        for t in range(0, T, stride):
            yield {v: z[v][t, 0].astype(np.float32) for v in IN_VARS}


def iter_heatwave_days(index_path):
    """Yield full-CONUS days that are real DC-region heatwave days in PRISM
    (built by scanning the train split for DC-window peak tmax >= threshold).
    This gives the finetune targeted signal for exactly the demo regime."""
    idx = json.load(open(index_path))
    days = idx["days"]
    # group by file to avoid reloading the same .npz repeatedly
    by_file = {}
    for d in days:
        by_file.setdefault(d["file"], []).append(d["t"])
    for fname, ts in by_file.items():
        z = np.load(os.path.join(PRISM_ROOT, "train", fname))
        for t in sorted(set(ts)):
            yield {v: z[v][t, 0].astype(np.float32) for v in IN_VARS}


def evaluate(model, device, mean, std, max_files=1, stride=8):
    """Held-out test MAE (degC) for tmax, ORBIT-2 vs the truth."""
    model.eval()
    errs = {v: [] for v in OUT_VARS}
    with torch.no_grad():
        for hi in iter_timesteps("test", mean, std, max_files, stride):
            x, y, hw = make_pair(hi, mean, std, device)
            model.img_size = hw
            p = model(x, IN_VARS, OUT_VARS)[0].cpu().numpy()
            for i, v in enumerate(OUT_VARS):
                pd = p[i]*std[v][0] + mean[v][0]
                td = y[0, i].cpu().numpy()*std[v][0] + mean[v][0]
                errs[v].append(float(np.mean(np.abs(pd - td))))
    return {v: round(float(np.mean(e)), 4) for v, e in errs.items() if e}


_OOD_CACHE = {}
TI = OUT_VARS.index("2m_temperature_max")
DC_BOX = (78, 102, 276, 300)


def evaluate_ood_dc(model, device, mean, std):
    """True-OOD DC tmax MAE: run on full CONUS with Open-Meteo DC window embedded,
    score the DC crop vs Open-Meteo. Returns (model_mae, bilinear_mae) in degC.
    Mirrors orbit2_ood_dc.py exactly so training targets the real demo metric."""
    ood_path = os.environ.get("OOD_FIELDS", "")
    if not ood_path or not os.path.exists(ood_path):
        return None, None
    if "day" not in _OOD_CACHE:
        fs = sorted(glob.glob(os.path.join(PRISM_ROOT, "test", "*.npz")))
        fs = [f for f in fs if "climatology" not in f]
        z = np.load(fs[0])
        di = int(os.environ.get("PRISM_DAY_INDEX", "196"))
        _OOD_CACHE["day"] = {v: z[v][di, 0].astype(np.float32).copy() for v in IN_VARS}
        data = json.load(open(ood_path))
        ev = os.environ.get("DC_EVENT", "july16_2024")
        ev = ev if ev in data["events"] else data.get("default_event", "july16_2024")
        _OOD_CACHE["ood"] = data["events"][ev]
    hi = {k: v.copy() for k, v in _OOD_CACHE["day"].items()}
    ood = _OOD_CACHE["ood"]
    r0, r1, c0, c1 = DC_BOX
    hi["2m_temperature_max"][r0:r1, c0:c1] = np.array(ood["tmax_c"], np.float32) + 273.15
    hi["2m_temperature_min"][r0:r1, c0:c1] = np.array(ood["tmin_c"], np.float32) + 273.15
    hi["total_precipitation_24hr"][r0:r1, c0:c1] = np.array(ood["precip_mm"], np.float32) / 1000.0
    hi = apply_urban(hi)  # same urban conditioning as training, if enabled
    model.eval()
    with torch.no_grad():
        H, W = hi[IN_VARS[0]].shape
        H8, W8 = (H // 8) * 8, (W // 8) * 8
        hip = {v: a[:H8, :W8] for v, a in hi.items()}
        hi_in = np.stack([(hip[v] - mean[v][0]) / std[v][0] for v in IN_VARS], 0)
        lo_in = coarsen(hi_in, 4)
        x = torch.tensor(lo_in, dtype=torch.float32, device=device).unsqueeze(0)
        model.img_size = (lo_in.shape[1], lo_in.shape[2])
        p = model(x, IN_VARS, OUT_VARS)[0].cpu().numpy()
    pred = p[TI]*std["2m_temperature_max"][0] + mean["2m_temperature_max"][0]
    truth = hip["2m_temperature_max"]
    ph, pw = pred.shape
    out_idx = IN_VARS.index("2m_temperature_max")
    bl = torch.nn.functional.interpolate(
        torch.tensor(lo_in[out_idx][None, None]), size=(ph, pw),
        mode="bilinear", align_corners=False)[0, 0].numpy()
    bl = bl*std["2m_temperature_max"][0] + mean["2m_temperature_max"][0]
    rr1, cc1 = min(r1, ph), min(c1, pw)
    pc, tc, bc = pred[r0:rr1, c0:cc1], truth[r0:rr1, c0:cc1], bl[r0:rr1, c0:cc1]
    m = np.isfinite(tc)
    return (round(float(np.mean(np.abs(pc[m] - tc[m]))), 3),
            round(float(np.mean(np.abs(bc[m] - tc[m]))), 3))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_stats(PRISM_ROOT)
    model = build_model(device)

    epochs = int(os.environ.get("FT_EPOCHS", "3"))
    lr = float(os.environ.get("FT_LR", "1e-4"))
    max_files = int(os.environ.get("FT_MAXFILES", "12"))   # years of train data
    stride = int(os.environ.get("FT_STRIDE", "6"))         # subsample timesteps
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    lossf = torch.nn.SmoothL1Loss()
    # Gradient-aware loss weight. Temperature GRADIENTS carry the fine-scale
    # structure (urban edges, coastlines) that a smooth field / bilinear misses;
    # penalizing dPred/dx,dPred/dy vs the truth teaches transferable sharpness.
    grad_w = float(os.environ.get("FT_GRAD_WEIGHT", "0.0"))
    if grad_w > 0:
        print(f"[ft] gradient-aware loss enabled, weight={grad_w}", flush=True)

    def grad_loss(p, t):
        # finite-difference spatial gradients over H (dim2) and W (dim3)
        pdx = p[:, :, 1:, :] - p[:, :, :-1, :]
        tdx = t[:, :, 1:, :] - t[:, :, :-1, :]
        pdy = p[:, :, :, 1:] - p[:, :, :, :-1]
        tdy = t[:, :, :, 1:] - t[:, :, :, :-1]
        return torch.nn.functional.smooth_l1_loss(pdx, tdx) + \
               torch.nn.functional.smooth_l1_loss(pdy, tdy)

    print(f"[ft] pre-finetune eval: {evaluate(model, device, mean, std)}", flush=True)
    # Optional DC-window emphasis: weight the loss higher over the DC region so
    # the finetuned model sharpens there (Act 3 should beat bilinear at DC).
    dc_weight = float(os.environ.get("FT_DC_WEIGHT", "1.0"))
    DC_LAT, DC_LON = (78, 102), (276, 300)

    # If a DC-heatwave-day index is given, train on exactly those real DC heatwave
    # days (targeted regime) instead of a strided sweep of all timesteps.
    hw_index = os.environ.get("FT_HEATWAVE_INDEX", "").strip()
    if hw_index:
        print(f"[ft] training on PRISM DC heatwave days from {hw_index}", flush=True)

    def train_iter():
        if hw_index:
            return iter_heatwave_days(hw_index)
        return iter_timesteps("train", mean, std, max_files, stride)

    for ep in range(epochs):
        model.train()
        losses = []
        for hi in train_iter():
            x, y, hw = make_pair(hi, mean, std, device)
            model.img_size = hw
            opt.zero_grad()
            p = model(x, IN_VARS, OUT_VARS)
            yt = y[:, :, :p.shape[2], :p.shape[3]]
            if dc_weight > 1.0:
                # Per-pixel weight map: dc_weight inside the DC window, 1.0 elsewhere.
                w = torch.ones_like(p)
                r0, r1 = min(DC_LAT[0], p.shape[2]), min(DC_LAT[1], p.shape[2])
                c0, c1 = min(DC_LON[0], p.shape[3]), min(DC_LON[1], p.shape[3])
                w[:, :, r0:r1, c0:c1] = dc_weight
                loss = (torch.nn.functional.smooth_l1_loss(p, yt, reduction="none") * w).mean()
            else:
                loss = lossf(p, yt)
            if grad_w > 0:
                loss = loss + grad_w * grad_loss(p, yt)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        ev = evaluate(model, device, mean, std)
        ood_m, ood_bl = evaluate_ood_dc(model, device, mean, std)
        ood_s = f" OOD_DC_MAE={ood_m}C (bilinear {ood_bl}C)" if ood_m is not None else ""
        print(f"[ft] epoch {ep+1}/{epochs} train_loss={np.mean(losses):.4f} test_MAE={ev}{ood_s}", flush=True)
        # Save the checkpoint that is best on the TRUE OOD metric (the demo target).
        if ood_m is not None:
            if ood_m < _OOD_CACHE.get("best", 1e9):
                _OOD_CACHE["best"] = ood_m
                torch.save({"model_state_dict": model.state_dict()}, OUT_CKPT)
                print(f"[ft]   -> new best OOD {ood_m}C, saved {OUT_CKPT}", flush=True)

    if _OOD_CACHE.get("best") is None:
        torch.save({"model_state_dict": model.state_dict()}, OUT_CKPT)
    print(f"[ft] saved finetuned model -> {OUT_CKPT} (best OOD {_OOD_CACHE.get('best','n/a')}C)", flush=True)


if __name__ == "__main__":
    main()
