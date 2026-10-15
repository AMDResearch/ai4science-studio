#!/usr/bin/env python3
"""Bake REAL HydraGNN predictions on multiple held-out Alexandria structures.

Runs the actual trained model (1-GPU and/or 8-GPU checkpoint) on a set of real
held-out structures and records real predicted-vs-DFT energies + atom coordinates.
The studio demo replays these instead of fabricating molecules.

Env:
    MODEL_PATH        checkpoint (.pk)
    HG_MODEL_VARIANT  label (1gpu | 8gpu)
    N_STRUCTURES      how many held-out structures to predict (default 8)
    BAKE_OUT          output JSON path
    HG_DATASET_BP     Alexandria ADIOS file (default: $HG_DATA_DIR/Alexandria-v2.bp)
"""
import sys, os, json, numpy as np, torch
INFER = os.environ["HG_INFER_REPO"]; sys.path.insert(0, INFER)
from adios2 import FileReader
from torch_geometric.data import Data
from torch_geometric.transforms import Distance, RadiusGraph
import hydragnn

RADIUS, MAX_NBR = 5.0, 20
_rg = RadiusGraph(RADIUS, loop=False, max_num_neighbors=MAX_NBR)
_dist = Distance(norm=False, cat=False)

_SYM = {1:"H",2:"He",3:"Li",4:"Be",5:"B",6:"C",7:"N",8:"O",9:"F",10:"Ne",11:"Na",
        12:"Mg",13:"Al",14:"Si",15:"P",16:"S",17:"Cl",18:"Ar",19:"K",20:"Ca",21:"Sc",
        22:"Ti",23:"V",24:"Cr",25:"Mn",26:"Fe",27:"Co",28:"Ni",29:"Cu",30:"Zn",31:"Ga",
        32:"Ge",33:"As",34:"Se",35:"Br",36:"Kr",37:"Rb",38:"Sr",39:"Y",40:"Zr",41:"Nb",
        42:"Mo",43:"Tc",44:"Ru",45:"Rh",46:"Pd",47:"Ag",48:"Cd",49:"In",50:"Sn",51:"Sb",
        52:"Te",53:"I",54:"Xe",55:"Cs",56:"Ba",57:"La",58:"Ce",59:"Pr",60:"Nd",61:"Pm",
        62:"Sm",63:"Eu",64:"Gd",65:"Tb",66:"Dy",67:"Ho",68:"Er",69:"Tm",70:"Yb",71:"Lu",
        72:"Hf",73:"Ta",74:"W",75:"Re",76:"Os",77:"Ir",78:"Pt",79:"Au",80:"Hg",81:"Tl",
        82:"Pb",83:"Bi",89:"Ac",90:"Th",91:"Pa",92:"U"}


def featurize(Z, pos):
    positions = torch.as_tensor(pos, dtype=torch.float32)
    an = torch.as_tensor(Z, dtype=torch.float32).view(-1, 1)
    d = Data(x=torch.cat([an, positions], dim=1), pos=positions)
    d = _rg(d); d = _dist(d)
    d.batch = torch.zeros(len(Z), dtype=torch.long)
    return d


def formula(Z):
    Z = np.asarray(Z, dtype=int)
    parts = []
    for z in sorted(set(Z.tolist())):
        c = int((Z == z).sum())
        parts.append(_SYM.get(z, f"Z{z}") + (str(c) if c > 1 else ""))
    return "".join(parts)


def main():
    model_path = os.environ["MODEL_PATH"]
    variant = os.environ.get("HG_MODEL_VARIANT", "8gpu")
    n_struct = int(os.environ.get("N_STRUCTURES", "8"))
    out_path = os.environ["BAKE_OUT"]

    ck = torch.load(model_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = hydragnn.models.create_model_config(config=ck["config"]["NeuralNetwork"], verbosity=0).to(device)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    metrics = ck.get("metrics", {})
    print(f"[bake] loaded {variant} model: corr={metrics.get('corr')}, MAE={metrics.get('mae')}", flush=True)

    bp = os.environ.get("HG_DATASET_BP") or os.path.join(
        os.environ.get("HG_DATA_DIR") or sys.exit("set HG_DATASET_BP or HG_DATA_DIR"),
        "Alexandria-v2.bp")
    predictions = []
    with FileReader(bp) as f:
        rd = f.read
        xc = rd("valset/x/variable_count").astype(int)
        xo = rd("valset/x/variable_offset").astype(int)
        X = rd("valset/x"); POS = rd("valset/pos"); Y = rd("valset/y")

        # Pick diverse structures: spread indices across the valset
        total = len(xc)
        indices = [int(i * total / (n_struct + 1)) for i in range(1, n_struct + 1)]

        for idx in indices:
            n = xc[idx]
            Z = X[xo[idx]:xo[idx]+n, 0]
            pos = POS[xo[idx]:xo[idx]+n]
            true_e = float(Y[idx][0])
            with torch.no_grad():
                p = model(featurize(Z, pos).to(device))
            pred_e = float(p[0].flatten()[0].item())

            Zc = np.asarray(Z, dtype=int); posc = np.asarray(pos, dtype=float)
            atoms = [{"element": _SYM.get(int(z), f"Z{int(z)}"), "Z": int(z),
                      "x": round(float(px), 4), "y": round(float(py), 4), "z": round(float(pz), 4)}
                     for z, (px, py, pz) in zip(Zc, posc)]
            predictions.append({
                "struct_index": idx,
                "formula": formula(Z),
                "n_atoms": int(n),
                "atoms": atoms,
                "predicted_energy_ev_per_atom": round(pred_e, 4),
                "dft_reference_ev_per_atom": round(true_e, 4),
                "abs_error_ev_per_atom": round(abs(pred_e - true_e), 4),
            })
            print(f"[bake] idx={idx} {formula(Z)} ({n} atoms): pred={pred_e:.3f} dft={true_e:.3f}", flush=True)

    out = {
        "model_variant": variant,
        "model_label": f"HydraGNN (PNA, trained by us on Alexandria DFT) — {variant} model",
        "val_corr": metrics.get("corr"),
        "val_mae": metrics.get("mae"),
        "val_r2": metrics.get("r2"),
        "predictions": predictions,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[bake] wrote {len(predictions)} real predictions to {out_path}", flush=True)


if __name__ == "__main__":
    main()
