# HydraGNN: Distributed Training on AMD Instinct MI355X

> **Ready-to-run scripts:** see [`../../examples/`](../../examples/) for `sbatch_train_amd.sh` and `run_train.sh`.

This recipe documents multi-node GFM (Graph Foundation Model) training using HydraGNN on AMD Instinct MI355X GPUs with Pensando/ionic interconnect.

---

## 1. Environment & Dependencies

No module loads or conda environments are needed. Everything runs inside an Apptainer container with a pre-built overlay:

| Component | Path |
|-----------|------|
| SIF image | `$AI4S_SHARED_DIR/images/pytorch_rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif` |
| Overlay   | `$AI4S_SHARED_DIR/models/HydraGNN/overlays/hydragnn-overlay.img` |

**Overlay contents:** PyTorch 2.10.0+rocm7.2.2, torch-geometric, torch-scatter/sparse/cluster (ROCm wheels), mpi4py, adios2, hydragnn, and runtime deps (tqdm, pyyaml, tensorboard, scikit-learn, scipy, h5py, ase).

**Hardware:** AMD Instinct MI355X (gfx950), 8 GPUs per node, 288 GB VRAM per GPU, Pensando ionic interconnect.

## 2. Build / Setup Instructions

The overlay is already built. The only setup step is cloning the HydraGNN source (done automatically by the sbatch script on first run):

```bash
# Manual clone (optional, script does this automatically):
git clone --depth=1 https://github.com/ORNL/HydraGNN.git \
    $AI4S_SHARED_DIR/models/HydraGNN/code/HydraGNN
```

The script also symlinks the staged ADIOS datasets into the expected location (`examples/multidataset_hpo_sc26/dataset/`).

### Staged datasets

| Dataset | Path | Size |
|---------|------|------|
| ANI1x   | `$AI4S_SHARED_DIR/models/HydraGNN/weights/ANI1x-v2.bp` | 30 GB |
| Alexandria | `$AI4S_SHARED_DIR/models/HydraGNN/weights/Alexandria-v2.bp` | 23 GB |

These are pre-processed ADIOS datasets from the [HydraGNN Predictive GFM 2024 model card](https://huggingface.co/mlupopa/HydraGNN_Predictive_GFM_2024).

## 3. Run Instructions

### Single node (8 GPUs)

```bash
export AI4S_SHARED_DIR=/path/to/shared   # set via /init-cluster
sbatch sbatch_train_amd.sh
```

### 2 nodes (16 GPUs)

```bash
sbatch --nodes=2 sbatch_train_amd.sh
```

### 4 nodes (32 GPUs)

```bash
sbatch --nodes=4 sbatch_train_amd.sh
```

### Custom parameters

```bash
export HG_BATCH_SIZE=200
export HG_NUM_EPOCH=4
export HG_PRECISION=bf16
sbatch --nodes=4 --time=04:00:00 sbatch_train_amd.sh
```

### Environment variable reference

| Variable | Default | Description |
|----------|---------|-------------|
| `AI4S_SHARED_DIR` | (required) | Base path for shared assets |
| `HG_SIF` | `$AI4S_SHARED_DIR/images/pytorch_rocm7.2.2_...sif` | Apptainer SIF |
| `HG_OVERLAY` | `$AI4S_SHARED_DIR/models/HydraGNN/overlays/hydragnn-overlay.img` | Overlay image |
| `HG_DATASETS` | `ANI1x,Alexandria` | Comma-separated dataset names |
| `HG_DATA_DIR` | `$AI4S_SHARED_DIR/models/HydraGNN/weights` | Dir containing `<name>-v2.bp` |
| `HG_BATCH_SIZE` | 128 (from config) | Per-rank batch size |
| `HG_NUM_EPOCH` | 1 (from config) | Training epochs |
| `HG_PRECISION` | `fp64` | `fp32`, `fp64`, or `bf16` |
| `HG_OUTPUT_DIR` | `$AI4S_SHARED_DIR/models/HydraGNN/outputs` | Output working directory |
| `HG_REPO_DIR` | `$AI4S_SHARED_DIR/models/HydraGNN/code/HydraGNN` | HydraGNN source clone |
| `HYDRAGNN_MAX_NUM_BATCH` | (unset = unlimited) | Cap batches per epoch (sanity: 50) |
| `HYDRAGNN_VALTEST` | `0` | Run validation/test loops (0=skip) |
| `SCRATCH_LOCAL` | `/scratch` | Node-local fast storage for MIOpen cache |

### How the launch works

```
sbatch → SLURM allocates N nodes
  └─ srun --mpi=pmix (N*8 ranks)
       └─ apptainer exec --rocm --overlay hydragnn-overlay.img:ro $SIF
            └─ bash rank_script.sh
                 └─ python -u gfm_mlip_all_mpnn.py --multi --multi_model_list=ANI1x,Alexandria
```

MPI is initialized by `hydragnn.utils.distributed.setup_ddp()`. Each rank opens the ADIOS datasets via `AdiosMultiDataset` with the MPI communicator. No DDStore in phase 1.

## 4. Config File

The upstream `gfm_mlip.json` (in `examples/multidataset_hpo_sc26/`) is used as-is. It is hardware-agnostic:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `mpnn_type` | MACE | Equivariant GNN architecture |
| `hidden_dim` | 128 | |
| `num_conv_layers` | 4 | |
| `force_weight` | 10.0 | Energy + forces training |
| `batch_size` | 128 | Overridable via `--batch_size` |
| `num_epoch` | 1 | Overridable via `--num_epoch` |
| `precision` | fp64 | Overridable via `--precision` |
| `learning_rate` | 0.001 | AdamW optimizer |
| `Checkpoint` | true | Writes to `logs/` in working dir |

CLI flags (`--batch_size`, `--num_epoch`, `--precision`) override these at runtime.

## 5. Expected Model Behavior

With `num_epoch=1` (validation/smoke-test):
- Training should complete without errors
- Loss values should be finite and decreasing within the epoch
- All ranks should report consistent loss values

For longer training runs (`num_epoch >= 10`):
- MAE loss on energy should decrease steadily
- Force prediction error should also decrease
- Early stopping (patience=10) will halt training if validation loss plateaus

### Validated sanity test (1 node / 8 GPUs, 50 batches)

Validated on MI355X (2026-05-19) with `HYDRAGNN_MAX_NUM_BATCH=50`, `HG_BATCH_SIZE=200`, `HG_PRECISION=fp64`, datasets `ANI1x,Alexandria`:

| Metric | Value |
|--------|-------|
| Total training time | 183 s (50 batches) |
| Steady-state iteration time | 2.9–3.0 s/batch |
| Peak GPU memory (allocated) | 7.5 GB |
| Peak GPU memory (reserved) | 9.0 GB |
| Data load time | 2.9 s |
| Model creation time | 0.75 s |
| RCCL errors | None |
| All ranks converged | Yes |

## 6. Performance & Scaling

Matched strong-scaling sweep on MI355X (2026-06-06): identical env on all node counts, steady-state train s/batch (mean of epochs 2–5), `HYDRAGNN_VALTEST=0`, RCCL high-priority enabled.

| Nodes | GPUs | Train s/batch | samples/s | Strong-scaling eff. |
|------:|-----:|--------------:|----------:|--------------------:|
| 1     | 8    | 2.69 s        | 74        | 1.00                |
| 2     | 16   | 2.73 s        | 73        | 0.98                |
| 4     | 32   | 2.71 s        | 74        | 0.99                |
| 8     | 64   | 2.85 s        | 70        | 0.94                |

Submit a matched sweep: `./examples/run_scaling_study.sh`  
Regenerate the table from SLURM logs: `python examples/collate_scaling_study.py --jobs <id1>,<id2>,... -o scaling_study`

### Multi-node network architecture

On Pensando/ionic fabrics, data NICs use `/31` point-to-point subnets that don't route between nodes — standard IB verbs cannot work for inter-node MPI. The architecture separates traffic into three paths:

| Path | Interface | Protocol | Purpose |
|------|-----------|----------|---------|
| MPI (data bcast, OOB) | management NIC | TCP | ADIOS metadata bcast, rank coordination |
| RCCL bootstrap | same mgmt NIC | TCP socket | RCCL topology exchange |
| RCCL data (GPU allreduce) | IB HCA devices (400G each) | ANP plugin (RoCEv2, GDRDMA) | GPU-to-GPU collectives |

**MPI configuration (in `sbatch_train_amd.sh`):**

```bash
--env OMPI_MCA_pml=ob1              # ob1 PML (UCX PML hangs on ionic fabrics)
--env OMPI_MCA_btl=tcp,self         # TCP BTL over management NIC
--env OMPI_MCA_btl_tcp_if_include=$NCCL_SOCKET_IFNAME  # Management interface
--env MPI4PY_RC_THREADS=false       # Avoid MPI_THREAD_MULTIPLE
```

**RCCL configuration:**

```bash
--bind $RCCL_ANP_PLUGIN:$RCCL_ANP_PLUGIN:ro   # ANP plugin (not exposed by --rocm)
--bind $LIBIONIC_PATH:$LIBIONIC_PATH:ro        # ionic userspace driver
--env NCCL_NET_PLUGIN=$RCCL_ANP_PLUGIN         # Select ANP transport
--env NCCL_IB_HCA=$NCCL_IB_HCA                  # All IB HCAs (from .cluster-config.yaml)
```

Without the ANP/libionic bind-mounts, RCCL falls back to socket transport over the management NIC (functional but significantly slower).

**Why ob1/tcp for MPI is correct:**
- MPI over TCP on the mgmt NIC is sufficient for ADIOS I/O (metadata bcast at startup, not on the training hot path)
- GPU gradient allreduce uses RCCL over ANP with GDRDMA — completely independent of MPI

## 7. Hyperparameter Recommendations

| Scale | Batch size | Learning rate | Precision | Notes |
|-------|-----------|---------------|-----------|-------|
| 1 node (smoke test) | 128 | 0.001 | fp64 | Config defaults |
| 1 node (training) | 200 | 0.001 | fp64 | Per upstream scaling test |
| 2-4 nodes | 200 | 0.001 | fp64 | Linear scaling rule may apply for LR |

## 8. Validation & Reproducibility

- Random seed is fixed at `random_state = 0` in `gfm_mlip_all_mpnn.py`
- Config is saved to `logs/<run_name>/` at start of training
- Checkpoints (if enabled) write to `logs/<run_name>/`
- Existing pretrained weights in `$AI4S_SHARED_DIR/models/HydraGNN/weights/` are never modified

## 9. Convergence Tracking

HydraGNN prints epoch-level loss metrics when `HYDRAGNN_VALTEST=1`. Use the bundled parser to extract structured convergence data:

```bash
# Run training with validation enabled:
export HYDRAGNN_VALTEST=1
export HG_NUM_EPOCH=10
export HYDRAGNN_MAX_NUM_BATCH=50   # cap batches per epoch for faster iteration
sbatch sbatch_train_amd.sh

# Parse the output:
python examples/parse_convergence.py --log hydragnn-train-<jobid>.out -o convergence.csv
python examples/parse_convergence.py --log hydragnn-train-<jobid>.out --plot convergence.png
```

### Output format (SLURM log parsing)

HydraGNN emits per-epoch lines:
```
0: Epoch: 01, Train Loss: 8.01219610, Val Loss: 7.77382998, Test Loss: 12.61618049
0: Tasks Train Loss: [1.351, 0.106, 0.791]
```

The parser extracts these into CSV columns: `epoch, batch, train_loss, val_loss, test_loss, tasks_train, tasks_val, wall_time_s, source`.

### Validated convergence (1 node / 8 GPUs, 5 epochs, 50 batches/epoch)

Validated on MI355X (2026-05-19) with `HYDRAGNN_VALTEST=1`, `HYDRAGNN_MAX_NUM_BATCH=50`, `HG_BATCH_SIZE=200`, `HG_PRECISION=fp64`:

| Epoch | Train Loss | Val Loss | Test Loss | Time/batch |
|-------|-----------|----------|-----------|------------|
| 0     | 8.2051    | 8.2576   | 13.2743   | 2.44 s     |
| 1     | 8.0122    | 7.7738   | 12.6162   | 2.44 s     |
| 2     | 7.3135    | 6.8208   | 11.1788   | 2.41 s     |
| 3     | 6.5180    | 6.2752   | 10.1997   | 2.43 s     |
| 4     | 6.0819    | 6.1667   | 10.1823   | 2.39 s     |

Total training time: 720 s (12 min). Loss reduced 25.9% over 5 epochs with clear downward trend.

### TensorBoard (alternative)

HydraGNN also writes TensorBoard events when `HYDRAGNN_VALTEST=1`. Event files appear in `$HG_OUTPUT_DIR/logs/<run_name>/`. The parser supports this mode:

```bash
python examples/parse_convergence.py --tbdir $AI4S_SHARED_DIR/models/HydraGNN/outputs/logs/hydragnn-train-<jobid>-N1/ -o convergence.csv
```

Note: TensorBoard scalars are only populated when validation runs (they remain empty with `HYDRAGNN_VALTEST=0`).

## Phase 2: DDStore (future)

For larger-scale runs, add DDStore for improved data distribution:

```bash
# Phase 2 (not yet implemented):
# Add --ddstore flag and set:
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_DDSTORE_METHOD=1
export HYDRAGNN_CUSTOM_DATALOADER=1
export HYDRAGNN_NUM_WORKERS=2
```

This requires verifying MPI DDStore functionality on MI355X and may need RCCL tuning for inter-node communication.
