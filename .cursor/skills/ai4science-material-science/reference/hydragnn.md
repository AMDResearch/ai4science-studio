# HydraGNN multi-node training and inference patterns

## HydraGNN multi-node training pattern

HydraGNN uses MPI-based distributed training with ADIOS datasets. Key patterns for AMD clusters:

- **Launch:** `srun --mpi=pmix apptainer exec --rocm --overlay <overlay>:ro $SIF bash <rank_script>`. The `--mpi=pmix` is required because `mpi4py` calls `MPI_Init` inside the container.
- **Non-DDStore path (phase 1):** Use `--multi --multi_model_list=<datasets>` without `--ddstore`. Each rank opens ADIOS files directly via `AdiosMultiDataset`. Simpler and sufficient for moderate scales (1-8 nodes).
- **DDStore path (phase 2):** Add `--ddstore` flag plus env vars `HYDRAGNN_AGGR_BACKEND=mpi`, `HYDRAGNN_DDSTORE_METHOD=1`, `HYDRAGNN_CUSTOM_DATALOADER=1`. Required for large-scale (32+ nodes) where per-rank ADIOS I/O becomes a bottleneck.
- **Dataset convention:** ADIOS datasets are expected at `./dataset/<name>-v2.bp` relative to the training script. Symlink from shared storage rather than copying.
- **Config file:** The upstream `gfm_mlip.json` is hardware-agnostic and works on any cluster. Use CLI flags (`--batch_size`, `--num_epoch`, `--precision`) to override parameters at runtime without modifying the file.
- **Env vars inside container:** Set `OMP_NUM_THREADS` to match `--cpus-per-task`, `MIOPEN_DISABLE_CACHE=1`, `MIOPEN_USER_DB_PATH=$SCRATCH_LOCAL/<jobid>/miopen` (node-local fast storage from `.cluster-config.yaml`), `HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1`.
- **Multi-node MPI transport (ob1/tcp):** On Pensando/ionic fabrics, data NICs use `/31` subnets that don't route between nodes — IB verbs cannot work for MPI. Use `OMPI_MCA_pml=ob1`, `OMPI_MCA_btl=tcp,self`, `OMPI_MCA_btl_tcp_if_include=$MGMT_NIC`, `MPI4PY_RC_THREADS=false`. RCCL uses ANP plugin (`librccl-anp.so`) over ionic native transport (RoCEv2/GDRDMA) for GPU allreduce, independent of MPI. The ANP plugin and `libionic.so.1` must be bind-mounted from the host into the container.
- **Convergence capture:** Run with `HYDRAGNN_VALTEST=1` to get epoch-level loss reporting. Parse with `examples/parse_convergence.py --log <slurm_output>`.
- **Strong-scaling sweep:** Submit matched 1/2/4/8-node runs via `examples/run_scaling_study.sh` (identical env, `HYDRAGNN_VALTEST=0`, 6 epochs). Collate steady-state throughput (epochs 2–5) with `examples/collate_scaling_study.py`. Results table and methodology in `recipes/train/README.md` §6. `sbatch_train_amd.sh` defaults `TORCH_NCCL_HIGH_PRIORITY=1` and `GPU_MAX_HW_QUEUES=2`.

## HydraGNN inference pattern

GFM 2024 checkpoints use an older model architecture (branch `Predictive_GFM_2024`) that is structurally incompatible with the training-pinned SHA. The inference script auto-clones the correct branch and uses `sys.path.insert(0, ...)` to load matching code. Key points:

- **Code version mismatch:** Never load old GFM checkpoints with new HydraGNN code — state_dict keys differ (e.g. `heads_NN.0.0.weight` vs `heads_NN.0.energy.0.weight`).
- **Config format:** Old configs use `model_type`; new code expects `mpnn_type`. Old `output_heads` is a flat dict; new expects list-of-dicts.
- **Inference repo:** Set `HG_INFER_REPO` or let `run_inference.sh` clone `Predictive_GFM_2024` to `$HG_OUTPUT_DIR/HydraGNN-infer` on first run.

