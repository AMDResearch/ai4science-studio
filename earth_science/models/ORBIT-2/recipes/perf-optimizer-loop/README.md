# ORBIT-2 Iterative Systems Optimizer Loop (deferred)

**Status:** Not started. Complete the scaling sweep (1/2/4/8 nodes) and review [../perf-analysis/HANDOFF.md](../perf-analysis/HANDOFF.md) before porting the HydraGNN `perf-optimizer-loop` recipe.

Planned entry points:

- `examples/run_optimizer_loop.sh`
- `lever_catalog.yaml` (RCCL, batch_size, num_workers, FSDP layout, TILES)
- Agent prompts under `agents/`

See [material_science/models/HydraGNN/recipes/perf-optimizer-loop/](../../../material_science/models/HydraGNN/recipes/perf-optimizer-loop/) for the reference implementation.
