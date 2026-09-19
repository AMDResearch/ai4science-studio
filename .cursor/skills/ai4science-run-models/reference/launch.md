# Launch details (overlay build, weight sources)

## Overlay build: always use $TMPDIR, not NFS

When directing users to build an Apptainer ext3 overlay, always point them to the `build_overlay_amd.sh` script in the model's `examples/` directory. That script already handles the correct pattern: build on `$TMPDIR` (node-local fast disk), then copy the finished `.img` file to NFS.

**Do not suggest building the overlay directly on NFS paths** (`/shared/`, `/scratch/`, etc.). An ext3 image served via FUSE over NFS incurs massive per-file overhead when writing thousands of Python package files — both `cp -r` and `tar | tar` are equally slow because the bottleneck is the FUSE layer, not the copy tool. The fix is to create and populate the image on local disk and copy the completed file to NFS as one large sequential write.

## Model-specific notes

### Models with auto-download weights
StormCast, Walrus, GP-MoLFormer — weights download automatically from HF on first run. No manual download step.

### Models requiring manual weight download
ORBIT-2, HydraGNN, MatterGen — user must download from HF Hub first. SemlaFlow — Google Drive. REINVENT4 — bundled with upstream.

### Models trained from scratch
MATEY, SwinUNETR — no pretrained checkpoints distributed; training recipe is the primary entry point.
