"""AMD pAIr™ — workflow video export.

POST /api/video/convert
  Multipart form: file=<webm blob>
  Converts WebM to MPEG-2 via ffmpeg and returns { download_url }.

GET /api/video/download/{vid}
  Streams the .mpg back to the browser as an attachment.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

router = APIRouter()

RUNS_DIR = Path(__file__).parent / "runs"
RUNS_DIR.mkdir(exist_ok=True)


def _find_ffmpeg() -> str:
    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        return sys_ff
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    raise RuntimeError(
        "ffmpeg not found — install with: sudo apt-get install ffmpeg  "
        "or: pip install imageio-ffmpeg"
    )


@router.post("/video/convert")
async def convert_video(file: UploadFile = File(...)):
    vid = f"video_{int(time.time() * 1000)}"
    work_dir = RUNS_DIR / vid
    work_dir.mkdir(parents=True)

    try:
        ffmpeg = _find_ffmpeg()
    except RuntimeError as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(503, str(e))

    webm_path = work_dir / "recording.webm"
    content = await file.read()
    webm_path.write_bytes(content)

    out_path = work_dir / "workflow.mpg"
    cmd = [
        ffmpeg, "-y",
        "-i", str(webm_path),
        "-vcodec", "mpeg2video",
        "-acodec", "mp2",
        "-qscale:v", "4",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out_path.exists():
        detail = result.stderr[-2000:] if result.stderr else "unknown ffmpeg error"
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(500, f"ffmpeg failed: {detail}")

    return {
        "path": str(out_path),
        "download_url": f"/api/video/download/{vid}",
    }


@router.get("/video/download/{vid}")
def download_video(vid: str):
    if ".." in vid or "/" in vid:
        raise HTTPException(400, "invalid video id")
    mpg = RUNS_DIR / vid / "workflow.mpg"
    if not mpg.exists():
        raise HTTPException(404, "video not found")
    return FileResponse(
        path=str(mpg),
        media_type="video/mpeg",
        filename="pair_workflow.mpg",
        headers={"Content-Disposition": 'attachment; filename="pair_workflow.mpg"'},
    )
