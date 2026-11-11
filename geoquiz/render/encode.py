"""FFmpeg encoder for the Reels format: H.264, yuv420p, AAC, moov atom first."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from ..scene import Frame


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:                                    # the bundled binary, if imageio-ffmpeg is installed
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as e:
        raise RuntimeError("ffmpeg not found: install it or `pip install imageio-ffmpeg`") from e


class VideoWriter:
    """Pipe raw RGBA frames into ffmpeg. Use as a context manager.

    Audio is a silent AAC track unless ``audio`` names a file: Reels expect an
    audio stream, and music must be licensed and embedded by you.
    """

    def __init__(self, path: Path, frame: Frame, crf: int = 18, preset: str = "medium", audio: Path | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        audio_in = (["-stream_loop", "-1", "-i", str(audio)] if audio else       # loop short tracks
                    ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"])
        cmd = [
            ffmpeg_exe(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{frame.width}x{frame.height}", "-r", str(frame.fps), "-i", "-",
            *audio_in,
            "-map", "0:v", "-map", "1:a", "-shortest",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-profile:v", "high", "-level", "4.2", "-r", str(frame.fps),
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(self.path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.frames = 0

    def write(self, rgba: np.ndarray) -> None:
        self.proc.stdin.write(rgba.tobytes() if not rgba.flags.c_contiguous else memoryview(rgba))
        self.frames += 1

    def close(self) -> None:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace")
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err.strip()}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.close()
        else:
            self.proc.kill()
