"""
Phase 1 — Audio Extraction
Supports: local video files + YouTube URLs
Uses:     imageio-ffmpeg (bundled ffmpeg, no system install needed)
          yt-dlp (YouTube download)
Output:   audio_output/audio.wav  (mono, 16 kHz)
"""

import argparse
import os
import subprocess
import sys
import tempfile


# ──────────────────────────────────────────────
# GUARD: must run inside the rag_env venv
# ──────────────────────────────────────────────
def _check_venv() -> None:
    if sys.prefix == sys.base_prefix:
        print(
            "\n⛔  Virtual environment is NOT active!\n"
            "   Activate it first, then re-run:\n\n"
            "       rag_env\\Scripts\\activate        (Windows)\n"
            "       source rag_env/bin/activate      (Mac/Linux)\n"
        )
        sys.exit(1)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def _is_youtube_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def _get_ffmpeg() -> str:
    """Return the bundled ffmpeg binary path from imageio-ffmpeg."""
    try:
        import imageio_ffmpeg  # type: ignore  # noqa: PLC0415
        path = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"[INFO] ffmpeg binary: {path}")
        return path
    except ImportError:
        print(
            "\n⛔  imageio-ffmpeg is not installed in the active venv.\n"
            "   Run:  pip install -r requirements_phase1.txt\n"
        )
        sys.exit(1)


def _ensure_output_dir(directory: str) -> None:
    try:
        os.makedirs(directory, exist_ok=True)
        print(f"[INFO] Output directory ready: {directory}")
    except OSError as exc:
        print(f"\n⛔  Cannot create output directory '{directory}': {exc}\n")
        sys.exit(1)


def _download_youtube(url: str, tmp_dir: str, cookies_from_browser: str = "") -> str:
    """Download a YouTube video to a temp file and return the file path."""
    print(f"[INFO] Detected YouTube URL — downloading with yt-dlp …")
    try:
        import yt_dlp  # type: ignore  # noqa: PLC0415
    except ImportError:
        print(
            "\n⛔  yt-dlp is not installed in the active venv.\n"
            "   Run:  pip install -r requirements_phase1.txt\n"
        )
        sys.exit(1)

    # Point yt-dlp at the bundled ffmpeg so it can merge video+audio streams
    ffmpeg_dir = os.path.dirname(_get_ffmpeg())

    tmp_path = os.path.join(tmp_dir, "yt_video.%(ext)s")
    ydl_opts = {
        # Prefer a single-file format first; fall back to merge only if needed
        "format": "best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "outtmpl": tmp_path,
        "ffmpeg_location": ffmpeg_dir,   # ← tell yt-dlp where bundled ffmpeg lives
        "quiet": False,
        "no_warnings": False,
    }
    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)

    try:
# Recommended fix in _download_youtube():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
         info = ydl.extract_info(url, download=True)
         actual_path = ydl.prepare_filename(info)  # ← correct, handles all extensions
         print(f"[INFO] YouTube video downloaded to: {actual_path}")
         return actual_path
    except Exception as exc:
        print(f"\n⛔  yt-dlp failed to download '{url}':\n   {exc}\n")
        sys.exit(1)


def _run_ffmpeg(ffmpeg_path: str, input_path: str, output_path: str) -> None:
    """Run ffmpeg to extract mono 16 kHz WAV audio."""
    print(f"[INFO] Extracting audio …")
    print(f"       Input : {input_path}")
    print(f"       Output: {output_path}")

    cmd = [
        ffmpeg_path,
        "-i", input_path,
        "-vn",          # drop video stream
        "-ac", "1",     # mono
        "-ar", "16000", # 16 kHz sample rate
        "-acodec", "pcm_s16le",  # standard PCM WAV
        "-y",           # overwrite without asking
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("\n⛔  ffmpeg failed. Full stderr output:\n")
        print(result.stderr)
        sys.exit(1)

    print(f"[INFO] ffmpeg completed successfully.")


# ──────────────────────────────────────────────
# MAIN PUBLIC FUNCTION
# ──────────────────────────────────────────────
def _video_audio_dir(video_id: str) -> str:
    """Canonical audio dir for a video_id  →  videos/<id>/audio/"""
    return os.path.join("videos", video_id, "audio")


def extract_audio(
    source:     str,
    output_dir: str = "",
    video_id:   str = "",
    cookies_from_browser: str = "",
) -> str:
    """
    Extract audio from a local video file or YouTube URL.

    Parameters
    ----------
    source     : local file path  OR  YouTube URL
    video_id   : unique identifier for this video (e.g. 'lecture1').
                 Output → videos/<video_id>/audio/audio.wav
                 Preferred — prevents old runs being overwritten.
    output_dir : legacy override; ignored when video_id is given.

    Returns
    -------
    Absolute path to the saved audio.wav file.
    """
    _check_venv()

    if video_id:
        output_dir = _video_audio_dir(video_id)
    elif not output_dir:
        output_dir = "audio_output"

    ffmpeg_path = _get_ffmpeg()
    output_path = os.path.join(output_dir, "audio.wav")
    _ensure_output_dir(output_dir)

    if _is_youtube_url(source):
        # ── YouTube path ──────────────────────────────
        with tempfile.TemporaryDirectory() as tmp_dir:
            print(f"[INFO] Temporary directory: {tmp_dir}")
            video_path = _download_youtube(source, tmp_dir, cookies_from_browser=cookies_from_browser)

            if not os.path.exists(video_path):
                print(f"\n⛔  Downloaded file not found at: {video_path}\n")
                sys.exit(1)

            _run_ffmpeg(ffmpeg_path, video_path, output_path)
        # tmp_dir and its contents are auto-deleted here
        print("[INFO] Temporary video file cleaned up.")

    else:
        # ── Local file path ───────────────────────────
        if not os.path.exists(source):
            print(f"\n⛔  Source file not found: '{source}'\n"
                  f"   Check the path and try again.\n")
            sys.exit(1)

        print(f"[INFO] Local file detected: {source}")
        _run_ffmpeg(ffmpeg_path, source, output_path)

    abs_output = os.path.abspath(output_path)
    print(f"\n✅  Audio saved to: {abs_output}\n")

    # Save meta.json alongside audio — used by Streamlit for clickable timestamps
    if video_id:
        import json as _json
        meta_path = os.path.join(os.path.dirname(output_dir), "meta.json")
        if not os.path.exists(meta_path):
            with open(meta_path, "w", encoding="utf-8") as _mf:
                _json.dump({"source": source, "video_id": video_id}, _mf)
            print(f"[INFO] meta.json saved → {meta_path}")

    return abs_output


# ──────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1 — Extract mono 16 kHz WAV audio from video or YouTube URL"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to a local video file  OR  a YouTube URL",
    )
    parser.add_argument(
        "--video_id",
        default="",
        help="Unique video id — output → videos/<id>/audio/  (recommended)",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Legacy: explicit output folder (use --video_id instead)",
    )
    args = parser.parse_args()

    extract_audio(source=args.source, output_dir=args.output_dir, video_id=args.video_id)
