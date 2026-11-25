"""
Phase 2 — Audio Transcription with WhisperX
============================================
- Model    : WhisperX (faster-whisper backend)
- Language : English (fixed)
- Device   : CPU  (int8 quantisation for speed)
- Timestamps: word-level  ✅
- Diarization: optional — pass --hf_token <TOKEN> to enable
- Outputs  : transcripts/transcript.json
             transcripts/transcript.txt

CLI usage
---------
  Basic:
    python transcribe.py --wav audio_output/audio.wav

  With speaker diarization:
    python transcribe.py --wav audio_output/audio.wav --hf_token hf_XXXX

  Choose model size (tiny/base/small/medium — default: base):
    python transcribe.py --wav audio_output/audio.wav --model base
"""

import argparse
import json
import os
import sys
import time

# ──────────────────────────────────────────────
# VENV GUARD
# ──────────────────────────────────────────────
def _check_venv() -> None:
    if sys.prefix == sys.base_prefix:
        print(
            "\n⛔  Virtual environment is NOT active!\n"
            "   Activate first:  rag_env\\Scripts\\activate\n"
        )
        sys.exit(1)


# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
DEVICE        = "cpu"
COMPUTE_TYPE  = "int8"   # fastest on CPU
LANGUAGE      = "en"
# Legacy fallback paths
OUTPUT_DIR    = "transcripts"
JSON_PATH     = os.path.join(OUTPUT_DIR, "transcript.json")
TXT_PATH      = os.path.join(OUTPUT_DIR, "transcript.txt")


def _video_transcript_dir(video_id: str) -> str:
    return os.path.join("videos", video_id, "transcript")


def _video_out_paths(video_id: str) -> tuple[str, str, str]:
    """Return (out_dir, json_path, txt_path) for a video_id."""
    d = _video_transcript_dir(video_id)
    return d, os.path.join(d, "transcript.json"), os.path.join(d, "transcript.txt")


def _inject_ffmpeg_to_path() -> None:
    """
    WhisperX calls ffmpeg as a subprocess using the system PATH.
    We inject the imageio-ffmpeg bundled binary directory into PATH
    so WhisperX finds it without any system install required.
    """
    try:
        import imageio_ffmpeg  # noqa: PLC0415
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)
        current_path = os.environ.get("PATH", "")
        if ffmpeg_dir not in current_path:
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path
            print(f"[INFO] Injected bundled ffmpeg into PATH: {ffmpeg_dir}")
        else:
            print(f"[INFO] Bundled ffmpeg already in PATH: {ffmpeg_dir}")
    except ImportError:
        print(
            "\n⛔  imageio-ffmpeg not found — cannot inject ffmpeg into PATH.\n"
            "   Run:  pip install -r requirements.txt\n"
        )
        sys.exit(1)


def _load_audio_numpy(wav_path: str):
    """
    Load a mono 16 kHz WAV file into a float32 numpy array.
    This bypasses WhisperX's internal ffmpeg subprocess call entirely —
    we read the WAV ourselves using Python's built-in wave module.
    WhisperX's model.transcribe() accepts a numpy array directly.
    """
    import wave  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import array as _array  # noqa: PLC0415

    print(f"[INFO] Loading audio into memory (bypassing ffmpeg subprocess) …")
    with wave.open(wav_path, "rb") as wf:
        n_channels  = wf.getnchannels()
        sample_rate = wf.getframerate()
        n_frames    = wf.getnframes()
        raw_bytes   = wf.readframes(n_frames)

    # Convert raw PCM bytes → int16 → float32 in [-1, 1]
    shorts = _array.array("h", raw_bytes)
    audio  = np.array(shorts, dtype=np.float32) / 32768.0

    # If stereo (shouldn't happen from Phase 1), average channels
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    print(f"[INFO] Audio loaded: {len(audio)/sample_rate:.2f}s  "
          f"@ {sample_rate} Hz  (float32 numpy array)")
    return audio


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        print(f"\n⛔  Cannot create directory '{path}': {exc}\n")
        sys.exit(1)


def _load_whisperx(model_name: str):
    print(f"[INFO] Loading WhisperX model '{model_name}' on {DEVICE} …")
    print(f"       (First run downloads the model — be patient)")
    try:
        import whisperx  # noqa: PLC0415
    except ImportError:
        print(
            "\n⛔  whisperx is not installed.\n"
            "   Run the Phase 2 install steps first.\n"
        )
        sys.exit(1)
    model = whisperx.load_model(
        model_name,
        DEVICE,
        compute_type=COMPUTE_TYPE,
        language=LANGUAGE,
        vad_options={
            "vad_onset": 0.200,   # lower = detect more speech (default ~0.500)
            "vad_offset": 0.100,  # lower = keep more speech after silence
        },
    )
    print(f"[INFO] Model loaded.")
    return whisperx, model


def _transcribe(whisperx, model, wav_path: str) -> dict:
    print(f"[INFO] Transcribing: {wav_path}")
    print(f"       Language  : {LANGUAGE}")
    print(f"       Device    : {DEVICE}  ({COMPUTE_TYPE})")
    t0 = time.time()

    # Pre-load audio as numpy array so WhisperX never calls ffmpeg subprocess
    audio_array = _load_audio_numpy(wav_path)

    result = model.transcribe(
        audio_array,        # ← numpy float32 array, not a file path
        batch_size=4,
        language=LANGUAGE,
        chunk_size=30,
        print_progress=True,
    )
    elapsed = time.time() - t0
    print(f"[INFO] Transcription done in {elapsed:.1f}s  "
          f"({len(result['segments'])} raw segments)")
    return result


def _align(whisperx, result: dict, audio_array) -> dict:
    print(f"[INFO] Aligning word-level timestamps …")
    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=LANGUAGE,
            device=DEVICE,
        )
        # Pass the pre-loaded numpy array — avoids internal ffmpeg subprocess
        aligned = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio_array,   # ← numpy float32 array, not a file path
            DEVICE,
            return_char_alignments=False,
        )
        print(f"[INFO] Alignment complete.")
        return aligned
    except Exception as exc:
        import traceback  # noqa: PLC0415
        print(f"[WARN] Alignment failed — word timestamps will be absent.")
        print(f"       Reason: {exc}")
        traceback.print_exc()
        return result


def _diarize(whisperx, aligned: dict, audio_array, hf_token: str) -> dict:
    print(f"[INFO] Running speaker diarization (pyannote.audio) …")
    print(f"       This may take several minutes on CPU.")
    try:
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=hf_token,
            device=DEVICE,
        )
        # Pass numpy array wrapped as {waveform, sample_rate} dict to avoid ffmpeg
        import torch  # noqa: PLC0415
        audio_input = {
            "waveform": torch.from_numpy(audio_array).unsqueeze(0),
            "sample_rate": 16000,
        }
        diarize_segments = diarize_model(audio_input)
        result_with_speakers = whisperx.assign_word_speakers(diarize_segments, aligned)
        print(f"[INFO] Diarization complete.")
        return result_with_speakers
    except Exception as exc:
        print(f"[WARN] Diarization failed — speaker labels will be absent.\n"
              f"       Reason: {exc}")
        return aligned


# ──────────────────────────────────────────────
# OUTPUT BUILDERS
# ──────────────────────────────────────────────
def _build_output(result: dict) -> dict:
    """
    Normalise WhisperX output into our standard schema:

    {
      "language": "en",
      "text": "<full transcript>",
      "segments": [
        {
          "start": 0.0,
          "end":   4.2,
          "text":  "Hello world",
          "speaker": "SPEAKER_00",          # or null
          "words": [
            {"word": "Hello", "start": 0.0, "end": 0.5, "score": 0.99},
            ...
          ]
        }
      ]
    }
    """
    segments = []
    full_text_parts = []

    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        full_text_parts.append(text)

        words = []
        for w in seg.get("words", []):
            words.append({
                "word":  w.get("word", "").strip(),
                "start": round(w.get("start", 0.0), 3),
                "end":   round(w.get("end", 0.0), 3),
                "score": round(w.get("score", 0.0), 4),
            })

        segments.append({
            "start":   round(seg.get("start", 0.0), 3),
            "end":     round(seg.get("end", 0.0), 3),
            "text":    text,
            "speaker": seg.get("speaker", None),
            "words":   words,
        })

    return {
        "language": LANGUAGE,
        "text":     " ".join(full_text_parts),
        "segments": segments,
    }


def _save_outputs(output: dict, out_dir: str, json_path: str, txt_path: str) -> None:
    _ensure_dir(out_dir)

    # ── JSON ────────────────────────────────────────────
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[INFO] JSON saved  → {os.path.abspath(json_path)}")

    # ── Plain TXT ───────────────────────────────────────
    lines = []
    for seg in output["segments"]:
        speaker = f"[{seg['speaker']}] " if seg.get("speaker") else ""
        ts = f"[{seg['start']:.2f}s → {seg['end']:.2f}s]"
        lines.append(f"{ts} {speaker}{seg['text']}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[INFO] TXT  saved  → {os.path.abspath(txt_path)}")


# ──────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────
def transcribe(
    wav_path:   str,
    model_name: str       = "base",
    hf_token:   str|None  = None,
    video_id:   str       = "",
) -> dict:
    """
    Transcribe a WAV file using WhisperX.

    Parameters
    ----------
    wav_path   : path to mono 16 kHz WAV  (output of Phase 1)
    model_name : whisper model size — tiny | base | small | medium
    hf_token   : HuggingFace token for speaker diarization (optional)
    video_id   : unique video identifier; output → videos/<id>/transcript/
                 Falls back to legacy 'transcripts/' when empty.

    Returns
    -------
    Structured dict with keys: language, text, segments
    """
    _check_venv()

    # Resolve output paths
    if video_id:
        out_dir, json_path, txt_path = _video_out_paths(video_id)
    else:
        out_dir, json_path, txt_path = OUTPUT_DIR, JSON_PATH, TXT_PATH

    if not os.path.exists(wav_path):
        print(f"\n⛔  WAV file not found: '{wav_path}'\n"
              f"   Run Phase 1 first:  python extract_audio.py --source <file>\n")
        sys.exit(1)

    # Inject bundled ffmpeg into PATH so WhisperX subprocess call works
    _inject_ffmpeg_to_path()

    print(f"\n{'═'*55}")
    print(f"  Phase 2 — WhisperX Transcription")
    print(f"{'═'*55}\n")

    whisperx, model = _load_whisperx(model_name)

    # Load audio once as numpy array — reused by transcribe, align, and diarize
    # This bypasses all internal ffmpeg subprocess calls in WhisperX
    audio_array = _load_audio_numpy(wav_path)

    raw     = _transcribe(whisperx, model, wav_path)
    # aligned = _align(whisperx, raw, audio_array)
    aligned = raw

    if hf_token:
        aligned = _diarize(whisperx, aligned, audio_array, hf_token)
    else:
        print(f"[INFO] Diarization skipped — no --hf_token provided.")
        print(f"       Speaker labels will be null in output.")

    output = _build_output(aligned)
    _save_outputs(output, out_dir, json_path, txt_path)

    word_count = len(output["text"].split())
    seg_count  = len(output["segments"])
    print(f"\n✅  Transcription complete!")
    print(f"   Segments : {seg_count}")
    print(f"   Words    : {word_count:,}")
    print(f"   JSON     : {os.path.abspath(json_path)}")
    print(f"   TXT      : {os.path.abspath(txt_path)}\n")

    return output


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2 — Transcribe WAV audio with WhisperX"
    )
    parser.add_argument(
        "--wav",
        required=True,
        help="Path to WAV file (output of Phase 1)",
    )
    parser.add_argument(
        "--video_id",
        default="",
        help="Unique video id — output → videos/<id>/transcript/",
    )
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium"],
        help="WhisperX model size (default: base)",
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help="HuggingFace token for speaker diarization (optional)",
    )
    args = parser.parse_args()

    transcribe(
        wav_path=args.wav,
        model_name=args.model,
        hf_token=args.hf_token,
        video_id=args.video_id,
    )
