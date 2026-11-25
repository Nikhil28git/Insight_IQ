"""
run_pipeline.py  —  End-to-end Video RAG Pipeline (single command)
===================================================================

Architecture
------------
  video_input.mp4  (or YouTube URL)
      │
      ▼  Phase 1 ── extract_audio.py
  videos/<id>/audio/audio.wav
      │
      ▼  Phase 2 ── transcribe.py
  videos/<id>/transcript/transcript.json
      │
      ▼  Phase 3 ── chunk_transcript.py   (semantic boundary detection)
  videos/<id>/chunks/chunks.json
  videos/<id>/chunks/chunk_embeddings.npy  ← saved once, reused in Phase 4
      │
      ▼  Phase 4 ── embed_and_store.py   (zero re-encoding)
  chroma_db/vid_<id>/                      ← isolated collection per video
      │
      ▼  Phase 5 ── rag_query.py          (LangChain LCEL)
  stdout: timestamp-aware Gemini answer

Caching / skip logic
--------------------
  Each phase checks whether its output artefact already exists on disk.
  If it does the phase is skipped automatically — re-run is instant.
  Use --force to ignore the cache and reprocess everything from scratch.

Usage
-----
  # Process a video and enter interactive Q&A
  python run_pipeline.py --source lecture.mp4 --video_id lecture1

  # Process a YouTube video
  python run_pipeline.py --source https://youtu.be/XYZ --video_id yt_xyz

  # Ask a single question without interactive mode
  python run_pipeline.py --source lecture.mp4 --video_id lecture1 \\
      --query "What is RAG?"

  # Force reprocess even if outputs already exist
  python run_pipeline.py --source lecture.mp4 --video_id lecture1 --force

  # Skip ingestion and only query an already-processed video
  python run_pipeline.py --video_id lecture1 --query "Explain embeddings"
"""

import argparse
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
# PATH HELPERS  (mirrors per-phase helpers)
# ──────────────────────────────────────────────
def _paths(video_id: str) -> dict:
    """Return all expected artefact paths for a video_id."""
    base  = os.path.join("videos", video_id)
    audio = os.path.join(base, "audio")
    trans = os.path.join(base, "transcript")
    chunk = os.path.join(base, "chunks")
    return {
        "base":         base,
        "audio_wav":    os.path.join(audio, "audio.wav"),
        "transcript":   os.path.join(trans, "transcript.json"),
        "chunks_json":  os.path.join(chunk, "chunks.json"),
        "chunk_embs":   os.path.join(chunk, "chunk_embeddings.npy"),
        "collection":   f"vid_{video_id}",
    }


def _phase_done(label: str, artefact: str) -> bool:
    """Return True (and print a skip notice) when an artefact already exists."""
    if os.path.exists(artefact):
        print(f"  ⏩  [{label}] artefact found — skipping  ({artefact})")
        return True
    return False


# ──────────────────────────────────────────────
# STEP BANNER
# ──────────────────────────────────────────────
def _banner(n: int, title: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  STEP {n}/5 — {title}")
    print(f"{'═'*60}")


# ──────────────────────────────────────────────
# PIPELINE PHASES
# ──────────────────────────────────────────────
def run_phase1(source: str, video_id: str, p: dict, force: bool,
               cookies_from_browser: str = "") -> str:
    """Phase 1 — Audio extraction."""
    _banner(1, "Audio Extraction")

    if not force and _phase_done("Phase 1", p["audio_wav"]):
        return p["audio_wav"]

    from extract_audio import extract_audio
    wav = extract_audio(source=source, video_id=video_id,
                        cookies_from_browser=cookies_from_browser)
    return wav


def run_phase2(video_id: str, p: dict, force: bool,
               model_name: str = "base", hf_token: str | None = None) -> dict:
    """Phase 2 — WhisperX transcription."""
    _banner(2, "WhisperX Transcription")
    if not force and _phase_done("Phase 2", p["transcript"]):
        import json
        with open(p["transcript"], encoding="utf-8") as f:
            return json.load(f)

    from transcribe import transcribe
    return transcribe(
        wav_path=p["audio_wav"],
        model_name=model_name,
        hf_token=hf_token,
        video_id=video_id,
    )


def run_phase3(video_id: str, p: dict, force: bool,
               min_words: int = 80, max_words: int = 350,
               threshold_pct: int = 75) -> list[dict]:
    """Phase 3 — Semantic chunking."""
    _banner(3, "Semantic Chunking")
    if not force and _phase_done("Phase 3", p["chunks_json"]):
        import json
        with open(p["chunks_json"], encoding="utf-8") as f:
            return json.load(f).get("chunks", [])

    from chunk_transcript import chunk_transcript
    return chunk_transcript(
        min_words=min_words,
        max_words=max_words,
        threshold_pct=threshold_pct,
        video_id=video_id,
    )


def run_phase4(video_id: str, p: dict, force: bool) -> None:
    """Phase 4 — Embed + store in ChromaDB (zero re-encoding)."""
    _banner(4, "ChromaDB Indexing  (LangChain, zero re-encoding)")

    # Check if this video's collection already has docs
    if not force:
        try:
            import chromadb
            client     = chromadb.PersistentClient(path="chroma_db")
            collection = client.get_collection(p["collection"])
            if collection.count() > 0:
                print(f"  ⏩  [Phase 4] collection '{p['collection']}' already has "
                      f"{collection.count()} docs — skipping")
                return
        except Exception:
            pass  # collection doesn't exist yet → proceed

    # from embed_and_store import embed_and_store
    embed_and_store(video_id=video_id, reset=force)


def run_phase5(video_id: str, query: str | None, top_k: int = 4,
               interactive: bool = True) -> None:
    """Phase 5 — LangChain LCEL RAG query."""
    _banner(5, "RAG Query  (LangChain LCEL + Gemini)")

    from rag_query import rag_query, _display

    if interactive and not query:
        print(f"\n  Video : {video_id}  |  Collection : vid_{video_id}")
        print(f"  Type your question below.  Type 'exit' to quit.\n")
        while True:
            try:
                q = input("  ❓ Question: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n  Goodbye!\n")
                break
            if not q:
                continue
            if q.lower() in ("exit", "quit", "q"):
                print("  Goodbye!\n")
                break
            result = rag_query(q, top_k=top_k, video_id=video_id)
            _display(result)
    elif query:
        result = rag_query(query, top_k=top_k, video_id=video_id)
        _display(result)


# ──────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ──────────────────────────────────────────────
def run_pipeline(
    video_id:      str,
    source:        str        = "",
    query:         str | None = None,
    force:         bool       = False,
    model_name:    str        = "base",
    hf_token:      str | None = None,
    top_k:         int        = 4,
    min_words:     int        = 80,
    max_words:     int        = 350,
    threshold_pct: int        = 75,
    skip_query:    bool       = False,
    interactive:   bool       = True,
    cookies_from_browser: str = "",
) -> None:
    """
    Run the full pipeline for a given video.

    Parameters
    ----------
    video_id      : unique label for this video  (e.g. 'lecture1')
    source        : local file path OR YouTube URL.  Can be omitted if the
                    video was already processed in a previous run.
    query         : optional single question to ask after indexing.
    force         : if True, reprocess all phases even if outputs exist.
    model_name    : WhisperX model  (tiny | base | small | medium)
    hf_token      : HuggingFace token for speaker diarization (optional)
    top_k         : number of chunks to retrieve for RAG
    min_words     : minimum chunk size (words)
    max_words     : maximum chunk size (words)
    threshold_pct : semantic boundary sensitivity (higher = more chunks)
    skip_query    : if True, stop after indexing without entering query mode
    interactive   : if True, launch interactive Q&A after indexing
    """
    _check_venv()

    p = _paths(video_id)
    t0 = time.time()

    print(f"\n{'═'*60}")
    print(f"  🎬  Video RAG Pipeline")
    print(f"{'═'*60}")
    print(f"  Video ID : {video_id}")
    print(f"  Source   : {source or '(skip — already processed)'}")
    print(f"  Force    : {force}")
    print(f"{'═'*60}\n")

    # ── Phases 1–4 (ingestion) ──
    if source:
        run_phase1(source, video_id, p, force, cookies_from_browser=cookies_from_browser)
        run_phase2(video_id, p, force, model_name, hf_token)
        run_phase3(video_id, p, force, min_words, max_words, threshold_pct)
        run_phase4(video_id, p, force)
        # meta.json is saved inside extract_audio.py (Phase 1) ✅
    else:
        # Source not provided — assume already processed, go straight to query
        print("  ℹ️  No --source given; skipping ingestion phases.")
        print(f"      Querying existing collection: vid_{video_id}\n")
        # Fallback: save meta.json if source was passed later via --source flag
        # (already handled in extract_audio.py when source is present)

    elapsed = time.time() - t0
    print(f"\n✅  Ingestion complete in {elapsed:.1f}s")
    print(f"   All artefacts in : videos/{video_id}/")
    print(f"   ChromaDB         : chroma_db/  (collection: vid_{video_id})\n")

    # ── Phase 5 (query) ──
    if not skip_query:
        run_phase5(video_id, query, top_k, interactive=(query is None))


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end Video RAG Pipeline — single command to process + query a video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline + interactive Q&A
  python run_pipeline.py --source lecture.mp4 --video_id lecture1

  # YouTube video
  python run_pipeline.py --source https://youtu.be/XYZ --video_id yt1

  # Single query (no interactive mode)
  python run_pipeline.py --source lecture.mp4 --video_id lecture1 \\
      --query "What is RAG?"

  # Query a video that was already processed
  python run_pipeline.py --video_id lecture1 --query "Explain embeddings"

  # Force reprocess
  python run_pipeline.py --source lecture.mp4 --video_id lecture1 --force
        """,
    )

    parser.add_argument("--source",        default="",
                        help="Local video file path or YouTube URL")
    parser.add_argument("--video_id",      required=True,
                        help="Unique label for this video (e.g. 'lecture1')")
    parser.add_argument("--query",  "-q",  default=None,
                        help="Ask a single question (skips interactive mode)")
    parser.add_argument("--force",         action="store_true",
                        help="Reprocess all phases even if outputs exist")
    parser.add_argument("--model",         default="base",
                        choices=["tiny", "base", "small", "medium"],
                        help="WhisperX model size (default: base)")
    parser.add_argument("--hf_token",      default=None,
                        help="HuggingFace token for speaker diarization")
    parser.add_argument("--top_k",         type=int, default=4,
                        help="Chunks to retrieve for RAG (default: 4)")
    parser.add_argument("--min_words",     type=int, default=80,
                        help="Min words per chunk (default: 80)")
    parser.add_argument("--max_words",     type=int, default=350,
                        help="Max words per chunk (default: 350)")
    parser.add_argument("--threshold_pct", type=int, default=75,
                        help="Semantic boundary sensitivity 0-100 (default: 75)")
    parser.add_argument("--index_only",    action="store_true",
                        help="Only run ingestion phases, skip querying")
    parser.add_argument("--cookies_from_browser", default="",
                        choices=["", "chrome", "firefox", "edge", "brave"],
                        help="Browser to pull YouTube cookies from (fixes bot detection)")

    args = parser.parse_args()

    run_pipeline(
        video_id=args.video_id,
        source=args.source,
        query=args.query,
        force=args.force,
        model_name=args.model,
        hf_token=args.hf_token,
        top_k=args.top_k,
        min_words=args.min_words,
        max_words=args.max_words,
        threshold_pct=args.threshold_pct,
        skip_query=args.index_only,
        interactive=(args.query is None),
        cookies_from_browser=getattr(args, 'cookies_from_browser', ''),
    )
