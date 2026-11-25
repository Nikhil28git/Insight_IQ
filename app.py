"""
Video RAG — Streamlit Client Presentation App
=============================================
Run:
    streamlit run app.py
"""

import os
import sys
import re
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from ui import components as C

load_dotenv(Path(__file__).with_name(".env"), override=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InsightIQ",
    page_icon=":material/movie:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS (single stylesheet: styles/theme.css) ─────────────────────────────────
C.load_css()


# ── Persistent chat history (survives app restarts) ───────────────────────────
HISTORY_FILE = Path(__file__).with_name("chat_history.json")
HISTORY_LIMIT = 10


def _load_history() -> list[dict]:
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data[-HISTORY_LIMIT:] if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _save_history() -> None:
    tmp = HISTORY_FILE.with_suffix(".json.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=2)
        os.replace(tmp, HISTORY_FILE)
    except OSError as e:
        st.warning(f"Could not save chat history: {e}", icon=":material/warning:")


def _add_to_history(entry: dict) -> None:
    """Append a query (skipping an exact repeat of the last one), keep the newest N, persist."""
    hist = st.session_state.chat_history
    key = lambda e: (e.get("question") or e.get("query"), e.get("source") or e.get("video_id"))
    if hist and key(hist[-1]) == key(entry):
        return
    st.session_state.chat_history = (hist + [entry])[-HISTORY_LIMIT:]
    _save_history()


# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history"    not in st.session_state: st.session_state.chat_history    = _load_history()
if "processed_videos" not in st.session_state: st.session_state.processed_videos = {}
if "active_video"    not in st.session_state: st.session_state.active_video    = ""
if "processed_pdfs"  not in st.session_state: st.session_state.processed_pdfs  = {}
if "active_pdf"      not in st.session_state: st.session_state.active_pdf      = ""
if "pipeline_log"    not in st.session_state: st.session_state.pipeline_log    = []
if "pipeline_done"   not in st.session_state: st.session_state.pipeline_done   = False
if "source_mode"     not in st.session_state: st.session_state.source_mode     = "Video"
if "_last_applied_suggestion" not in st.session_state: st.session_state["_last_applied_suggestion"] = ""


# ── Helpers ───────────────────────────────────────────────────────────────────
def _fmt_time(seconds: float) -> str:
    """Convert seconds to mm:ss, e.g. 1348 → '22:28'."""
    seconds = max(0, int(round(seconds)))
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"


def _list_processed_videos() -> list[str]:
    videos_dir = Path("videos")
    if not videos_dir.exists():
        return []
    result = []
    for vid_dir in videos_dir.iterdir():
        if vid_dir.is_dir():
            chunks_json = vid_dir / "chunks" / "chunks.json"
            if chunks_json.exists():
                result.append(vid_dir.name)
    return sorted(result)


def _get_chunk_count(video_id: str) -> int:
    p = Path("videos") / video_id / "chunks" / "chunks.json"
    if not p.exists():
        return 0
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("total_chunks", len(data.get("chunks", [])))
    except Exception:
        return 0


def _get_video_duration(video_id: str) -> str:
    p = Path("videos") / video_id / "transcript" / "transcript.json"
    if not p.exists():
        return "N/A"
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        segs = data.get("segments", [])
        if segs:
            end = segs[-1].get("end", 0)
            m, s = divmod(int(end), 60)
            h, m = divmod(m, 60)
            if h:
                return f"{h}h {m}m {s}s"
            return f"{m}m {s}s"
    except Exception:
        pass
    return "N/A"


def _normalise_timestamp(ts: str) -> str:
    """Convert a raw-seconds timestamp string to mm:ss → mm:ss.
    Handles both '424 → 713' (raw seconds) and '04:24 → 07:13' (already formatted).
    """
    # Match two numbers separated by → (with or without colons inside)
    parts = re.split(r"\s*→\s*|\s*->\s*", ts)
    result = []
    for p in parts:
        p = p.strip()
        if re.fullmatch(r"\d+:\d{2}", p):
            result.append(p)          # already mm:ss
        elif re.fullmatch(r"\d+", p):
            result.append(_fmt_time(float(p)))   # raw seconds → mm:ss
        else:
            result.append(p)          # unknown format — keep as-is
    return " → ".join(result) if result else ts


# ── PDF Helpers ──────────────────────────────────────────────────────────────
def _list_processed_pdfs() -> list[str]:
    """List all processed PDFs from pdfs/ directory."""
    pdfs_dir = Path("pdfs")
    if not pdfs_dir.exists():
        return []
    result = []
    for pdf_dir in pdfs_dir.iterdir():
        if pdf_dir.is_dir():
            meta_json = pdf_dir / "meta.json"
            if meta_json.exists():
                result.append(pdf_dir.name)
    return sorted(result)


def _get_pdf_page_count(pdf_id: str) -> int:
    """Get total pages from PDF metadata."""
    p = Path("pdfs") / pdf_id / "meta.json"
    if not p.exists():
        return 0
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("total_pages", 0)
    except Exception:
        return 0


def _get_pdf_chunk_count(pdf_id: str) -> int:
    """Get chunk count for a PDF."""
    p = Path("pdfs") / pdf_id / "meta.json"
    if not p.exists():
        return 0
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("chunks_count", 0)
    except Exception:
        return 0


def process_pdf_upload(uploaded_file, pdf_id: str) -> dict | None:
    """Process uploaded PDF: save, chunk, embed, and store in ChromaDB."""
    try:
        # Step 1: Save uploaded PDF to temp location
        temp_pdf_path = f"temp_{pdf_id}.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.info(f"PDF saved: {uploaded_file.name}", icon=":material/upload_file:")
        
        # Step 2: Run PDF processing pipeline
        st.info("Processing PDF (chunking, embedding)...", icon=":material/hourglass_top:")
        
        from process_pdf import process_pdf_end_to_end
        from embed_and_store import embed_and_store
        
        result = process_pdf_end_to_end(
            pdf_path=temp_pdf_path,
            pdf_id=pdf_id,
            base_dir="pdfs"
        )
        
        # Step 3: Store chunks in ChromaDB
        # if result['success']:
        #     st.info("🔗 Storing in ChromaDB...")
        #     try:
        #         embed_and_store(
        #             pdf_id=pdf_id,
        #             source_type="pdf",
        #             reset=False
        #         )
        #         st.success("✅ Stored in ChromaDB")
        #     except Exception as e:
        #         st.error(f"❌ ChromaDB storage failed: {str(e)}")
        #         return None

        if result['success']:
            st.success("Stored in ChromaDB", icon=":material/database:")
            # Show extracted images (if any) from the PDF output dir
            images_dir = os.path.join("pdfs", pdf_id, "images")
            if os.path.exists(images_dir):
                try:
                    imgs = [str(p) for p in Path(images_dir).iterdir() if p.is_file()]
                    if imgs:
                        st.info("Extracted images:")
                        cols = st.columns(min(4, len(imgs)))
                        for i, img_path in enumerate(imgs):
                            try:
                                with open(img_path, "rb") as f:
                                    cols[i % len(cols)].image(f.read(), width=200, caption=os.path.basename(img_path))
                            except Exception:
                                continue
                except Exception:
                    pass
            # Show extracted tables from chunks.json (if any)
            chunks_file = os.path.join("pdfs", pdf_id, "chunks", "chunks.json")
            if os.path.exists(chunks_file):
                try:
                    with open(chunks_file, encoding="utf-8") as f:
                        chunks_data = json.load(f)

                    # chunks.json is expected to be a list of chunk dicts
                    table_chunks = [c for c in chunks_data if c.get("chunk_type") == "table"]
                    if table_chunks:
                        st.info("Extracted tables:")
                        for t in table_chunks:
                            try:
                                hdr = f"**{t.get('chunk_id')} — Page {t.get('start_page', '?')}**"
                                st.markdown(hdr)
                                md = t.get("text", "")
                                # Render markdown table; if large, Streamlit will handle scrolling
                                st.markdown(md)
                            except Exception:
                                continue
                except Exception:
                    pass
        
        # Clean up temp file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        
        if result['success']:
            # Update session state
            st.session_state.active_pdf = pdf_id
            st.session_state.pipeline_done = True
            
            st.success(f"""
            **PDF processing complete!**
            
            **Stats:**
            - Chunks created: {result['chunks_stored']}
            - Total pages: {result['metadata'].get('total_pages', 'N/A')}
            - Estimated words: {result['metadata'].get('estimated_words', 'N/A')}
            - Model: BAAI/bge-small-en-v1.5
            
            Ready to query!
            """, icon=":material/check_circle:")
            return result
        else:
            st.error(f"Processing failed: {result.get('error', 'Unknown error')}", icon=":material/error:")

            return None
            
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}", icon=":material/error:")
        return None


# Pre-import rag_query modules at startup for caching
def _parse_response_to_cards(response_text: str) -> list[dict]:
    """Parse the structured Gemini response into card dicts."""
    cards = []
    blocks = [b.strip() for b in response_text.split("---") if b.strip()]
    for block in blocks:
        card = {"timestamp": "", "topic": "", "summary": ""}
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("📍"):
                raw_ts = line.replace("📍", "").replace("Timestamp", "").replace(":", "", 1).strip()
                card["timestamp"] = _normalise_timestamp(raw_ts)
            elif line.startswith("🏷"):
                card["topic"] = line.replace("🏷️", "").replace("🏷", "").replace("Topic", "").replace(":", "", 1).strip()
            elif line.startswith("📝"):
                card["summary"] = line.replace("📝", "").replace("Summary", "").replace(":", "", 1).strip()
        if card["topic"] or card["summary"]:
            cards.append(card)
    return cards


# Pre-import rag_query modules at startup for caching
sys.path.insert(0, str(Path(__file__).parent))
try:
    from rag_query import rag_query as _rag_query_fn
    _RAG_IMPORT_OK = True
    _RAG_IMPORT_ERR = ""
except Exception as _e:
    _RAG_IMPORT_OK = False
    _RAG_IMPORT_ERR = str(_e)

try:
    from rag_query_pdf import rag_query_pdf as _rag_query_pdf_fn
    _RAG_PDF_IMPORT_OK = True
    _RAG_PDF_IMPORT_ERR = ""
except Exception as _e:
    _RAG_PDF_IMPORT_OK = False
    _RAG_PDF_IMPORT_ERR = str(_e)


def run_rag_query(query: str, video_id: str, top_k: int = 4) -> dict | None:
    if not _RAG_IMPORT_OK:
        st.error(f"Failed to import rag_query: {_RAG_IMPORT_ERR}")
        return None
    try:
        result = _rag_query_fn(
            query=query,
            top_k=top_k,
            verbose=True,          # show per-stage timings in terminal
            video_id=video_id,
            _skip_venv_check=True, # venv already active in Streamlit process
        )
        return result
    except RuntimeError as e:
        # Quota / RESOURCE_EXHAUSTED raised after all retries
        st.error(str(e))
        return None
    except Exception as e:
        st.error(f"Query error: {e}")
        return None


def run_rag_query_pdf(query: str, pdf_id: str) -> dict | None:
    """Run RAG query on a PDF."""
    if not _RAG_PDF_IMPORT_OK:
        st.error(f"Failed to import rag_query_pdf: {_RAG_PDF_IMPORT_ERR}")
        return None
    try:
        chunks_dir = os.path.join("pdfs", pdf_id, "chunks")
        api_key = os.environ.get("GEMINI_API_KEY", "")
        result = _rag_query_pdf_fn(
            query=query,
            pdf_id=pdf_id,
            chunks_dir=chunks_dir,
            api_key=api_key,
        )
        return result
    except Exception as e:
        st.error(f"PDF Query error: {e}")
        return None


def _get_video_url(video_id: str) -> str:
    """Read saved YouTube source URL from videos/<id>/meta.json."""
    meta = Path("videos") / video_id / "meta.json"
    if meta.exists():
        try:
            with open(meta, encoding="utf-8") as f:
                return json.load(f).get("source", "")
        except Exception:
            pass
    return ""


def _yt_watch_url(video_id: str, start_sec: float) -> str:
    """
    Build a YouTube watch URL with timestamp from stored meta.json.
    Returns empty string if no URL saved.
    """
    vid = _youtube_id(video_id)
    if not vid:
        return ""
    return f"https://www.youtube.com/watch?v={vid}&t={int(start_sec)}s"


def _load_transcript_segments(video_id: str) -> list[dict]:
    """Load transcript segments for a video (returns list of {'start','end','text'}).
    Returns empty list on error or missing file."""
    p = Path("videos") / video_id / "transcript" / "transcript.json"
    if not p.exists():
        return []
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("segments", [])
    except Exception:
        return []


# def _find_segment_start_for_time(video_id: str, target_sec: float) -> float:
#     """Find the best sentence-level start time for a given target second.
#     Prefers a segment that covers the target; otherwise returns nearest segment start.
#     Falls back to the original target_sec if transcript not available."""
#     segs = _load_transcript_segments(video_id)
#     if not segs:
#         return float(target_sec)
#     # Look for a segment that contains the target time
#     for s in segs:
#         try:
#             if s.get("start", 0) <= target_sec <= s.get("end", 0):
#                 return float(s.get("start", target_sec))
#         except Exception:
#             continue
#     # If none contain it, find nearest start
#     starts = [float(s.get("start", 0)) for s in segs if "start" in s]
#     if not starts:
#         return float(target_sec)
#     nearest = min(starts, key=lambda x: abs(x - target_sec))
#     return float(nearest)

# ...existing code...
def _find_segment_start_for_time(video_id: str, target_sec: float) -> float:
    """Find best sentence-level start for a target second.
    If a transcript segment contains target_sec, return target_sec (so YouTube jumps
    to the exact semantic alignment). Otherwise return the nearest segment start.
    This avoids jumping earlier to the enclosing segment's start when you prefer
    the precise second found by semantic alignment.
    """
    segs = _load_transcript_segments(video_id)
    if not segs:
        return float(target_sec)

    # Prefer returning the exact target if it's inside any segment
    for s in segs:
        try:
            start = float(s.get("start", 0))
            end   = float(s.get("end",   0))
            if start <= float(target_sec) <= end:
                # Return the precise second (no shift to the segment's start)
                return float(target_sec)
        except Exception:
            continue

    # If none contain it, fall back to nearest segment start (existing behavior)
    starts = [float(s.get("start", 0)) for s in segs if "start" in s]
    if not starts:
        return float(target_sec)
    nearest = min(starts, key=lambda x: abs(x - float(target_sec)))
    return float(nearest)
# ...existing code...

# ── View-model helpers (presentation only; read existing files) ──────────────
def _youtube_id(video_id: str) -> str | None:
    """Extract the YouTube video id from the saved source URL, if any."""
    url = _get_video_url(video_id)
    if not url:
        return None
    for p in (
        r"youtu\.be/([^?&/]+)",
        r"youtube\.com/watch\?v=([^&]+)",
        r"youtube\.com/embed/([^?&/]+)",
        r"youtube\.com/shorts/([^?&/]+)",
    ):
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def _video_duration_seconds(video_id: str) -> float:
    segs = _load_transcript_segments(video_id)
    try:
        return float(segs[-1].get("end", 0)) if segs else 0.0
    except Exception:
        return 0.0


def _video_info(video_id: str) -> dict:
    chunks_json = Path("videos") / video_id / "chunks" / "chunks.json"
    try:
        processed_ts = chunks_json.stat().st_mtime
    except OSError:
        processed_ts = 0.0
    return {
        "id":        video_id,
        "chunks":    _get_chunk_count(video_id),
        "duration":  _get_video_duration(video_id),
        "seconds":   _video_duration_seconds(video_id),
        "processed": processed_ts,
        "yt":        _youtube_id(video_id),
        # Only fully-indexed videos (chunks.json present) are listed → ready.
        "status":    "ready",
    }


def _pdf_info(pdf_id: str) -> dict:
    meta = {}
    p = Path("pdfs") / pdf_id / "meta.json"
    try:
        with open(p, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        pass
    try:
        processed_ts = datetime.fromisoformat(meta.get("processing_date", "")).timestamp()
    except Exception:
        processed_ts = p.stat().st_mtime if p.exists() else 0.0
    return {
        "id":        pdf_id,
        "chunks":    meta.get("chunks_count", 0),
        "pages":     meta.get("total_pages", 0),
        "processed": processed_ts,
        "status":    "ready",
    }


def _fmt_date(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("Processed %b %d, %Y") if ts else ""


_STATUS_COLOR = {"ready": "green", "processing": "orange", "failed": "red"}


# ═══════════════════════════════════════════════════════════════
# PAGE: HOME (Video + PDF)
# ═══════════════════════════════════════════════════════════════
def page_home():
    is_video = st.session_state.source_mode == "Video"

    # ── Hero ────────────────────────────────────────────────────
    with st.container(key="hero"):
        if is_video:
            st.markdown(C.hero(
                "Video Intelligence Platform",
                "Upload any video or YouTube URL → Get AI-powered answers with precise timestamps",
                "movie",
            ), unsafe_allow_html=True)
        else:
            st.markdown(C.hero(
                "PDF Intelligence Platform",
                "Upload PDFs → Get AI-powered answers with page references",
                "file",
            ), unsafe_allow_html=True)
        with st.container(horizontal=True, gap="small"):
            if st.button("Process a video" if is_video else "Process a PDF",
                         type="primary", icon=":material/add_circle:"):
                st.switch_page(_page("process"))
            if st.button("Ask AI" if is_video else "Ask PDF", icon=":material/chat:"):
                st.switch_page(_page("ask"))

    # ── Stats ───────────────────────────────────────────────────
    n_queries = len(st.session_state.chat_history)
    ai_card = {
        "icon": "sparkles", "label": "AI Status", "tone": "sky",
        "pill": C.status_pill(gemini_ok, "Ready", "Not ready"),
        "context": _MODEL_LABEL if gemini_ok else "Add GEMINI_API_KEY to .env",
    }
    if is_video:
        items = [_video_info(v) for v in _list_processed_videos()]
        total_chunks = sum(i["chunks"] for i in items)
        hours = sum(i["seconds"] for i in items) / 3600
        cards = [
            {"icon": "movie", "label": "Videos Indexed", "value": len(items),
             "context": "ready to query" if items else "none yet"},
            {"icon": "layers", "label": "Total Chunks", "value": f"{total_chunks:,}",
             "context": f"~{total_chunks // len(items)} per video" if items else ""},
            {"icon": "clock", "label": "Hours Indexed", "value": f"{hours:.1f}", "tone": "emerald",
             "context": f"{n_queries} saved quer{'y' if n_queries == 1 else 'ies'}"},
            ai_card,
        ]
    else:
        items = [_pdf_info(p) for p in _list_processed_pdfs()]
        total_chunks = sum(i["chunks"] for i in items)
        total_pages = sum(i["pages"] for i in items)
        cards = [
            {"icon": "file", "label": "PDFs Indexed", "value": len(items),
             "context": "ready to query" if items else "none yet"},
            {"icon": "layers", "label": "Total Chunks", "value": f"{total_chunks:,}",
             "context": f"~{total_chunks // len(items)} per PDF" if items else ""},
            {"icon": "pages", "label": "Pages Indexed", "value": f"{total_pages:,}", "tone": "emerald",
             "context": f"{n_queries} saved quer{'y' if n_queries == 1 else 'ies'}"},
            ai_card,
        ]
    st.markdown(C.stat_grid(cards), unsafe_allow_html=True)

    # ── How it works ────────────────────────────────────────────
    if is_video:
        steps = [("Input Video", "YouTube URL"), ("Extract Audio", "FFmpeg"),
                 ("Transcribe", "WhisperX"), ("Index Chunks", "ChromaDB"), ("Ask AI", "Gemini")]
    else:
        steps = [("Upload PDF", "Choose document"), ("Extract Text", "PyMuPDF"),
                 ("Semantic Chunking", "LangChain"), ("Embeddings", "BGE model"), ("Ask PDF", "Gemini")]
    if not items:
        st.markdown(C.section_title("How it works", "sparkles"), unsafe_allow_html=True)
        st.markdown(C.stepper(steps), unsafe_allow_html=True)
    else:
        with st.expander("How it works", icon=":material/info:"):
            st.markdown(C.stepper(steps), unsafe_allow_html=True)

    # ── Processed items ─────────────────────────────────────────
    noun = "videos" if is_video else "PDFs"
    if not items:
        st.markdown(C.empty_state(
            f"No {noun} yet",
            "Paste a YouTube link to build your first searchable index." if is_video
            else "Upload a PDF to build your first searchable index.",
            "movie" if is_video else "file",
        ), unsafe_allow_html=True)
        with st.container(horizontal=True, horizontal_alignment="center"):
            if st.button("Process your first video" if is_video else "Process your first PDF",
                         type="primary", icon=":material/add_circle:"):
                st.switch_page(_page("process"))
        return

    st.markdown(C.section_title(f"Processed {noun} ({len(items)})",
                                "layers"), unsafe_allow_html=True)
    col_search, col_sort = st.columns([3, 1], vertical_alignment="bottom")
    with col_search:
        search = st.text_input("Search", placeholder=f"Search {noun} by name",
                               icon=":material/search:", label_visibility="collapsed",
                               key="home_search")
    with col_sort:
        sort_opts = ["Newest", "Name", "Duration" if is_video else "Pages"]
        sort_by = st.selectbox("Sort", sort_opts, label_visibility="collapsed", key="home_sort")

    shown = [i for i in items if search.strip().lower() in i["id"].lower()]
    if sort_by == "Newest":
        shown.sort(key=lambda i: i["processed"], reverse=True)
    elif sort_by == "Name":
        shown.sort(key=lambda i: i["id"].lower())
    elif sort_by == "Duration":
        shown.sort(key=lambda i: i["seconds"], reverse=True)
    else:
        shown.sort(key=lambda i: i["pages"], reverse=True)

    if not shown:
        st.caption(f"No {noun} match “{search}”.")

    for it in shown:
        with st.container(key=f"row_{it['id']}"):
            c_thumb, c_meta, c_act = st.columns([1.5, 5, 1.3], vertical_alignment="center")
            with c_thumb:
                st.markdown(C.thumb(it.get("yt"), "movie" if is_video else "file"),
                            unsafe_allow_html=True)
            with c_meta:
                if is_video:
                    badges = (C.badge(f"{it['chunks']} chunks", "violet", "layers")
                              + C.badge(it["duration"], "emerald", "clock"))
                else:
                    badges = (C.badge(f"{it['pages']} pages", "emerald", "pages")
                              + C.badge(f"{it['chunks']} chunks", "violet", "layers"))
                st.markdown(C.item_meta(it["id"], _fmt_date(it["processed"]), badges),
                            unsafe_allow_html=True)
            with c_act:
                if st.button("Ask", key=f"home_ask_{it['id']}", type="primary",
                             icon=":material/chat:", width="stretch"):
                    if is_video:
                        st.session_state.active_video = it["id"]
                    else:
                        st.session_state.active_pdf = it["id"]
                    st.switch_page(_page("ask"))


# ═══════════════════════════════════════════════════════════════
# PAGE: PROCESS VIDEO
# ═══════════════════════════════════════════════════════════════
def page_process_video():
    st.markdown(C.page_header("Process a New Video",
                              "Index a YouTube video or local file for AI-powered Q&A", "upload"),
                unsafe_allow_html=True)

    col_form, col_info = st.columns([3, 2])

    with col_form:
        with st.form("process_form", border=False):
            source = st.text_input(
                "Video source",
                placeholder="https://youtu.be/... or /path/to/video.mp4",
                help="Paste a YouTube URL or provide a local video file path",
                icon=":material/link:",
            )
            video_id = st.text_input(
                "Video ID",
                placeholder="e.g. lecture1, meeting_jan, product_demo",
                help="A unique name to identify this video",
                icon=":material/label:",
            )

            col_a, col_b = st.columns(2)
            with col_a:
                model_size = st.selectbox("Whisper model", ["base", "small", "medium", "tiny"],
                                          help="Larger = more accurate but slower")
                top_k = st.slider("Top-K chunks", 2, 8, 4, help="Chunks retrieved per query")
            with col_b:
                min_words = st.number_input("Min words / chunk", 40, 200, 80)
                max_words = st.number_input("Max words / chunk", 100, 600, 350)

            force = st.checkbox("Force reprocess", value=False,
                                help="Re-run all phases even if outputs already exist")

            cookies_browser = st.selectbox(
                "YouTube bot fix — use cookies from browser",
                ["", "chrome", "firefox", "edge", "brave"],
                format_func=lambda x: "None (default)" if x == "" else x.capitalize(),
                help="If YouTube shows 'Sign in to confirm not a bot', select your browser. Must be logged into YouTube in that browser."
            )

            submitted = st.form_submit_button("Start processing", type="primary",
                                              icon=":material/rocket_launch:", width="stretch")

    with col_info:
        st.markdown(C.info_panel([
            ("timer", "Estimated times", [
                ("10-min video", "~3 min"),
                ("30-min video", "~8 min"),
                ("1-hour video", "~15 min"),
            ]),
            ("mic", "Model guide", [
                ("tiny", "fastest, lower accuracy"),
                ("base", "good balance (default)"),
                ("small", "better accuracy"),
                ("medium", "best accuracy, slower"),
            ]),
        ]), unsafe_allow_html=True)

    if submitted:
        if not source.strip():
            st.error("Please enter a video source URL or file path.", icon=":material/error:")
        elif not video_id.strip():
            st.error("Please enter a Video ID.", icon=":material/error:")
        else:
            st.markdown(C.section_title("Pipeline progress", "cpu"), unsafe_allow_html=True)

            phases = [
                ("Phase 1", "Audio Extraction"),
                ("Phase 2", "Transcription (WhisperX)"),
                ("Phase 3", "Semantic Chunking"),
                ("Phase 4", "ChromaDB Indexing"),
            ]
            phase_icons = ["audio", "mic", "scissors", "database"]
            phase_cols = st.columns(4)
            phase_placeholders = []
            for i, (col, (ph, label)) in enumerate(zip(phase_cols, phases)):
                with col:
                    ph_ph = st.empty()
                    ph_ph.markdown(C.phase_card(ph, label, "wait", phase_icons[i]),
                                   unsafe_allow_html=True)
                    phase_placeholders.append(ph_ph)

            log_box = st.empty()
            status_box = st.empty()

            # Always use the venv Python so all packages are available
            venv_python = str(Path(__file__).parent / "rag_env" / "Scripts" / "python.exe")
            if not Path(venv_python).exists():
                venv_python = sys.executable  # fallback

            cmd = [
                venv_python, "run_pipeline.py",
                "--source", source.strip(),
                "--video_id", video_id.strip(),
                "--model", model_size,
                "--top_k", str(top_k),
                "--min_words", str(min_words),
                "--max_words", str(max_words),
                "--index_only",
            ]
            if force:
                cmd.append("--force")
            if cookies_browser:
                cmd += ["--cookies_from_browser", cookies_browser]

            phase_keywords = [
                ("STEP 1", "Audio"),
                ("STEP 2", "Transcri"),
                ("STEP 3", "Chunk"),
                ("STEP 4", "ChromaDB"),
            ]

            def update_phase(idx, done=False):
                ph, label = phases[idx]
                phase_placeholders[idx].markdown(
                    C.phase_card(ph, label, "done" if done else "active", phase_icons[idx]),
                    unsafe_allow_html=True,
                )

            logs = []
            try:
                env = os.environ.copy()
                env["PYTHONUTF8"] = "1"
                env["PYTHONIOENCODING"] = "utf-8"
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    cwd=str(Path(__file__).parent),
                )
                for line in proc.stdout:
                    line = line.rstrip()
                    logs.append(line)
                    for i, (kw1, kw2) in enumerate(phase_keywords):
                        if kw1 in line or kw2 in line:
                            if i > 0:
                                update_phase(i - 1, done=True)
                            update_phase(i, done=False)
                    log_box.code("\n".join(logs[-30:]), language=None)

                proc.wait()
                for i in range(4):
                    update_phase(i, done=True)

                if proc.returncode == 0:
                    st.session_state.active_video = video_id.strip()
                    status_box.success(f"Video **{video_id}** processed successfully! Go to **Ask AI** to start querying.",
                                       icon=":material/check_circle:")
                else:
                    status_box.error("Pipeline failed. Check logs above for errors.", icon=":material/error:")

            except Exception as e:
                st.error(f"Failed to start pipeline: {e}", icon=":material/error:")


# ═══════════════════════════════════════════════════════════════
# PAGE: PROCESS PDF
# ═══════════════════════════════════════════════════════════════
def page_process_pdf():
    st.markdown(C.page_header("Process PDF",
                              "Upload a PDF → Automatically chunk, embed, and index for Q&A", "upload"),
                unsafe_allow_html=True)

    st.markdown(C.stepper([
        ("Upload", "your PDF file"),
        ("Name it", "e.g. thesis_2024"),
        ("Process", "chunk + embed (BGE)"),
        ("Query", "in Ask PDF"),
    ]), unsafe_allow_html=True)

    with st.expander("Features", icon=":material/info:"):
        st.markdown("""
- **Semantic chunking** — chunks grouped by meaning, not page breaks
- **Page numbers** — answers include exact page references
- **Fast retrieval** — MMR-based search for relevance
- **Zero manual steps** — one-click processing (text, embeddings with BAAI/bge-small-en-v1.5, ChromaDB)
""")

    st.markdown(C.section_title("Upload PDF", "upload"), unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        uploaded_pdf = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            accept_multiple_files=False,
            help="Max file size: typically 50MB (depends on Streamlit config)"
        )

    with col2:
        pdf_id = st.text_input(
            "PDF ID",
            placeholder="e.g. thesis, whitepaper, manual",
            help="A unique name to identify this PDF",
            icon=":material/label:",
        )

    if uploaded_pdf and pdf_id:
        # Validate PDF ID
        if not pdf_id.replace("_", "").replace("-", "").isalnum():
            st.error("PDF ID can only contain letters, numbers, underscores, and hyphens",
                     icon=":material/error:")
        else:
            # Check if already processed
            if Path(f"pdfs/{pdf_id}/meta.json").exists():
                st.warning(f"PDF '{pdf_id}' already processed. Overwriting...", icon=":material/warning:")

            col_process, col_info = st.columns([2, 3])

            with col_process:
                if st.button("Process PDF", width="stretch", type="primary",
                             icon=":material/rocket_launch:"):
                    result = process_pdf_upload(uploaded_file=uploaded_pdf, pdf_id=pdf_id)
                    if result and result.get('success'):
                        st.session_state.active_pdf = pdf_id
                        st.success("Navigating to Ask PDF...", icon=":material/arrow_forward:")
                        time.sleep(1)
                        st.switch_page(_page("ask"))

            with col_info:
                st.info(f"""
                **File info:**
                - Name: {uploaded_pdf.name}
                - Size: {uploaded_pdf.size / 1024 / 1024:.2f} MB
                """, icon=":material/description:")

    elif uploaded_pdf and not pdf_id:
        st.warning("Please enter a PDF ID to proceed", icon=":material/warning:")

    elif pdf_id and not uploaded_pdf:
        st.info("Upload a PDF file to get started", icon=":material/info:")

    # Show recently processed PDFs
    st.markdown(C.section_title("Recently processed PDFs", "layers"), unsafe_allow_html=True)

    processed_pdfs = _list_processed_pdfs()

    if processed_pdfs:
        for pdf in processed_pdfs[-5:]:  # Show last 5
            info = _pdf_info(pdf)
            with st.container(key=f"row_recent_{pdf}"):
                c_thumb, c_meta, c_act = st.columns([0.7, 5, 1.1], vertical_alignment="center")
                with c_thumb:
                    st.markdown(C.thumb(None, "file"), unsafe_allow_html=True)
                with c_meta:
                    st.markdown(C.item_meta(
                        pdf, _fmt_date(info["processed"]),
                        C.badge(f"{info['pages']} pages", "emerald", "pages")
                        + C.badge(f"{info['chunks']} chunks", "violet", "layers"),
                    ), unsafe_allow_html=True)
                with c_act:
                    if st.button("Query", key=f"query_pdf_{pdf}", width="stretch",
                                 icon=":material/chat:"):
                        st.session_state.active_pdf = pdf
                        st.switch_page(_page("ask"))
    else:
        st.info("No PDFs processed yet. Upload one above to get started!", icon=":material/info:")


# ═══════════════════════════════════════════════════════════════
# PAGE: ASK AI
# ═══════════════════════════════════════════════════════════════
def page_ask_video():
    st.markdown(C.page_header("Ask AI", "Get timestamp-aware answers from your indexed videos", "chat"),
                unsafe_allow_html=True)

    processed = _list_processed_videos()

    if not processed:
        st.markdown(C.empty_state("No videos indexed yet",
                                  "Process a video first, then come back to ask questions."),
                    unsafe_allow_html=True)
        with st.container(horizontal=True, horizontal_alignment="center"):
            if st.button("Process a video", type="primary", icon=":material/add_circle:"):
                st.switch_page(_page("process"))
        return

    # Video selector
    col_sel, col_meta = st.columns([2, 3], vertical_alignment="bottom")
    with col_sel:
        default_idx = processed.index(st.session_state.active_video) if st.session_state.active_video in processed else 0
        selected_video = st.selectbox("Video", processed, index=default_idx)
        st.session_state.active_video = selected_video

    with col_meta:
        chunks = _get_chunk_count(selected_video)
        dur    = _get_video_duration(selected_video)
        st.markdown(
            '<div class="iq-badges" style="padding-bottom:0.55rem">'
            + C.badge(f"{chunks} chunks indexed", "violet", "layers")
            + C.badge(f"{dur} duration", "emerald", "clock")
            + C.badge(_MODEL_LABEL, "slate", "sparkles")
            + "</div>",
            unsafe_allow_html=True,
        )

    # Suggested questions — dropdown
    suggestions = [
        "— Pick a suggested question —",
        "What is the main topic of this video?",
        "Summarize the key points",
        "What tools or technologies are mentioned?",
        "What are the practical use cases discussed?",
        "Explain the most important concept",
    ]
    picked = st.selectbox(
        ":material/lightbulb: Suggested questions",
        suggestions,
        index=0,
        key="suggested_q",
    )
    # Only copy to text box when user actually picks a new suggestion (not on every rerun)
    if picked and picked != "— Pick a suggested question —":
        if st.session_state.get("_last_applied_suggestion") != picked:
            st.session_state["query_input"] = picked
            st.session_state["_last_applied_suggestion"] = picked

    # Query input
    col_q, col_k, col_btn = st.columns([6, 1, 1], vertical_alignment="bottom")
    with col_q:
        query = st.text_input(
            "Your question",
            placeholder="Ask anything about the video...",
            key="query_input",
            icon=":material/search:",
        )
    with col_k:
        top_k = st.selectbox("Top-K", [3, 4, 5, 6], index=1)
    with col_btn:
        ask_clicked = st.button("Ask", type="primary", icon=":material/send:", width="stretch")

    if ask_clicked:
        if not query.strip():
            st.warning("Please type a question in the box above before clicking Ask.", icon=":material/warning:")
        elif not gemini_ok:
            st.error("GEMINI_API_KEY is not set. Please add it to your .env file.", icon=":material/key_off:")
        else:
            with st.spinner("Thinking…"):
                t0 = time.time()
                result = run_rag_query(query.strip(), selected_video, top_k)
                t1 = time.time()
            if result:
                st.caption(f"Query completed in {t1-t0:.1f}s")
                _add_to_history({
                    "query":    query.strip(),
                    # only the answer text is needed by Chat History (keeps the file small)
                    "result":   {"gemini_response": result["gemini_response"]},
                    "video_id": selected_video,
                    "time":     time.strftime("%b %d, %H:%M"),
                })

                cards = _parse_response_to_cards(result["gemini_response"])

                st.markdown(C.section_title("Answer", "sparkles"), unsafe_allow_html=True)

                if cards:
                    for card in cards:
                        # Build YouTube link from first timestamp in card
                        _ts_sec = 0
                        _raw = card.get("timestamp", "")
                        _first = re.split(r"\s*→\s*", _raw)[0].strip()
                        if re.fullmatch(r"\d+", _first):
                            _ts_sec = int(_first)
                        elif re.fullmatch(r"\d+:\d{2}", _first):
                            _mm, _ss = _first.split(":")
                            _ts_sec = int(_mm) * 60 + int(_ss)
                        # prefer sentence-level start time when available
                        _sent_start = _find_segment_start_for_time(selected_video, _ts_sec)
                        _yt = _yt_watch_url(selected_video, _sent_start)
                        st.markdown(C.answer_card(card["timestamp"], card["topic"], card["summary"], _yt),
                                    unsafe_allow_html=True)
                else:
                    st.markdown(C.answer_card("", "", result["gemini_response"]), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: ASK PDF
# ═══════════════════════════════════════════════════════════════
def page_ask_pdf():
    st.markdown(C.page_header("Ask PDF", "Get page-aware answers from your indexed PDFs", "chat"),
                unsafe_allow_html=True)

    processed_pdfs = _list_processed_pdfs()

    if not processed_pdfs:
        st.markdown(C.empty_state("No PDFs indexed yet",
                                  "Process a PDF first, then come back to ask questions.", "file"),
                    unsafe_allow_html=True)
        with st.container(horizontal=True, horizontal_alignment="center"):
            if st.button("Process a PDF", type="primary", icon=":material/add_circle:"):
                st.switch_page(_page("process"))
        return

    # PDF selector
    col_sel, col_meta = st.columns([2, 3], vertical_alignment="bottom")
    with col_sel:
        default_idx = processed_pdfs.index(st.session_state.active_pdf) if st.session_state.active_pdf in processed_pdfs else 0
        selected_pdf = st.selectbox("PDF", processed_pdfs, index=default_idx)
        st.session_state.active_pdf = selected_pdf

    with col_meta:
        pages = _get_pdf_page_count(selected_pdf)
        chunks = _get_pdf_chunk_count(selected_pdf)
        st.markdown(
            '<div class="iq-badges" style="padding-bottom:0.55rem">'
            + C.badge(f"{pages} pages", "emerald", "pages")
            + C.badge(f"{chunks} chunks indexed", "violet", "layers")
            + C.badge(_MODEL_LABEL, "slate", "sparkles")
            + "</div>",
            unsafe_allow_html=True,
        )

    # Query input
    user_query = st.text_input("Ask a question about this PDF",
                               placeholder="E.g., What are the main findings?",
                               icon=":material/search:")

    if user_query:
        with st.spinner("Retrieving and generating answer..."):
            result = run_rag_query_pdf(user_query, selected_pdf)

        if result:
            # Add to chat history
            _add_to_history({
                "source": f"PDF: {selected_pdf}",
                "question": user_query,
                "answer": result['answer'],
                "pages": list(result['sources']),
                "time": time.strftime("%b %d, %H:%M"),
            })

            # Display answer
            st.markdown(C.section_title("Answer", "sparkles"), unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(result['answer'])

                # Display sources (page numbers)
                if result['sources']:
                    st.markdown(
                        '<div class="iq-badges">'
                        + "".join(C.badge(f"Page {p}", "violet", "pin") for p in result['sources'])
                        + "</div>",
                        unsafe_allow_html=True,
                    )

            # Display query time
            st.caption(f"Query time: {result['query_time']:.2f}s")

            # Display retrieved chunks
            if result['retrieved_chunks']:
                with st.expander("Retrieved chunks", icon=":material/layers:"):
                    for i, chunk in enumerate(result['retrieved_chunks'], 1):
                        with st.container(border=True):
                            st.markdown(f"**Chunk {i}:** {chunk['page_spans']}")
                            st.text(chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'])


# ═══════════════════════════════════════════════════════════════
# PAGE: CHAT HISTORY
# ═══════════════════════════════════════════════════════════════
def page_history():
    st.markdown(C.page_header("Chat History", f"Your {HISTORY_LIMIT} most recent queries, saved across restarts", "chat"),
                unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown(C.empty_state("No queries yet",
                                  "Go to Ask AI or Ask PDF to start a conversation.", "chat"),
                    unsafe_allow_html=True)
        return

    col_count, col_clear = st.columns([4, 1], vertical_alignment="center")
    with col_count:
        st.markdown(f"**{len(st.session_state.chat_history)} saved quer{'y' if len(st.session_state.chat_history) == 1 else 'ies'}**")
    with col_clear:
        if st.button("Clear history", icon=":material/delete:", width="stretch"):
            st.session_state.chat_history = []
            _save_history()
            st.rerun()

    for i, entry in enumerate(reversed(st.session_state.chat_history), 1):
        q_num = len(st.session_state.chat_history) - i + 1

        # Handle both video and PDF entries
        if 'source' in entry and 'PDF:' in entry.get('source', ''):
            source_label = entry.get('source', 'PDF')
            expander_label = f"**Q{q_num}:** {entry['question'][:50]}... — {source_label}"
            exp_icon = ":material/description:"
        else:
            source_label = entry.get('video_id', 'Video')
            time_label = entry.get('time', 'N/A')
            expander_label = f"**Q{q_num}:** {entry['query'][:50]}... — {time_label} | {source_label}"
            exp_icon = ":material/movie:"

        with st.expander(expander_label, icon=exp_icon):
            query_text = entry.get('question') or entry.get('query', '')
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(query_text)

            answer_text = entry.get('answer') or entry.get('result', {}).get('gemini_response', '')
            with st.chat_message("assistant", avatar=":material/smart_toy:"):
                if not entry.get('answer'):  # Video format
                    cards = _parse_response_to_cards(answer_text)
                    if cards:
                        st.markdown("".join(
                            C.answer_card(c['timestamp'], c['topic'], c['summary']) for c in cards
                        ), unsafe_allow_html=True)
                    else:
                        st.markdown(answer_text)
                else:  # PDF format
                    st.markdown(answer_text)
                    if entry.get('pages'):
                        st.markdown(
                            '<div class="iq-badges">'
                            + "".join(C.badge(f"Page {p}", "violet", "pin") for p in entry['pages'])
                            + "</div>",
                            unsafe_allow_html=True,
                        )


# ═══════════════════════════════════════════════════════════════
# NAVIGATION
# ═══════════════════════════════════════════════════════════════
_PAGES = {
    "home":          st.Page(page_home,          title="Home",          icon=":material/home:",     default=True),
    "process_video": st.Page(page_process_video, title="Process Video", icon=":material/settings:", url_path="process-video"),
    "ask_video":     st.Page(page_ask_video,     title="Ask AI",        icon=":material/chat:",     url_path="ask-ai"),
    "process_pdf":   st.Page(page_process_pdf,   title="Process PDF",   icon=":material/settings:", url_path="process-pdf"),
    "ask_pdf":       st.Page(page_ask_pdf,       title="Ask PDF",       icon=":material/chat:",     url_path="ask-pdf"),
    "history":       st.Page(page_history,       title="Chat History",  icon=":material/history:",  url_path="history"),
}


def _page(kind: str):
    """Resolve 'home' | 'process' | 'ask' | 'history' to the page for the current mode."""
    if kind in ("process", "ask"):
        suffix = "video" if st.session_state.source_mode == "Video" else "pdf"
        return _PAGES[f"{kind}_{suffix}"]
    return _PAGES[kind]


gemini_ok = bool(os.environ.get("GEMINI_API_KEY", "").strip())
from image_description import GEMINI_MODEL
_MODEL_LABEL = GEMINI_MODEL.replace("gemini-", "Gemini ").replace("-", " ").title()

# ── SIDEBAR: brand + mode switch (before navigation so a mode change re-routes) ─
with st.sidebar:
    st.markdown(f"""
<div class="iq-brand">
  <div class="iq-brand-mark">{C.icon("movie", 20)}</div>
  <div>
    <div class="iq-brand-name">InsightIQ</div>
    <div class="iq-brand-tag">Ask questions about any video</div>
  </div>
</div>""", unsafe_allow_html=True)

    source_mode = st.segmented_control(
        "Source mode",
        options=["Video", "PDF"],
        default=st.session_state.source_mode,
        required=True,
        format_func=lambda o: ":material/movie: Video" if o == "Video" else ":material/description: PDF",
        key="mode_seg",
        label_visibility="collapsed",
        width="stretch",
    )
    if source_mode and source_mode != st.session_state.source_mode:
        st.session_state.source_mode = source_mode
        st.switch_page(_PAGES["home"])

_nav_pages = [_PAGES["home"], _page("process"), _page("ask"), _PAGES["history"]]
pg = st.navigation(_nav_pages, position="hidden")

# ── SIDEBAR: menu + processed items + API status ──────────────────────────────
with st.sidebar:
    st.markdown('<div class="iq-side-label">Menu</div>', unsafe_allow_html=True)
    with st.container(key="nav"):
        for p in _nav_pages:
            state = "navon" if p.url_path == pg.url_path else "navoff"
            with st.container(key=f"{state}_{p.url_path or 'home'}"):
                st.page_link(p, label=p.title, icon=p.icon, width="stretch")

    is_video = st.session_state.source_mode == "Video"
    if is_video:
        sb_items = sorted((_video_info(v) for v in _list_processed_videos()),
                          key=lambda i: i["processed"], reverse=True)
        active_id = st.session_state.active_video
    else:
        sb_items = sorted((_pdf_info(p) for p in _list_processed_pdfs()),
                          key=lambda i: i["processed"], reverse=True)
        active_id = st.session_state.active_pdf

    if sb_items:
        st.markdown(
            f'<div class="iq-side-label">Processed {"videos" if is_video else "PDFs"} · {len(sb_items)}</div>',
            unsafe_allow_html=True,
        )
        with st.container(key="sb_list"):
            for it in sb_items[:5]:
                state = "sbon" if it["id"] == active_id else "sboff"
                dot = _STATUS_COLOR.get(it["status"], "green")
                meta = (f"{it['duration']} · {it['chunks']} chunks" if is_video
                        else f"{it['pages']} pages · {it['chunks']} chunks")
                with st.container(key=f"{state}_{it['id']}"):
                    if st.button(f":{dot}[●]  {it['id']}  :gray[{meta}]",
                                 key=f"sb_{it['id']}", width="stretch",
                                 help=f"Ask about {it['id']}"):
                        if is_video:
                            st.session_state.active_video = it["id"]
                        else:
                            st.session_state.active_pdf = it["id"]
                        st.switch_page(_page("ask"))
        if len(sb_items) > 5:
            with st.container(key="sb_viewall"):
                st.page_link(_PAGES["home"], label=f"View all ({len(sb_items)})",
                             icon=":material/arrow_forward:")

    st.markdown('<div class="iq-side-label">Gemini API</div>', unsafe_allow_html=True)
    st.markdown(C.status_pill(gemini_ok, "Connected", "Set GEMINI_API_KEY"), unsafe_allow_html=True)


pg.run()
