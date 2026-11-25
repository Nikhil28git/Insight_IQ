# """
# Phase 5 — LangChain LCEL RAG Query  (Gemini Flash)
# ====================================================
# Architecture
# ------------
#   LangChain HuggingFaceEmbeddings        ← same model as Phase 3/4
#       ↓  (query embedding)
#   LangChain Chroma MMR retriever         ← MMR search (diversity + relevance)
#       ↓  (list[Document])
#   _expand_neighbors()                    ← adds prev/next chunk neighbours
#       ↓  (expanded list[dict])
#   _merge_windows()                       ← custom transcript-aware merging
#       ↓  (time windows with combined text)
#   _semantic_answer_start()               ← fine-grained sentence alignment
#       ↓  (precise answer_start_sec per window)
#   LCEL chain:
#       ChatPromptTemplate
#           | ChatGoogleGenerativeAI       ← Gemini Flash via LangChain
#           | StrOutputParser
#       ↓  (plain-text response)

# Where LangChain is used (and why)
# ----------------------------------
#   ✅  HuggingFaceEmbeddings  — consistent model wrapper; same object reused
#       by embed_and_store.py so query encoding matches stored vectors exactly.
#   ✅  Chroma retriever        — as_retriever() gives a standard Runnable that
#       integrates cleanly into LCEL pipelines.
#   ✅  ChatGoogleGenerativeAI  — LangChain-native LLM wrapper; supports LCEL
#       pipe syntax, streaming, and future swap to any other provider.
#   ✅  ChatPromptTemplate + StrOutputParser — standard LCEL building blocks.

# Where custom logic is kept (and why)
# --------------------------------------
#   🔧  _merge_windows()  — no LangChain equivalent; merges retrieved chunks
#       that are adjacent in video time into continuous viewing windows.
#       This is transcript-domain knowledge, not retrieval logic.

# CLI usage:
#   python rag_query.py --query "What is generative AI?"
#   python rag_query.py --query "How does RAG work?" --top_k 5
#   python rag_query.py --interactive
# """

# import argparse
# import os
# import sys
# import time

# import streamlit as st

# # Load .env once at import time (not per query)
# try:
#     from dotenv import load_dotenv
#     load_dotenv()
# except ImportError:
#     pass

# # ──────────────────────────────────────────────
# # VENV GUARD
# # ──────────────────────────────────────────────
# def _check_venv() -> None:
#     if sys.prefix == sys.base_prefix:
#         print(
#             "\n⛔  Virtual environment is NOT active!\n"
#             "   Activate first:  rag_env\\Scripts\\activate\n"
#         )
#         sys.exit(1)


# # ──────────────────────────────────────────────
# # CONSTANTS
# # ──────────────────────────────────────────────
# CHROMA_DIR    = "chroma_db"
# COLLECTION    = "video_transcript"   # legacy fallback
# EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
# GEMINI_MODEL  = "gemini-2.5-flash"
# DEFAULT_TOP_K = 4
# # _merge_windows() dual-gate: two MMR chunks are fused into one viewing window
# # only when BOTH conditions are true:
# #   1. Time proximity  : start of next chunk <= end of current + MERGE_GAP_SEC
# #   2. Semantic similarity: cosine(emb_i, emb_j) >= MERGE_SIM_THRESHOLD
# # This avoids both (a) chain-merging across topics that share timestamps due
# # to boundary-aware overlap, and (b) the brittleness of a single hardcoded gap.
# # The threshold is derived dynamically per query from the actual embeddings:
# # it is set to the MEDIAN inter-chunk cosine of all retrieved chunks, so it
# # self-calibrates to the density of the video's topic structure.
# MERGE_GAP_SEC       = 45   # generous time window; semantic gate does the real work
# MERGE_SIM_THRESHOLD = None  # None = auto (median cosine of retrieved chunks)


# def _video_collection(video_id: str) -> str:
#     return f"vid_{video_id}"


# # NOTE:
# # Streamlit resource caching is used for all heavy reusable objects below.
# # This replaces the old custom dict caches so reruns keep reusing objects
# # reliably inside a Streamlit session and across repeated queries.

# # ──────────────────────────────────────────────
# # PROMPT TEMPLATE  (used in LCEL chain)
# # ──────────────────────────────────────────────
# _PROMPT_TEMPLATE = """\
# You are an expert assistant that helps users navigate video content.

# A user asked: "{query}"

# Below are the most relevant transcript segments retrieved from the video, \
# each labelled with their timestamps (in mm:ss format).

# {context}

# Based primarily on the content of these segments, respond in the following exact \
# format for EACH relevant segment:

# ---
# 📍 Timestamp : <start_time> → <end_time>
# 🏷️  Topic     : <short descriptive topic title (max 8 words)>
# 📝 Summary   : <2-4 sentences explaining what is discussed in this segment \
# that directly answers or relates to the user's query>
# ---

# Rules:
# - Use the exact timestamps provided for each segment.
# - Only include segments that are genuinely relevant to the query.
# - Keep each summary focused and factual — based strictly on the transcript text.
# - If multiple segments discuss the same topic, merge them into one entry with \
# the combined timestamp range.
# - Do not add information not present in the transcript.
# - If none of the retrieved segments directly answer the query, respond with \
# exactly this message (do not fabricate content):
#   ⚠️ The retrieved segments do not cover this topic directly. \
# Try rephrasing your question, increasing Top-K, or check that the correct \
# video is selected.
# """


# # ──────────────────────────────────────────────
# # HELPERS
# # ──────────────────────────────────────────────
# def _load_env() -> str:
#     key = os.environ.get("GEMINI_API_KEY", "").strip()
#     if not key or key.startswith("your-"):
#         print(
#             "\n⛔  GEMINI_API_KEY is not set!\n"
#             "   Edit .env and add:  GEMINI_API_KEY=your-key-here\n"
#             "   Get a free key at:  https://aistudio.google.com/app/apikey\n"
#         )
#         sys.exit(1)
#     return key


# def _fmt_time(seconds: float) -> str:
#     """Convert seconds → mm:ss string,  e.g. 150.4 → '2:30'"""
#     seconds = max(0, int(round(seconds)))
#     m, s    = divmod(seconds, 60)
#     return f"{m}:{s:02d}"


# def _merge_windows(
#     raw_chunks:   list[dict],
#     extra_texts:  dict[int, list[str]] | None = None,
#     chunk_embs:   dict | None = None,
# ) -> list[dict]:
#     """
#     Merge retrieved chunks into continuous viewing windows.

#     Dual-gate merge condition — a consecutive pair of chunks is fused into
#     one window only when BOTH conditions hold:

#       1. Time proximity  :  next.start_sec  <=  current.end  +  MERGE_GAP_SEC
#       2. Semantic similarity (when chunk_embs provided):
#              cosine(emb_cur, emb_next)  >=  sim_threshold
#          where sim_threshold = MEDIAN cosine of all consecutive retrieved pairs.
#          Auto-calibrates per query/video — no hardcoded per-video values.

#     When embeddings are not available only the time gate is used (safe fallback).

#     Only `raw_chunks` (MMR-retrieved) define window BOUNDARIES (timestamps).
#     `extra_texts` supplies neighbour text injected for Gemini context
#     WITHOUT extending any window's start/end.
#     """
#     import numpy as _np

#     if not raw_chunks:
#         return []

#     sorted_chunks = sorted(raw_chunks, key=lambda x: x["start_sec"])

#     # ── Auto-calibrate similarity threshold ────────────────────────────────────
#     # Compute median cosine similarity across all consecutive pairs of retrieved
#     # chunks.  Pairs MORE similar than median → same topic → merge.
#     # Pairs LESS similar → different topic → separate window.
#     sim_threshold: float | None = None
#     if chunk_embs and len(sorted_chunks) >= 2:
#         cosines = []
#         for i in range(len(sorted_chunks) - 1):
#             a = chunk_embs.get(sorted_chunks[i]["chunk_id"])
#             b = chunk_embs.get(sorted_chunks[i + 1]["chunk_id"])
#             if a is not None and b is not None:
#                 denom = float(_np.linalg.norm(a) * _np.linalg.norm(b)) + 1e-9
#                 cosines.append(float(_np.dot(a, b) / denom))
#         if cosines:
#             sim_threshold = float(_np.median(cosines))

#     def _should_merge(cur_window: dict, nxt: dict) -> bool:
#         # Gate 1: time
#         if nxt["start_sec"] > cur_window["end"] + MERGE_GAP_SEC:
#             return False
#         # Gate 2: semantic (only when available)
#         if sim_threshold is not None and chunk_embs:
#             a = chunk_embs.get(cur_window["chunk_ids"][-1])
#             b = chunk_embs.get(nxt["chunk_id"])
#             if a is not None and b is not None:
#                 denom = float(_np.linalg.norm(a) * _np.linalg.norm(b)) + 1e-9
#                 if float(_np.dot(a, b) / denom) < sim_threshold:
#                     return False   # different topic → keep as separate window
#         return True

#     # ── Build windows ──────────────────────────────────────────────────────────
#     windows = []
#     current = {
#         "start":     sorted_chunks[0]["start_sec"],
#         "end":       sorted_chunks[0]["end_sec"],
#         "chunk_ids": [sorted_chunks[0]["chunk_id"]],
#         "texts":     [sorted_chunks[0]["text"]],
#     }
#     for meta in sorted_chunks[1:]:
#         if _should_merge(current, meta):
#             current["end"] = max(current["end"], meta["end_sec"])
#             current["chunk_ids"].append(meta["chunk_id"])
#             current["texts"].append(meta["text"])
#         else:
#             windows.append(current)
#             current = {
#                 "start":     meta["start_sec"],
#                 "end":       meta["end_sec"],
#                 "chunk_ids": [meta["chunk_id"]],
#                 "texts":     [meta["text"]],
#             }
#     windows.append(current)

#     # ── Inject neighbour context text (no timestamp change) ────────────────────
#     if extra_texts:
#         for cid, texts in extra_texts.items():
#             best_win, best_dist = 0, float("inf")
#             for wi, w in enumerate(windows):
#                 for wcid in w["chunk_ids"]:
#                     dist = abs(wcid - cid)
#                     if dist < best_dist:
#                         best_dist, best_win = dist, wi
#             windows[best_win]["texts"].extend(texts)

#     return windows


# def _format_context(windows: list[dict]) -> str:
#     """
#     Format merged time windows into the prompt context block.

#     Uses answer_start_sec (set by _semantic_answer_start) as the segment's
#     start timestamp when available, so that Gemini echoes back the precise
#     answer-aligned time rather than the earlier merged-window start.

#     Example:
#       Before: [Segment 1 | 03:26 → 07:07]
#       After : [Segment 1 | 04:52 → 07:07]
#     """
#     blocks = []
#     for i, w in enumerate(windows, 1):
#         # Use semantically-aligned start if available, else fall back to merge start
#         display_start = w.get("answer_start_sec", w["start"])
#         ts   = f"{_fmt_time(display_start)} → {_fmt_time(w['end'])}"
#         text = " ".join(w["texts"])
#         blocks.append(f"[Segment {i} | {ts}]\n{text}")
#     return "\n\n".join(blocks)


# @st.cache_resource(show_spinner=False)
# def _load_embeddings(model_name: str = EMBED_MODEL):
#     """Load the HuggingFace embedding model once per Streamlit session."""
#     try:
#         from langchain_huggingface import HuggingFaceEmbeddings
#     except ImportError as exc:
#         print(f"\n⛔  Missing LangChain dependency: {exc}\n"
#               "   Run: pip install langchain-huggingface langchain-chroma "
#               "langchain-google-genai\n")
#         sys.exit(1)

#     return HuggingFaceEmbeddings(
#         model_name=model_name,
#         model_kwargs={"device": "cpu"},
#         encode_kwargs={"normalize_embeddings": True},  # required for BGE models
#     )


# @st.cache_resource(show_spinner=False)
# def _load_vectorstore(collection_name: str, model_name: str = EMBED_MODEL):
#     """Reconnect to the persisted Chroma collection once per Streamlit session."""
#     try:
#         from langchain_chroma import Chroma
#     except ImportError as exc:
#         print(f"\n⛔  Missing LangChain dependency: {exc}\n"
#               "   Run: pip install langchain-huggingface langchain-chroma "
#               "langchain-google-genai\n")
#         sys.exit(1)

#     embedding_fn = _load_embeddings(model_name)
#     return Chroma(
#         collection_name=collection_name,
#         embedding_function=embedding_fn,
#         persist_directory=CHROMA_DIR,
#     )


# @st.cache_resource(show_spinner=False)
# def _load_gemini_llm(api_key: str, model_name: str = GEMINI_MODEL, temperature: float = 0.2):
#     """Create the Gemini client once per Streamlit session."""
#     try:
#         from langchain_google_genai import ChatGoogleGenerativeAI
#     except ImportError as exc:
#         print(f"\n⛔  Missing LangChain dependency: {exc}\n"
#               "   Run: pip install langchain-huggingface langchain-chroma "
#               "langchain-google-genai\n")
#         sys.exit(1)

#     return ChatGoogleGenerativeAI(
#         model=model_name,
#         google_api_key=api_key,
#         temperature=temperature,
#     )


# @st.cache_resource(show_spinner=False)
# def _load_prompt():
#     """Cache the prompt template object because it is reused for every query."""
#     try:
#         from langchain_core.prompts import ChatPromptTemplate
#     except ImportError as exc:
#         print(f"\n⛔  Missing LangChain dependency: {exc}\n"
#               "   Run: pip install langchain-huggingface langchain-chroma "
#               "langchain-google-genai\n")
#         sys.exit(1)

#     return ChatPromptTemplate.from_template(_PROMPT_TEMPLATE)


# @st.cache_resource(show_spinner=False)
# def _load_chain(api_key: str, model_name: str = GEMINI_MODEL, temperature: float = 0.2):
#     """Build the LCEL prompt → Gemini → parser chain once per Streamlit session."""
#     try:
#         from langchain_core.output_parsers import StrOutputParser
#     except ImportError as exc:
#         print(f"\n⛔  Missing LangChain dependency: {exc}\n"
#               "   Run: pip install langchain-huggingface langchain-chroma "
#               "langchain-google-genai\n")
#         sys.exit(1)

#     prompt = _load_prompt()
#     llm = _load_gemini_llm(api_key, model_name, temperature)
#     return prompt | llm | StrOutputParser()


# @st.cache_resource(show_spinner=False)
# def _load_retriever(collection_name: str, top_k: int, model_name: str = EMBED_MODEL):
#     """
#     MMR retriever — balances relevance with diversity.

#     fetch_k : candidate pool Chroma scores before MMR re-ranking
#               (max(15, top_k*4) so there are enough candidates to diversify from)
#     k       : final chunks returned to the pipeline  (= top_k from UI)
#     lambda_mult : 0 = max diversity, 1 = max relevance.
#                   0.75 → slight diversity bias; avoids returning 3 near-identical
#                   adjacent chunks while still prioritising relevance.

#     Cached per (collection_name, top_k) — changing Top-K rebuilds only this
#     object; embeddings / vectorstore / chain stay cached.
#     """
#     vectorstore = _load_vectorstore(collection_name, model_name)
#     return vectorstore.as_retriever(
#         search_type="mmr",
#         search_kwargs={
#             "k":           top_k,
#             "fetch_k":     max(15, top_k * 4),
#             "lambda_mult": 0.75,
#         },
#     )


# # ──────────────────────────────────────────────
# # PART 2 — NEIGHBOUR EXPANSION
# # ──────────────────────────────────────────────
# def _expand_neighbors(
#     raw_chunks:  list[dict],
#     vectorstore,
#     prev_n:      int = 1,
#     next_n:      int = 1,
# ) -> list[dict]:
#     """
#     For each retrieved chunk_id N, also fetch chunks N-prev_n … N-1
#     and N+1 … N+next_n from Chroma.

#     Why: MMR picks diverse chunks; an answer straddling a chunk boundary may
#     be split between chunk N and N+1.  Adding neighbours before merge_windows()
#     fuses both halves into one coherent reading window.

#     Deduplication : chunk_ids already in raw_chunks are never fetched twice.
#     Order         : returned list is sorted chronologically by start_sec.
#     """
#     retrieved_ids = {c["chunk_id"] for c in raw_chunks}
#     neighbor_ids: set[int] = set()
#     for c in raw_chunks:
#         cid = c["chunk_id"]
#         for offset in range(-prev_n, next_n + 1):
#             if offset != 0:
#                 nid = cid + offset
#                 if nid >= 0 and nid not in retrieved_ids:
#                     neighbor_ids.add(nid)

#     if not neighbor_ids:
#         return sorted(raw_chunks, key=lambda x: x["start_sec"])

#     fetch_ids = [f"chunk_{i:04d}" for i in sorted(neighbor_ids)]
#     try:
#         result = vectorstore.get(ids=fetch_ids, include=["documents", "metadatas"])
#     except Exception as exc:
#         print(f"[WARN] Neighbour fetch failed: {exc}")
#         return sorted(raw_chunks, key=lambda x: x["start_sec"])

#     expanded = list(raw_chunks)
#     for doc_text, meta in zip(
#         result.get("documents") or [],
#         result.get("metadatas") or [],
#     ):
#         if meta is None:
#             continue
#         expanded.append({
#             "chunk_id":  meta.get("chunk_id",  -1),
#             "start_sec": meta.get("start_sec", 0.0),
#             "end_sec":   meta.get("end_sec",   0.0),
#             "words":     meta.get("words",     0),
#             "text":      doc_text or "",
#         })

#     # Dedup by chunk_id, sort chronologically
#     seen: set[int] = set()
#     unique: list[dict] = []
#     for c in sorted(expanded, key=lambda x: x["start_sec"]):
#         if c["chunk_id"] not in seen:
#             seen.add(c["chunk_id"])
#             unique.append(c)
#     return unique


# # ──────────────────────────────────────────────
# # PART 3 — FINE-GRAINED SEMANTIC TIMESTAMP
# # ──────────────────────────────────────────────
# def _semantic_answer_start(
#     query:        str,
#     window:       dict,
#     video_id:     str,
#     embeddings_fn,
# ) -> float:
#     """
#     Determine the exact sentence inside a merged window where the answer
#     semantically begins, instead of using the window's merged start time
#     (which may include early contextual chunks).

#     Algorithm
#     ---------
#     1. Load raw transcript segments for this video.
#     2. Keep only segments whose time overlaps window [start, end].
#     3. Embed all segment texts + the query in ONE batch using the already-cached
#        HuggingFaceEmbeddings object  — no extra model load, no re-encoding.
#     4. Find the segment with the highest cosine similarity to the query.
#     5. Return its start time as answer_start_sec.

#     Performance: 10-20 short segments embed in < 0.05 s on CPU.
#     Falls back to window["start"] on any error.

#     Example
#     -------
#     Current (merged start): 03:26
#     After alignment       : 04:52   ← sentence where the actual topic begins
#     """
#     import json as _json
#     import numpy as _np

#     if video_id:
#         transcript_path = os.path.join("videos", video_id, "transcript", "transcript.json")
#     else:
#         transcript_path = os.path.join("transcripts", "transcript.json")

#     if not os.path.exists(transcript_path):
#         return window["start"]

#     try:
#         with open(transcript_path, encoding="utf-8") as _f:
#             segments = _json.load(_f).get("segments", [])
#     except Exception:
#         return window["start"]

#     win_start, win_end = window["start"], window["end"]

#     in_window = [
#         s for s in segments
#         if s.get("end",   0.0) >= win_start
#         and s.get("start", 0.0) <= win_end
#         and s.get("text",  "").strip()
#     ]
#     if not in_window:
#         return win_start

#     seg_texts = [s["text"].strip() for s in in_window]

#     try:
#         vecs  = _np.array(
#             embeddings_fn.embed_documents([query] + seg_texts),
#             dtype=_np.float32,
#         )
#         # vectors are L2-normalised (encode_kwargs normalize_embeddings=True)
#         sims  = vecs[1:] @ vecs[0]
#         best  = int(_np.argmax(sims))
#         return float(in_window[best].get("start", win_start))
#     except Exception:
#         return win_start


# # ──────────────────────────────────────────────
# # CORE RAG FUNCTION
# # ──────────────────────────────────────────────
# def rag_query(
#     query:    str,
#     top_k:    int  = DEFAULT_TOP_K,
#     verbose:  bool = True,
#     video_id: str  = "",
#     _skip_venv_check: bool = False,
# ) -> dict:
#     """
#     Run a full LangChain LCEL RAG query against the video transcript.

#     Parameters
#     ----------
#     query    : user question
#     top_k    : number of chunks to retrieve
#     video_id : which video to query (collection = vid_<video_id>).
#                Falls back to legacy 'video_transcript' collection when empty.
#     _skip_venv_check : set True when calling from Streamlit (venv already active).
#     """
#     if not _skip_venv_check:
#         _check_venv()
#     api_key = _load_env()

#     collection_name = _video_collection(video_id) if video_id else COLLECTION

#     if verbose:
#         print("[INFO] Loading LangChain components (Streamlit cached resources)…")
#         if video_id:
#             print(f"[INFO] Video  : {video_id}  →  collection: {collection_name}")

#     # Heavy resources: embeddings, Chroma, Gemini, and LCEL chain are all
#     # cached with @st.cache_resource so Streamlit reruns reuse them reliably.
#     t_emb = time.time()
#     _load_embeddings(EMBED_MODEL)
#     if verbose:
#         print(f"[INFO] Embeddings ready in {time.time() - t_emb:.1f}s")

#     t_vec = time.time()
#     _load_vectorstore(collection_name, EMBED_MODEL)
#     if verbose:
#         print(f"[INFO] Vectorstore ready in {time.time() - t_vec:.1f}s")

#     t_llm_load = time.time()
#     _load_gemini_llm(api_key, GEMINI_MODEL, 0.2)
#     if verbose:
#         print(f"[INFO] Gemini client ready in {time.time() - t_llm_load:.1f}s")

#     t_chain = time.time()
#     chain = _load_chain(api_key, GEMINI_MODEL, 0.2)
#     if verbose:
#         print(f"[INFO] LCEL chain ready in {time.time() - t_chain:.1f}s")

#     t_ret_create = time.time()
#     retriever = _load_retriever(collection_name, top_k, EMBED_MODEL)
#     if verbose:
#         print(f"[INFO] Retriever ready in {time.time() - t_ret_create:.1f}s")

#     # ── 1. MMR retrieval ──────────────────────────────────────────────────────
#     # fetch_k=max(15, top_k*4) candidate pool → MMR re-ranks for diversity
#     # lambda_mult=0.75 → slight diversity bias to avoid near-duplicate chunks
#     if verbose:
#         print(f"[INFO] MMR retrieval: top_k={top_k}  "
#               f"fetch_k={max(15, top_k * 4)}  lambda_mult=0.75 …")

#     t_ret = time.time()
#     docs  = retriever.invoke(query)   # returns list[Document]

#     raw_chunks = []
#     for doc in docs:
#         m = doc.metadata
#         raw_chunks.append({
#             "chunk_id":  m.get("chunk_id",  -1),
#             "start_sec": m.get("start_sec", 0.0),
#             "end_sec":   m.get("end_sec",   0.0),
#             "words":     m.get("words",     0),
#             "text":      doc.page_content,
#         })

#     if verbose:
#         mmr_ids = [f"chunk_{c['chunk_id']:04d}" for c in raw_chunks]
#         print(f"[INFO] Retrieved {len(raw_chunks)} chunks in {time.time()-t_ret:.1f}s")
#         print(f"[INFO] MMR-selected chunks: {mmr_ids}")
#         for c in raw_chunks:
#             print(f"       chunk_{c['chunk_id']:04d}  "
#                   f"{_fmt_time(c['start_sec'])} → {_fmt_time(c['end_sec'])}")

#     # ── 2. Neighbour expansion ────────────────────────────────────────────────
#     # Fetch chunk_(N-1) and chunk_(N+1) for every MMR-selected chunk N.
#     # IMPORTANT: neighbours contribute CONTEXT TEXT only — their timestamps
#     # are NOT used for window boundaries.  This prevents chain-merging across
#     # topic boundaries caused by the 15% overlap making adjacent chunks share
#     # timestamps (e.g. chunk_0003 end=5:25, chunk_0004 start=4:56 overlap).
#     vectorstore = _load_vectorstore(collection_name, EMBED_MODEL)
#     all_expanded = _expand_neighbors(raw_chunks, vectorstore, prev_n=1, next_n=1)

#     # Split: boundary chunks (MMR only) vs neighbour-only chunks (text extras)
#     retrieved_ids  = {c["chunk_id"] for c in raw_chunks}
#     extra_texts: dict[int, list[str]] = {}
#     for c in all_expanded:
#         if c["chunk_id"] not in retrieved_ids and c["text"].strip():
#             extra_texts[c["chunk_id"]] = [c["text"]]

#     if verbose:
#         exp_ids = [f"chunk_{c['chunk_id']:04d}" for c in all_expanded]
#         new_ids = [f"chunk_{cid:04d}" for cid in extra_texts]
#         print(f"[INFO] Neighbour-expanded context (+{len(new_ids)} text-only): {exp_ids}")

#     # ── 3. Merge MMR chunks into transcript time windows (boundaries only) ─────
#     # Load pre-computed per-chunk embeddings from disk (written by Phase 3/4,
#     # no re-encoding needed) and pass to _merge_windows for the semantic gate.
#     import numpy as _np
#     chunk_embs: dict[int, _np.ndarray] = {}
#     _emb_path = (
#         os.path.join("videos", video_id, "chunks", "chunk_embeddings.npy")
#         if video_id else os.path.join("chunks", "chunk_embeddings.npy")
#     )
#     if os.path.exists(_emb_path):
#         try:
#             _arr = _np.load(_emb_path)   # shape (N, dim), float32
#             chunk_embs = {i: _arr[i] for i in range(len(_arr))}
#             if verbose:
#                 print(f"[INFO] Loaded chunk embeddings from {_emb_path} "
#                       f"({len(chunk_embs)} vectors) for semantic merge gate")
#         except Exception as _exc:
#             if verbose:
#                 print(f"[WARN] Could not load chunk embeddings: {_exc}")

#     # Neighbour texts injected via extra_texts without altering timestamps.
#     windows = _merge_windows(raw_chunks, extra_texts, chunk_embs)
#     if verbose:
#         print(f"[INFO] Merged into {len(windows)} time window(s)")
#     for w in windows:
#         truncated, total = [], 0
#         for t in w["texts"]:
#             wds = t.split()
#             if total + len(wds) > 600:
#                 rem = 600 - total
#                 if rem > 0:
#                     truncated.append(" ".join(wds[:rem]))
#                 break
#             truncated.append(t)
#             total += len(wds)
#         w["texts"] = truncated

#     # ── 4. Fine-grained semantic timestamp alignment ───────────────────────────
#     # For each window, find the transcript sentence most semantically similar
#     # to the query and store its timestamp as answer_start_sec.
#     # The Gemini context still uses the full merged window; only the *displayed*
#     # start timestamp is sharpened.
#     embeddings_fn = _load_embeddings(EMBED_MODEL)
#     for w in windows:
#         raw_start    = w["start"]
#         answer_start = _semantic_answer_start(query, w, video_id, embeddings_fn)
#         w["answer_start_sec"] = answer_start
#         if verbose:
#             print(f"[INFO] Window {_fmt_time(raw_start)} → {_fmt_time(w['end'])}  |  "
#                   f"semantic answer start → {_fmt_time(answer_start)}"
#                   + (f"  (shifted +{_fmt_time(answer_start - raw_start)})"
#                      if answer_start > raw_start else ""))

#     # ── 3. LCEL chain:  ChatPromptTemplate | Gemini | StrOutputParser ─────────
#     if verbose:
#         print(f"[INFO] Querying {GEMINI_MODEL} via LCEL chain …\n")

#     t_llm = time.time()
#     context  = _format_context(windows)

#     # Retry with backoff on RESOURCE_EXHAUSTED (429 quota) or transient errors
#     _max_retries = 3
#     _delay = 15   # seconds
#     response = None
#     for _attempt in range(_max_retries):
#         try:
#             response = chain.invoke({"query": query, "context": context})
#             break
#         except Exception as _e:
#             _msg = str(_e)
#             if "RESOURCE_EXHAUSTED" in _msg or "429" in _msg:
#                 if _attempt < _max_retries - 1:
#                     if verbose:
#                         print(f"[WARN] Quota limit hit (attempt {_attempt+1}/{_max_retries}). "
#                               f"Retrying in {_delay}s…")
#                     time.sleep(_delay)
#                     _delay *= 2   # exponential backoff
#                 else:
#                     raise RuntimeError(
#                         "⚠️ Gemini quota exceeded (free tier: 20 req/day). "
#                         "Please wait a few minutes and try again, or upgrade your API plan at "
#                         "https://ai.google.dev/gemini-api/docs/rate-limits"
#                     ) from _e
#             else:
#                 raise
#     if verbose:
#         print(f"[INFO] Gemini responded in {time.time()-t_llm:.1f}s")

#     return {
#         "query":           query,
#         "windows":         windows,
#         "gemini_response": response,
#         "raw_chunks":      raw_chunks,
#     }


# # ──────────────────────────────────────────────
# # DISPLAY
# # ──────────────────────────────────────────────
# def _display(result: dict) -> None:
#     print(f"\n{'═'*60}")
#     print(f"  🔍  Query: {result['query']}")
#     print(f"{'═'*60}\n")
#     print(result["gemini_response"])
#     print(f"\n{'─'*60}")
#     print(f"  Retrieved chunks: ", end="")
#     for c in result["raw_chunks"]:
#         sim = f"sim={c['similarity']}" if 'similarity' in c else ""
#         label = f"chunk_{c['chunk_id']:04d}"
#         print(f"{label}({sim})" if sim else label, end="  ")
#     print(f"\n{'─'*60}\n")


# # ──────────────────────────────────────────────
# # CLI
# # ──────────────────────────────────────────────
# def main() -> None:
#     _check_venv()

#     parser = argparse.ArgumentParser(
#         description="Phase 5 — RAG query over video transcript using Gemini Flash 2.5"
#     )
#     parser.add_argument(
#         "--query", "-q", type=str, default=None,
#         help="Question to ask about the video"
#     )
#     parser.add_argument(
#         "--top_k", type=int, default=DEFAULT_TOP_K,
#         help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K})"
#     )
#     parser.add_argument(
#         "--interactive", "-i", action="store_true",
#         help="Run in interactive loop (ask multiple questions)"
#     )
#     args = parser.parse_args()

#     if args.interactive:
#         print(f"\n{'═'*60}")
#         print(f"  Phase 5 — Interactive RAG Query")
#         print(f"  Model  : {GEMINI_MODEL}")
#         print(f"  Top-K  : {args.top_k}")
#         print(f"  Type 'exit' or 'quit' to stop.")
#         print(f"{'═'*60}\n")

#         while True:
#             try:
#                 query = input("  ❓ Your question: ").strip()
#             except (KeyboardInterrupt, EOFError):
#                 print("\n  Goodbye!\n")
#                 break
#             if not query:
#                 continue
#             if query.lower() in ("exit", "quit", "q"):
#                 print("  Goodbye!\n")
#                 break
#             result = rag_query(query, top_k=args.top_k)
#             _display(result)

#     elif args.query:
#         result = rag_query(args.query, top_k=args.top_k)
#         _display(result)

#     else:
#         parser.print_help()
#         print("\n  Example:\n"
#               "    python rag_query.py --query \"What is generative AI?\"\n"
#               "    python rag_query.py --interactive\n")


# if __name__ == "__main__":
#     main()


"""
Phase 5 — LangChain LCEL RAG Query  (Gemini Flash)
====================================================
Architecture
------------
  LangChain HuggingFaceEmbeddings        ← same model as Phase 3/4
      ↓  (query embedding)
  LangChain Chroma MMR retriever         ← MMR search (diversity + relevance)
      ↓  (list[Document])
  _expand_neighbors()                    ← adds prev/next chunk neighbours
      ↓  (expanded list[dict])
  _merge_windows()                       ← custom transcript-aware merging
      ↓  (time windows with combined text)
  _semantic_answer_start()               ← fine-grained sentence alignment
      ↓  (precise answer_start_sec per window)
  LCEL chain:
      ChatPromptTemplate
          | ChatGoogleGenerativeAI       ← Gemini Flash via LangChain
          | StrOutputParser
      ↓  (plain-text response)

Where LangChain is used (and why)
----------------------------------
  ✅  HuggingFaceEmbeddings  — consistent model wrapper; same object reused
      by embed_and_store.py so query encoding matches stored vectors exactly.
  ✅  Chroma retriever        — as_retriever() gives a standard Runnable that
      integrates cleanly into LCEL pipelines.
  ✅  ChatGoogleGenerativeAI  — LangChain-native LLM wrapper; supports LCEL
      pipe syntax, streaming, and future swap to any other provider.
  ✅  ChatPromptTemplate + StrOutputParser — standard LCEL building blocks.

Where custom logic is kept (and why)
--------------------------------------
  🔧  _merge_windows()  — no LangChain equivalent; merges retrieved chunks
      that are adjacent in video time into continuous viewing windows.
      This is transcript-domain knowledge, not retrieval logic.

  🔧  _semantic_answer_start()  — sentence-level (not just segment-level)
      semantic alignment within a merged window. Splits transcript segments
      into sentences, interpolates per-sentence timestamps, and picks the
      sentence most similar to the query (with overlap + similarity gating).

CLI usage:
  python rag_query.py --query "What is generative AI?"
  python rag_query.py --query "How does RAG work?" --top_k 5
  python rag_query.py --interactive
"""

import argparse
import os
import re
import sys
import time

import streamlit as st

# Load .env once at import time (not per query)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)
except ImportError:
    pass

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
CHROMA_DIR    = "chroma_db"
COLLECTION    = "video_transcript"   # legacy fallback
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
from image_description import GEMINI_MODEL  # shared model setting (override via .env)
DEFAULT_TOP_K = 4
# _merge_windows() dual-gate: two MMR chunks are fused into one viewing window
# only when BOTH conditions are true:
#   1. Time proximity  : start of next chunk <= end of current + MERGE_GAP_SEC
#   2. Semantic similarity: cosine(emb_i, emb_j) >= MERGE_SIM_THRESHOLD
# This avoids both (a) chain-merging across topics that share timestamps due
# to boundary-aware overlap, and (b) the brittleness of a single hardcoded gap.
# The threshold is derived dynamically per query from the actual embeddings:
# it is set to the MEDIAN inter-chunk cosine of all retrieved chunks, so it
# self-calibrates to the density of the video's topic structure.
MERGE_GAP_SEC       = 45   # generous time window; semantic gate does the real work
MERGE_SIM_THRESHOLD = None  # None = auto (median cosine of retrieved chunks)

# _semantic_answer_start() tuning:
#   ANSWER_SIM_THRESHOLD : minimum cosine similarity for a sentence to be
#                          considered the "answer start". Below this, fall
#                          back to the window's merged start time.
#   ANSWER_MIN_OVERLAP   : a transcript segment must overlap the window by at
#                          least this fraction of its own duration to be
#                          considered for sentence-level alignment.
ANSWER_SIM_THRESHOLD = 0.25
ANSWER_MIN_OVERLAP   = 0.3


def _video_collection(video_id: str) -> str:
    return f"vid_{video_id}"


# NOTE:
# Streamlit resource caching is used for all heavy reusable objects below.
# This replaces the old custom dict caches so reruns keep reusing objects
# reliably inside a Streamlit session and across repeated queries.

# ──────────────────────────────────────────────
# PROMPT TEMPLATE  (used in LCEL chain)
# ──────────────────────────────────────────────
_PROMPT_TEMPLATE = """\
You are an expert assistant that helps users navigate video content.

A user asked: "{query}"

Below are the most relevant transcript segments retrieved from the video, \
each labelled with their timestamps (in mm:ss format).

{context}

Based primarily on the content of these segments, respond in the following exact \
format for EACH relevant segment:

---
📍 Timestamp : <start_time> → <end_time>
🏷️  Topic     : <short descriptive topic title (max 8 words)>
📝 Summary   : <2-4 sentences explaining what is discussed in this segment \
that directly answers or relates to the user's query>
---

Rules:
- Use the exact timestamps provided for each segment.
- Only include segments that are genuinely relevant to the query.
- Keep each summary focused and factual — based strictly on the transcript text.
- If multiple segments discuss the same topic, merge them into one entry with \
the combined timestamp range.
- Do not add information not present in the transcript.
- If none of the retrieved segments directly answer the query, respond with \
exactly this message (do not fabricate content):
  ⚠️ The retrieved segments do not cover this topic directly. \
Try rephrasing your question, increasing Top-K, or check that the correct \
video is selected.
"""


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def _load_env() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key or key.startswith("your-"):
        print(
            "\n⛔  GEMINI_API_KEY is not set!\n"
            "   Edit .env and add:  GEMINI_API_KEY=your-key-here\n"
            "   Get a free key at:  https://aistudio.google.com/app/apikey\n"
        )
        sys.exit(1)
    return key


def _fmt_time(seconds: float) -> str:
    """Convert seconds → mm:ss string,  e.g. 150.4 → '2:30'"""
    seconds = max(0, int(round(seconds)))
    m, s    = divmod(seconds, 60)
    return f"{m}:{s:02d}"


def _merge_windows(
    raw_chunks:   list[dict],
    extra_texts:  dict[int, list[str]] | None = None,
    chunk_embs:   dict | None = None,
) -> list[dict]:
    """
    Merge retrieved chunks into continuous viewing windows.

    Dual-gate merge condition — a consecutive pair of chunks is fused into
    one window only when BOTH conditions hold:

      1. Time proximity  :  next.start_sec  <=  current.end  +  MERGE_GAP_SEC
      2. Semantic similarity (when chunk_embs provided):
             cosine(emb_cur, emb_next)  >=  sim_threshold
         where sim_threshold = MEDIAN cosine of all consecutive retrieved pairs.
         Auto-calibrates per query/video — no hardcoded per-video values.

    When embeddings are not available only the time gate is used (safe fallback).

    Only `raw_chunks` (MMR-retrieved) define window BOUNDARIES (timestamps).
    `extra_texts` supplies neighbour text injected for Gemini context
    WITHOUT extending any window's start/end.
    """
    import numpy as _np

    if not raw_chunks:
        return []

    sorted_chunks = sorted(raw_chunks, key=lambda x: x["start_sec"])

    # ── Auto-calibrate similarity threshold ────────────────────────────────────
    # Compute median cosine similarity across all consecutive pairs of retrieved
    # chunks.  Pairs MORE similar than median → same topic → merge.
    # Pairs LESS similar → different topic → separate window.
    sim_threshold: float | None = None
    if chunk_embs and len(sorted_chunks) >= 2:
        cosines = []
        for i in range(len(sorted_chunks) - 1):
            a = chunk_embs.get(sorted_chunks[i]["chunk_id"])
            b = chunk_embs.get(sorted_chunks[i + 1]["chunk_id"])
            if a is not None and b is not None:
                denom = float(_np.linalg.norm(a) * _np.linalg.norm(b)) + 1e-9
                cosines.append(float(_np.dot(a, b) / denom))
        if cosines:
            sim_threshold = float(_np.median(cosines))

    def _should_merge(cur_window: dict, nxt: dict) -> bool:
        # Gate 1: time
        if nxt["start_sec"] > cur_window["end"] + MERGE_GAP_SEC:
            return False
        # Gate 2: semantic (only when available)
        if sim_threshold is not None and chunk_embs:
            a = chunk_embs.get(cur_window["chunk_ids"][-1])
            b = chunk_embs.get(nxt["chunk_id"])
            if a is not None and b is not None:
                denom = float(_np.linalg.norm(a) * _np.linalg.norm(b)) + 1e-9
                if float(_np.dot(a, b) / denom) < sim_threshold:
                    return False   # different topic → keep as separate window
        return True

    # ── Build windows ──────────────────────────────────────────────────────────
    windows = []
    current = {
        "start":     sorted_chunks[0]["start_sec"],
        "end":       sorted_chunks[0]["end_sec"],
        "chunk_ids": [sorted_chunks[0]["chunk_id"]],
        "texts":     [sorted_chunks[0]["text"]],
    }
    for meta in sorted_chunks[1:]:
        if _should_merge(current, meta):
            current["end"] = max(current["end"], meta["end_sec"])
            current["chunk_ids"].append(meta["chunk_id"])
            current["texts"].append(meta["text"])
        else:
            windows.append(current)
            current = {
                "start":     meta["start_sec"],
                "end":       meta["end_sec"],
                "chunk_ids": [meta["chunk_id"]],
                "texts":     [meta["text"]],
            }
    windows.append(current)

    # ── Inject neighbour context text (no timestamp change) ────────────────────
    if extra_texts:
        for cid, texts in extra_texts.items():
            best_win, best_dist = 0, float("inf")
            for wi, w in enumerate(windows):
                for wcid in w["chunk_ids"]:
                    dist = abs(wcid - cid)
                    if dist < best_dist:
                        best_dist, best_win = dist, wi
            windows[best_win]["texts"].extend(texts)

    return windows


def _format_context(windows: list[dict]) -> str:
    """
    Format merged time windows into the prompt context block.

    Uses answer_start_sec (set by _semantic_answer_start) as the segment's
    start timestamp when available, so that Gemini echoes back the precise
    answer-aligned time rather than the earlier merged-window start.

    Example:
      Before: [Segment 1 | 03:26 → 07:07]
      After : [Segment 1 | 04:52 → 07:07]
    """
    blocks = []
    for i, w in enumerate(windows, 1):
        # Use semantically-aligned start if available, else fall back to merge start
        display_start = w.get("answer_start_sec", w["start"])
        ts   = f"{_fmt_time(display_start)} → {_fmt_time(w['end'])}"
        text = " ".join(w["texts"])
        blocks.append(f"[Segment {i} | {ts}]\n{text}")
    return "\n\n".join(blocks)


@st.cache_resource(show_spinner=False)
def _load_embeddings(model_name: str = EMBED_MODEL):
    """Load the HuggingFace embedding model once per Streamlit session."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        print(f"\n⛔  Missing LangChain dependency: {exc}\n"
              "   Run: pip install langchain-huggingface langchain-chroma "
              "langchain-google-genai\n")
        sys.exit(1)

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # required for BGE models
    )


@st.cache_resource(show_spinner=False)
def _load_vectorstore(collection_name: str, model_name: str = EMBED_MODEL):
    """Reconnect to the persisted Chroma collection once per Streamlit session."""
    try:
        from langchain_chroma import Chroma
    except ImportError as exc:
        print(f"\n⛔  Missing LangChain dependency: {exc}\n"
              "   Run: pip install langchain-huggingface langchain-chroma "
              "langchain-google-genai\n")
        sys.exit(1)

    embedding_fn = _load_embeddings(model_name)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_fn,
        persist_directory=CHROMA_DIR,
    )


@st.cache_resource(show_spinner=False)
def _load_gemini_llm(api_key: str, model_name: str = GEMINI_MODEL, temperature: float = 0.2):
    """Create the Gemini client once per Streamlit session."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        print(f"\n⛔  Missing LangChain dependency: {exc}\n"
              "   Run: pip install langchain-huggingface langchain-chroma "
              "langchain-google-genai\n")
        sys.exit(1)

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
    )


@st.cache_resource(show_spinner=False)
def _load_prompt():
    """Cache the prompt template object because it is reused for every query."""
    try:
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError as exc:
        print(f"\n⛔  Missing LangChain dependency: {exc}\n"
              "   Run: pip install langchain-huggingface langchain-chroma "
              "langchain-google-genai\n")
        sys.exit(1)

    return ChatPromptTemplate.from_template(_PROMPT_TEMPLATE)


@st.cache_resource(show_spinner=False)
def _load_chain(api_key: str, model_name: str = GEMINI_MODEL, temperature: float = 0.2):
    """Build the LCEL prompt → Gemini → parser chain once per Streamlit session."""
    try:
        from langchain_core.output_parsers import StrOutputParser
    except ImportError as exc:
        print(f"\n⛔  Missing LangChain dependency: {exc}\n"
              "   Run: pip install langchain-huggingface langchain-chroma "
              "langchain-google-genai\n")
        sys.exit(1)

    prompt = _load_prompt()
    llm = _load_gemini_llm(api_key, model_name, temperature)
    return prompt | llm | StrOutputParser()


@st.cache_resource(show_spinner=False)
def _load_retriever(collection_name: str, top_k: int, model_name: str = EMBED_MODEL):
    """
    MMR retriever — balances relevance with diversity.

    fetch_k : candidate pool Chroma scores before MMR re-ranking
              (max(15, top_k*4) so there are enough candidates to diversify from)
    k       : final chunks returned to the pipeline  (= top_k from UI)
    lambda_mult : 0 = max diversity, 1 = max relevance.
                  0.75 → slight diversity bias; avoids returning 3 near-identical
                  adjacent chunks while still prioritising relevance.

    Cached per (collection_name, top_k) — changing Top-K rebuilds only this
    object; embeddings / vectorstore / chain stay cached.
    """
    vectorstore = _load_vectorstore(collection_name, model_name)
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":           top_k,
            "fetch_k":     max(15, top_k * 4),
            "lambda_mult": 0.75,
        },
    )


# ──────────────────────────────────────────────
# PART 2 — NEIGHBOUR EXPANSION
# ──────────────────────────────────────────────
def _expand_neighbors(
    raw_chunks:  list[dict],
    vectorstore,
    prev_n:      int = 1,
    next_n:      int = 1,
) -> list[dict]:
    """
    For each retrieved chunk_id N, also fetch chunks N-prev_n … N-1
    and N+1 … N+next_n from Chroma.

    Why: MMR picks diverse chunks; an answer straddling a chunk boundary may
    be split between chunk N and N+1.  Adding neighbours before merge_windows()
    fuses both halves into one coherent reading window.

    Deduplication : chunk_ids already in raw_chunks are never fetched twice.
    Order         : returned list is sorted chronologically by start_sec.
    """
    retrieved_ids = {c["chunk_id"] for c in raw_chunks}
    neighbor_ids: set[int] = set()
    for c in raw_chunks:
        cid = c["chunk_id"]
        for offset in range(-prev_n, next_n + 1):
            if offset != 0:
                nid = cid + offset
                if nid >= 0 and nid not in retrieved_ids:
                    neighbor_ids.add(nid)

    if not neighbor_ids:
        return sorted(raw_chunks, key=lambda x: x["start_sec"])

    fetch_ids = [f"chunk_{i:04d}" for i in sorted(neighbor_ids)]
    try:
        result = vectorstore.get(ids=fetch_ids, include=["documents", "metadatas"])
    except Exception as exc:
        print(f"[WARN] Neighbour fetch failed: {exc}")
        return sorted(raw_chunks, key=lambda x: x["start_sec"])

    expanded = list(raw_chunks)
    for doc_text, meta in zip(
        result.get("documents") or [],
        result.get("metadatas") or [],
    ):
        if meta is None:
            continue
        expanded.append({
            "chunk_id":  meta.get("chunk_id",  -1),
            "start_sec": meta.get("start_sec", 0.0),
            "end_sec":   meta.get("end_sec",   0.0),
            "words":     meta.get("words",     0),
            "text":      doc_text or "",
        })

    # Dedup by chunk_id, sort chronologically
    seen: set[int] = set()
    unique: list[dict] = []
    for c in sorted(expanded, key=lambda x: x["start_sec"]):
        if c["chunk_id"] not in seen:
            seen.add(c["chunk_id"])
            unique.append(c)
    return unique


# ──────────────────────────────────────────────
# PART 3 — FINE-GRAINED SEMANTIC TIMESTAMP
# ──────────────────────────────────────────────
def _split_into_sentence_spans(seg: dict) -> list[dict]:
    """
    Split a transcript segment's text into sentences and proportionally
    distribute the segment's [start, end] time range across them based
    on character length.

    This converts coarse 10-20s WhisperX segments into shorter sentence-level
    spans so _semantic_answer_start() can land much closer to the exact
    moment the answer begins.
    """
    text = seg.get("text", "").strip()
    if not text:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []

    seg_start = seg.get("start", 0.0)
    seg_end   = seg.get("end", seg_start)
    duration  = max(seg_end - seg_start, 0.0)

    char_total = sum(len(s) for s in sentences)
    if char_total == 0:
        char_total = len(sentences)

    spans = []
    cursor = seg_start
    for s in sentences:
        frac  = (len(s) / char_total) if char_total else (1.0 / len(sentences))
        s_dur = duration * frac
        spans.append({
            "text":  s,
            "start": cursor,
            "end":   cursor + s_dur,
        })
        cursor += s_dur

    return spans


def _semantic_answer_start(
    query:         str,
    window:        dict,
    video_id:      str,
    embeddings_fn,
    sim_threshold:    float = ANSWER_SIM_THRESHOLD,
    min_overlap_frac: float = ANSWER_MIN_OVERLAP,
) -> float:
    """
    Determine the exact sentence inside a merged window where the answer
    semantically begins, instead of using the window's merged start time
    (which may include early contextual chunks).

    Algorithm
    ---------
    1. Load raw transcript segments for this video.
    2. Keep only segments whose time overlaps window [start, end] by at
       least `min_overlap_frac` of their own duration.
    3. Split each kept segment into sentence-level spans, with start/end
       times interpolated proportionally by character length.
    4. Embed all sentence spans + the query in ONE batch using the
       already-cached HuggingFaceEmbeddings object — no extra model load.
    5. Explicitly L2-normalise the embeddings and compute cosine similarity
       to the query.
    6. Return the start time of the highest-scoring sentence span,
       provided it clears `sim_threshold`. Otherwise fall back to
       window["start"].

    Performance: typically 10-40 short sentence spans embed in < 0.1 s on CPU.
    Falls back to window["start"] on any error or low-confidence match.

    Example
    -------
    Current (merged start): 03:26
    After alignment       : 04:52   ← sentence where the actual topic begins
    """
    import json as _json
    import numpy as _np

    if video_id:
        transcript_path = os.path.join("videos", video_id, "transcript", "transcript.json")
    else:
        transcript_path = os.path.join("transcripts", "transcript.json")

    if not os.path.exists(transcript_path):
        return window["start"]

    try:
        with open(transcript_path, encoding="utf-8") as _f:
            segments = _json.load(_f).get("segments", [])
    except Exception:
        return window["start"]

    win_start, win_end = window["start"], window["end"]

    # ── Filter segments by meaningful overlap with window ──────────────────
    in_window_segments = []
    for s in segments:
        s_start = s.get("start", 0.0)
        s_end   = s.get("end", s_start)
        text    = s.get("text", "").strip()
        if not text:
            continue

        overlap = min(s_end, win_end) - max(s_start, win_start)
        if overlap <= 0:
            continue

        seg_dur = max(s_end - s_start, 1e-6)
        overlap_frac = overlap / seg_dur

        if overlap_frac >= min_overlap_frac:
            in_window_segments.append(s)

    if not in_window_segments:
        return win_start

    # ── Split into sentence-level spans ─────────────────────────────────────
    sentence_spans = []
    for seg in in_window_segments:
        sentence_spans.extend(_split_into_sentence_spans(seg))

    if not sentence_spans:
        return win_start

    seg_texts = [sp["text"] for sp in sentence_spans]

    try:
        vecs = _np.array(
            embeddings_fn.embed_documents([query] + seg_texts),
            dtype=_np.float32,
        )

        # Explicit normalisation — don't silently rely on external config
        norms = _np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms

        sims = vecs[1:] @ vecs[0]
        best = int(_np.argmax(sims))
        best_sim = float(sims[best])

        if best_sim < sim_threshold:
            return win_start

        return float(sentence_spans[best]["start"])
    except Exception:
        return win_start


# ──────────────────────────────────────────────
# CORE RAG FUNCTION
# ──────────────────────────────────────────────
def rag_query(
    query:    str,
    top_k:    int  = DEFAULT_TOP_K,
    verbose:  bool = True,
    video_id: str  = "",
    _skip_venv_check: bool = False,
) -> dict:
    """
    Run a full LangChain LCEL RAG query against the video transcript.

    Parameters
    ----------
    query    : user question
    top_k    : number of chunks to retrieve
    video_id : which video to query (collection = vid_<video_id>).
               Falls back to legacy 'video_transcript' collection when empty.
    _skip_venv_check : set True when calling from Streamlit (venv already active).
    """
    if not _skip_venv_check:
        _check_venv()
    api_key = _load_env()

    collection_name = _video_collection(video_id) if video_id else COLLECTION

    if verbose:
        print("[INFO] Loading LangChain components (Streamlit cached resources)…")
        if video_id:
            print(f"[INFO] Video  : {video_id}  →  collection: {collection_name}")

    # Heavy resources: embeddings, Chroma, Gemini, and LCEL chain are all
    # cached with @st.cache_resource so Streamlit reruns reuse them reliably.
    t_emb = time.time()
    _load_embeddings(EMBED_MODEL)
    if verbose:
        print(f"[INFO] Embeddings ready in {time.time() - t_emb:.1f}s")

    t_vec = time.time()
    _load_vectorstore(collection_name, EMBED_MODEL)
    if verbose:
        print(f"[INFO] Vectorstore ready in {time.time() - t_vec:.1f}s")

    t_llm_load = time.time()
    _load_gemini_llm(api_key, GEMINI_MODEL, 0.2)
    if verbose:
        print(f"[INFO] Gemini client ready in {time.time() - t_llm_load:.1f}s")

    t_chain = time.time()
    chain = _load_chain(api_key, GEMINI_MODEL, 0.2)
    if verbose:
        print(f"[INFO] LCEL chain ready in {time.time() - t_chain:.1f}s")

    t_ret_create = time.time()
    retriever = _load_retriever(collection_name, top_k, EMBED_MODEL)
    if verbose:
        print(f"[INFO] Retriever ready in {time.time() - t_ret_create:.1f}s")

    # ── 1. MMR retrieval ──────────────────────────────────────────────────────
    # fetch_k=max(15, top_k*4) candidate pool → MMR re-ranks for diversity
    # lambda_mult=0.75 → slight diversity bias to avoid near-duplicate chunks
    if verbose:
        print(f"[INFO] MMR retrieval: top_k={top_k}  "
              f"fetch_k={max(15, top_k * 4)}  lambda_mult=0.75 …")

    t_ret = time.time()
    docs  = retriever.invoke(query)   # returns list[Document]

    raw_chunks = []
    for doc in docs:
        m = doc.metadata
        raw_chunks.append({
            "chunk_id":  m.get("chunk_id",  -1),
            "start_sec": m.get("start_sec", 0.0),
            "end_sec":   m.get("end_sec",   0.0),
            "words":     m.get("words",     0),
            "text":      doc.page_content,
        })

    if verbose:
        mmr_ids = [f"chunk_{c['chunk_id']:04d}" for c in raw_chunks]
        print(f"[INFO] Retrieved {len(raw_chunks)} chunks in {time.time()-t_ret:.1f}s")
        print(f"[INFO] MMR-selected chunks: {mmr_ids}")
        for c in raw_chunks:
            print(f"       chunk_{c['chunk_id']:04d}  "
                  f"{_fmt_time(c['start_sec'])} → {_fmt_time(c['end_sec'])}")

    # ── 2. Neighbour expansion ────────────────────────────────────────────────
    # Fetch chunk_(N-1) and chunk_(N+1) for every MMR-selected chunk N.
    # IMPORTANT: neighbours contribute CONTEXT TEXT only — their timestamps
    # are NOT used for window boundaries.  This prevents chain-merging across
    # topic boundaries caused by the 15% overlap making adjacent chunks share
    # timestamps (e.g. chunk_0003 end=5:25, chunk_0004 start=4:56 overlap).
    vectorstore = _load_vectorstore(collection_name, EMBED_MODEL)
    all_expanded = _expand_neighbors(raw_chunks, vectorstore, prev_n=1, next_n=1)

    # Split: boundary chunks (MMR only) vs neighbour-only chunks (text extras)
    retrieved_ids  = {c["chunk_id"] for c in raw_chunks}
    extra_texts: dict[int, list[str]] = {}
    for c in all_expanded:
        if c["chunk_id"] not in retrieved_ids and c["text"].strip():
            extra_texts[c["chunk_id"]] = [c["text"]]

    if verbose:
        exp_ids = [f"chunk_{c['chunk_id']:04d}" for c in all_expanded]
        new_ids = [f"chunk_{cid:04d}" for cid in extra_texts]
        print(f"[INFO] Neighbour-expanded context (+{len(new_ids)} text-only): {exp_ids}")

    # ── 3. Merge MMR chunks into transcript time windows (boundaries only) ─────
    # Load pre-computed per-chunk embeddings from disk (written by Phase 3/4,
    # no re-encoding needed) and pass to _merge_windows for the semantic gate.
    import numpy as _np
    chunk_embs: dict[int, _np.ndarray] = {}
    _emb_path = (
        os.path.join("videos", video_id, "chunks", "chunk_embeddings.npy")
        if video_id else os.path.join("chunks", "chunk_embeddings.npy")
    )
    if os.path.exists(_emb_path):
        try:
            _arr = _np.load(_emb_path)   # shape (N, dim), float32
            chunk_embs = {i: _arr[i] for i in range(len(_arr))}
            if verbose:
                print(f"[INFO] Loaded chunk embeddings from {_emb_path} "
                      f"({len(chunk_embs)} vectors) for semantic merge gate")
        except Exception as _exc:
            if verbose:
                print(f"[WARN] Could not load chunk embeddings: {_exc}")

    # Neighbour texts injected via extra_texts without altering timestamps.
    windows = _merge_windows(raw_chunks, extra_texts, chunk_embs)
    if verbose:
        print(f"[INFO] Merged into {len(windows)} time window(s)")
    for w in windows:
        truncated, total = [], 0
        for t in w["texts"]:
            wds = t.split()
            if total + len(wds) > 600:
                rem = 600 - total
                if rem > 0:
                    truncated.append(" ".join(wds[:rem]))
                break
            truncated.append(t)
            total += len(wds)
        w["texts"] = truncated

    # ── 4. Fine-grained semantic timestamp alignment ───────────────────────────
    # For each window, find the transcript SENTENCE most semantically similar
    # to the query and store its timestamp as answer_start_sec.
    # The Gemini context still uses the full merged window; only the *displayed*
    # start timestamp is sharpened.
    embeddings_fn = _load_embeddings(EMBED_MODEL)
    for w in windows:
        raw_start    = w["start"]
        answer_start = _semantic_answer_start(query, w, video_id, embeddings_fn)
        w["answer_start_sec"] = answer_start
        if verbose:
            shift = answer_start - raw_start
            shift_str = f"  (shifted +{_fmt_time(shift)})" if shift > 0 else (
                f"  (shifted -{_fmt_time(-shift)})" if shift < 0 else ""
            )
            print(f"[INFO] Window {_fmt_time(raw_start)} → {_fmt_time(w['end'])}  |  "
                  f"semantic answer start → {_fmt_time(answer_start)}{shift_str}")

    # ── 3. LCEL chain:  ChatPromptTemplate | Gemini | StrOutputParser ─────────
    if verbose:
        print(f"[INFO] Querying {GEMINI_MODEL} via LCEL chain …\n")

    t_llm = time.time()
    context  = _format_context(windows)

    # Retry with backoff on RESOURCE_EXHAUSTED (429 quota) or transient errors
    _max_retries = 3
    _delay = 15   # seconds
    response = None
    for _attempt in range(_max_retries):
        try:
            response = chain.invoke({"query": query, "context": context})
            break
        except Exception as _e:
            _msg = str(_e)
            if "RESOURCE_EXHAUSTED" in _msg or "429" in _msg:
                if _attempt < _max_retries - 1:
                    if verbose:
                        print(f"[WARN] Quota limit hit (attempt {_attempt+1}/{_max_retries}). "
                              f"Retrying in {_delay}s…")
                    time.sleep(_delay)
                    _delay *= 2   # exponential backoff
                else:
                    raise RuntimeError(
                        "⚠️ Gemini quota exceeded (free tier: 20 req/day). "
                        "Please wait a few minutes and try again, or upgrade your API plan at "
                        "https://ai.google.dev/gemini-api/docs/rate-limits"
                    ) from _e
            else:
                raise
    if verbose:
        print(f"[INFO] Gemini responded in {time.time()-t_llm:.1f}s")

    return {
        "query":           query,
        "windows":         windows,
        "gemini_response": response,
        "raw_chunks":      raw_chunks,
    }


# ──────────────────────────────────────────────
# DISPLAY
# ──────────────────────────────────────────────
def _display(result: dict) -> None:
    print(f"\n{'═'*60}")
    print(f"  🔍  Query: {result['query']}")
    print(f"{'═'*60}\n")
    print(result["gemini_response"])
    print(f"\n{'─'*60}")
    print(f"  Retrieved chunks: ", end="")
    for c in result["raw_chunks"]:
        sim = f"sim={c['similarity']}" if 'similarity' in c else ""
        label = f"chunk_{c['chunk_id']:04d}"
        print(f"{label}({sim})" if sim else label, end="  ")
    print(f"\n{'─'*60}\n")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main() -> None:
    _check_venv()

    parser = argparse.ArgumentParser(
        description="Phase 5 — RAG query over video transcript using Gemini Flash 2.5"
    )
    parser.add_argument(
        "--query", "-q", type=str, default=None,
        help="Question to ask about the video"
    )
    parser.add_argument(
        "--top_k", type=int, default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K})"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Run in interactive loop (ask multiple questions)"
    )
    args = parser.parse_args()

    if args.interactive:
        print(f"\n{'═'*60}")
        print(f"  Phase 5 — Interactive RAG Query")
        print(f"  Model  : {GEMINI_MODEL}")
        print(f"  Top-K  : {args.top_k}")
        print(f"  Type 'exit' or 'quit' to stop.")
        print(f"{'═'*60}\n")

        while True:
            try:
                query = input("  ❓ Your question: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n  Goodbye!\n")
                break
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("  Goodbye!\n")
                break
            result = rag_query(query, top_k=args.top_k)
            _display(result)

    elif args.query:
        result = rag_query(args.query, top_k=args.top_k)
        _display(result)

    else:
        parser.print_help()
        print("\n  Example:\n"
              "    python rag_query.py --query \"What is generative AI?\"\n"
              "    python rag_query.py --interactive\n")


if __name__ == "__main__":
    main()