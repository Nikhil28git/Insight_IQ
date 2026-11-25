"""
Phase 4 — ChromaDB Storage  (LangChain-based, zero re-encoding)
================================================================
Why no re-encoding?
  Phase 3 (chunk_transcript.py) already embeds every sentence for boundary
  detection.  After chunking it aggregates those sentence vectors per-chunk
  (mean + L2-normalise) and writes:

    chunks/chunk_embeddings.npy   shape (n_chunks, 384)  float32
    chunks/chunk_ids.json         ["chunk_0000", "chunk_0001", …]

  This phase simply loads those pre-computed vectors and upserts them
  straight into ChromaDB via the LangChain Chroma wrapper — no model
  inference is repeated.

LangChain usage here
--------------------
  * HuggingFaceEmbeddings  — registered as the collection's embedding
    function so that rag_query.py can later open the same collection and
    the retriever knows how to embed queries at search time.
  * langchain_chroma.Chroma — creates / opens the persistent collection
    and exposes .as_retriever() for Phase 5.
  * Direct upsert via vectorstore._collection (underlying chromadb client)
    is used only to inject the pre-computed vectors; all other interactions
    go through the LangChain API.

Fallback
--------
  If chunk_embeddings.npy is absent (old Phase 3 run), chunk texts are
  encoded on-the-fly via the same model (no accuracy difference).

CLI usage:
  python embed_and_store.py
  python embed_and_store.py --reset
  python embed_and_store.py --collection my_collection
"""

import argparse
import json
import os
import sys

import numpy as np

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
# CONSTANTS  (legacy fallback paths)
# ──────────────────────────────────────────────
CHUNKS_JSON     = os.path.join("chunks", "chunks.json")
CHUNK_EMBS_NPY  = os.path.join("chunks", "chunk_embeddings.npy")
CHUNK_IDS_JSON  = os.path.join("chunks", "chunk_ids.json")
CHROMA_DIR      = "chroma_db"
COLLECTION_NAME = "video_transcript"
EMBED_MODEL     = "BAAI/bge-small-en-v1.5"
EMBED_DIM       = 384
BATCH_SIZE      = 64


def _video_embed_paths(video_id: str) -> dict:
    """Return per-video artifact paths and the Chroma collection name."""
    c = os.path.join("videos", video_id, "chunks")
    return {
        "chunks_json":    os.path.join(c, "chunks.json"),
        "chunk_embs_npy": os.path.join(c, "chunk_embeddings.npy"),
        "chunk_ids_json": os.path.join(c, "chunk_ids.json"),
        "collection":     f"vid_{video_id}",
    }


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
# def _load_chunks(path: str) -> list[dict]:
#     print(f"[INFO] Loading chunks: {path}")
#     if not os.path.exists(path):
#         print(f"\n⛔  File not found: {path}\n"
#               f"   Run Phase 3 first: python chunk_transcript.py\n")
#         sys.exit(1)
#     with open(path, encoding="utf-8") as f:
#         data = json.load(f)
#     chunks = data.get("chunks", [])
#     print(f"[INFO] Loaded {len(chunks)} chunks")
#     return chunks

def _load_chunks(path: str) -> list[dict]:
    print(f"[INFO] Loading chunks: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        chunks = data
    elif isinstance(data, dict):
        chunks = data.get("chunks", [])
    else:
        raise ValueError(f"Unexpected chunks format: {type(data)}")

    print(f"[INFO] Loaded {len(chunks)} chunks")
    return chunks


def _load_langchain_embeddings():
    """
    LangChain HuggingFaceEmbeddings wrapper — stored as the collection's
    embedding function so the retriever in Phase 5 uses the same model
    to encode queries automatically.
    """
    print(f"[INFO] Loading LangChain HuggingFaceEmbeddings ({EMBED_MODEL}) …")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        print("\n⛔  langchain-huggingface not installed.\n"
              "   Run: pip install langchain-huggingface\n")
        sys.exit(1)
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _init_vectorstore(embedding_fn, collection_name: str, reset: bool):
    """Create or open the LangChain Chroma vectorstore."""
    print(f"[INFO] Initialising LangChain Chroma at: {os.path.abspath(CHROMA_DIR)}")
    try:
        from langchain_chroma import Chroma
        import chromadb
    except ImportError:
        print("\n⛔  langchain-chroma / chromadb not installed.\n"
              "   Run: pip install langchain-chroma chromadb\n")
        sys.exit(1)

    os.makedirs(CHROMA_DIR, exist_ok=True)

    if reset:
        try:
            raw = chromadb.PersistentClient(path=CHROMA_DIR)
            raw.delete_collection(collection_name)
            print(f"[INFO] Dropped existing collection '{collection_name}'")
        except Exception:
            pass

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_fn,
        persist_directory=CHROMA_DIR,
        collection_metadata={
            "hnsw:space":      "cosine",
            "embedding_model": EMBED_MODEL,
            "embedding_dim":   str(EMBED_DIM),
        },
    )
    print(f"[INFO] Collection '{collection_name}' ready  "
          f"(existing docs: {vectorstore._collection.count()})")
    return vectorstore


def _already_indexed(vectorstore, ids: list[str]) -> set[str]:
    try:
        return set(vectorstore._collection.get(ids=ids, include=[])["ids"])
    except Exception:
        return set()


def _encode_fallback(chunks: list[dict]) -> np.ndarray:
    """Encode chunk texts when pre-computed .npy is absent (backward compat)."""
    print("[INFO] No pre-computed embeddings — encoding via sentence-transformers …")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("\n⛔  sentence-transformers not installed.\n"
              "   Run: pip install sentence-transformers\n")
        sys.exit(1)
    model = SentenceTransformer(EMBED_MODEL)
    return model.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=BATCH_SIZE,
    )


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def embed_and_store(collection_name: str = COLLECTION_NAME, reset: bool = False,
                    video_id: str = "", source_type: str = "video", pdf_id: str = "") -> None:
    _check_venv()

    # Resolve per-video or per-PDF paths
    if video_id:
        p = _video_embed_paths(video_id)
        _chunks_json    = p["chunks_json"]
        _chunk_embs_npy = p["chunk_embs_npy"]
        _chunk_ids_json = p["chunk_ids_json"]
        collection_name = p["collection"]
        source_type     = "video"
    elif pdf_id:
        _chunks_json    = os.path.join("pdfs", pdf_id, "chunks", "chunks.json")
        _chunk_embs_npy = os.path.join("pdfs", pdf_id, "chunks", "chunk_embeddings.npy")
        _chunk_ids_json = os.path.join("pdfs", pdf_id, "chunks", "chunk_ids.json")
        collection_name = f"pdf_{pdf_id}"
        source_type     = "pdf"
    else:
        _chunks_json    = CHUNKS_JSON
        _chunk_embs_npy = CHUNK_EMBS_NPY
        _chunk_ids_json = CHUNK_IDS_JSON

    print(f"\n{'═'*60}")
    print(f"  Phase 4 — ChromaDB Storage  (LangChain, zero re-encoding)")
    print(f"{'═'*60}\n")
    print(f"  Embed model : {EMBED_MODEL}")
    print(f"  Collection  : {collection_name}")
    print(f"  Source type : {source_type}")
    print(f"  Reset DB    : {reset}\n")

    chunks = _load_chunks(_chunks_json)

    # ── Try to load pre-computed chunk embeddings (from Phase 3) ──
    precomputed          = False
    chunk_embs: np.ndarray = np.array([])
    chunk_ids_pre: list[str] = []

    if os.path.exists(_chunk_embs_npy) and os.path.exists(_chunk_ids_json):
        try:
            chunk_embs    = np.load(_chunk_embs_npy)
            with open(_chunk_ids_json, encoding="utf-8") as f:
                chunk_ids_pre = json.load(f)
            if chunk_embs.shape[0] == len(chunks):
                precomputed = True
                print(f"[INFO] ✅  Pre-computed embeddings loaded "
                      f"(shape={chunk_embs.shape}) — skipping re-encoding")
            else:
                print(f"[WARN] Embedding count mismatch "
                      f"({chunk_embs.shape[0]} vs {len(chunks)}) — will re-encode")
        except Exception as exc:
            print(f"[WARN] Could not load pre-computed embeddings ({exc}) — will re-encode")

    if not precomputed:
        chunk_embs    = _encode_fallback(chunks)
        chunk_ids_pre = [f"chunk_{c['chunk_id']:04d}" for c in chunks]

    # ── Build LangChain vectorstore ────────────────────────────
    embedding_fn = _load_langchain_embeddings()
    vectorstore  = _init_vectorstore(embedding_fn, collection_name, reset)

    print(type(chunks[0]["chunk_id"]))
    print(chunks[0]["chunk_id"])

    # ── Skip already-indexed chunks ────────────────────────────
    # all_ids    = [f"chunk_{c['chunk_id']:04d}" for c in chunks]
    all_ids = [str(c["chunk_id"]) for c in chunks]
    indexed    = _already_indexed(vectorstore, all_ids)
    to_process = [(i, c) for i, c in enumerate(chunks) if all_ids[i] not in indexed]

    if not to_process:
        print(f"\n✅  All {len(chunks)} chunks already indexed — nothing to do.\n")
        return

    print(f"[INFO] Upserting {len(to_process)} chunks "
          f"(skipping {len(indexed)} already indexed) …\n")

    total_batches = (len(to_process) - 1) // BATCH_SIZE + 1

    for batch_num, batch_start in enumerate(range(0, len(to_process), BATCH_SIZE), 1):
        batch    = to_process[batch_start : batch_start + BATCH_SIZE]
        b_orig_i = [i for i, _ in batch]
        b_chunks = [c for _, c in batch]
        # b_ids    = [f"chunk_{c['chunk_id']:04d}" for c in b_chunks]
        b_ids = [str(c["chunk_id"]) for c in b_chunks]
        b_texts  = [c["text"] for c in b_chunks]
        
        # Build metadata based on source type (video or PDF)
        if source_type == "video":
            b_metas  = [
                {
                    "chunk_id":  c["chunk_id"],
                    "start_sec": round(c["start"], 2),
                    "end_sec":   round(c["end"],   2),
                    "words":     c["words"],
                    "sentences": c.get("sentences", 0),
                    "duration":  round(c["end"] - c["start"], 2),
                    "source_type": "video",
                }
                for c in b_chunks
            ]
        else:  # PDF
            b_metas  = [
                {
                    "chunk_id":  c["chunk_id"],
                    "start_page": c.get("start_page", 1),
                    "end_page":   c.get("end_page", 1),
                    "page_spans": c.get("page_spans", "Page 1"),
                    "words":      c["word_count"],
                    "source_type": "pdf",
                    "chunk_type": c.get("chunk_type", "text"),
                    **({"image_path": c.get("image_path")} if c.get("image_path") else {}),
                }
                for c in b_chunks
            ]
        
        b_embs = [chunk_embs[i].tolist() for i in b_orig_i]

        print(f"  Batch {batch_num}/{total_batches} — "
              f"chunks {b_chunks[0]['chunk_id']}–{b_chunks[-1]['chunk_id']} …",
              end=" ", flush=True)

        # Inject pre-computed vectors directly via underlying chromadb client;
        # this bypasses LangChain's own encode step while keeping the
        # LangChain wrapper valid for retrieval in Phase 5.
        vectorstore._collection.upsert(
            ids        = b_ids,
            embeddings = b_embs,
            documents  = b_texts,
            metadatas  = b_metas,
        )
        print("done ✅")

    final_count = vectorstore._collection.count()

    print(f"\n{'═'*60}")
    print(f"  ✅  Phase 4 complete!")
    print(f"   Chunks stored    : {len(to_process)}")
    print(f"   Total in DB      : {final_count}")
    print(f"   Re-encoded       : {'No — reused Phase 3 embeddings 🚀' if precomputed else 'Yes (fallback)'}")
    print(f"   Embed model      : {EMBED_MODEL}  ({EMBED_DIM}-dim)")
    print(f"   ChromaDB path    : {os.path.abspath(CHROMA_DIR)}")
    print(f"   Collection       : {collection_name}")
    print(f"{'═'*60}\n")

    # ── Sanity check via LangChain retriever ───────────────────
    print("[INFO] Sanity query via LangChain retriever: 'What is generative AI?' …\n")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    results   = retriever.invoke("What is generative AI?")
    for i, doc in enumerate(results, 1):
        m = doc.metadata
        # print(f"  [{i}] chunk_{m['chunk_id']:04d}  "
        print(f"[{i}] {m['chunk_id']}"
              f"| {m['start_sec']:.0f}s → {m['end_sec']:.0f}s\n"
              f"      {doc.page_content[:140]}…\n")
    # for i, doc in enumerate(results, 1):
    #     m = doc.metadata

    #     if m.get("source_type") == "pdf":
    #       print(
    #         f"[{i}] {m['chunk_id']} "
    #         f"| Page {m.get('start_page', 1)}\n"
    #         f"      {doc.page_content[:140]}...\n"
    #       )
    #     else:
    #       print(
    #         f"[{i}] {m['chunk_id']} "
    #         f"| {m.get('start_sec', 0):.0f}s → {m.get('end_sec', 0):.0f}s\n"
    #         f"      {doc.page_content[:140]}...\n"
    #     )
# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 4 — Store chunks in ChromaDB (LangChain, zero re-encoding)"
    )
    parser.add_argument(
        "--video_id",
        default="",
        help="Unique video id — reads videos/<id>/chunks/, uses collection vid_<id>",
    )
    parser.add_argument(
        "--pdf_id",
        default="",
        help="Unique PDF id — reads pdfs/<id>/chunks/, uses collection pdf_<id>",
    )
    parser.add_argument(
        "--collection", default=COLLECTION_NAME,
        help=f"ChromaDB collection (ignored when --video_id or --pdf_id set, default: {COLLECTION_NAME})"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Drop and recreate the collection before inserting"
    )
    args = parser.parse_args()
    
    source_type = "pdf" if args.pdf_id else "video"
    embed_and_store(
        collection_name=args.collection,
        reset=args.reset,
        video_id=args.video_id,
        pdf_id=args.pdf_id,
        source_type=source_type
    )
