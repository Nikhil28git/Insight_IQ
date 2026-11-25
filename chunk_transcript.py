"""
Phase 3 — Semantic Chunking of Transcript
==========================================
Strategy: Embedding-based topic boundary detection

How it works:
  1. Load segments from transcripts/transcript.json
  2. Group short segments into "sentences" (by pause / word count)
  3. Use LangChain SemanticChunker (HuggingFaceEmbeddings all-MiniLM-L6-v2)
     to detect topic boundaries via percentile-based cosine distance
     → replaces manual embed / cosine-sim / percentile-threshold logic
  4. Map SemanticChunker output back to timestamped sentence units
  5. Apply boundary-aware overlap (default 15%) so boundary-spanning
     answers are fully captured by at least one chunk
  6. Enforce min/max token limits — split oversized chunks, merge tiny ones
  7. Aggregate sentence embeddings → chunk_embeddings.npy (no re-encoding
     in Phase 4; same model object reused from step 3)

Outputs:
  chunks/chunks.json   — structured chunks with timestamps + metadata
  chunks/chunks.txt    — human-readable view

CLI usage:
  python chunk_transcript.py
  python chunk_transcript.py --min_words 80 --max_words 300 --threshold_pct 75 --overlap_pct 0.15
"""

import argparse
import json
import os
import sys
import re

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
# Legacy fallback paths
INPUT_JSON   = os.path.join("transcripts", "transcript.json")
OUTPUT_DIR   = "chunks"
OUTPUT_JSON  = os.path.join(OUTPUT_DIR, "chunks.json")
OUTPUT_TXT   = os.path.join(OUTPUT_DIR, "chunks.txt")

DEFAULT_MIN_WORDS      = 80
DEFAULT_MAX_WORDS      = 350
DEFAULT_THRESHOLD_PCT  = 75
DEFAULT_OVERLAP_PCT    = 0.15   # 15% boundary-aware overlap (mentor recommendation)
SENTENCE_GAP_SEC       = 1.5
SENTENCE_MAX_WORDS     = 60


def _video_chunk_paths(video_id: str) -> dict:
    """Return all I/O paths for a given video_id."""
    base = os.path.join("videos", video_id)
    c    = os.path.join(base, "chunks")
    return {
        "input_json":  os.path.join(base, "transcript", "transcript.json"),
        "output_dir":  c,
        "output_json": os.path.join(c, "chunks.json"),
        "output_txt":  os.path.join(c, "chunks.txt"),
        "emb_path":    os.path.join(c, "chunk_embeddings.npy"),
        "ids_path":    os.path.join(c, "chunk_ids.json"),
    }


# ──────────────────────────────────────────────
# STEP 1 — Load transcript segments
# ──────────────────────────────────────────────
def load_segments(path: str) -> list[dict]:
    print(f"[INFO] Loading transcript: {path}")
    if not os.path.exists(path):
        print(f"\n⛔  File not found: {path}\n"
              f"   Run Phase 2 first: python transcribe.py --wav audio_output/audio.wav\n")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    segs = data.get("segments", [])
    print(f"[INFO] Loaded {len(segs)} segments  |  {len(data['text'].split())} total words")
    return segs


# ──────────────────────────────────────────────
# STEP 2 — Group raw segments into "sentences"
# A sentence = natural speech unit for embedding
# ──────────────────────────────────────────────
def group_into_sentences(segments: list[dict]) -> list[dict]:
    """
    Merge consecutive short segments into sentence-level units.
    Split on:
      - sentence-ending punctuation  (. ! ?)
      - long pause (gap > SENTENCE_GAP_SEC)
      - word-count ceiling (SENTENCE_MAX_WORDS)
    """
    sentences = []
    current_texts  = []
    current_segs   = []
    current_words  = 0

    for i, seg in enumerate(segments):
        text  = seg["text"].strip()
        words = len(text.split())

        current_texts.append(text)
        current_segs.append(seg)
        current_words += words

        # Determine if we should close this sentence
        ends_sentence = bool(re.search(r"[.!?]$", text))
        word_limit    = current_words >= SENTENCE_MAX_WORDS

        # Check pause to next segment
        long_pause = False
        if i + 1 < len(segments):
            gap = segments[i + 1]["start"] - seg["end"]
            long_pause = gap >= SENTENCE_GAP_SEC

        if ends_sentence or word_limit or long_pause:
            sentences.append({
                "text":     " ".join(current_texts),
                "start":    current_segs[0]["start"],
                "end":      current_segs[-1]["end"],
                "words":    current_words,
                "segments": list(current_segs),
            })
            current_texts  = []
            current_segs   = []
            current_words  = 0

    # Flush any remaining
    if current_texts:
        sentences.append({
            "text":     " ".join(current_texts),
            "start":    current_segs[0]["start"],
            "end":      current_segs[-1]["end"],
            "words":    current_words,
            "segments": list(current_segs),
        })

    print(f"[INFO] Grouped into {len(sentences)} sentence units")
    return sentences


# ──────────────────────────────────────────────
# STEP 3 — LangChain SemanticChunker boundary detection
# Replaces: manual embed → cosine-sim → percentile-threshold → build_chunks
# ──────────────────────────────────────────────
def _langchain_semantic_boundaries(
    sentences:    list[dict],
    hf_embeddings,
    threshold_pct: int = DEFAULT_THRESHOLD_PCT,
) -> list[dict]:
    """
    Use LangChain SemanticChunker (HuggingFaceEmbeddings) to detect topic
    boundaries and group sentences into semantic chunks.

    Internally:
      - SemanticChunker embeds each sentence with hf_embeddings
      - Computes cosine DISTANCE between consecutive sentence embeddings
      - Splits where distance exceeds the (threshold_pct) percentile
        → this is equivalent to the old manual percentile-threshold logic
      - Returns LangChain Documents, one per semantic topic block

    Timestamp preservation:
      - After SemanticChunker produces text chunks, each chunk is mapped
        back to the original timestamped sentences via a sequential
        greedy word-count scan (contiguous order is guaranteed because
        SemanticChunker never reorders sentences).
      - start, end, sentence_idxs are all faithfully preserved for the
        Streamlit UI timestamp links and merge_windows() in rag_query.py.
    """
    from langchain_experimental.text_splitter import SemanticChunker

    chunker = SemanticChunker(
        embeddings=hf_embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=threshold_pct,
    )

    full_text = " ".join(s["text"] for s in sentences)
    print(f"[INFO] Running LangChain SemanticChunker "
          f"(breakpoint_threshold_type=percentile, amount={threshold_pct}) …")
    lc_docs = chunker.create_documents([full_text])
    print(f"[INFO] SemanticChunker produced {len(lc_docs)} raw topic blocks")

    # ── Map LangChain chunks → our timestamped sentences ──────────────────
    # SemanticChunker preserves sentence order, so we do a greedy forward
    # scan: for each output chunk, consume sentences until accumulated word
    # count reaches the chunk's word count.
    chunks      = []
    sent_cursor = 0

    for chunk_idx, doc in enumerate(lc_docs):
        lc_word_count = len(doc.page_content.split())
        matched_idxs  = []
        accumulated   = 0

        while sent_cursor < len(sentences) and accumulated < lc_word_count:
            matched_idxs.append(sent_cursor)
            accumulated += sentences[sent_cursor]["words"]
            sent_cursor += 1

        if not matched_idxs:
            continue

        chunk_sents = [sentences[i] for i in matched_idxs]
        text = " ".join(s["text"] for s in chunk_sents)
        chunks.append({
            "chunk_id":      chunk_idx,
            "start":         chunk_sents[0]["start"],    # ← timestamp preserved
            "end":           chunk_sents[-1]["end"],      # ← timestamp preserved
            "text":          text,
            "words":         len(text.split()),
            "sentences":     len(matched_idxs),
            "sentence_idxs": matched_idxs,               # ← needed for emb reuse
        })

    # Flush any tail sentences not yet consumed
    if sent_cursor < len(sentences):
        remaining = list(range(sent_cursor, len(sentences)))
        if chunks:
            last  = chunks[-1]
            extra = " ".join(sentences[i]["text"] for i in remaining)
            last["text"]          += " " + extra
            last["words"]          = len(last["text"].split())
            last["end"]            = sentences[remaining[-1]]["end"]
            last["sentences"]     += len(remaining)
            last["sentence_idxs"].extend(remaining)
        else:
            sents = [sentences[i] for i in remaining]
            text  = " ".join(s["text"] for s in sents)
            chunks.append({
                "chunk_id":      0,
                "start":         sents[0]["start"],
                "end":           sents[-1]["end"],
                "text":          text,
                "words":         len(text.split()),
                "sentences":     len(remaining),
                "sentence_idxs": remaining,
            })

    return chunks


def _embed_sentences_lc(sentences: list[dict], hf_embeddings) -> "np.ndarray":
    """
    Embed sentence texts using the SAME HuggingFaceEmbeddings instance that
    was passed to SemanticChunker — no second model load.
    Returns a (N, dim) float32 numpy array used to build chunk_embeddings.npy
    via mean-pooling of sentence vectors per chunk (Phase 4 reuse).
    """
    import numpy as np
    texts = [s["text"] for s in sentences]
    print(f"[INFO] Embedding {len(texts)} sentences for chunk_embeddings.npy …")
    vecs = hf_embeddings.embed_documents(texts)
    arr  = np.array(vecs, dtype=np.float32)
    print(f"[INFO] Sentence embeddings shape: {arr.shape}")
    return arr


# ──────────────────────────────────────────────
# STEP 4 — Boundary-aware overlap
# Extends each chunk by overlap_pct of its sentences on both sides
# ──────────────────────────────────────────────
def _apply_boundary_overlap(
    chunks:      list[dict],
    sentences:   list[dict],
    overlap_pct: float = DEFAULT_OVERLAP_PCT,
) -> list[dict]:
    """
    Extend each chunk's sentence window by (overlap_pct * chunk_length)
    sentences on BOTH sides, clamped to corpus boundaries.

    Example: chunk has 10 sentences, overlap_pct=0.15
      → extend by max(1, round(10*0.15)) = 2 sentences left + 2 right.

    Why this matters for MMR retrieval:
      - Non-overlapping chunks can split a relevant answer across two chunks.
      - Overlap ensures boundary-spanning content appears in at least one
        retrieved chunk, giving MMR (fetch_k=20, top_k=4) better candidates
        to select diverse but complete windows from.
    """
    if not overlap_pct:
        return chunks

    total_sents = len(sentences)
    new_chunks  = []

    for c in chunks:
        s_idxs = c.get("sentence_idxs", [])
        if not s_idxs:
            new_chunks.append(c)
            continue

        extend    = max(1, int(round(len(s_idxs) * overlap_pct)))
        new_start = max(0, s_idxs[0] - extend)
        new_end   = min(total_sents, s_idxs[-1] + 1 + extend)
        new_idxs  = list(range(new_start, new_end))

        texts    = [sentences[i]["text"] for i in new_idxs]
        new_text = " ".join(texts)
        new_chunks.append({
            "chunk_id":      c["chunk_id"],
            "start":         sentences[new_idxs[0]]["start"],
            "end":           sentences[new_idxs[-1]]["end"],
            "text":          new_text,
            "words":         len(new_text.split()),
            "sentences":     len(new_idxs),
            "sentence_idxs": new_idxs,
        })

    # Re-number so chunk_ids are contiguous (required for emb_arr indexing)
    for i, c in enumerate(new_chunks):
        c["chunk_id"] = i

    return new_chunks


# ──────────────────────────────────────────────
# STEP 6 — Enforce min / max word limits
# ──────────────────────────────────────────────
def enforce_limits(chunks: list[dict], sentences: list[dict],
                   boundaries: list[int],
                   min_words: int, max_words: int) -> list[dict]:
    """
    Post-process chunks:
      - Merge tiny chunks (< min_words) with their neighbour
      - Split oversized chunks (> max_words) by halving
    """
    print(f"[INFO] Enforcing limits — min: {min_words} words, max: {max_words} words …")

    # ── Merge tiny chunks into previous ──
    merged = []
    for chunk in chunks:
        if merged and chunk["words"] < min_words:
            prev = merged[-1]
            prev["text"]          = prev["text"] + " " + chunk["text"]
            prev["words"]         = len(prev["text"].split())
            prev["end"]           = chunk["end"]
            prev["sentences"]     = prev["sentences"] + chunk["sentences"]
            prev["sentence_idxs"] = prev.get("sentence_idxs", []) + chunk.get("sentence_idxs", [])
        else:
            merged.append(dict(chunk))

    # ── Split oversized chunks ──
    final = []
    for chunk in merged:
        if chunk["words"] <= max_words:
            final.append(chunk)
        else:
            # Split by sentences roughly in half
            words_list = chunk["text"].split()
            mid = len(words_list) // 2
            # Find a sentence boundary near mid
            split_text  = " ".join(words_list)
            # Approximate split at mid-word sentence boundary
            sentences_in_chunk = re.split(r"(?<=[.!?])\s+", split_text)
            if len(sentences_in_chunk) < 2:
                # No sentence boundary found — split by word count
                part1 = " ".join(words_list[:mid])
                part2 = " ".join(words_list[mid:])
            else:
                # Accumulate sentences until we pass mid
                acc, accumulated = [], 0
                for sent in sentences_in_chunk:
                    acc.append(sent)
                    accumulated += len(sent.split())
                    if accumulated >= mid:
                        break
                part1 = " ".join(acc)
                part2 = " ".join(sentences_in_chunk[len(acc):])

            s_idxs = chunk.get("sentence_idxs", [])
            mid_s  = max(1, len(s_idxs) // 2)
            if part1.strip():
                final.append({
                    "chunk_id":      -1,
                    "start":         chunk["start"],
                    "end":           chunk["end"],   # approximate
                    "text":          part1.strip(),
                    "words":         len(part1.split()),
                    "sentences":     chunk["sentences"],
                    "sentence_idxs": s_idxs[:mid_s],
                })
            if part2.strip():
                final.append({
                    "chunk_id":      -1,
                    "start":         chunk["start"],  # approximate
                    "end":           chunk["end"],
                    "text":          part2.strip(),
                    "words":         len(part2.split()),
                    "sentences":     chunk["sentences"],
                    "sentence_idxs": s_idxs[mid_s:],
                })

    # Re-number
    for i, c in enumerate(final):
        c["chunk_id"] = i

    return final


# ──────────────────────────────────────────────
# STEP 7 — Save outputs
# ──────────────────────────────────────────────
def save_outputs(chunks: list[dict], output_dir=None, output_json=None, output_txt=None) -> None:
    _out_dir  = output_dir  if output_dir  is not None else OUTPUT_DIR
    _out_json = output_json if output_json is not None else OUTPUT_JSON
    _out_txt  = output_txt  if output_txt  is not None else OUTPUT_TXT

    os.makedirs(_out_dir, exist_ok=True)

    # ── JSON ─────────────────────────────────
    output = {
        "total_chunks": len(chunks),
        "chunks": chunks,
    }
    with open(_out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[INFO] JSON saved → {os.path.abspath(_out_json)}")

    # ── TXT ──────────────────────────────────
    lines = []
    for c in chunks:
        duration = c['end'] - c['start']
        lines.append(
            f"{'═'*60}\n"
            f"CHUNK {c['chunk_id']:03d}  |  "
            f"{c['start']:.1f}s → {c['end']:.1f}s  "
            f"({duration/60:.1f} min)  |  {c['words']} words\n"
            f"{'─'*60}\n"
            f"{c['text']}\n"
        )
    with open(_out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[INFO] TXT  saved → {os.path.abspath(_out_txt)}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def chunk_transcript(
    min_words:     int   = DEFAULT_MIN_WORDS,
    max_words:     int   = DEFAULT_MAX_WORDS,
    threshold_pct: int   = DEFAULT_THRESHOLD_PCT,
    video_id:      str   = "",
    overlap_pct:   float = DEFAULT_OVERLAP_PCT,
) -> list[dict]:

    _check_venv()

    # Resolve paths
    if video_id:
        p           = _video_chunk_paths(video_id)
        input_json  = p["input_json"]
        output_dir  = p["output_dir"]
        output_json = p["output_json"]
        output_txt  = p["output_txt"]
        emb_path    = p["emb_path"]
        ids_path    = p["ids_path"]
    else:
        input_json  = INPUT_JSON
        output_dir  = OUTPUT_DIR
        output_json = OUTPUT_JSON
        output_txt  = OUTPUT_TXT
        emb_path    = os.path.join(OUTPUT_DIR, "chunk_embeddings.npy")
        ids_path    = os.path.join(OUTPUT_DIR, "chunk_ids.json")

    print(f"\n{'═'*60}")
    print(f"  Phase 3 — Semantic Chunking")
    print(f"{'═'*60}\n")
    print(f"  Settings: min_words={min_words}  max_words={max_words}  "
          f"threshold_pct={threshold_pct}  overlap_pct={overlap_pct}\n")

    # 1. Load
    segments  = load_segments(input_json)

    # 2. Group into sentences
    sentences = group_into_sentences(segments)

    # 3. Load HuggingFaceEmbeddings once — shared by SemanticChunker (boundary
    #    detection) AND sentence embedding aggregation (chunk_embeddings.npy).
    #    No duplicate model loads.
    from langchain_huggingface import HuggingFaceEmbeddings
    print("[INFO] Loading HuggingFaceEmbeddings (BAAI/bge-small-en-v1.5) …")
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},  # BGE models need this
    )

    # 4. LangChain SemanticChunker — detects topic boundaries + builds chunks
    #    (replaces manual embed → cosine-sim → percentile → build_chunks)
    raw_chunks = _langchain_semantic_boundaries(sentences, hf_embeddings, threshold_pct)
    print(f"[INFO] Raw chunks: {len(raw_chunks)}  |  "
          f"avg {sum(c['words'] for c in raw_chunks) // max(len(raw_chunks), 1)} words each")

    # 5. Enforce min/max word limits (unchanged)
    final_chunks = enforce_limits(raw_chunks, sentences, [], min_words, max_words)

    # 6. Boundary-aware overlap — extend each chunk by overlap_pct sentences
    #    on both sides so boundary-spanning answers land in one chunk.
    #    Mentor recommended 10–20%; default is 15%.
    if overlap_pct > 0:
        final_chunks = _apply_boundary_overlap(final_chunks, sentences, overlap_pct)
        print(f"[INFO] Overlap applied ({overlap_pct:.0%} per side): "
              f"{len(final_chunks)} chunks after overlap")

    # 7. Embed sentences → aggregate per-chunk embeddings (no re-encoding in Phase 4)
    try:
        import numpy as np
        embeddings = _embed_sentences_lc(sentences, hf_embeddings)
        os.makedirs(output_dir, exist_ok=True)
        n        = len(final_chunks)
        dim      = embeddings.shape[1]
        emb_arr  = np.zeros((n, dim), dtype=np.float32)
        id_list  = []
        for c in final_chunks:
            cid    = c["chunk_id"]
            id_list.append(f"chunk_{cid:04d}")
            s_idxs = [i for i in c.get("sentence_idxs", []) if i < len(embeddings)]
            if s_idxs:
                mean         = embeddings[s_idxs].mean(axis=0)
                norm         = np.linalg.norm(mean) + 1e-9
                emb_arr[cid] = (mean / norm).astype(np.float32)
        np.save(emb_path, emb_arr)
        with open(ids_path, "w", encoding="utf-8") as _f:
            json.dump(id_list, _f, indent=2)
        print(f"[INFO] Saved chunk embeddings → {os.path.abspath(emb_path)}  "
              f"({n} chunks, dim={dim}) — no re-encoding needed in Phase 4")
    except Exception as _exc:
        print(f"[WARN] Could not save chunk embeddings: {_exc}")

    # 8. Save outputs (unchanged — same JSON structure, downstream compatible)
    save_outputs(final_chunks, output_dir, output_json, output_txt)

    # ── Summary ──────────────────────────────
    word_counts = [c["words"] for c in final_chunks]
    total_words = sum(word_counts)
    print(f"\n{'═'*60}")
    print(f"  ✅  Chunking complete!")
    print(f"   Total chunks : {len(final_chunks)}")
    print(f"   Total words  : {total_words:,}")
    print(f"   Avg words    : {total_words // len(final_chunks)}")
    print(f"   Min words    : {min(word_counts)}")
    print(f"   Max words    : {max(word_counts)}")
    print(f"   JSON         : {os.path.abspath(OUTPUT_JSON)}")
    print(f"   TXT          : {os.path.abspath(OUTPUT_TXT)}")
    print(f"{'═'*60}\n")

    return final_chunks


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 3 — Semantic chunking of WhisperX transcript"
    )
    parser.add_argument(
        "--min_words", type=int, default=DEFAULT_MIN_WORDS,
        help=f"Minimum words per chunk (default: {DEFAULT_MIN_WORDS})"
    )
    parser.add_argument(
        "--max_words", type=int, default=DEFAULT_MAX_WORDS,
        help=f"Maximum words per chunk (default: {DEFAULT_MAX_WORDS})"
    )
    parser.add_argument(
        "--threshold_pct", type=int, default=DEFAULT_THRESHOLD_PCT,
        help=(
            f"Percentile of similarity drops used as topic-boundary threshold "
            f"(default: {DEFAULT_THRESHOLD_PCT}). "
            f"Higher = more chunks (more sensitive). Lower = fewer, bigger chunks."
        )
    )
    parser.add_argument(
        "--overlap_pct", type=float, default=DEFAULT_OVERLAP_PCT,
        help=(
            f"Boundary-aware overlap as a fraction of each chunk's sentences "
            f"(default: {DEFAULT_OVERLAP_PCT}). "
            f"0.15 = 15%% overlap on each side. 0 = no overlap."
        )
    )
    args = parser.parse_args()

    chunk_transcript(
        min_words=args.min_words,
        max_words=args.max_words,
        threshold_pct=args.threshold_pct,
        overlap_pct=args.overlap_pct,
    )
