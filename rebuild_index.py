"""
Rebuild the ChromaDB search index from the chunk files already on disk.

chroma_db/ is not committed to git, so run this once after cloning
(or whenever chroma_db/ is deleted). It uses local embeddings only —
no API calls.

Run:
    python rebuild_index.py
"""

import json
import os
import subprocess
import sys

import numpy as np

from process_pdf import load_chroma_client, store_chunks_in_chromadb

ROOT = os.path.dirname(os.path.abspath(__file__))


def rebuild_videos() -> None:
    videos_dir = os.path.join(ROOT, "videos")
    if not os.path.isdir(videos_dir):
        return
    for video_id in sorted(os.listdir(videos_dir)):
        if not os.path.exists(os.path.join(videos_dir, video_id, "chunks", "chunks.json")):
            continue
        print(f"[video] {video_id}")
        subprocess.run(
            [sys.executable, "embed_and_store.py", "--video_id", video_id, "--reset"],
            cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
        )


def rebuild_pdfs() -> None:
    pdfs_dir = os.path.join(ROOT, "pdfs")
    if not os.path.isdir(pdfs_dir):
        return
    client = load_chroma_client()
    existing = {c.name for c in client.list_collections()}
    for pdf_id in sorted(os.listdir(pdfs_dir)):
        chunks_dir = os.path.join(pdfs_dir, pdf_id, "chunks")
        if not os.path.exists(os.path.join(chunks_dir, "chunks.json")):
            continue
        with open(os.path.join(chunks_dir, "chunks.json"), encoding="utf-8") as f:
            chunks = json.load(f)
        embeddings = np.load(os.path.join(chunks_dir, "chunk_embeddings.npy"))
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb.tolist()
        if f"pdf_{pdf_id}" in existing:
            client.delete_collection(f"pdf_{pdf_id}")
        print(f"[pdf]   {pdf_id}")
        store_chunks_in_chromadb(chunks, pdf_id, None, client)


if __name__ == "__main__":
    os.chdir(ROOT)
    rebuild_videos()
    rebuild_pdfs()
    print("\nDone — search index rebuilt in chroma_db/")
