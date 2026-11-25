"""
PDF RAG query module: Hybrid retrieval (Vector + BM25) and LLM answer generation.
Implements table-aware chunks (chunk_type) and removes MMR.
"""

import json
import os
import numpy as np
import re
from typing import List, Dict, Any, Tuple

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from pdf_utils import get_page_for_chunk, format_pdf_source, get_chunks_for_page
from image_description import describe_image_with_gemini, GEMINI_MODEL

# Embedding model (must match chunk_pdf.py)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def load_embeddings():
    """Load HuggingFace embeddings model with normalization."""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )


def load_vectorstore(pdf_id: str, embeddings) -> Chroma:
    """Load a LangChain Chroma vector store wrapping the persisted collection."""
    collection_name = f"pdf_{pdf_id}"

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory="chroma_db",
    )

    try:
        count = vectorstore._collection.count()
        print(f"Loaded collection: {collection_name} ({count} chunks)")
    except Exception:
        print(f"Error: Collection '{collection_name}' not found or inaccessible.")
        raise

    return vectorstore


def load_pdf_chunks(chunks_dir: str) -> Tuple[List[Dict], np.ndarray, List[int]]:
    """Load chunks and embeddings from disk (chunks.json + chunk_embeddings.npy).

    Returns (chunks_list, embeddings_array, page_breaks)
    """
    chunks_file = os.path.join(chunks_dir, 'chunks.json')
    embeddings_file = os.path.join(chunks_dir, 'chunk_embeddings.npy')

    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    embeddings_array = None
    if os.path.exists(embeddings_file):
        try:
            embeddings_array = np.load(embeddings_file)
        except Exception:
            embeddings_array = None

    # Reconstruct page breaks from chunks
    page_breaks = [0]
    # Some chunks (images or synthetic table chunks) may not have start_index/end_index.
    # Sort robustly using get with default 0, and append end positions using sensible defaults.
    def start_key(c):
        try:
            return int(c.get('start_index', 0))
        except Exception:
            return 0

    def end_pos(c):
        try:
            return int(c.get('end_index', c.get('start_index', 0) + len(c.get('text', ''))))
        except Exception:
            return page_breaks[-1]

    for chunk in sorted(chunks, key=start_key):
        page_breaks.append(end_pos(chunk))

    return chunks, embeddings_array, page_breaks


def build_bm25_index(chunks: List[Dict[str, Any]]) -> Tuple[BM25Okapi, List[List[str]]]:
    """Build BM25 index from chunk texts (tokenized lower-case word split).

    Returns BM25 object and tokenized_corpus aligned with chunks.
    """
    tokenized_corpus = [chunk['text'].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus


def bm25_search(query: str, bm25: BM25Okapi, tokenized_corpus: List[List[str]], chunks: List[Dict[str, Any]], k: int = 10) -> List[Dict[str, Any]]:
    """Search BM25 and return top-k chunks with scores and ranks."""
    q_tokens = query.lower().split()
    scores = bm25.get_scores(q_tokens)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

    results = []
    for rank, (idx, score) in enumerate(ranked, start=1):
        chunk = chunks[idx]
        results.append({
            'chunk_id': chunk['chunk_id'],
            'text': chunk['text'],
            'metadata': chunk,
            'bm25_score': float(score),
            'rank': rank,
        })

    return results


def vector_similarity_search(query: str, vectorstore: Chroma, k: int = 10) -> List[Dict[str, Any]]:
    """Perform vector similarity search using LangChain Chroma wrapper.

    Returns list of dicts with chunk_id, text, metadata, and similarity score.
    """
    docs = vectorstore.similarity_search_with_score(query, k=k)
    results = []
    for rank, (doc, score) in enumerate(docs, start=1):
        meta = doc.metadata or {}
        results.append({
            'chunk_id': meta.get('chunk_id'),
            'text': doc.page_content,
            'metadata': meta,
            'similarity': float(score),
            'rank': rank,
        })
    return results


def hybrid_retrieve_chunks(query: str, vectorstore: Chroma, chunks: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    """Hybrid retrieval: vector similarity + BM25 + Reciprocal Rank Fusion (RRF).

    Steps:
      - Vector top-10
      - BM25 top-10
      - RRF combining ranks: score += 1/(60 + rank)
      - Return top-k unique chunks
    """
    # Vector search
    vec_results = vector_similarity_search(query, vectorstore, k=10)

    # BM25 index
    bm25, tokenized = build_bm25_index(chunks)
    bm25_results = bm25_search(query, bm25, tokenized, chunks, k=10)

    # Aggregate via RRF
    agg_scores: Dict[str, float] = {}
    details: Dict[str, Dict[str, Any]] = {}

    for res in vec_results:
        cid = res['chunk_id']
        rank = res['rank']
        agg_scores[cid] = agg_scores.get(cid, 0.0) + 1.0 / (60.0 + rank)
        details.setdefault(cid, {}).update({'vector_rank': rank, 'vector_score': res.get('similarity'), 'text': res.get('text'), 'metadata': res.get('metadata')})

    for res in bm25_results:
        cid = res['chunk_id']
        rank = res['rank']
        agg_scores[cid] = agg_scores.get(cid, 0.0) + 1.0 / (60.0 + rank)
        details.setdefault(cid, {}).update({'bm25_rank': rank, 'bm25_score': res.get('bm25_score'), 'text': res.get('text'), 'metadata': res.get('metadata')})

    # Create ranking
    ranked = sorted(agg_scores.items(), key=lambda x: x[1], reverse=True)

    final = []
    for cid, score in ranked:
        if len(final) >= k:
            break
        info = details.get(cid, {})
        final.append({
            'chunk_id': cid,
            'score': float(score),
            'text': info.get('text'),
            'metadata': info.get('metadata'),
            'vector_rank': info.get('vector_rank'),
            'bm25_rank': info.get('bm25_rank'),
        })

    return final


def format_context_for_llm(retrieved_chunks: List[Dict[str, Any]], pdf_id: str) -> str:
    """
    Format retrieved chunks as context string for LLM.

    Args:
        retrieved_chunks: List of retrieved chunk dicts
        pdf_id: PDF identifier

    Returns:
        Formatted context string
    """
    context_parts = [f"Context from PDF ('{pdf_id}'):\n"]

    for i, chunk in enumerate(retrieved_chunks, 1):
        source = format_pdf_source(
            chunk.get('chunk_id', 'unknown'),
            chunk.get('start_page', 1),
            pdf_id
        )
        # Include chunk type and image marker when present to help LLM reason
        ctype = chunk.get('chunk_type', 'text')
        img = chunk.get('image_path')
        header = f"[{i}] {source} (type={ctype})"
        if img:
            header += f" [image: {img}]"
        context_parts.append(f"\n{header}\n{chunk.get('text','')}")

    return "\n---\n".join(context_parts)


def rag_query_pdf(
    query: str,
    pdf_id: str,
    chunks_dir: str,
    api_key: str = None,
) -> Dict[str, Any]:
    """
    End-to-end PDF RAG query pipeline using LangChain MMR retrieval.

    Args:
        query: User query
        pdf_id: PDF identifier (e.g., "thesis")
        chunks_dir: Directory with chunks.json and embeddings.npy
        api_key: Google Gemini API key

    Returns:
        Dict with 'answer', 'sources' (list of pages), 'retrieved_chunks', 'query_time'
    """
    import time
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"PDF RAG Query: {query[:50]}...")
    print(f"{'='*60}")

    # Load embeddings
    embeddings = load_embeddings()

    # Load chunks from disk (for metadata lookups)
    chunks, embeddings_array, page_breaks = load_pdf_chunks(chunks_dir)

    # Page-aware query detection: if user asked about a specific page like
    # "explain page 62" or "what is on page 62", bypass BM25/Vector/RRF
    m = re.search(r"page\s+(\d+)", query, flags=re.IGNORECASE)
    if m:
        try:
            page_num = int(m.group(1))
        except Exception:
            page_num = None

        if page_num is not None:
            # Retrieve all chunks for the page directly from disk metadata
            page_chunks = get_chunks_for_page(page_num, chunks, page_breaks)
        
            print("\nPAGE CHUNKS FOUND:")
            for c in page_chunks:
              print(
              c.get("chunk_id"),
              c.get("chunk_type"),
              c.get("start_page"),
              c.get("end_page")
              )  

            # Transform into retrieved_chunks format
            retrieved_chunks = []
            for pos, c in enumerate(page_chunks):
                retrieved_chunks.append({
                    'chunk_id': c.get('chunk_id'),
                    'text': c.get('text', ''),
                    'start_index': c.get('start_index'),
                    'end_index': c.get('end_index'),
                    'start_page': c.get('start_page', page_num),
                    'end_page': c.get('end_page', page_num),
                    'page_spans': c.get('page_spans', f"Page {page_num}"),
                    'chunk_type': c.get('chunk_type', 'text'),
                    'image_path': c.get('image_path'),
                    'position': pos,
                    'score': None,
                })

            # # For any image chunks, try on-demand description so the LLM has context
            # for rc in retrieved_chunks:
            #     img_path = rc.get('image_path')
            #     text = (rc.get('text') or "").strip()
            #     should_describe = False
            #     if img_path:
            #         if not text or text.startswith("[IMAGE_DESCRIPTION_PENDING]") or len(text.split()) < 6:
            #             should_describe = True
            #     if should_describe and img_path:
            #         try:
            #             desc, meta = describe_image_with_gemini(img_path, api_key=api_key)
            #             if desc and not desc.startswith("[IMAGE_DESCRIPTION_PENDING]") and desc != "[NO_DESCRIPTION_RETURNED]":
            #                 rc['text'] = desc
            #         except Exception:
            #             pass
            for rc in retrieved_chunks:

                if rc.get("chunk_type") == "image":

                    image_path = rc.get("image_path")

                    if image_path and os.path.exists(image_path):

                        try:

                            desc, _ = describe_image_with_gemini(
                              image_path,
                              api_key=api_key
                            )

                            if desc and not desc.startswith("[IMAGE_DESCRIPTION_PENDING]"):

                                rc["text"] = desc

                        except Exception as e:

                            print(f"Vision error: {e}")

            # Skip the standard retrieval pipeline and go straight to LLM
            context = format_context_for_llm(retrieved_chunks, pdf_id)

            llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, api_key=api_key, temperature=0.3)
            system_prompt ="""
You are an expert technical assistant answering questions about a PDF document.

Use the provided PDF context as the primary source of truth.

Always cite page numbers when information comes from the PDF.

If the PDF contains brief instructions, bullet points, tables,
captions, diagrams, or summaries, you may expand and explain them
using your own engineering, technical, or domain knowledge.

Clearly distinguish between:

1. Information directly found in the PDF.
2. Additional explanation generated from your knowledge.

Never contradict information present in the PDF.

If the requested information is completely absent from the PDF,
state that clearly and then provide helpful general knowledge if relevant.

When users ask for detailed explanations, tutorials, examples,
step-by-step guidance, or learning-oriented answers, provide
comprehensive explanations instead of only repeating the PDF text.
"""

            messages = [
                ("system", system_prompt),
                ("human", f"Context:\n{context}\n\nQuestion: {query}"),
            ]

            print("Generating answer with Gemini (page-mode)...")
            llm_resp = ChatGoogleGenerativeAI(model=GEMINI_MODEL, api_key=api_key, temperature=0.3).invoke(messages)
            answer = llm_resp.content

            sources = sorted(set(chunk['start_page'] for chunk in retrieved_chunks))
            query_time = time.time() - start_time

            return {
                'answer': answer,
                'sources': sources,
                'retrieved_chunks': retrieved_chunks,
                'query_time': query_time,
            }

    # Not a page-specific query: continue with hybrid retrieval

    # Load LangChain Chroma vector store wrapping the existing collection
    vectorstore = load_vectorstore(pdf_id, embeddings)

    # Retrieve relevant chunks using hybrid Vector + BM25 RRF
    hybrid_results = hybrid_retrieve_chunks(query, vectorstore, chunks, k=5)

    # Transform hybrid results into expected retrieved_chunks format
    retrieved_chunks = []
    for pos, item in enumerate(hybrid_results):
        meta = item.get('metadata') or {}
        retrieved_chunks.append({
            'chunk_id': item.get('chunk_id'),
            'text': item.get('text') or meta.get('text') or "",
            'start_index': meta.get('start_index'),
            'end_index': meta.get('end_index'),
            'start_page': meta.get('start_page', 1),
            'end_page': meta.get('end_page', 1),
            'page_spans': meta.get('page_spans', f"Page {meta.get('start_page',1)}"),
            'chunk_type': meta.get('chunk_type', 'text'),
            'image_path': meta.get('image_path'),
            'position': pos,
            'score': item.get('score'),
        })

    # On-demand image description fallback:
    # If a retrieved chunk has an image but its text is a placeholder
    # or too short, call Gemini Vision at query time to get a usable
    # description so the LLM has context for image-only pages.
    # for rc in retrieved_chunks:
    #     img_path = rc.get('image_path')
    #     text = (rc.get('text') or "").strip()
    #     should_describe = False
    #     if img_path:
    #         if not text:
    #             should_describe = True
    #         elif text.startswith("[IMAGE_DESCRIPTION_PENDING]") or text.startswith("[NO_DESCRIPTION_RETURNED]"):
    #             should_describe = True
    #         elif len(text.split()) < 6:
    #             # Very short descriptions are unlikely to be useful
    #             should_describe = True

    #     if should_describe and img_path:
    #         try:
    #             desc, meta = describe_image_with_gemini(img_path, api_key=api_key)
    #             if desc and not desc.startswith("[IMAGE_DESCRIPTION_PENDING]") and desc != "[NO_DESCRIPTION_RETURNED]":
    #                 rc['text'] = desc
    #                 # Update metadata fields where present
    #                 if isinstance(rc.get('metadata'), dict):
    #                     rc['metadata']['text'] = desc
    #                     rc['metadata']['word_count'] = len(desc.split())
    #         except Exception:
    #             # If image description fails, leave original text unchanged
    #             pass
    for rc in retrieved_chunks:

       if rc.get("chunk_type") == "image":

          image_path = rc.get("image_path")

          if image_path and os.path.exists(image_path):

            try:

                desc, _ = describe_image_with_gemini(
                    image_path,
                    api_key=api_key
                )

                if desc and not desc.startswith("[IMAGE_DESCRIPTION_PENDING]"):

                    rc["text"] = desc

            except Exception as e:

                print(f"Vision error: {e}")

    if not retrieved_chunks:
        return {
            'answer': f"No relevant information found in PDF '{pdf_id}' for your query.",
            'sources': [],
            'retrieved_chunks': [],
            'query_time': time.time() - start_time,
        }

    # Format context for LLM
    context = format_context_for_llm(retrieved_chunks, pdf_id)

    # Call Gemini with context
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        api_key=api_key,
        temperature=0.3,
    )

    system_prompt = """You are a helpful assistant answering questions about a PDF document.
Use the provided context to answer the user's question accurately.
Always cite the page numbers where you found the information.
If the information is not in the provided context, say so clearly."""

    messages = [
        ("system", system_prompt),
        ("human", f"Context:\n{context}\n\nQuestion: {query}"),
    ]

    print("Generating answer with Gemini...")
    response = llm.invoke(messages)
    answer = response.content

    # Extract unique pages mentioned
    sources = sorted(set(chunk['start_page'] for chunk in retrieved_chunks))

    query_time = time.time() - start_time

    result = {
        'answer': answer,
        'sources': sources,  # List of page numbers
        'retrieved_chunks': retrieved_chunks,
        'query_time': query_time,
    }

    print(f"\n✅ Query complete in {query_time:.2f}s")
    print(f"Sources: {sources}")

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python rag_query_pdf.py <pdf_id> <chunks_dir> <query>")
        print("Example: python rag_query_pdf.py thesis pdfs/thesis_2024/chunks/ 'What is machine learning?'")
        sys.exit(1)

    pdf_id = sys.argv[1]
    chunks_dir = sys.argv[2]
    query = " ".join(sys.argv[3:])

    result = rag_query_pdf(query, pdf_id, chunks_dir)

    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources (pages): {result['sources']}")