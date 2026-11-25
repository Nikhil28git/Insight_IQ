"""
PDF chunking pipeline: Extract text, apply semantic chunking, track pages, embed.
Outputs chunks with page metadata to chunks/chunks.json and chunk embeddings to embeddings.npy
"""

import json
import os
import numpy as np
from typing import List, Dict, Any

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError(
        "\n❌ PyMuPDF not installed!\n"
        "   Run: pip install PyMuPDF\n"
        "   Or: pip install -r requirements.txt\n"
    )

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from pdf_utils import (
    extract_pdf_text_with_pages,
    get_page_for_chunk,
    get_page_spans,
    estimate_pdf_stats,
)

from image_utils import extract_images_from_pdf, extract_image_region
from image_description import describe_image_with_gemini

# Embedding model
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_THRESHOLD = 0.5  # Semantic similarity threshold for chunk boundaries


def load_embeddings():
    """Load HuggingFace embeddings model with normalization."""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )


def extract_pdf_text(pdf_path: str) -> tuple:
    """
    Extract text from PDF and track page breaks.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        (full_text, page_breaks) tuple
    """
    print(f"Extracting text from PDF: {pdf_path}")
    # Note: pdf_utils.extract_pdf_text_with_pages now also returns table_spans
    full_text, page_breaks, table_spans = extract_pdf_text_with_pages(pdf_path)

    stats = estimate_pdf_stats(pdf_path)
    print(f"  Pages: {stats['total_pages']}, Estimated words: {stats['estimated_words']}")

    return full_text, page_breaks, table_spans


def apply_semantic_chunking(text: str, embeddings) -> List[Dict[str, Any]]:
    """
    Apply LangChain SemanticChunker to text.
    Returns chunks with text and character position metadata.
    
    Args:
        text: Full PDF text
        embeddings: HuggingFaceEmbeddings instance
        
    Returns:
        List of chunk dicts with 'text', 'start_index', 'end_index'
    """
    print("Applying semantic chunking...")
    
    chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=70,  # Use top 30% of similarity drops as breaks
    )
    
    # Get chunks as strings
    chunk_texts = chunker.split_text(text)
    
    # Map chunks back to character positions
    chunks = []
    current_pos = 0
    
    for i, chunk_text in enumerate(chunk_texts):
        # Find where this chunk starts in remaining text
        start_pos = text.find(chunk_text, current_pos)
        
        if start_pos == -1:
            # Fallback if exact match not found (shouldn't happen)
            start_pos = current_pos
        
        end_pos = start_pos + len(chunk_text)
        current_pos = end_pos
        
        chunks.append({
            'chunk_id': f'chunk_{i:04d}',
            'text': chunk_text,
            'start_index': start_pos,
            'end_index': end_pos,
            'word_count': len(chunk_text.split()),
            # default type; may be switched to 'table' in add_page_metadata
            'chunk_type': 'text',
        })
    
    print(f"  Created {len(chunks)} chunks")
    return chunks


def add_page_metadata(chunks: List[Dict[str, Any]], page_breaks: List[int], table_spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add page number metadata to each chunk.
    
    Args:
        chunks: List of chunks from semantic chunking
        page_breaks: List of character indices where pages start
        
    Returns:
        Chunks with page metadata added
    """
    print("Adding page metadata to chunks...")
    
    for chunk in chunks:
        start_page, end_page = get_page_for_chunk(chunk, page_breaks)
        chunk['start_page'] = start_page
        chunk['end_page'] = end_page
        chunk['page_spans'] = get_page_spans(chunk, page_breaks)

        # Mark chunk_type as 'table' if it overlaps any detected table spans
        try:
            from pdf_utils import chunk_overlaps_tables
            if chunk_overlaps_tables(chunk, table_spans):
                chunk['chunk_type'] = 'table'
        except Exception:
            # If helper not available for any reason, leave as 'text'
            pass
    
    return chunks


def embed_chunks(chunks: List[Dict[str, Any]], embeddings) -> tuple:
    """
    Embed chunk texts using HuggingFaceEmbeddings.
    
    Args:
        chunks: List of chunks with 'text' field
        embeddings: HuggingFaceEmbeddings instance
        
    Returns:
        (chunks_with_embeddings, embeddings_array) tuple
    """
    print("Embedding chunks...")
    
    texts = [chunk['text'] for chunk in chunks]
    embeddings_array = embeddings.embed_documents(texts)
    
    for i, chunk in enumerate(chunks):
        chunk['embedding'] = embeddings_array[i]
    
    print(f"  Embedded {len(chunks)} chunks")
    return chunks, np.array(embeddings_array)


def save_chunks(chunks: List[Dict[str, Any]], output_dir: str):
    """
    Save chunks to JSON and embeddings to NPY file.
    
    Args:
        chunks: List of chunks with metadata and embeddings
        output_dir: Output directory (e.g., pdfs/my_pdf/chunks/)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save chunks.json (without embeddings for readability)
    chunks_for_json = []
    embeddings_array = []

    for chunk in chunks:
        # Do not mutate original chunk dict; create shallow copy for JSON
        chunk_copy = {k: v for k, v in chunk.items() if k != 'embedding'}
        embeddings_array.append(chunk.get('embedding'))
        chunks_for_json.append(chunk_copy)
    
    with open(os.path.join(output_dir, 'chunks.json'), 'w', encoding='utf-8') as f:
        json.dump(chunks_for_json, f, indent=2, ensure_ascii=False)
    
    # Save embeddings as NPY
    np.save(os.path.join(output_dir, 'chunk_embeddings.npy'), np.array(embeddings_array))
    
    print(f"Saved {len(chunks_for_json)} chunks to {output_dir}")

# Save embeddings
#     np.save(
#     os.path.join(output_dir, "chunk_embeddings.npy"),
#     np.array(embeddings_array)
#     )

# # Save chunk IDs
#     chunk_ids = [chunk["chunk_id"] for chunk in chunks_for_json]

#     with open(
#        os.path.join(output_dir, "chunk_ids.json"),
#        "w",
#         encoding="utf-8"
#     ) as f:
#         json.dump(chunk_ids, f, indent=2)

#     print(f"Saved {len(chunks_for_json)} chunks to {output_dir}")
#     print(f"Saved chunk_embeddings.npy")
#     print(f"Saved chunk_ids.json")

def process_pdf_pipeline(pdf_path: str, output_dir: str):
    """
    End-to-end PDF chunking pipeline.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Output directory for chunks and embeddings
    """
    print(f"\n{'='*60}")
    print(f"PDF Chunking Pipeline")
    print(f"{'='*60}")
    
    # Load embeddings
    embeddings = load_embeddings()
    
    # Step 1: Extract text and track page breaks and tables
    full_text, page_breaks, table_spans = extract_pdf_text(pdf_path)
    
    # Step 2: Apply semantic chunking for text
    chunks = apply_semantic_chunking(full_text, embeddings)

    # Image extraction: save images and create placeholder image chunks
    images_dir = os.path.join(os.path.dirname(output_dir), "images")
    os.makedirs(images_dir, exist_ok=True)
    try:
        image_items = extract_images_from_pdf(pdf_path, images_dir)
    except Exception:
        image_items = []

    # Create dedicated image chunks (will be described via Gemini Vision)
    image_chunks = []
    # for img in image_items:
    #     page = img.get("page")
    #     image_path = img.get("image_path")
    #     image_id = img.get("image_id")

    #     # Describe image using Gemini Vision (best-effort). This returns a short
    #     # structured text suitable for embedding. If Gemini client is unavailable
    #     # or fails, the description is a placeholder.
    #     desc, meta = describe_image_with_gemini(image_path)

    #     ic = {
    #         'chunk_id': image_id,
    #         'chunk_type': 'image',
    #         'text': desc,
    #         'start_page': page,
    #         'end_page': page,
    #         'page_spans': f"Page {page}",
    #         'image_path': image_path,
    #         'word_count': len(desc.split()),
    #     }
    #     image_chunks.append(ic)
    for img in image_items:
        page = img.get("page")
        image_path = img.get("image_path")
        image_id = img.get("image_id")

        ic = {
          'chunk_id': image_id,
          'chunk_type': 'image',
          'text': f"Image on page {page}",
          'start_page': page,
          'end_page': page,
          'page_spans': f"Page {page}",
          'image_path': image_path,
          'word_count': 4,
        }

        image_chunks.append(ic)
    
    # Step 3: Add page metadata and detect table chunks
    chunks = add_page_metadata(chunks, page_breaks, table_spans)

    # Create dedicated table chunks from table_spans (use markdown stored in spans)
    table_chunks = []
    for t_i, t in enumerate(table_spans):
        try:
            page = t.get('page')
            tbl_id = t.get('table_id')
            md = t.get('markdown', '')
            start = t.get('start')
            end = t.get('end')

            # If the table region was detected as an image, render the region
            # and describe it with Gemini Vision instead of layout parsing.
            image_path_for_table = None
            if t.get('is_image'):
                try:
                    bbox = t.get('bbox')
                    if bbox:
                        region_meta = extract_image_region(pdf_path, page, bbox, images_dir, image_id=f"tbl_{page:03d}_{t_i:02d}")
                        image_path_for_table = region_meta.get('image_path')
                        # Describe the cropped table image using Gemini Vision
                        desc, meta = describe_image_with_gemini(image_path_for_table)
                        # Use description as markdown/text for table chunk when available
                        if desc and desc.strip():
                            md = desc
                except Exception:
                    # If region extraction or description fails, fall back to raw markdown
                    pass

            tbl_chunk = {
                'chunk_id': f"tbl_{page:03d}_{tbl_id.split('_')[-1]}",
                'chunk_type': 'table',
                'text': md,
                'start_index': start,
                'end_index': end,
                'start_page': page,
                'end_page': page,
                'page_spans': f"Page {page}",
                'word_count': len(md.split()),
            }

            if image_path_for_table:
                tbl_chunk['image_path'] = image_path_for_table

            table_chunks.append(tbl_chunk)
        except Exception:
            continue

    # Append table chunks and image chunks to the chunk list so they get embedded and stored
    if table_chunks:
        chunks.extend(table_chunks)
    if image_chunks:
        chunks.extend(image_chunks)
    
    # Step 4: Embed chunks
    chunks, embeddings_array = embed_chunks(chunks, embeddings)
    
    # Step 5: Save chunks and embeddings
    save_chunks(chunks, output_dir)
    
    print(f"\n✅ Pipeline complete!")
    print(f"Output directory: {output_dir}")
    
    return chunks, embeddings_array


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python chunk_pdf.py <pdf_path> [output_dir]")
        print("Example: python chunk_pdf.py thesis.pdf pdfs/thesis_2024/chunks/")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "chunks/"
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    process_pdf_pipeline(pdf_path, output_dir)
