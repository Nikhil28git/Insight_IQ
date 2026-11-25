"""
End-to-end PDF processing orchestrator: Chunk, embed, and store in ChromaDB.
Equivalent to run_pipeline.py but for PDF documents.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

from chunk_pdf import process_pdf_pipeline
from pdf_utils import estimate_pdf_stats

# Embedding model
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def load_embeddings():
    """Load HuggingFace embeddings model."""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )


def load_chroma_client():
    """Load persisted ChromaDB client."""
    return chromadb.PersistentClient(path="chroma_db")


def setup_output_directory(pdf_id: str, base_dir: str = "pdfs") -> str:
    """
    Create output directory structure for PDF.
    
    Args:
        pdf_id: PDF identifier (e.g., "thesis_2024")
        base_dir: Base directory for PDFs
        
    Returns:
        Path to chunks directory
    """
    pdf_dir = os.path.join(base_dir, pdf_id)
    chunks_dir = os.path.join(pdf_dir, "chunks")
    
    os.makedirs(chunks_dir, exist_ok=True)
    
    return chunks_dir, pdf_dir


def store_chunks_in_chromadb(
    chunks: list,
    pdf_id: str,
    embeddings,
    client,
) -> int:
    """
    Store chunks in ChromaDB collection.
    
    Args:
        chunks: List of chunks with embeddings and metadata
        pdf_id: PDF identifier
        embeddings: HuggingFaceEmbeddings instance
        client: ChromaDB client
        
    Returns:
        Number of chunks stored
    """
    collection_name = f"pdf_{pdf_id}"
    
    # Get or create collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"source": "pdf", "pdf_id": pdf_id}
    )
    
    print(f"\nStoring chunks in ChromaDB collection: {collection_name}")
    
    # Prepare data for ChromaDB
    ids = []
    documents = []
    embeddings_list = []
    metadatas = []
    
    for chunk in chunks:
        ids.append(chunk['chunk_id'])
        documents.append(chunk['text'])
        embeddings_list.append(chunk['embedding'])
        
        metad = {
            'chunk_id': chunk['chunk_id'],
            'start_page': chunk.get('start_page'),
            'end_page': chunk.get('end_page'),
            'page_spans': chunk.get('page_spans'),
            'word_count': chunk.get('word_count'),
            'source_type': 'pdf',
            'pdf_id': pdf_id,
            'chunk_type': chunk.get('chunk_type', 'text'),
        }
        # include image_path for image chunks
        if 'image_path' in chunk:
            metad['image_path'] = chunk.get('image_path')

        metadatas.append(metad)
    
    # Add to collection
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings_list,
        metadatas=metadatas,
    )
    
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB")
    
    return len(chunks)


def save_metadata(pdf_path: str, pdf_id: str, pdf_dir: str, chunks_count: int):
    """
    Save PDF metadata to JSON file.
    
    Args:
        pdf_path: Path to original PDF
        pdf_id: PDF identifier
        pdf_dir: PDF output directory
        chunks_count: Number of chunks created
    """
    stats = estimate_pdf_stats(pdf_path)
    
    metadata = {
        'pdf_id': pdf_id,
        'pdf_path': os.path.abspath(pdf_path),
        'processing_date': datetime.now().isoformat(),
        'total_pages': stats['total_pages'],
        'estimated_words': stats['estimated_words'],
        'file_size_mb': stats['file_size_mb'],
        'chunks_count': chunks_count,
        'embedding_model': EMBED_MODEL,
        'chromadb_collection': f'pdf_{pdf_id}',
    }
    
    metadata_file = os.path.join(pdf_dir, 'meta.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_file}")
    
    return metadata


def process_pdf_end_to_end(
    pdf_path: str,
    pdf_id: str,
    base_dir: str = "pdfs",
) -> Dict[str, Any]:
    """
    End-to-end PDF processing pipeline.
    
    Steps:
    1. Setup output directory structure
    2. Extract and chunk PDF
    3. Embed chunks
    4. Store in ChromaDB
    5. Save metadata
    
    Args:
        pdf_path: Path to PDF file
        pdf_id: PDF identifier (e.g., "thesis_2024")
        base_dir: Base directory for PDFs
        
    Returns:
        Dict with processing summary
    """
    print(f"\n{'='*70}")
    print(f"PDF Processing Pipeline")
    print(f"{'='*70}")
    print(f"PDF: {pdf_path}")
    print(f"ID: {pdf_id}")
    
    # Validate PDF exists
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found: {pdf_path}")
        return {'success': False, 'error': 'PDF not found'}
    
    # Step 1: Setup output directory
    chunks_dir, pdf_dir = setup_output_directory(pdf_id, base_dir)
    print(f"Output directory: {pdf_dir}")
    
    # Step 2: Chunk PDF
    try:
        chunks, embeddings_array = process_pdf_pipeline(pdf_path, chunks_dir)
    except Exception as e:
        print(f"❌ Error during chunking: {e}")
        return {'success': False, 'error': f'Chunking failed: {str(e)}'}
    
    # Step 3: Store in ChromaDB
    try:
        embeddings = load_embeddings()
        client = load_chroma_client()
        # Use embeddings returned by the chunking pipeline when available
        # (avoids writing then re-loading chunk_embeddings.npy)
        if 'embedding' not in chunks[0]:
            # If chunks don't include embeddings, try to use embeddings_array
            try:
                # embeddings_array is returned by process_pdf_pipeline
                for i, chunk in enumerate(chunks):
                    chunk['embedding'] = embeddings_array[i].tolist()
            except Exception:
                # Fallback: attempt to load from disk (backwards compatibility)
                try:
                    import numpy as np
                    disk_embeddings = np.load(os.path.join(chunks_dir, 'chunk_embeddings.npy'))
                    for i, chunk in enumerate(chunks):
                        chunk['embedding'] = disk_embeddings[i].tolist()
                except Exception as e:
                    print(f"❌ Error: embeddings not available: {e}")
                    raise

        chunks_stored = store_chunks_in_chromadb(chunks, pdf_id, embeddings, client)
    except Exception as e:
        print(f"❌ Error storing in ChromaDB: {e}")
        return {'success': False, 'error': f'Storage failed: {str(e)}'}
    
    # Step 4: Save metadata
    try:
        metadata = save_metadata(pdf_path, pdf_id, pdf_dir, chunks_stored)
    except Exception as e:
        print(f"⚠️  Warning: Failed to save metadata: {e}")
        metadata = {}
    
    print(f"\n{'='*70}")
    print(f"✅ PDF Processing Complete!")
    print(f"{'='*70}")
    print(f"Chunks created: {chunks_stored}")
    print(f"Pages: {metadata.get('total_pages', 'N/A')}")
    print(f"ChromaDB collection: pdf_{pdf_id}")
    print(f"Output directory: {pdf_dir}")
    
    return {
        'success': True,
        'pdf_id': pdf_id,
        'chunks_stored': chunks_stored,
        'pdf_dir': pdf_dir,
        'metadata': metadata,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process PDF: Extract, chunk, embed, and store in ChromaDB"
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to PDF file"
    )
    parser.add_argument(
        "--pdf_id",
        required=True,
        help="PDF identifier (e.g., 'thesis_2024', 'whitepaper')"
    )
    parser.add_argument(
        "--base_dir",
        default="pdfs",
        help="Base directory for PDF outputs (default: 'pdfs')"
    )
    
    args = parser.parse_args()
    
    result = process_pdf_end_to_end(
        pdf_path=args.pdf,
        pdf_id=args.pdf_id,
        base_dir=args.base_dir,
    )
    
    if not result['success']:
        exit(1)

# """
# process_pdf.py

# PDF orchestration pipeline.

# Flow:
# PDF
#  ↓
# chunk_pdf.py
#  ↓
# chunks.json + chunk_embeddings.npy
#  ↓
# embed_and_store.py
#  ↓
# ChromaDB
#  ↓
# meta.json
# """

# import os
# import json
# import argparse
# from datetime import datetime
# from typing import Dict, Any

# from chunk_pdf import process_pdf_pipeline
# from embed_and_store import embed_and_store
# from pdf_utils import estimate_pdf_stats

# EMBED_MODEL = "BAAI/bge-small-en-v1.5"


# def setup_output_directory(pdf_id: str, base_dir: str = "pdfs"):
#     pdf_dir = os.path.join(base_dir, pdf_id)
#     chunks_dir = os.path.join(pdf_dir, "chunks")

#     os.makedirs(chunks_dir, exist_ok=True)

#     return chunks_dir, pdf_dir


# def save_metadata(
#     pdf_path: str,
#     pdf_id: str,
#     pdf_dir: str,
# ):
#     stats = estimate_pdf_stats(pdf_path)

#     chunks_file = os.path.join(
#         pdf_dir,
#         "chunks",
#         "chunks.json"
#     )

#     chunks_count = 0

#     if os.path.exists(chunks_file):
#         with open(chunks_file, "r", encoding="utf-8") as f:
#             chunks = json.load(f)

#         if isinstance(chunks, list):
#             chunks_count = len(chunks)
#         elif isinstance(chunks, dict):
#             chunks_count = len(chunks.get("chunks", []))

#     metadata = {
#         "pdf_id": pdf_id,
#         "pdf_path": os.path.abspath(pdf_path),
#         "processing_date": datetime.now().isoformat(),
#         "total_pages": stats["total_pages"],
#         "estimated_words": stats["estimated_words"],
#         "file_size_mb": stats["file_size_mb"],
#         "chunks_count": chunks_count,
#         "embedding_model": EMBED_MODEL,
#         "chromadb_collection": f"pdf_{pdf_id}",
#     }

#     metadata_file = os.path.join(pdf_dir, "meta.json")

#     with open(metadata_file, "w", encoding="utf-8") as f:
#         json.dump(metadata, f, indent=2)

#     return metadata


# def process_pdf_end_to_end(
#     pdf_path: str,
#     pdf_id: str,
#     base_dir: str = "pdfs",
# ) -> Dict[str, Any]:

#     print("\n" + "=" * 70)
#     print("PDF Processing Pipeline")
#     print("=" * 70)
#     print(f"PDF : {pdf_path}")
#     print(f"ID  : {pdf_id}")

#     if not os.path.exists(pdf_path):
#         return {
#             "success": False,
#             "error": f"PDF not found: {pdf_path}"
#         }

#     chunks_dir, pdf_dir = setup_output_directory(
#         pdf_id,
#         base_dir
#     )

#     try:
#         print("\n[1/3] Chunking PDF...")

#         chunks, _ = process_pdf_pipeline(
#             pdf_path=pdf_path,
#             output_dir=chunks_dir
#         )

#     except Exception as e:
#         return {
#             "success": False,
#             "error": f"Chunking failed: {e}"
#         }

#     try:
#         print("\n[2/3] Storing in ChromaDB...")

#         embed_and_store(
#             pdf_id=pdf_id,
#             source_type="pdf",
#             reset=False
#         )

#     except Exception as e:
#         return {
#             "success": False,
#             "error": f"Storage failed: {e}"
#         }

#     try:
#         print("\n[3/3] Saving metadata...")

#         metadata = save_metadata(
#             pdf_path=pdf_path,
#             pdf_id=pdf_id,
#             pdf_dir=pdf_dir
#         )

#     except Exception as e:
#         metadata = {}
#         print(f"Warning: metadata save failed: {e}")

#     chunks_count = metadata.get("chunks_count", 0)

#     print("\n" + "=" * 70)
#     print("✅ PDF Processing Complete")
#     print("=" * 70)

#     return {
#         "success": True,
#         "pdf_id": pdf_id,
#         "chunks_stored": chunks_count,
#         "pdf_dir": pdf_dir,
#         "metadata": metadata,
#     }


# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(
#         description="Process PDF and index into ChromaDB"
#     )

#     parser.add_argument(
#         "--pdf",
#         required=True,
#         help="Path to PDF file"
#     )

#     parser.add_argument(
#         "--pdf_id",
#         required=True,
#         help="Unique PDF identifier"
#     )

#     parser.add_argument(
#         "--base_dir",
#         default="pdfs"
#     )

#     args = parser.parse_args()

#     result = process_pdf_end_to_end(
#         pdf_path=args.pdf,
#         pdf_id=args.pdf_id,
#         base_dir=args.base_dir
#     )

#     if not result["success"]:
#         print(result["error"])
#         exit(1)