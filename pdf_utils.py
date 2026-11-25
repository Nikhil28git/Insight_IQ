"""
PDF utility functions for text extraction, page tracking, and formatting.
Shared across chunk_pdf.py, rag_query_pdf.py, and app.py.
"""

import os
from typing import List, Tuple, Dict, Any

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError(
        "\n❌ PyMuPDF not installed!\n"
        "   Run: pip install PyMuPDF\n"
        "   Or: pip install -r requirements.txt\n"
    )


def extract_pdf_text_with_pages(pdf_path: str) -> Tuple[str, List[int], List[Dict[str, Any]]]:
    """
    Extract full text from PDF and track page boundaries.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        (full_text, page_breaks) where page_breaks[i] = character index where page i starts
        
    Example:
        full_text, page_breaks = extract_pdf_text_with_pages("thesis.pdf")
        # full_text = "Introduction...Chapter 1...Results..."
        # page_breaks = [0, 2450, 5890, 8120, ...]  (character indices)
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    page_breaks = [0]
    table_spans: List[Dict[str, Any]] = []  # list of {start, end, page, table_id}

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_num = page_index + 1

        # Extract normal selectable text
        text = page.get_text(option="text")

        # Append page text and remember start index for page
        start_idx = len(full_text)
        full_text += text

        # Detect tables on the page and inject markdown placeholders
        try:
            tables = page.find_tables()
        except Exception:
            tables = []

        for t_i, table in enumerate(tables):
            try:
                df = table.to_pandas()
                md = df.to_markdown(index=False)
            except Exception:
                # Fallback: try multiple table representations safely.
                md = ""
                rows = None
                # table may expose `cells`, `rows`, or row objects; handle gracefully
                if hasattr(table, 'cells'):
                    rows = table.cells
                elif hasattr(table, 'rows'):
                    rows = table.rows

                if rows is None:
                    md = "[UNSUPPORTED_TABLE_FORMAT]"
                else:
                    md_lines = []
                    try:
                        for r in rows:
                            # r might be an iterable of cells or a TableRow object
                            if hasattr(r, '__iter__') and not isinstance(r, (str, bytes)):
                                cells = []
                                for c in r:
                                    if isinstance(c, dict):
                                        cells.append(str(c.get('text', '')).strip())
                                    elif hasattr(c, 'text'):
                                        cells.append(str(c.text).strip())
                                    else:
                                        cells.append(str(c).strip())
                                md_lines.append("| " + " | ".join(cells) + " |")
                            else:
                                # Try attribute access for non-iterable row objects
                                if hasattr(r, 'cells'):
                                    cells = [getattr(cell, 'text', str(cell)).strip() for cell in r.cells]
                                    md_lines.append("| " + " | ".join(cells) + " |")
                                else:
                                    md_lines.append("|UNSUPPORTED_ROW|")
                        md = "\n".join(md_lines) if md_lines else "[UNSUPPORTED_TABLE_FORMAT]"
                    except Exception:
                        md = "[UNSUPPORTED_TABLE_FORMAT]"

            # Attempt to detect whether the table is actually an embedded image
            bbox = None
            is_image = False
            try:
                # table may expose a rect/bbox as a fitz.Rect or tuple
                if hasattr(table, 'bbox'):
                    r = table.bbox
                elif hasattr(table, 'rect'):
                    r = table.rect
                else:
                    r = None

                if r is not None:
                    # Normalize to tuple (x0, y0, x1, y1)
                    if hasattr(r, 'x0'):
                        bbox = (r.x0, r.y0, r.x1, r.y1)
                    elif isinstance(r, (list, tuple)) and len(r) == 4:
                        bbox = tuple(r)

                # If we have a bbox, inspect page blocks for image blocks overlapping it
                if bbox is not None:
                    try:
                        d = page.get_text("dict")
                        blocks = d.get('blocks', [])

                        def rects_overlap(a, b):
                            return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])

                        for block in blocks:
                            if block.get('type') == 1:  # image block
                                bbb = block.get('bbox')
                                if bbb and rects_overlap(bbox, tuple(bbb)):
                                    is_image = True
                                    break
                    except Exception:
                        # If page dict extraction fails, don't crash; leave is_image False
                        pass
            except Exception:
                bbox = None
                is_image = False

            table_marker_start = f"[TABLE_{page_num}_{t_i}]\n"
            table_marker_end = "\n[/TABLE]\n"

            # Inject marker and markdown into full_text
            insert_pos = len(full_text)
            full_text += "\n" + table_marker_start + md + table_marker_end

            table_spans.append({
                "start": insert_pos,
                "end": len(full_text),
                "page": page_num,
                "table_id": f"table_{page_num}_{t_i}",
                "markdown": md,
                "bbox": bbox,
                "is_image": is_image,
            })

        # After processing page, append a page boundary index
        page_breaks.append(len(full_text))

    doc.close()

    return full_text, page_breaks, table_spans


def get_page_for_chunk(chunk: Dict[str, Any], page_breaks: List[int]) -> Tuple[int, int]:
    """
    Get (start_page, end_page) for a chunk based on character positions.
    
    Args:
        chunk: Dict with 'start_index' and 'end_index' (character positions in full text)
        page_breaks: List of character indices where pages start
        
    Returns:
        (start_page, end_page) as 1-indexed page numbers
        
    Example:
        chunk = {'text': 'Results...', 'start_index': 5890, 'end_index': 6200}
        page_breaks = [0, 2450, 5890, 8120]
        page_num = get_page_for_chunk(chunk, page_breaks)
        # Returns (3, 3) because chunk is entirely on page 3
    """
    start_idx = chunk.get('start_index', 0)
    end_idx = chunk.get('end_index', 0)
    
    # Binary search to find page numbers (1-indexed)
    def find_page(char_idx):
        for i in range(len(page_breaks) - 1):
            if page_breaks[i] <= char_idx < page_breaks[i + 1]:
                return i + 1  # 1-indexed
        return len(page_breaks) - 1  # Last page if beyond
    
    start_page = find_page(start_idx)
    end_page = find_page(end_idx)
    
    return (start_page, end_page)


def chunk_overlaps_tables(chunk: Dict[str, Any], table_spans: List[Dict[str, Any]]) -> bool:
    """
    Return True if the chunk overlaps any table span.
    """
    s = chunk.get('start_index', 0)
    e = chunk.get('end_index', 0)
    for t in table_spans:
        if not (e <= t['start'] or s >= t['end']):
            return True
    return False


def get_page_spans(chunk: Dict[str, Any], page_breaks: List[int]) -> str:
    """
    Get formatted page span string for a chunk (e.g., "Page 5" or "Pages 2-3").
    
    Args:
        chunk: Dict with 'start_index' and 'end_index'
        page_breaks: List of character indices where pages start
        
    Returns:
        String like "Page 5" or "Pages 2-3"
        
    Example:
        page_span = get_page_spans(chunk, page_breaks)
        # Returns "Page 5" or "Pages 2-3"
    """
    start_page, end_page = get_page_for_chunk(chunk, page_breaks)
    
    if start_page == end_page:
        return f"Page {start_page}"
    else:
        return f"Pages {start_page}-{end_page}"


def get_chunks_for_page(page_num: int, chunks: List[Dict[str, Any]], 
                        page_breaks: List[int]) -> List[Dict[str, Any]]:
    """
    Get all chunks that appear on a specific page.
    
    Args:
        page_num: 1-indexed page number
        chunks: List of chunk dicts with 'start_index', 'end_index', etc.
        page_breaks: List of character indices where pages start
        
    Returns:
        List of chunks that appear on page_num
        
    Example:
        page_5_chunks = get_chunks_for_page(5, all_chunks, page_breaks)
        # Returns [chunk_42, chunk_43] if they both appear on page 5
    """
    result = []
    for chunk in chunks:
        # Prefer explicit page metadata when available (image chunks/tables)
        if 'start_page' in chunk or 'end_page' in chunk:
            try:
                sp = int(chunk.get('start_page', chunk.get('start_page', page_num)))
                ep = int(chunk.get('end_page', chunk.get('end_page', page_num)))
            except Exception:
                sp, ep = get_page_for_chunk(chunk, page_breaks)
        else:
            sp, ep = get_page_for_chunk(chunk, page_breaks)

        if sp <= page_num <= ep:
            result.append(chunk)

    return result


def format_pdf_source(chunk_id: str, page_num: int, pdf_id: str) -> str:
    """
    Format source string for display (e.g., in Streamlit UI or answer context).
    
    Args:
        chunk_id: Chunk identifier (e.g., "chunk_0042")
        page_num: Page number (1-indexed)
        pdf_id: PDF identifier (e.g., "thesis")
        
    Returns:
        Formatted string like "thesis (page 5, chunk_0042)"
        
    Example:
        source = format_pdf_source("chunk_0042", 5, "thesis")
        # Returns "thesis (page 5, chunk_0042)"
    """
    return f"{pdf_id} (page {page_num}, {chunk_id})"


def estimate_pdf_stats(pdf_path: str) -> Dict[str, Any]:
    """
    Estimate PDF statistics without full text extraction.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dict with 'total_pages', 'estimated_words', 'file_size_mb'
        
    Example:
        stats = estimate_pdf_stats("thesis.pdf")
        # Returns {'total_pages': 42, 'estimated_words': 15000, 'file_size_mb': 2.5}
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    # Estimate words by sampling first 3 pages
    sample_text = ""
    for page_num in range(min(3, total_pages)):
        page = doc[page_num]
        sample_text += page.get_text(option="text")
    
    avg_words_per_char = 1 / 5.0  # Average English word is ~5 chars
    doc.close()
    
    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    estimated_words = int(len(sample_text) * avg_words_per_char * total_pages / min(3, total_pages))
    
    return {
        'total_pages': total_pages,
        'estimated_words': estimated_words,
        'file_size_mb': round(file_size_mb, 2)
    }


def extract_text_span(full_text: str, start_pos: int, end_pos: int, 
                      context_words: int = 0) -> str:
    """
    Extract a text span (chunk) with optional context words before and after.
    
    Args:
        full_text: Full PDF text
        start_pos: Character index where chunk starts
        end_pos: Character index where chunk ends
        context_words: How many words to add before/after for context
        
    Returns:
        Text snippet of the chunk with optional context
        
    Example:
        snippet = extract_text_span(full_text, 5890, 6200, context_words=5)
        # Returns the chunk text with 5 words of context before/after
    """
    chunk_text = full_text[start_pos:end_pos]
    
    if context_words == 0:
        return chunk_text
    
    # Find context before
    before_start = max(0, start_pos)
    before_text = full_text[max(0, start_pos - 500):start_pos]  # Up to 500 chars before
    before_words = before_text.split()[-context_words:]
    
    # Find context after
    after_end = min(len(full_text), end_pos + 500)
    after_text = full_text[end_pos:after_end]  # Up to 500 chars after
    after_words = after_text.split()[:context_words]
    
    context = " ".join(before_words) + " " if before_words else ""
    context += chunk_text
    context += " " + " ".join(after_words) if after_words else ""
    
    return context.strip()
