"""
Image extraction utilities using PyMuPDF.

Functions:
 - extract_images_from_pdf(pdf_path, out_dir) -> List[Dict]

Saves images as PNG files under `out_dir` and returns a list of image metadata dicts:
  {"page": int, "image_id": str, "image_path": str}

This module is minimal and avoids external deps.
"""
import os
from typing import List, Dict, Any

try:
    import fitz
except ImportError:
    raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")


def extract_images_from_pdf(pdf_path: str, out_dir: str) -> List[Dict[str, Any]]:
    """Extract inline images from a PDF and save to `out_dir`.

    Returns a list of dicts with keys: page (1-indexed), image_id, image_path.
    """
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    images = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_num = page_index + 1

        # page.get_images returns tuples with xref as first element
        try:
            img_list = page.get_images(full=True)
        except Exception:
            img_list = []

        for img_i, img in enumerate(img_list, start=1):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image.get("image")
                ext = base_image.get("ext", "png")
            except Exception:
                # Fallback: render the page region (no bbox available here)
                try:
                    pix = page.get_pixmap()
                    image_bytes = pix.tobytes()
                    ext = "png"
                except Exception:
                    continue

            image_id = f"img_{page_num:03d}_{img_i:02d}"
            image_filename = f"{image_id}.{ext}"
            image_path = os.path.join(out_dir, image_filename)

            try:
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
            except Exception:
                continue

            images.append({
                "page": page_num,
                "image_id": image_id,
                "image_path": image_path,
            })

    doc.close()
    return images


def extract_image_region(pdf_path: str, page_num: int, rect: tuple, out_dir: str, image_id: str = None) -> Dict[str, Any]:
    """Render a region of a page (rect = (x0,y0,x1,y1)) to an image file and return metadata.

    Returns a dict: {"page": int, "image_id": str, "image_path": str}
    """
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    try:
        page_index = page_num - 1
        if page_index < 0 or page_index >= len(doc):
            raise IndexError("page_num out of range")

        page = doc[page_index]
        r = None
        if isinstance(rect, (list, tuple)) and len(rect) == 4:
            r = fitz.Rect(rect)
        elif hasattr(rect, 'x0'):
            r = rect

        if r is None:
            # Render the whole page as a fallback
            pix = page.get_pixmap()
        else:
            pix = page.get_pixmap(clip=r)

        if image_id is None:
            image_id = f"tbl_region_{page_num:03d}"
        image_path = os.path.join(out_dir, f"{image_id}.png")
        pix.save(image_path)

        return {"page": page_num, "image_id": image_id, "image_path": image_path}
    finally:
        doc.close()
