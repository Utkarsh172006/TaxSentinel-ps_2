from __future__ import annotations

import io
import shutil


def extract_pdf_text(payload: bytes, enable_ocr: bool = False) -> str:
    """Extract PDF text; OCR is opt-in and requires the Tesseract executable."""
    try:
        import pdfplumber
    except ModuleNotFoundError as exc:
        if exc.name != "pdfplumber":
            raise
        raise RuntimeError(
            "PDF extraction requires pdfplumber. Install project dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    pages_text: list[str] = []
    with pdfplumber.open(io.BytesIO(payload)) as document:
        pages_text = [page.extract_text() or "" for page in document.pages]
        if enable_ocr and not any(text.strip() for text in pages_text):
            if shutil.which("tesseract") is None:
                raise RuntimeError("OCR was enabled, but the Tesseract executable is not installed.")
            import pytesseract
            from PIL import Image

            pages_text = []
            for page in document.pages:
                image = page.to_image(resolution=200).original
                pages_text.append(pytesseract.image_to_string(image))
    text = "\n".join(pages_text).strip()
    if not text:
        raise ValueError("No readable text was extracted from the PDF.")
    return text
