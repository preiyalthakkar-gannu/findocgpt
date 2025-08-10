# core/loader.py
from pathlib import Path
import zipfile
from docx import Document
import PyPDF2

def load_text_from_file(path: str):
    """
    Load text from PDF, DOCX, or TXT.
    - DOCX is validated as a real zipped package to avoid 'Package not found' errors.
    - PDF is extracted via PyPDF2 (best-effort).
    - TXT is read as UTF-8 (ignores decoding errors).
    Returns: (text, source_name)
    """
    p = Path(path)
    ext = p.suffix.lower()
    name = p.name

    if ext == ".txt":
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(), name

    if ext == ".docx":
        # Ensure it's a valid DOCX package (ZIP)
        if not zipfile.is_zipfile(p):
            raise ValueError("File is not a valid DOCX package. Please re-save as DOCX or upload as PDF/TXT.")
        doc = Document(str(p))
        text = "\n".join(par.text for par in doc.paragraphs)
        return text, name

    if ext == ".pdf":
        text = ""
        with open(p, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                try:
                    text += page.extract_text() or ""
                except Exception:
                    # Some pages may fail to extract text cleanly; continue
                    continue
        return text, name

    raise ValueError(f"Unsupported file type: {ext} (use PDF, DOCX, or TXT)")
