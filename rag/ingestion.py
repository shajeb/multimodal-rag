"""PDF text/table/image extraction, with one-based source page numbers."""
import re
from pathlib import Path
from .embeddings import cosine
from .models import Chunk, digest


def semantic_chunks(text, embedder, limit=1400, minimum=250, threshold=0.45):
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\s*\n", text) if s.strip()]
    sentences = [s[i:i+limit] for s in sentences for i in range(0, len(s), limit)]
    if not sentences:
        return []
    vectors = embedder.embed(sentences)
    groups, current = [], ""
    for i, sentence in enumerate(sentences):
        boundary = i and cosine(vectors[i-1], vectors[i]) < threshold
        if current and (len(current) + len(sentence) + 1 > limit or (boundary and len(current) >= minimum)):
            groups.append(current)
            current = ""
        current = f"{current} {sentence}".strip()
    if current:
        groups.append(current)
    return groups


def table_chunks(text, limit=1400):
    """Keep row boundaries and repeat the header for table context."""
    rows = text.splitlines()
    if not rows:
        return []
    header = rows[0][:limit//3]
    groups, current = [], header
    for row in rows[1:]:
        for start in range(0, max(1, len(row)), limit - len(header) - 1):
            part = row[start:start + limit - len(header) - 1]
            if len(current) + len(part) + 1 > limit:
                groups.append(current)
                current = header
            current += "\n" + part
    groups.append(current)
    return groups


def extract_pdf(path, assets: Path, vision=None, max_pages=500):
    import pymupdf as fitz
    import pdfplumber
    assets.mkdir(parents=True, exist_ok=True)
    elements, seen_images = [], {}
    with fitz.open(path) as pdf, pdfplumber.open(path) as tables_pdf:
        if pdf.is_encrypted:
            raise ValueError("Password-protected PDFs are not supported")
        if len(pdf) > max_pages:
            raise ValueError(f"PDF exceeds the {max_pages}-page limit")
        for number, page in enumerate(pdf):
            table_page = tables_pdf.pages[number]
            tables = table_page.find_tables()
            def outside_tables(obj):
                x = (obj.get("x0", 0) + obj.get("x1", 0)) / 2
                y = (obj.get("top", 0) + obj.get("bottom", 0)) / 2
                return not any(t.bbox[0] <= x <= t.bbox[2] and t.bbox[1] <= y <= t.bbox[3] for t in tables)
            prose = table_page.filter(outside_tables).extract_text() or ""
            if prose.strip():
                elements.append((number+1, "text", prose, ""))
            for table in tables:
                text = "\n".join(" | ".join((c or "").replace("\n", " ") for c in row)
                                 for row in table.extract())
                if text.strip():
                    elements.append((number+1, "table", text, ""))
            for item in page.get_images(full=True):
                extracted = pdf.extract_image(item[0])
                pix = fitz.Pixmap(extracted["image"])
                if pix.n - pix.alpha > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                raw = pix.tobytes("png")
                key = digest(raw)
                if key not in seen_images:
                    asset = assets / f"{key}.png"
                    asset.write_bytes(raw)
                    caption = vision(raw) if vision else "Image extracted; live vision is required to index its visual content."
                    seen_images[key] = (caption, str(asset.resolve()))
                caption, asset_path = seen_images[key]
                elements.append((number+1, "image", caption, asset_path))
            if vision and (not prose.strip() or page.get_drawings()):
                raw = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).tobytes("png")
                asset = assets / f"{digest(raw)}.png"
                asset.write_bytes(raw)
                elements.append((number+1, "image", vision(raw), str(asset.resolve())))
    return list(dict.fromkeys(elements))


def make_chunks(elements, source, document, version, embedder):
    chunks = []
    for page, kind, text, asset in elements:
        parts = table_chunks(text) if kind == "table" else semantic_chunks(text, embedder)
        for part in parts:
            key = digest(f"{document}:{version}:{page}:{kind}:{part}")
            chunks.append(Chunk(key, document, version, source, page, kind, part, asset))
    return list({chunk.id: chunk for chunk in chunks}.values())
