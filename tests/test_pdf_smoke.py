"""Minimal PDF-ingestion smoke test: the fulltext toolchain must parse a
valid PDF end-to-end (MarkItDown). Guards the access layer's core path."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from citens.grounding.fulltext import _pdf_bytes_to_text  # noqa: E402

markitdown = pytest.importorskip("markitdown", reason="PDF toolchain not installed")


def _make_pdf(text: str) -> bytes:
    """Hand-rolled single-page PDF wrapping `text` (no external deps)."""
    lines = [text[i : i + 80] for i in range(0, len(text), 80)]
    content = "BT /F1 10 Tf 40 750 Td 14 TL " + " ".join(
        f"({line}) Tj T*" for line in lines
    ) + " ET"
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF"
    ).encode()
    return out


def test_pdf_bytes_to_text_extracts_body():
    body = ("Sample methods text. " * 40).strip()  # > 500 chars
    pdf = _make_pdf(body)
    text = _pdf_bytes_to_text(pdf)
    assert text and "Sample methods text" in text


def test_non_pdf_bytes_rejected():
    assert _pdf_bytes_to_text(b"not a pdf at all") is None
