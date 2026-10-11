"""Preserve page content/timestamps while making bounded, named PDF parts."""

from pathlib import Path

from pypdf import PdfReader, PdfWriter

from src.cancellation import check_cancel


def split_pdf(source, destination, stem, max_pages=100):
    if (
        isinstance(max_pages, bool)
        or not isinstance(max_pages, int)
        or not 1 <= max_pages <= 100
    ):
        raise ValueError("PDF parts must contain between 1 and 100 pages.")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(source)
    count = len(reader.pages)
    if not count:
        raise ValueError("Cannot split a PDF with no pages.")
    paths = [
        destination
        / (
            f"{stem}_{start // max_pages + 1:02d}.pdf"
            if count > max_pages
            else f"{stem}.pdf"
        )
        for start in range(0, count, max_pages)
    ]
    if any(p.exists() for p in paths):
        raise ValueError(
            "PDF output already exists; preserve it and use another output folder."
        )
    result = []
    for start, path in zip(range(0, count, max_pages), paths):
        check_cancel()
        writer = PdfWriter()
        for page in reader.pages[start : start + max_pages]:
            check_cancel()
            writer.add_page(page)
        writer.add_metadata(
            {
                "/Title": stem
                + (f" â€” part {start // max_pages + 1}" if count > max_pages else ""),
                "/Author": "TransVision",
            }
        )
        partial = path.with_suffix(".partial.pdf")
        with partial.open("wb") as stream:
            writer.write(stream)
        partial.replace(path)
        result.append(
            {
                "file": str(path),
                "pages": min(max_pages, count - start),
                "source_page_start": start + 1,
                "source_page_end": min(count, start + max_pages),
            }
        )
    return result
