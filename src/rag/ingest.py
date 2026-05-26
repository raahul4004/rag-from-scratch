from pathlib import Path

import fitz
import requests
from tqdm.auto import tqdm


def download_pdf(url: str, pdf_path: Path) -> Path:
    if pdf_path.exists():
        print(f"File {pdf_path} exists")
        return pdf_path

    print(f"[INFO] File doesn't exist, downloading from {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    pdf_path.write_bytes(response.content)
    print(f"[INFO] Saved to {pdf_path}")
    return pdf_path


def text_formatter(text: str) -> str:
    return text.replace("\n", " ").strip()


def open_and_read_pdf(pdf_path: Path, page_number_offset: int = -41) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages_and_texts: list[dict] = []
    for page_number, page in tqdm(enumerate(doc), desc="Reading PDF"):
        text = text_formatter(page.get_text())
        pages_and_texts.append(
            {
                "page_number": page_number + page_number_offset,
                "page_char_count": len(text),
                "page_word_count": len(text.split(" ")),
                "page_sentence_count_raw": len(text.split(". ")),
                "page_token_count": len(text) / 4,
                "text": text,
            }
        )
    doc.close()
    return pages_and_texts
