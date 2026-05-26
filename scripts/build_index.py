"""Download the textbook, chunk it, embed it, and save the embedding CSV."""
from rag.chunking import (
    attach_sentence_chunks,
    attach_sentences,
    filter_short_chunks,
    flatten_chunks,
)
from rag.config import Config
from rag.embeddings import (
    embed_chunks,
    load_embedding_model,
    save_chunks_with_embeddings,
)
from rag.ingest import download_pdf, open_and_read_pdf


def main() -> None:
    cfg = Config()
    print(f"[INFO] Device: {cfg.device}")

    download_pdf(cfg.pdf_url, cfg.pdf_path)

    pages = open_and_read_pdf(cfg.pdf_path, page_number_offset=cfg.page_number_offset)
    pages = attach_sentences(pages)
    pages = attach_sentence_chunks(pages, slice_size=cfg.num_sentence_chunk_size)
    chunks = flatten_chunks(pages)
    chunks = filter_short_chunks(chunks, min_token_length=cfg.min_token_length)
    print(f"[INFO] {len(chunks)} chunks after filtering")

    model = load_embedding_model(cfg.embedding_model_name, device=cfg.device)
    embed_chunks(chunks, model=model, batch_size=cfg.embedding_batch_size)

    save_chunks_with_embeddings(chunks, cfg.embeddings_csv)
    print(f"[INFO] Saved embeddings to {cfg.embeddings_csv}")


if __name__ == "__main__":
    main()
