"""Run only the retrieval step — no LLM. Useful to check chunks for a query."""
import argparse

from rag.config import Config
from rag.embeddings import load_chunks_with_embeddings, load_embedding_model
from rag.retrieval import print_top_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="foods high in fiber")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    args = parser.parse_args()

    cfg = Config(n_resources_to_return=args.top_k)
    print(f"[INFO] Device: {cfg.device}")

    pages_and_chunks, embeddings = load_chunks_with_embeddings(
        cfg.embeddings_csv, device=cfg.device
    )
    embedding_model = load_embedding_model(cfg.embedding_model_name, device=cfg.device)

    print_top_results(
        query=args.query,
        embeddings=embeddings,
        pages_and_chunks=pages_and_chunks,
        model=embedding_model,
        n_resources_to_return=cfg.n_resources_to_return,
    )


if __name__ == "__main__":
    main()
