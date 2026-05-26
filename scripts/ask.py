"""Load a prebuilt index and answer a query end-to-end with the local LLM."""
import argparse

from rag.config import Config
from rag.embeddings import load_chunks_with_embeddings, load_embedding_model
from rag.llm import ask, load_llm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="What are Vitamins?")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    cfg = Config(
        max_new_tokens=args.max_new_tokens, temperature=args.temperature
    )
    print(f"[INFO] Device: {cfg.device}")

    pages_and_chunks, embeddings = load_chunks_with_embeddings(
        cfg.embeddings_csv, device=cfg.device
    )
    embedding_model = load_embedding_model(cfg.embedding_model_name, device=cfg.device)
    tokenizer, llm_model = load_llm(
        cfg.llm_model_id, device=cfg.device, use_quantization=cfg.use_quantization
    )

    answer = ask(
        query=args.query,
        embeddings=embeddings,
        pages_and_chunks=pages_and_chunks,
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        llm_model=llm_model,
        n_resources_to_return=cfg.n_resources_to_return,
        temperature=cfg.temperature,
        max_new_tokens=cfg.max_new_tokens,
    )

    print(f"\nQuery: {args.query}")
    print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()
