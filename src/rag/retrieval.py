import textwrap
from time import perf_counter as timer

import torch
from sentence_transformers import SentenceTransformer, util


def retrieve_relevant_resources(
    query: str,
    embeddings: torch.Tensor,
    model: SentenceTransformer,
    n_resources_to_return: int = 5,
    print_time: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_embedding = model.encode(query, convert_to_tensor=True).to(embeddings.device)

    start = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    end = timer()

    if print_time:
        print(
            f"[INFO] Time to score {len(embeddings)} embeddings: {end - start:.5f}s"
        )

    return torch.topk(dot_scores, k=n_resources_to_return)


def print_top_results(
    query: str,
    embeddings: torch.Tensor,
    pages_and_chunks: list[dict],
    model: SentenceTransformer,
    n_resources_to_return: int = 5,
    wrap_length: int = 80,
) -> None:
    scores, indices = retrieve_relevant_resources(
        query=query,
        embeddings=embeddings,
        model=model,
        n_resources_to_return=n_resources_to_return,
    )
    print(f"Query: '{query}'\nResults:")
    for score, idx in zip(scores, indices):
        chunk = pages_and_chunks[int(idx)]
        print(f"\nScore: {float(score):.4f}")
        print(textwrap.fill(chunk["sentence_chunk"], wrap_length))
        print(f"Page: {chunk['page_number']}")


def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    return torch.dot(v1, v2) / (torch.linalg.norm(v1) * torch.linalg.norm(v2))
