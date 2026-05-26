from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


def load_embedding_model(model_name: str, device: str) -> SentenceTransformer:
    return SentenceTransformer(model_name_or_path=model_name, device=device)


def embed_chunks(
    pages_and_chunks: list[dict],
    model: SentenceTransformer,
    batch_size: int = 16,
) -> torch.Tensor:
    text_chunks = [item["sentence_chunk"] for item in pages_and_chunks]
    embeddings = model.encode(
        text_chunks,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True,
    )
    for item, emb in zip(pages_and_chunks, embeddings):
        item["embedding"] = emb.cpu().numpy()
    return embeddings


def save_chunks_with_embeddings(pages_and_chunks: list[dict], csv_path: Path) -> Path:
    df = pd.DataFrame(pages_and_chunks)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path


def load_chunks_with_embeddings(
    csv_path: Path, device: str
) -> tuple[list[dict], torch.Tensor]:
    df = pd.read_csv(csv_path)
    df["embedding"] = df["embedding"].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=" ")
    )
    embeddings = torch.tensor(np.stack(df["embedding"].tolist(), axis=0)).float()
    embeddings = embeddings.to(device)
    pages_and_chunks = df.to_dict(orient="records")
    return pages_and_chunks, embeddings
