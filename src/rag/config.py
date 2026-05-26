from dataclasses import dataclass
from pathlib import Path


def pick_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class Config:
    pdf_url: str = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
    pdf_path: Path = DATA_DIR / "human-nutrition-text.pdf"
    embeddings_csv: Path = DATA_DIR / "text_chunks_and_embeddings_df.csv"

    page_number_offset: int = -41
    num_sentence_chunk_size: int = 10
    min_token_length: int = 30

    embedding_model_name: str = "all-mpnet-base-v2"
    embedding_batch_size: int = 16

    llm_model_id: str = "unsloth/Phi-3-mini-4k-instruct"
    use_quantization: bool = True
    max_new_tokens: int = 256
    temperature: float = 0.7

    n_resources_to_return: int = 5
    device: str = ""

    def __post_init__(self) -> None:
        if not self.device:
            self.device = pick_device()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
