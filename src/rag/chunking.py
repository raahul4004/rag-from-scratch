import re

from spacy.lang.en import English
from tqdm.auto import tqdm


def build_sentencizer() -> English:
    nlp = English()
    nlp.add_pipe("sentencizer")
    return nlp


def split_list(input_list: list, slice_size: int) -> list[list]:
    return [input_list[i : i + slice_size] for i in range(0, len(input_list), slice_size)]


def attach_sentences(pages_and_texts: list[dict], nlp: English | None = None) -> list[dict]:
    nlp = nlp or build_sentencizer()
    for item in tqdm(pages_and_texts, desc="Sentencizing"):
        sentences = list(nlp(item["text"]).sents)
        item["sentences"] = [str(s) for s in sentences]
        item["page_sentence_count_spacy"] = len(item["sentences"])
    return pages_and_texts


def attach_sentence_chunks(pages_and_texts: list[dict], slice_size: int = 10) -> list[dict]:
    for item in tqdm(pages_and_texts, desc="Chunking"):
        item["sentence_chunks"] = split_list(item["sentences"], slice_size=slice_size)
        item["num_chunks"] = len(item["sentence_chunks"])
    return pages_and_texts


def flatten_chunks(pages_and_texts: list[dict]) -> list[dict]:
    pages_and_chunks: list[dict] = []
    for item in tqdm(pages_and_texts, desc="Flattening"):
        for sentence_chunk in item["sentence_chunks"]:
            joined = "".join(sentence_chunk).replace("  ", " ").strip()
            joined = re.sub(r"\.([A-Z])", r". \1", joined)
            pages_and_chunks.append(
                {
                    "page_number": item["page_number"],
                    "sentence_chunk": joined,
                    "chunk_char_count": len(joined),
                    "chunk_word_count": len(joined.split(" ")),
                    "chunk_token_count": len(joined) / 4,
                }
            )
    return pages_and_chunks


def filter_short_chunks(pages_and_chunks: list[dict], min_token_length: int = 30) -> list[dict]:
    return [c for c in pages_and_chunks if c["chunk_token_count"] > min_token_length]
