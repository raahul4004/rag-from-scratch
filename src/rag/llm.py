import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_flash_attn_2_available

from .prompts import format_prompt
from .retrieval import retrieve_relevant_resources


def pick_attn_implementation() -> str:
    if torch.cuda.is_available() and is_flash_attn_2_available():
        if torch.cuda.get_device_capability(0)[0] >= 8:
            return "flash_attention_2"
    return "sdpa"


def load_llm(
    model_id: str,
    device: str,
    use_quantization: bool = True,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    attn_impl = pick_attn_implementation()
    print(f"[INFO] Using attention implementation: {attn_impl}")
    print(f"[INFO] Loading model: {model_id}")

    quant_config = None
    if use_quantization and device == "cuda":
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if device == "cpu":
        dtype = torch.float32
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    )

    if quant_config is None:
        model.to(device)

    return tokenizer, model


def ask(
    query: str,
    embeddings: torch.Tensor,
    pages_and_chunks: list[dict],
    embedding_model: SentenceTransformer,
    tokenizer: AutoTokenizer,
    llm_model: AutoModelForCausalLM,
    n_resources_to_return: int = 5,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    return_context: bool = False,
) -> str | tuple[str, list[dict]]:
    scores, indices = retrieve_relevant_resources(
        query=query,
        embeddings=embeddings,
        model=embedding_model,
        n_resources_to_return=n_resources_to_return,
    )
    context_items = [pages_and_chunks[int(i)] for i in indices]
    for i, item in enumerate(context_items):
        item["score"] = float(scores[i])

    prompt, base_prompt = format_prompt(query=query, context_items=context_items)

    input_ids = tokenizer(prompt, return_tensors="pt").to(llm_model.device)
    outputs = llm_model.generate(
        **input_ids,
        temperature=temperature,
        do_sample=True,
        max_new_tokens=max_new_tokens,
    )

    decoded = tokenizer.decode(outputs[0])
    for token in ("<s>", "<|user|>", "<|end|>", "<|assistant|>"):
        decoded = decoded.replace(token, "")
    answer = decoded[len(base_prompt) + 2 :].strip()

    if return_context:
        return answer, context_items
    return answer
