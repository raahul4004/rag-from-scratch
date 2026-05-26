SYSTEM_PROMPT = """You are a helpful nutrition assistant answering questions using passages from a human nutrition textbook.

Rules:
- Answer only from the provided context. If the context does not contain the answer, say so plainly.
- Write in clear, well-formed English sentences. No bullet headers, no scaffolding tokens, no meta commentary.
- Be specific and explanatory; aim for 3-6 sentences unless the question demands more.
- Do not repeat the question and do not preface your answer with phrases like "Based on the context"."""


USER_TEMPLATE = """Context passages from the textbook:
{context}

Question: {query}"""


def format_prompt(query: str, context_items: list[dict], tokenizer=None) -> tuple[str, str]:
    """Build a prompt for the LLM.

    If a tokenizer with a chat template is provided, use it (proper instruction
    formatting for Phi-3). Otherwise fall back to a plain string template.

    Returns (prompt, base_prompt) where base_prompt is what the model sees as
    its context (used to strip the prompt from the decoded output).
    """
    context = "\n\n".join(
        f"[page {item.get('page_number', '?')}] {item['sentence_chunk']}"
        for item in context_items
    )
    user_msg = USER_TEMPLATE.format(context=context, query=query)

    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt, prompt

    fallback = f"{SYSTEM_PROMPT}\n\n{user_msg}\n\nAnswer:"
    return fallback, fallback


SAMPLE_QUERIES = [
    "What are the macronutrients, and what roles do they play in the human body?",
    "How do vitamins and minerals differ in their roles and importance for health?",
    "Describe the process of digestion and absorption of nutrients in the human body.",
    "What role does fibre play in digestion? Name five fibre containing foods.",
    "Explain the concept of energy balance and its importance in weight management.",
    "How often should infants be breastfed?",
    "What are symptoms of pellagra?",
    "How does saliva help with digestion?",
    "What is the RDI for protein per day?",
    "water soluble vitamins",
]
