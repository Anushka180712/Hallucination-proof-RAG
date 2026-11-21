"""Prompt templates for grounded generation.

Each rule in the system prompt targets a specific, well-known hallucination
pattern. The comment next to each rule says which one.
"""

from __future__ import annotations

from hprag.types import ScoredChunk

ABSTAIN_TOKEN = "INSUFFICIENT_EVIDENCE"

GROUNDED_SYSTEM = f"""You are a careful assistant that answers ONLY from the numbered sources provided.

Rules:
1. Use only facts stated in the sources. Do not use outside knowledge, even if you are sure it is true.
   (Prevents answers drawn from training data that the user cannot verify.)
2. End every sentence with the citation number(s) of the source(s) that state it, like [1] or [2, 3].
   (Forces each claim to be anchored, which makes it checkable.)
3. If the sources do not contain the answer, reply with exactly: {ABSTAIN_TOKEN}
   (Gives the model a sanctioned way to say "I don't know" instead of guessing.)
4. If the sources only partly answer the question, answer that part and say plainly what is not covered.
   (Prevents filling gaps with plausible inventions.)
5. Copy numbers, names, dates and units exactly as written in the sources.
   (Numeric drift is the most common subtle hallucination.)
6. Never combine facts from different sources into a new conclusion the sources do not state.
   (Prevents unsupported multi-hop inference.)
7. Be concise: short, plain sentences, each one making a single claim.
   (Atomic sentences are easier to verify claim by claim.)"""


def format_sources(hits: list[ScoredChunk]) -> str:
    """Number the chunks [1]..[n] with their document title, as the model will cite them."""
    blocks = []
    for i, h in enumerate(hits, 1):
        title = h.chunk.metadata.get("title", h.chunk.doc_id)
        blocks.append(f"[{i}] (from: {title})\n{h.chunk.text.strip()}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, hits: list[ScoredChunk], feedback: str | None = None) -> str:
    """Sources + question, plus verifier feedback when regenerating a failed draft."""
    prompt = f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"
    if feedback:
        # Used by the "regenerate" repair strategy: tell the model exactly which
        # of its previous claims failed verification, so it can drop or fix them.
        prompt += (
            "\n\nYour previous draft contained statements that are NOT supported by the sources:\n"
            f"{feedback}\nRewrite the answer using only statements the sources support, with citations."
        )
    return prompt
