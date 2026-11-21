# Contributing

Thanks for helping make RAG more trustworthy.

## Setup

```bash
git clone https://github.com/Anushka180712/hallucination-proof-rag.git
cd hallucination-proof-rag
make install
make test lint
```

## Ground rules

1. **Every change keeps the test suite offline.** Tests must not need network access, API keys or model downloads. Use `ScriptedGenerator` or a fake client (see `tests/test_llm_providers.py`).
2. **Safety changes come with numbers.** If you change a threshold, a verifier or a gate, run `make eval` and put the before/after table in your pull request. Lowering `hallucination_rate` matters most; report `false_abstention_rate` too.
3. **Explain the why.** New modules start with a docstring explaining the problem they solve and the choice made. Significant decisions get an entry in `docs/DESIGN_DECISIONS.md`.
4. **Log decisions, not noise.** Use `get_logger(__name__)`, pass structured fields via `extra={...}`, and wrap timed steps in `stage(...)`. Don't use the reserved `LogRecord` names (`name`, `msg`, `args`, ...) as field keys.

## Adding a provider

```python
class MyGenerator:
    name = "my-provider"

    def generate(self, question, hits, feedback=None) -> str:
        prompt = build_user_prompt(question, hits, feedback)
        ...  # call your model with GROUNDED_SYSTEM + prompt
        return text  # must contain [n] citations, or INSUFFICIENT_EVIDENCE
```

Register it in `make_generator` in `src/hprag/generation/providers.py` and add a test with a fake client.

## Adding a verifier

Implement `verify(claims, sources, threshold) -> claims`, finishing each claim with `_finalise(...)` so citation validation and logging stay consistent. Register it in `make_verifier`.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/): `feat(retrieval): ...`, `fix(verification): ...`, `docs: ...`, `test: ...`.
