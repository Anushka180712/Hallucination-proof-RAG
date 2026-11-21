"""Command-line interface.

    hprag ingest data/corpus                 # build and save the index
    hprag ask "When was Halcyon commissioned?"
    hprag ask "..." --json                   # full machine-readable result
    hprag eval data/eval/halcyon_qa.jsonl --compare-baseline
    hprag ui                                 # web app: ask your own questions in the browser
    hprag serve --port 8000                  # REST API (needs hprag[api])
    hprag demo                               # guided tour, no setup needed

Built on ``argparse`` (standard library) so the CLI works on the core install.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hprag.config import Settings
from hprag.logging_utils import configure_logging, get_logger

log = get_logger("cli")
DEFAULT_INDEX = ".hprag_index"
# Repo checkout: <repo>/src/hprag/cli.py -> <repo>/data/corpus. Falls back to ./data/corpus
# so `hprag demo` also works from a non-editable install run inside the repo.
_REPO_CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus"
BUNDLED_CORPUS = _REPO_CORPUS if _REPO_CORPUS.exists() else Path("data") / "corpus"


def _settings(args: argparse.Namespace) -> Settings:
    s = Settings.load(args.config)
    if args.provider:
        s.llm.provider = args.provider
    if args.verifier:
        s.verifier.kind = args.verifier
    if args.log_level:
        s.log_level = args.log_level
    if args.log_format:
        s.log_format = args.log_format
    configure_logging(s.log_level, s.log_format)
    # One line that answers "which models and thresholds was this run using?"
    log.info("config.loaded", extra={"file": args.config or "-", "generator": s.llm.provider,
                                     "verifier": s.verifier.kind, "embedder": s.retrieval.embedder,
                                     "reranker": s.retrieval.reranker,
                                     "min_confidence": s.guard.min_retrieval_confidence,
                                     "min_faithfulness": s.guard.min_faithfulness})
    return s


def _pipeline(s: Settings, index: str, corpus: str | None = None):
    from hprag.pipeline import RAGPipeline
    from hprag.retrieval import HybridRetriever

    if corpus:
        rag = RAGPipeline(s)
        rag.ingest_path(corpus)
        return rag
    if not Path(index, "meta.json").exists():
        sys.exit(f"No index at {index!r}. Run `hprag ingest <path>` first (or pass --corpus).")
    return RAGPipeline(s, retriever=HybridRetriever.load(index, s.retrieval))


def _print_answer(a, as_json: bool) -> None:
    if as_json:
        print(json.dumps(a.to_dict(), indent=2, ensure_ascii=False))
        return
    print()
    print(a.answer)
    print()
    if a.abstained:
        print(f"  ⛔ abstained: {a.abstain_reason}")
    else:
        print(f"  ✅ faithfulness {a.faithfulness:.0%} · retrieval confidence {a.confidence:.0%}")
    for c in a.claims:
        mark = {"supported": "✔", "contradicted": "✘", "unsupported": "?"}[c.verdict.value]
        print(f"     {mark} {c.verdict.value:<12} {c.text[:90]}")
    if a.citations:
        print("\n  Sources:")
        for c in a.citations:
            quote = " ".join(c.quote.split())[:110]
            print(f"   [{c.number}] {c.doc_id} (chars {c.start}-{c.end}): {quote}…")
    print(f"\n  trace_id={a.trace_id}")


def cmd_ingest(args) -> None:
    """`hprag ingest PATH`: chunk and index documents, then save the index to disk."""
    from hprag.pipeline import RAGPipeline

    s = _settings(args)
    rag = RAGPipeline(s)
    n = rag.ingest_path(args.path)
    rag.retriever.save(args.index)
    print(f"Indexed {n} chunks from {args.path} -> {args.index}")


def cmd_ask(args) -> None:
    """`hprag ask QUESTION`: answer one question and print the answer, verdicts and sources."""
    s = _settings(args)
    rag = _pipeline(s, args.index, args.corpus)
    _print_answer(rag.ask(args.question), args.json)


def cmd_eval(args) -> None:
    """`hprag eval DATASET`: run the evaluation harness, optionally against the naive baseline."""
    from hprag.evaluation import evaluate, format_report, load_dataset
    from hprag.evaluation.baseline import naive_pipeline

    s = _settings(args)
    rag = _pipeline(s, args.index, args.corpus)
    items = load_dataset(args.dataset)
    reports = []
    if args.compare_baseline:
        reports.append(evaluate(naive_pipeline(s, retriever=rag.retriever), items, "naive_rag"))
    reports.append(evaluate(rag, items, "hallucination_proof"))
    print(format_report(reports))
    if args.out:
        Path(args.out).write_text(json.dumps([r.to_dict() for r in reports], indent=2))
        print(f"\nFull per-question results written to {args.out}")


def cmd_serve(args) -> None:
    """`hprag serve`: start the REST API (and the UI at /) without opening a browser."""
    try:
        import uvicorn
    except ImportError:
        sys.exit("The API needs extras: pip install 'hprag[api]'")
    from hprag.api.server import create_app

    s = _settings(args)
    rag = _pipeline(s, args.index, args.corpus)
    uvicorn.run(create_app(rag), host=args.host, port=args.port, log_config=None)


def cmd_ui(args) -> None:
    """Start the API and open the web UI in the browser.

    Uses --index if it exists, otherwise --corpus, otherwise the bundled demo
    corpus, so `hprag ui` works with no setup at all.
    """
    try:
        import uvicorn
    except ImportError:
        sys.exit("The UI needs extras: pip install -e '.[api]'")
    import threading
    import webbrowser

    from hprag.api.server import create_app

    s = _settings(args)
    corpus = args.corpus
    if not corpus and not Path(args.index, "meta.json").exists():
        corpus = str(BUNDLED_CORPUS)
    rag = _pipeline(s, args.index, corpus)
    url = f"http://{args.host}:{args.port}"
    print(f"\n  Hallucination-Proof RAG is running at {url}\n  Press Ctrl+C to stop.\n")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(rag), host=args.host, port=args.port, log_config=None)


def cmd_demo(args) -> None:
    """`hprag demo`: a scripted terminal tour showing each safety mechanism once."""
    from hprag.generation import ScriptedGenerator
    from hprag.pipeline import RAGPipeline

    s = _settings(args)
    rag = RAGPipeline(s)
    rag.ingest_path(args.corpus or str(BUNDLED_CORPUS))
    tour = [
        ("1) An answerable question", "How much power do the solar arrays generate?", None),
        ("2) A question the knowledge base cannot answer", "Who is the station commander?", None),
        ("3) A draft that invents a fact (the verifier drops it)",
         "When was Halcyon commissioned?",
         "Halcyon was commissioned on 14 March 2041 [1]. It orbits at 412 kilometres [1]. "
         "The station also carries a small nuclear reactor for backup power [1]."),
        ("4) A draft that contradicts the sources (the answer is blocked)",
         "Are open flames allowed on the station?",
         "Open flames are permitted anywhere on the station [1]."),
    ]
    original = rag.generator
    for title, q, scripted in tour:
        print("\n" + "=" * 78 + f"\n{title}\nQ: {q}")
        rag.generator = ScriptedGenerator(scripted) if scripted else original
        if scripted:
            print(f"Draft from the (simulated) LLM:\n  {scripted}")
        _print_answer(rag.ask(q), False)


def build_parser() -> argparse.ArgumentParser:
    """Define all sub-commands and the options they share."""
    p = argparse.ArgumentParser(prog="hprag", description="Hallucination-proof RAG")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="TOML config file")
    common.add_argument("--index", default=DEFAULT_INDEX, help="index directory")
    common.add_argument("--provider", choices=["extractive", "anthropic"], help="answer generator")
    common.add_argument("--verifier", choices=["lexical", "nli", "llm", "ensemble"])
    common.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    common.add_argument("--log-format", choices=["pretty", "json"])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", parents=[common], help="index a file or directory")
    s.add_argument("path")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("ask", parents=[common], help="ask one question")
    s.add_argument("question")
    s.add_argument("--corpus", help="ingest this path on the fly instead of loading --index")
    s.add_argument("--json", action="store_true", help="print the full result as JSON")
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("eval", parents=[common], help="run the evaluation harness")
    s.add_argument("dataset")
    s.add_argument("--corpus", help="ingest this path on the fly instead of loading --index")
    s.add_argument("--compare-baseline", action="store_true", help="also evaluate a naive RAG baseline")
    s.add_argument("--out", help="write per-question JSON results here")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("serve", parents=[common], help="start the REST API")
    s.add_argument("--corpus")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("ui", parents=[common], help="open the web UI to ask your own questions")
    s.add_argument("--corpus", help="documents to load (default: --index if it exists, else the demo corpus)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    s.set_defaults(func=cmd_ui)

    s = sub.add_parser("demo", parents=[common], help="guided tour on the bundled corpus")
    s.add_argument("--corpus")
    s.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> None:
    """Entry point for the `hprag` console script."""
    args = build_parser().parse_args(argv)
    # The first line of every run names the command, so a log file is self-describing.
    log.debug("cli.command", extra={"command": args.func.__name__.removeprefix("cmd_")})
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
