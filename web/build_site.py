"""Build the static GitHub Pages site: the web UI plus everything it needs to run in the browser.

Usage:  python web/build_site.py [OUT_DIR]      (default: _site)

Output layout:
    index.html      the same UI as `hprag ui`, with bridge.js loaded first
    bridge.js       starts Python (Pyodide) in the page and answers the UI's requests
    hprag-*.whl     this package, installed into the in-browser Python by micropip
    corpus/*.md     the demo documents, indexed in the browser on page load
    manifest.json   tells bridge.js the wheel's file name and the corpus files
    .nojekyll       stops GitHub Pages from running Jekyll over the files

Why build a wheel instead of copying the source? micropip installs wheels, and a wheel
is exactly what `pip install` would produce, so the browser runs the real package.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build(out: Path) -> Path:
    """Write the site into ``out`` (replacing it) and return ``out``."""
    if out.exists():
        shutil.rmtree(out)
    (out / "corpus").mkdir(parents=True)

    # 1. The package as a wheel. --no-deps: numpy comes from Pyodide's own build.
    subprocess.run([sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps",
                    "-q", "-w", str(out)], check=True)
    wheel = next(out.glob("hprag-*.whl")).name

    # 2. The documents.
    corpus = sorted(p.name for p in (ROOT / "data" / "corpus").iterdir() if p.is_file())
    for name in corpus:
        shutil.copy(ROOT / "data" / "corpus" / name, out / "corpus" / name)

    # 3. The UI, with the bridge loaded before the UI's own script so api() finds it.
    html = (ROOT / "src" / "hprag" / "api" / "static" / "index.html").read_text(encoding="utf-8")
    marker = "<script>"
    if marker not in html:
        raise SystemExit("index.html has no inline <script>; cannot inject bridge.js")
    html = html.replace(marker, '<script src="bridge.js"></script>\n' + marker, 1)
    (out / "index.html").write_text(html, encoding="utf-8")
    shutil.copy(ROOT / "web" / "bridge.js", out / "bridge.js")

    (out / "manifest.json").write_text(json.dumps({"wheel": wheel, "corpus": corpus}, indent=2))
    (out / ".nojekyll").write_text("")
    print(f"Built {out} with {wheel} and {len(corpus)} documents")
    return out


if __name__ == "__main__":
    build(Path(sys.argv[1] if len(sys.argv) > 1 else "_site").resolve())
