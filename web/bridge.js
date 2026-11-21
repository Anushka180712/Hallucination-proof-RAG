// Runs the Hallucination-Proof RAG pipeline inside the browser, for GitHub Pages.
//
// Why: GitHub Pages serves static files only, so there is no server to answer the
// UI's requests. This script loads Python (Pyodide, CPython compiled to WebAssembly),
// installs the hprag package from the wheel published next to this page, indexes
// the bundled documents, and routes the UI's api() calls to hprag.browser.BrowserApp.
// Everything runs on the visitor's machine; nothing is sent anywhere.
//
// The UI only needs one hook: if window.hpragBridge exists, api() calls it instead
// of fetch(). See src/hprag/api/static/index.html.
"use strict";

const PYODIDE_VERSION = "0.29.3";  // pinned: a known-good build that ships numpy for Python 3.13
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

function setLoading(text) {
  const s = document.getElementById("status");
  if (s) { s.replaceChildren(); const p = document.createElement("span"); p.className = "pill"; p.textContent = text; s.append(p); }
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const tag = document.createElement("script");
    tag.src = src; tag.onload = resolve; tag.onerror = () => reject(new Error("could not load " + src));
    document.head.append(tag);
  });
}

async function boot() {
  const t0 = performance.now();
  // manifest.json is written by web/build_site.py: the wheel's file name and the corpus files.
  const manifest = await (await fetch("manifest.json")).json();
  setLoading("Loading Python in your browser (first visit takes ~10 s)…");
  await loadScript(PYODIDE_URL + "pyodide.js");
  const pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
  setLoading("Installing numpy and hprag…");
  await pyodide.loadPackage(["numpy", "micropip"]);
  const micropip = pyodide.pyimport("micropip");
  await micropip.install(new URL(manifest.wheel, location.href).href);

  setLoading("Indexing the documents…");
  pyodide.FS.mkdirTree("/corpus");
  await Promise.all(manifest.corpus.map(async (name) => {
    const text = await (await fetch("corpus/" + name)).text();
    pyodide.FS.writeFile("/corpus/" + name, text);
  }));
  const app = pyodide.runPython(`
from hprag.browser import BrowserApp
BrowserApp("/corpus")
`);
  console.info(`[hprag] ready in ${Math.round(performance.now() - t0)} ms. Each answer's trace id appears in the log lines below.`);
  return app;
}

const ready = boot();
ready.catch((e) => { console.error(e); setLoading("Could not start Python in this browser: " + e.message); });

window.hpragBridge = async (path, body) => {
  const app = await ready;
  return JSON.parse(app.handle(path, body ? JSON.stringify(body) : null));
};
