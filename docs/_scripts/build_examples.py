"""Generate the Quarto example gallery from the notebooks under ``examples/``.

The gallery is metadata-driven: there is no hand-maintained list of examples.
Every ``examples/**/*.ipynb`` carries a ``metadata.nefes`` block

.. code-block:: json

    {"title": "...", "description": "...", "category": "flow", "render": "full"}

and this script turns that into two things:

* ``render == "full"`` -> a copy is written into ``docs/examples/`` (with a Quarto
  front matter cell and a hidden Plotly-renderer cell) together with any data files
  the notebook references, and is **executed here** (nbclient); Quarto then renders it
  from the stored outputs.  Quarto's own notebook execution does not capture Plotly,
  hence the pre-execution.
* every notebook (``full`` or ``list``) -> one row in the generated
  ``docs/examples/index.qmd`` gallery table, linking to its rendered page (``full``)
  or to the source on GitHub (``list``).

The source notebooks are only read; their output-free state is never modified.
This runs as the Quarto ``pre-render`` step configured in ``docs/_quarto.yml``.

Add a notebook and it appears automatically; flip its ``render`` flag to ``full``
to promote it from a listed link to an executed page.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import nbformat
from nbclient import NotebookClient

DOCS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = DOCS_DIR.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
OUT_DIR = DOCS_DIR / "examples"

#  Blob base for linking listed (non-rendered) notebooks to their source.
GITHUB_BLOB = "https://github.com/cetinalanyalioglu/Nefes/blob/master"

#  Display order of the example layers; anything unlisted sorts last, alphabetically.
LAYER_ORDER = ["getting-started", "flow", "combustion", "acoustics", "thermoacoustics", "validation"]

#  Quarto's own notebook execution does not capture Plotly output, so the gallery
#  copies are executed here (nbclient) and rendered from stored outputs. The
#  ``notebook_connected`` renderer stores each figure as a small CDN-backed HTML div
#  (not the multi-megabyte inline library); this hidden cell selects it.
RENDERER_CELL = 'import plotly.io as pio\n\npio.renderers.default = "notebook_connected"'

#  Some notebooks carry a local-viewing workaround (offline Plotly + an injected MathJax
#  <script>) that bloats the page and clashes with the site's own math rendering. These
#  anchored patterns drop exactly that block from the executed copy, nothing else.
WORKAROUND_PATTERNS = [
    r"^[ \t]*#.*plotly LaTeX rendering.*\n",
    r"^[ \t]*from IPython\.display import display, HTML\n",
    r"^[ \t]*import plotly\.offline as pyo\n",
    r"^[ \t]*pyo\.init_notebook_mode\(\)\n",
    r"^[ \t]*display\(HTML\(\s*\n[ \t]*'<script src=\"https://cdnjs\.cloudflare\.com/ajax/libs/mathjax"
    r"[^\n]*\n[ \t]*\)\)\n",
]


def strip_workaround(src: str) -> str:
    """Remove the offline-Plotly / MathJax local-viewing block from a cell's source."""
    for pat in WORKAROUND_PATTERNS:
        src = re.sub(pat, "", src, flags=re.M)
    return src


#  Front matter and prose for the generated gallery index page. Generated in full
#  (rather than included) so it never depends on a pre-render output existing at
#  Quarto's input-scan time.
INDEX_FRONT_MATTER = (
    "---\n"
    'title: "Examples"\n'
    'description: "Runnable notebooks spanning flow, combustion, acoustics, and thermoacoustics."\n'
    "---\n"
)
INDEX_INTRO = (
    "The examples live as notebooks under "
    "[`examples/`](https://github.com/cetinalanyalioglu/Nefes/tree/master/examples) in the repository.\n"
    "A selection is executed at build time and shown here as full pages; the rest link out to their "
    "source (marked ↗) so you can run them yourself.\n"
    "Each notebook declares its own `title`, `description`, and a `render` flag in its `metadata.nefes` "
    "block, so this gallery is generated straight from the notebooks with no separate list to maintain.\n"
)

#  Human-readable gallery headings, in display order.
LAYER_TITLES = {
    "getting-started": "Getting started",
    "flow": "Flow",
    "combustion": "Combustion",
    "acoustics": "Acoustics",
    "thermoacoustics": "Thermoacoustics",
    "validation": "Validation",
}


def read_nefes_meta(nb_path: Path) -> dict | None:
    """Return the ``metadata.nefes`` block of a notebook, or ``None`` if absent."""
    with open(nb_path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("metadata", {}).get("nefes")


def referenced_data_files(nb: nbformat.NotebookNode, src_dir: Path) -> list[Path]:
    """Sibling non-notebook files whose basename is mentioned in the notebook source."""
    text = "\n".join(c.source for c in nb.cells)
    out = []
    for sib in src_dir.iterdir():
        if sib.is_file() and sib.suffix != ".ipynb" and sib.name in text:
            out.append(sib)
    return out


def front_matter_cell(title: str, description: str) -> nbformat.NotebookNode:
    """A raw cell of Quarto front matter; ``execute.enabled: false`` keeps Quarto from
    re-executing the notebook (it is pre-executed here) and discarding the stored plots."""
    body = (
        f"---\n"
        f"title: {json.dumps(title)}\n"
        f"description: {json.dumps(description)}\n"
        f"execute:\n"
        f"  enabled: false\n"
        f"---"
    )
    return nbformat.v4.new_raw_cell(body)


def source_link_cell(category: str, name: str) -> nbformat.NotebookNode:
    """A callout linking the rendered page back to the source notebook on GitHub."""
    rel = f"examples/{category}/{name}.ipynb"
    body = (
        ":::{.callout-note appearance='simple'}\n"
        f"Source notebook: [`{rel}`]({GITHUB_BLOB}/{rel}). "
        "This page is executed from it at build time.\n"
        ":::"
    )
    return nbformat.v4.new_markdown_cell(body)


def strip_leading_heading(cells: list[nbformat.NotebookNode]) -> None:
    """Drop the first markdown cell's leading ``# H1`` so it does not duplicate the title."""
    for cell in cells:
        if cell.cell_type == "markdown" and cell.source.strip():
            lines = cell.source.splitlines()
            if lines and lines[0].lstrip().startswith("# "):
                cell.source = "\n".join(lines[1:]).lstrip("\n")
            return


def build_full_copy(src_path: Path, meta: dict, name: str) -> None:
    """Write an executed gallery copy of a ``full`` notebook into ``docs/examples/``.

    The copy is executed here (nbclient) rather than by Quarto, because Quarto's own
    notebook execution does not capture Plotly output; a failure to execute raises and
    fails the build, which doubles as a regression gate.
    """
    nb = nbformat.read(src_path, as_version=4)
    strip_leading_heading(nb.cells)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.source = strip_workaround(cell.source)
    renderer = nbformat.v4.new_code_cell(RENDERER_CELL)
    renderer.metadata["tags"] = ["remove-cell"]  # keep the injected setup off the page
    nb.cells = [
        front_matter_cell(meta["title"], meta["description"]),
        renderer,
        source_link_cell(meta["category"], name),
        *nb.cells,
    ]
    #  data files must sit next to the copy before execution (relative loads run in OUT_DIR)
    for data_file in referenced_data_files(nb, src_path.parent):
        shutil.copy2(data_file, OUT_DIR / data_file.name)
    NotebookClient(nb, kernel_name="python3", timeout=900, resources={"metadata": {"path": str(OUT_DIR)}}).execute()
    nbformat.write(nb, OUT_DIR / f"{name}.ipynb")


def clean_output_dir() -> None:
    """Remove all previously generated gallery files (the whole directory is generated)."""
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)


def layer_rank(category: str) -> int:
    """Rank placing known layers in ``LAYER_ORDER`` and unknown ones last."""
    return LAYER_ORDER.index(category) if category in LAYER_ORDER else len(LAYER_ORDER)


def md_escape(text: str) -> str:
    """Escape the few characters that would break a Markdown table cell."""
    return text.replace("|", "\\|")


def render_gallery_markdown(items: list[dict]) -> str:
    """Group the collected items by layer and render one Markdown table per layer."""
    lines: list[str] = []
    for category in sorted({it["category"] for it in items}, key=layer_rank):
        rows = [it for it in items if it["category"] == category]
        rows.sort(key=lambda it: (not it["full"], it["title"]))
        lines.append(f"## {LAYER_TITLES.get(category, category.title())}\n")
        lines.append("| Example | What it shows |")
        lines.append("| :-- | :-- |")
        for it in rows:
            marker = "" if it["full"] else " ↗"
            link = f"[{md_escape(it['title'])}{marker}]({it['path']})"
            lines.append(f"| {link} | {md_escape(it['description'])} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    clean_output_dir()
    notebooks = sorted(EXAMPLES_DIR.glob("**/*.ipynb"))
    items: list[dict] = []
    full_names: set[str] = set()
    for src_path in notebooks:
        meta = read_nefes_meta(src_path)
        if not meta:
            print(f"[build_examples] skipping (no metadata.nefes): {src_path.relative_to(REPO_ROOT)}")
            continue
        name = src_path.stem
        category = meta.get("category", src_path.parent.name)
        full = meta.get("render", "list") == "full"
        if full:
            if name in full_names:
                raise ValueError(f"duplicate full-notebook name '{name}'; names must be unique across layers")
            full_names.add(name)
            build_full_copy(src_path, meta, name)
            path = f"{name}.html"
        else:
            path = f"{GITHUB_BLOB}/{src_path.relative_to(REPO_ROOT).as_posix()}"
        items.append(
            {
                "title": meta["title"],
                "description": meta["description"],
                "category": category,
                "path": path,
                "full": full,
            }
        )
    index = INDEX_FRONT_MATTER + "\n" + INDEX_INTRO + "\n" + render_gallery_markdown(items)
    (OUT_DIR / "index.qmd").write_text(index, encoding="utf-8")
    print(f"[build_examples] {len(items)} examples ({len(full_names)} executed) -> {OUT_DIR}")


if __name__ == "__main__":
    main()
