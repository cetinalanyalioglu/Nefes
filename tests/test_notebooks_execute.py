"""Execute the example notebooks end-to-end as a regression gate against the live API.

Skipped by default to keep the local suite fast; set ``NEFES_TEST_NOTEBOOKS=1`` (as CI does)
to run it. Each notebook is executed in its own directory so relative data loads and the
``sys.path`` bootstrap resolve; an in-memory copy is executed, so the on-disk notebooks are
never modified and stay output-free.
"""

import os
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"
NOTEBOOKS = sorted(EXAMPLES.glob("**/*.ipynb"))

pytestmark = pytest.mark.skipif(
    os.environ.get("NEFES_TEST_NOTEBOOKS") != "1",
    reason="set NEFES_TEST_NOTEBOOKS=1 to execute the example notebooks (slow)",
)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: str(p.relative_to(EXAMPLES)))
def test_notebook_executes(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(path.parent)
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, kernel_name="python3", timeout=900).execute()
