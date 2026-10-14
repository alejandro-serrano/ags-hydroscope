"""notebooks/hydroscope_walkthrough.ipynb: saved with its outputs and without errors.

The notebook is not executed here (CI has no data); we only read the saved file.
"""

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "hydroscope_walkthrough.ipynb"


def code_cells() -> list[dict]:
    """The code cells of the saved notebook."""
    with open(NOTEBOOK) as f:
        notebook = json.load(f)
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_every_code_cell_was_executed():
    for cell in code_cells():
        assert cell["execution_count"] is not None


def test_no_output_is_an_error():
    for cell in code_cells():
        for output in cell["outputs"]:
            assert output["output_type"] != "error"
