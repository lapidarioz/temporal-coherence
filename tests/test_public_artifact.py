"""Dataset-free checks for public-artifact hygiene."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class NotebookHygieneTests(unittest.TestCase):
    def test_notebooks_have_no_stored_outputs_or_attachments(self) -> None:
        notebooks = sorted(REPOSITORY_ROOT.rglob("*.ipynb"))
        self.assertGreater(len(notebooks), 0, "No notebooks were found to audit")

        violations: list[str] = []
        for path in notebooks:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook.get("cells", [])):
                if cell.get("attachments"):
                    violations.append(f"{path}: cell {index} has attachments")
                if cell.get("cell_type") != "code":
                    continue
                if cell.get("outputs"):
                    violations.append(f"{path}: cell {index} has outputs")
                if cell.get("execution_count") is not None:
                    violations.append(f"{path}: cell {index} has an execution count")

        self.assertEqual(violations, [])


class CredentialConfigurationTests(unittest.TestCase):
    def test_code_server_credential_file_is_absent(self) -> None:
        self.assertFalse((REPOSITORY_ROOT / "code-server.yml").exists())

    def test_compose_publishes_jupyter_on_loopback_only(self) -> None:
        compose = (REPOSITORY_ROOT / "compose.yaml").read_text(encoding="utf-8")

        self.assertIn('"127.0.0.1:5555:5555"', compose)
        self.assertNotIn("7458", compose)
        self.assertNotIn("hashed-password", compose)


if __name__ == "__main__":
    unittest.main()
