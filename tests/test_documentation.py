"""Checks for public Markdown documentation."""

from __future__ import annotations

from pathlib import Path
import re
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")


def _public_markdown_files() -> list[Path]:
    files = list(REPOSITORY_ROOT.glob("*.md"))
    files.extend((REPOSITORY_ROOT / "docs").rglob("*.md"))
    private_sources = {
        "CODEX_INSTRUCTIONS_TEMPORAL_COHERENCE_WITH_APPROVALS.md",
        "IX-1_IEEEACCESS2025.md",
    }
    return sorted(
        path for path in files if path.exists() and path.name not in private_sources
    )


class DocumentationTests(unittest.TestCase):
    def test_local_markdown_links_resolve(self) -> None:
        missing: list[str] = []
        for document in _public_markdown_files():
            for target in MARKDOWN_LINK.findall(
                document.read_text(encoding="utf-8")
            ):
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                path_text = target.split("#", maxsplit=1)[0]
                target_path = (document.parent / path_text).resolve()
                if not target_path.exists():
                    missing.append(f"{document}: {target}")

        self.assertEqual(missing, [])

    def test_unapproved_doi_is_absent(self) -> None:
        occurrences = [
            str(document)
            for document in _public_markdown_files()
            if "10.1109/ACCESS" in document.read_text(encoding="utf-8")
        ]

        self.assertEqual(occurrences, [])


if __name__ == "__main__":
    unittest.main()
