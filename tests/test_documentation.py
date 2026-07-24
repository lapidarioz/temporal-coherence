"""Checks for public Markdown documentation."""

from __future__ import annotations

from pathlib import Path
import re
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")
ARTICLE_DOI = "10.1109/ACCESS.2025.3612820"


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

    def test_article_doi_is_scoped_to_readme(self) -> None:
        occurrences: dict[str, int] = {}
        for document in _public_markdown_files():
            count = document.read_text(encoding="utf-8").count(ARTICLE_DOI)
            if count:
                occurrences[str(document.relative_to(REPOSITORY_ROOT))] = count

        self.assertEqual(occurrences, {"README.md": 3})

    def test_no_other_ieee_access_doi_is_present(self) -> None:
        unexpected: list[str] = []
        for document in _public_markdown_files():
            text = document.read_text(encoding="utf-8")
            without_article_doi = text.replace(ARTICLE_DOI, "")
            if "10.1109/ACCESS" in without_article_doi:
                unexpected.append(str(document.relative_to(REPOSITORY_ROOT)))

        self.assertEqual(unexpected, [])


if __name__ == "__main__":
    unittest.main()
