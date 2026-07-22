"""Regression checks that expose, but do not silently repair, legacy splits."""

from __future__ import annotations

import unittest

import splits


def _mug_subjects(paths: list[str]) -> set[str]:
    subjects: set[str] = set()
    for path in paths:
        parts = path.split("/")
        try:
            subject_index = parts.index("mug128") + 1
        except ValueError as error:
            raise AssertionError(f"Unexpected legacy MUG path: {path}") from error
        subjects.add(parts[subject_index])
    return subjects


class LegacySplitAuditTests(unittest.TestCase):
    def test_main_split_snapshot(self) -> None:
        self.assertEqual(len(splits.TRAIN_PATHS), 722)
        self.assertEqual(len(splits.TEST_PATHS), 310)
        self.assertEqual(len(_mug_subjects(splits.TRAIN_PATHS)), 36)
        self.assertEqual(len(_mug_subjects(splits.TEST_PATHS)), 17)

    def test_known_subject_overlap_is_explicit(self) -> None:
        overlap = _mug_subjects(splits.TRAIN_PATHS) & _mug_subjects(
            splits.TEST_PATHS
        )

        self.assertEqual(overlap, {"084"})
        self.assertEqual(
            sum("/084/" in path for path in splits.TRAIN_PATHS),
            17,
        )
        self.assertEqual(
            sum("/084/" in path for path in splits.TEST_PATHS),
            5,
        )


if __name__ == "__main__":
    unittest.main()
