"""Offline checks for native Sudoku preparation and legal augmentation."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from local_lab import data


def solved(shift=0):
    return np.array([((row * 3 + row // 3 + col + shift) % 9) + 1
                     for row in range(9) for col in range(9)], dtype=np.uint8)


def record(shift=0, source="fixture"):
    solution = solved(shift)
    puzzle = solution.copy()
    puzzle[::2] = 0
    return source, "".join(str(x) if x else "." for x in puzzle), "".join(str(x) for x in solution), str(shift + 1)


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(data.HEADER)
    writer.writerows(rows)
    return stream.getvalue().encode()


class Response(io.BytesIO):
    def __init__(self, raw):
        super().__init__(raw)
        self.headers = {"ETag": '"fixture-etag"', "Content-Length": str(len(raw))}
        self.status = 206


class DataTests(unittest.TestCase):
    def test_valid_solution_and_clues(self):
        _, question, answer, _ = record()
        puzzle, solution = data.validate_sudoku(question, answer)
        self.assertEqual(puzzle.shape, (81,))
        self.assertTrue(np.all((puzzle == 0) | (puzzle == solution)))

    def test_invalid_solution_rejected(self):
        _, question, answer, _ = record()
        with self.assertRaises(ValueError):
            data.validate_sudoku(question, "1" * 81)
        with self.assertRaises(ValueError):
            data.validate_sudoku(question[:-1], answer)

    def test_conflicting_clue_rejected(self):
        _, question, answer, _ = record()
        with self.assertRaises(ValueError):
            data.validate_sudoku("2" + question[1:], answer)

    def test_augmentation_is_valid_reproducible_and_pure(self):
        puzzle, solution = data.validate_sudoku(*record()[1:3])
        inputs = np.stack([puzzle + 1] * 8)
        labels = np.stack([solution + 1] * 8)
        before_inputs, before_labels = inputs.copy(), labels.copy()
        first = data.augment_batch(inputs, labels, np.random.default_rng(17))
        second = data.augment_batch(inputs, labels, np.random.default_rng(17))
        self.assertTrue(np.array_equal(inputs, before_inputs))
        self.assertTrue(np.array_equal(labels, before_labels))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(first, second)))
        self.assertGreater(len({row.tobytes() for row in first[1]}), 1)
        self.assertTrue(np.all((first[0] == 1).sum(1) == (inputs == 1).sum(1)))
        for question, answer in zip(*first):
            data.validate_sudoku("".join(str(int(x) - 1) for x in question),
                                 "".join(str(int(x) - 1) for x in answer))

    def test_augmentation_rejects_non_native_shapes_tokens(self):
        with self.assertRaises(ValueError):
            data.augment_batch(np.ones((2, 80)), np.ones((2, 80)), np.random.default_rng())
        with self.assertRaises(ValueError):
            data.augment_batch(np.zeros((2, 81)), np.ones((2, 81)), np.random.default_rng())
        with self.assertRaises(ValueError):
            data.augment_batch(np.ones((2, 81)), np.full((2, 81), 2.), np.random.default_rng())

    def test_prepare_streams_prefix_deduplicates_and_caches(self):
        # Test row 0 is duplicated in training; row 1 repeats another accepted
        # training solution, so both are excluded before filling source/val.
        train_raw = csv_bytes([record(0), record(1), record(1), record(2), record(3)])
        test_raw = csv_bytes([record(0, "held-out")])
        metadata = json.dumps({"sha": data.REVISION, "cardData": {}}).encode()

        def opener(request, timeout=None):
            url = request.full_url if hasattr(request, "full_url") else request
            if url == data.API_URL:
                return Response(metadata)
            if url.endswith("README.md"):
                return Response(b"# Fixture dataset\n")
            if url.endswith("train.csv"):
                self.assertEqual(request.get_header("Range"), f"bytes=0-{data.MAX_STREAM_BYTES - 1}")
                return Response(train_raw)
            if url.endswith("test.csv"):
                return Response(test_raw)
            raise AssertionError(url)

        with tempfile.TemporaryDirectory() as temporary, patch.object(data.urllib.request, "urlopen", side_effect=opener) as network:
            root = Path(temporary) / "data"
            manifest = data.prepare_data(root, train_size=2, val_size=1, test_size=1)
            self.assertEqual(manifest["revision"], data.REVISION)
            self.assertIsNone(manifest["dataset_license_declared_in_card_or_api"])
            self.assertEqual(manifest["streams"]["train"]["skipped_rows"], {"duplicate_puzzle_or_solution": 2})
            pools = data.load_data(root)
            self.assertEqual(pools["train"]["inputs"].shape, (2, 81))
            self.assertEqual(pools["val"]["source_indices"].tolist(), [4])
            self.assertEqual(pools["test"]["base_ids"].tolist(), [1_000_000_000])
            self.assertEqual(pools["train"]["inputs"].dtype, np.uint8)
            self.assertEqual(network.call_count, 4)
            self.assertEqual(data.prepare_data(root, 2, 1, 1), manifest)
            self.assertEqual(network.call_count, 4)
            with self.assertRaises(ValueError):
                data.prepare_data(root, 3, 1, 1)
            path = root / "train.npz"
            path.write_bytes(path.read_bytes() + b"corruption")
            with self.assertRaises(ValueError):
                data.load_data(root)

    def test_sizes_and_nonempty_cache_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                data.prepare_data(temporary, 0, 1, 1)
            Path(temporary, "unrelated.txt").write_text("preserve")
            with self.assertRaises(FileExistsError):
                data.prepare_data(temporary, 1, 1, 1)

    def test_invalid_stream_caps_rejected_before_any_network_or_cache_write(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(data.urllib.request, "urlopen") as network:
            for cap in (0, -1, True, 1.5, data.MAX_STREAM_BYTES + 1):
                with self.assertRaises(ValueError, msg=str(cap)):
                    data.prepare_data(Path(temporary) / "unused", 1, 1, 1, max_stream_bytes=cap)
            network.assert_not_called()
            self.assertFalse(Path(temporary, "unused").exists())

    def test_explicit_stream_cap_is_enforced_even_if_server_ignores_range(self):
        raw = csv_bytes([record(0), record(1)])
        class CountingResponse(Response):
            def __init__(self, value):
                super().__init__(value)
                self.count = 0
            def read(self, amount=-1):
                value = super().read(amount)
                self.count += len(value)
                return value
        response = CountingResponse(raw)
        with tempfile.TemporaryDirectory() as temporary, patch.object(data.urllib.request, "urlopen", return_value=response) as network:
            root = Path(temporary)
            (root / "raw").mkdir()
            with self.assertRaisesRegex(RuntimeError, "bounded stream exhausted"):
                data._stream_records("test", root, 1, set(), set(), max_stream_bytes=64)
            self.assertEqual(response.count, 64)
            self.assertEqual(network.call_args.args[0].get_header("Range"), "bytes=0-63")

    def test_stream_records_actual_file_size_and_does_not_silently_claim_partial_download(self):
        raw = csv_bytes([record(0)])
        response = Response(raw)
        response.status = 200  # A small complete response rather than a range.
        with tempfile.TemporaryDirectory() as temporary, patch.object(data.urllib.request, "urlopen", return_value=response):
            root = Path(temporary)
            (root / "raw").mkdir()
            rows, audit = data._stream_records("test", root, 1, set(), set(), max_stream_bytes=4096)
            self.assertEqual(len(rows), 1)
            self.assertEqual(audit["official_file_bytes"], len(raw))
            self.assertTrue(audit["whole_official_file_downloaded"])
            self.assertEqual(audit["maximum_application_bytes"], 4096)

    def test_explicit_cap_and_official_range_are_retained_in_manifest(self):
        raw = csv_bytes([record(0), record(1)])
        response = Response(raw)
        response.headers["Content-Range"] = "bytes 0-4095/718819925"
        with tempfile.TemporaryDirectory() as temporary, patch.object(data.urllib.request, "urlopen", return_value=response):
            root = Path(temporary)
            (root / "raw").mkdir()
            _, audit = data._stream_records("test", root, 1, set(), set(), max_stream_bytes=4096)
            self.assertEqual(audit["official_file_bytes"], 718819925)
            self.assertFalse(audit["whole_official_file_downloaded"])
            self.assertEqual(audit["maximum_application_bytes"], 4096)


if __name__ == "__main__":
    unittest.main()
