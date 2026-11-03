"""Bounded, pinned native Sudoku data for the local TRM experiment.

Only a small sequential CSV prefix is streamed. This is intentionally a local
diagnostic sample, not a representative reproduction of the full benchmark.
No Torch dependency is needed for preparation, auditing or augmentation.
"""
from __future__ import annotations

import csv
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import urllib.request

import numpy as np

REPOSITORY = "sapientinc/sudoku-extreme"
REVISION = "58942f96baeb572ca3127e2a9e9c70f330783d6b"
API_URL = f"https://huggingface.co/api/datasets/{REPOSITORY}/revision/{REVISION}"
BASE_URL = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/"
MAX_STREAM_BYTES = 8 * 1024 * 1024
CHUNK_BYTES = 65536
HEADER = ("source", "question", "answer", "rating")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _small_download(url, limit=262144):
    with urllib.request.urlopen(url, timeout=45) as response:
        raw = response.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"Metadata exceeds the bounded download limit: {url}")
    return raw


def validate_sudoku(question, answer):
    """Validate one complete solution and all supplied clues; return raw digits."""
    if len(question) != 81 or len(answer) != 81:
        raise ValueError("Sudoku question and answer must each have 81 cells")
    question = question.replace(".", "0")
    if any(char not in "0123456789" for char in question) or any(char not in "123456789" for char in answer):
        raise ValueError("Invalid Sudoku digit")
    puzzle = np.frombuffer(question.encode("ascii"), dtype=np.uint8).copy() - ord("0")
    solution = np.frombuffer(answer.encode("ascii"), dtype=np.uint8).copy() - ord("0")
    board = solution.reshape(9, 9)
    expected = set(range(1, 10))
    if (any(set(row.tolist()) != expected for row in board)
            or any(set(col.tolist()) != expected for col in board.T)
            or any(set(board[r:r + 3, c:c + 3].ravel().tolist()) != expected
                   for r in (0, 3, 6) for c in (0, 3, 6))):
        raise ValueError("Answer is not a valid Sudoku solution")
    if np.any((puzzle != 0) & (puzzle != solution)):
        raise ValueError("Puzzle clues disagree with the supplied solution")
    return puzzle, solution


def _stream_records(name, root, target, seen_puzzles, seen_solutions, train_size=None):
    """Consume a bounded HTTP prefix, retaining complete original CSV rows."""
    url = BASE_URL + name + ".csv"
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{MAX_STREAM_BYTES - 1}",
                                                 "User-Agent": "TransformerLab-local-bounded-data/1"})
    prefix_path = root / "raw" / f"{name}_consumed_prefix.csv"
    accepted, skipped, sources = [], Counter(), Counter()
    read_bytes, source_index = 0, -1
    header = None
    buffer = b""
    with urllib.request.urlopen(request, timeout=60) as response, prefix_path.open("wb") as cached:
        headers = dict(response.headers)
        etag = response.headers.get("ETag")
        content_length = response.headers.get("Content-Length")
        status = getattr(response, "status", None)
        finished = False
        while not finished:
            amount = min(CHUNK_BYTES, MAX_STREAM_BYTES - read_bytes)
            if amount <= 0:
                raise RuntimeError(f"{name}: bounded stream exhausted before selecting {target} unique valid rows")
            raw = response.read(amount)
            if not raw:
                raise RuntimeError(f"{name}: stream ended before selecting {target} unique valid rows")
            read_bytes += len(raw)
            buffer += raw
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                original = line + b"\n"
                cached.write(original)
                row = next(csv.reader([line.decode("utf-8-sig")]))
                if header is None:
                    header = tuple(row)
                    if header != HEADER:
                        raise ValueError(f"Unexpected official CSV header: {header}")
                    continue
                source_index += 1
                if len(row) != 4:
                    skipped["invalid_csv_width"] += 1
                    continue
                source, question, answer, rating = row
                try:
                    puzzle, solution = validate_sudoku(question, answer)
                    difficulty = int(rating)
                except ValueError:
                    skipped["invalid_solution_or_clues_or_rating"] += 1
                    continue
                puzzle_hash = _sha(puzzle.tobytes())
                solution_hash = _sha(solution.tobytes())
                if puzzle_hash in seen_puzzles or solution_hash in seen_solutions:
                    skipped["duplicate_puzzle_or_solution"] += 1
                    continue
                seen_puzzles.add(puzzle_hash)
                seen_solutions.add(solution_hash)
                selected_split = "test" if name == "test" else "train" if len(accepted) < train_size else "val"
                accepted.append({"puzzle": puzzle, "solution": solution, "raw": original,
                                 "source_index": source_index, "source": source, "rating": difficulty,
                                 "puzzle_sha256": puzzle_hash, "solution_sha256": solution_hash,
                                 "split": selected_split})
                sources[source] += 1
                if len(accepted) == target:
                    finished = True
                    break
    return accepted, {"requested_url": url, "revision": REVISION, "http_status": status,
                      "etag": etag, "response_content_length": int(content_length) if content_length else None,
                      "content_range": headers.get("Content-Range") or headers.get("content-range"),
                      "application_bytes_read": read_bytes, "cached_complete_prefix_bytes": prefix_path.stat().st_size,
                      "read_ahead_bytes_not_selected_or_cached": len(buffer),
                      "maximum_application_bytes": MAX_STREAM_BYTES, "whole_official_file_downloaded": False,
                      "original_rows_consumed": source_index + 1, "accepted_rows": len(accepted),
                      "skipped_rows": dict(skipped), "selected_source_counts": dict(sources),
                      "row_index_basis": "zero-based data rows, excluding the CSV header"}


def prepare_data(root, train_size=8192, val_size=512, test_size=512):
    """Cache pinned, deduplicated prefix pools and return their audit manifest."""
    sizes = {"train": train_size, "val": val_size, "test": test_size}
    if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in sizes.values()):
        raise ValueError("All split sizes must be positive integers")
    root = Path(root)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("revision") != REVISION or manifest.get("requested_sizes") != sizes:
            raise ValueError("Existing cache has a different pinned revision or requested sizes")
        for name, expected in manifest["file_sha256"].items():
            if _sha((root / name).read_bytes()) != expected:
                raise ValueError(f"Cached data hash differs: {name}")
        return manifest
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError("Incomplete or unrelated files exist in the requested data cache; choose an empty directory")
    (root / "raw").mkdir()
    metadata_bytes = _small_download(API_URL)
    metadata = json.loads(metadata_bytes)
    if metadata.get("sha") != REVISION:
        raise ValueError("Hugging Face metadata does not match the pinned dataset revision")
    (root / "upstream_metadata.json").write_bytes(metadata_bytes)
    card = _small_download(BASE_URL + "README.md")
    (root / "upstream_dataset_card.md").write_bytes(card)
    seen_puzzles, seen_solutions = set(), set()
    # Choose the official held-out test pool first, so any exact duplicates in
    # the training prefix are skipped rather than entering source/adapt pools.
    test, test_stream = _stream_records("test", root, test_size, seen_puzzles, seen_solutions)
    training, train_stream = _stream_records("train", root, train_size + val_size, seen_puzzles, seen_solutions, train_size)
    records = {"train": training[:train_size], "val": training[train_size:], "test": test}
    split_manifest = {}
    selected_rows = []
    for split, rows in records.items():
        inputs = np.stack([row["puzzle"] + 1 for row in rows]).astype(np.uint8)
        labels = np.stack([row["solution"] + 1 for row in rows]).astype(np.uint8)
        source_indices = np.array([row["source_index"] for row in rows], dtype=np.int64)
        base_ids = source_indices + (1_000_000_000 if split == "test" else 0)
        np.savez_compressed(root / f"{split}.npz", inputs=inputs, labels=labels, base_ids=base_ids,
                            source_indices=source_indices, puzzle_identifiers=np.zeros(len(rows), dtype=np.int32))
        with (root / "raw" / f"{split}_selected.csv").open("wb") as selected:
            selected.write((",".join(HEADER) + "\n").encode())
            for row in rows:
                selected.write(row["raw"])
                selected_rows.append({key: row[key] for key in ("split", "source_index", "source", "rating", "puzzle_sha256", "solution_sha256")})
        split_manifest[split] = {"n": len(rows), "shape": list(inputs.shape),
                                "official_source_split": "test" if split == "test" else "train",
                                "original_index_min": int(source_indices.min()), "original_index_max": int(source_indices.max()),
                                "source_counts": dict(Counter(row["source"] for row in rows)),
                                "rating_min": min(row["rating"] for row in rows), "rating_max": max(row["rating"] for row in rows),
                                "valid_solution_count": len(rows), "clue_consistent_count": len(rows),
                                "unique_puzzle_sha256": len({row["puzzle_sha256"] for row in rows}),
                                "unique_solution_sha256": len({row["solution_sha256"] for row in rows})}
    _write_json(root / "selected_rows.json", selected_rows)
    declared_license = metadata.get("cardData", {}).get("license")
    manifest = {"repository": REPOSITORY, "revision": REVISION, "metadata_api_url": API_URL,
                "dataset_card_url": f"https://huggingface.co/datasets/{REPOSITORY}/blob/{REVISION}/README.md",
                "dataset_license_declared_in_card_or_api": declared_license,
                "dataset_license_note": "Not declared in the pinned dataset card/API" if declared_license is None else "As declared in pinned Hugging Face metadata",
                "upstream_code_license": "MIT; TinyRecursiveModels/LICENSE is preserved separately and is not asserted to license this dataset",
                "requested_sizes": sizes, "splits": split_manifest, "streams": {"train": train_stream, "test": test_stream},
                "encoding": {"pad": 0, "blank": 1, "digits_1_to_9": [2, 10], "vocab_size": 11,
                             "puzzle_identifier": 0, "seq_len": 81},
                "selection": "Official test sequential prefix first; official train sequential prefix supplies train then validation; reject invalid rows and any already-selected puzzle OR solution hash",
                "cross_split_duplicate_puzzle_or_solution_count": 0,
                "augmentation": "On-the-fly train only: independent digit permutation, optional transpose, band/stack and within-band/stack row/column permutations, matching upstream Sudoku symmetries",
                "limitations": ["Sequential prefixes can be source- and difficulty-biased; these are not random samples of the full official dataset.",
                                "Validation is an adjacent held-out portion of the official training prefix, not a separately published benchmark split.",
                                "Exact puzzle and solution hash separation is enforced; mathematical equivalence of local train/validation puzzles is not independently solved.",
                                "Official card claims unique solutions and mathematically inequivalent train/test sets; local checks validate provided solutions and clues, not independent uniqueness proofs.",
                                "Pool size is not training exposure; the runner must report actual base puzzles seen and repeated augmented presentations."]}
    manifest["file_sha256"] = {str(path.relative_to(root)).replace("\\", "/"): _sha(path.read_bytes())
                               for path in sorted(root.rglob("*")) if path.is_file()}
    _write_json(manifest_path, manifest)
    return manifest


def load_data(root):
    """Load only local arrays; never downloads or imports Torch."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    result = {}
    puzzle_sets, solution_sets = {}, {}
    for split in ("train", "val", "test"):
        path = root / f"{split}.npz"
        if _sha(path.read_bytes()) != manifest["file_sha256"][path.name]:
            raise ValueError(f"Local data checksum mismatch: {path.name}")
        with np.load(path, allow_pickle=False) as values:
            result[split] = {key: values[key].copy() for key in values.files}
        inputs, labels = result[split]["inputs"], result[split]["labels"]
        n = manifest["requested_sizes"][split]
        if inputs.shape != (n, 81) or labels.shape != inputs.shape or not np.all((inputs >= 1) & (inputs <= 10)) or not np.all((labels >= 2) & (labels <= 10)):
            raise ValueError(f"Invalid native encoding/shape in {split}")
        puzzles, solutions = set(), set()
        for puzzle, solution in zip(inputs, labels):
            validate_sudoku("".join(str(int(x) - 1) for x in puzzle), "".join(str(int(x) - 1) for x in solution))
            puzzles.add(_sha((puzzle - 1).tobytes()))
            solutions.add(_sha((solution - 1).tobytes()))
        if len(puzzles) != n or len(solutions) != n:
            raise ValueError(f"Duplicate puzzle/solution in {split}")
        puzzle_sets[split], solution_sets[split] = puzzles, solutions
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if puzzle_sets[left] & puzzle_sets[right] or solution_sets[left] & solution_sets[right]:
            raise ValueError(f"Puzzle or solution leakage between {left} and {right}")
    return result


def augment_batch(inputs, labels, rng):
    """Apply independent legal Sudoku symmetries to each encoded train item."""
    inputs, labels = np.asarray(inputs), np.asarray(labels)
    if inputs.ndim != 2 or inputs.shape[1] != 81 or labels.shape != inputs.shape:
        raise ValueError("Inputs and labels must have matching [N,81] shapes")
    if not np.issubdtype(inputs.dtype, np.integer) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Native token arrays must use integer dtypes")
    if not np.all((inputs >= 1) & (inputs <= 10)) or not np.all((labels >= 2) & (labels <= 10)):
        raise ValueError("Augmentation expects native blank=1, digit=2..10 tokens")
    out_inputs, out_labels = np.empty_like(inputs), np.empty_like(labels)
    for index, (puzzle, solution) in enumerate(zip(inputs, labels)):
        digit_map = np.concatenate(([0], rng.permutation(np.arange(1, 10))))
        transpose = rng.random() < .5
        rows = np.concatenate([band * 3 + rng.permutation(3) for band in rng.permutation(3)])
        cols = np.concatenate([stack * 3 + rng.permutation(3) for stack in rng.permutation(3)])
        for source, target in ((puzzle, out_inputs), (solution, out_labels)):
            board = (source - 1).reshape(9, 9)
            if transpose:
                board = board.T
            target[index] = (digit_map[board[np.ix_(rows, cols)]].ravel() + 1).astype(source.dtype)
    return out_inputs, out_labels
