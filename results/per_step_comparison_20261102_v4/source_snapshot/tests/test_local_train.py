import numpy as np
from local_lab.train import metrics_from_predictions, CandidateStream


def example_arrays(n=4):
    solution = np.array([[((r * 3 + r // 3 + c) % 9) + 2 for c in range(9)]
                         for r in range(9)], dtype=np.uint8).reshape(1, 81)
    labels = np.repeat(solution, n, axis=0)
    inputs = labels.copy()
    inputs[:, ::2] = 1
    return dict(inputs=inputs, labels=labels, base_ids=np.arange(n, dtype=np.int64))


def test_metrics_separate_blank_cells_from_clue_copying_and_valid_sudoku():
    data = example_arrays()
    perfect = metrics_from_predictions(data["labels"], data["inputs"], data["labels"])
    assert perfect["blank_accuracy"] == perfect["exact_accuracy"] == perfect["valid_grid_fraction"] == 1
    copied = metrics_from_predictions(data["inputs"], data["inputs"], data["labels"])
    assert copied["given_agreement"] == 1
    assert copied["blank_accuracy"] == copied["exact_accuracy"] == copied["valid_grid_fraction"] == 0
    assert copied["all_accuracy"] == copied["clue_copy_accuracy"]


def test_valid_grid_must_respect_given_clues():
    data = example_arrays()
    shifted = ((data["labels"] - 2 + 1) % 9) + 2
    metrics = metrics_from_predictions(shifted, data["inputs"], data["labels"])
    assert metrics["valid_grid_fraction"] == 0


def test_fixed_candidate_stream_is_reproducible_and_does_not_mutate_source():
    data = example_arrays()
    first, second = CandidateStream(data, 3, 17), CandidateStream(data, 3, 17)
    original = data["inputs"].copy()
    last = -1
    for _ in range(3):
        left, right = first.next(), second.next()
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
        assert left["sample_ids"].min() > last
        last = left["sample_ids"].max()
        assert (left["labels"][left["inputs"] != 1] == left["inputs"][left["inputs"] != 1]).all()
    np.testing.assert_array_equal(data["inputs"], original)
