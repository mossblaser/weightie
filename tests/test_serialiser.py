import pytest

from typing import Any, Iterable, NamedTuple

from pathlib import Path

from itertools import zip_longest

from collections import OrderedDict

from numpy.typing import NDArray
import numpy as np

from weightie.serialiser import load, dump


def iter_weights(weights: Any) -> Iterable[tuple[str, Any]]:
    """Iterate over (label, value) pairs in a weights dictionary."""
    to_yield = [("weights", weights)]
    while to_yield:
        prefix, weights = to_yield.pop()
        if hasattr(weights, "_fields"):
            to_yield.extend(
                (f"{prefix}.{field}", getattr(weights, field))
                for field in weights._fields
            )
        elif isinstance(weights, (list, tuple)):
            to_yield.extend((f"{prefix}[{i}]", x) for i, x in enumerate(weights))
        elif isinstance(weights, dict):
            to_yield.extend((f"{prefix}[{k!r}]", x) for k, x in weights.items())
        else:
            yield (prefix, weights)


class MyNamedTuple(NamedTuple):
    arr: NDArray
    str: str
    num: int
    obj: Any


@pytest.fixture
def fake_weights() -> Any:
    """
    A set of fake weights which excercises all of the supported data types.
    """
    np.random.seed(0)

    def make_random_array() -> NDArray:
        return np.random.normal(
            size=tuple(
                np.random.randint(1, 128) for _ in range(np.random.randint(1, 5))
            )
        ).astype(np.random.choice(np.array([np.int64, np.float32, np.float64])))

    return {
        "list": [make_random_array(), "foo", 123, print],
        "tuple": (make_random_array(), "foo", 123, print),
        "named_tuple": MyNamedTuple(make_random_array(), "foo", 123, print),
        "dict": {
            "arr": make_random_array(),
            "str": "foo",
            "num": 123,
            "obj": print,
        },
        "ordered_dict": OrderedDict(
            [
                ("arr", make_random_array()),
                ("str", "foo"),
                ("num", 123),
                ("obj", print),
            ]
        ),
    }


def test_roundtrip(fake_weights: Any, tmp_path: Path) -> None:
    filename = tmp_path / "weights"

    with filename.open("wb") as f:
        dump(fake_weights, f)

    with filename.open("rb") as f:
        loaded_weights = load(f)

    for (a_label, a_value), (b_label, b_value) in zip_longest(
        iter_weights(fake_weights), iter_weights(loaded_weights), fillvalue=("", None)
    ):
        assert a_label == b_label
        print(a_label)

        assert type(a_value) == type(b_value)
        if isinstance(a_value, np.ndarray) and isinstance(b_value, np.ndarray):
            assert a_value.dtype == b_value.dtype
            assert np.array_equal(a_value, b_value)
        else:
            assert a_value == b_value
