from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy.typing as npt
import pandas as pd

__all__ = [
    "ArrayLike1D",
    "BinaryCounts",
    "ConcordanceCounts",
    "CountTable",
    "DomainValues",
    "MeasureComputer",
    "MeasureMap",
    "MeasureValue",
    "NumericArrayLike1D",
    "PairwiseAssociationFn",
    "ScalarMeasure",
    "SupportsToNumpy",
]


@runtime_checkable
class SupportsToNumpy(Protocol):
    """Common protocol for pandas / NumPy-style containers."""

    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool | None = None,
        **kwargs: Any,
    ) -> npt.NDArray[Any]: ...


ArrayLike1D: TypeAlias = npt.NDArray[Any] | SupportsToNumpy | Sequence[Any] | Iterable[Any]
NumericArrayLike1D: TypeAlias = npt.NDArray[Any] | SupportsToNumpy | Sequence[int | float] | Iterable[int | float]
DomainValues: TypeAlias = Sequence[Any]
CountTable: TypeAlias = Sequence[Sequence[int]]
BinaryCounts: TypeAlias = tuple[int, int, int, int]
ConcordanceCounts: TypeAlias = tuple[int, int, int, int, int, int]
ScalarMeasure: TypeAlias = int | float
MeasureValue: TypeAlias = ScalarMeasure | tuple[ScalarMeasure, ...]
MeasureMap: TypeAlias = dict[str, Any]
PairwiseAssociationFn: TypeAlias = Callable[[pd.Series, pd.Series], ScalarMeasure]


class MeasureComputer(Protocol):
    """Protocol for classes exposing named measures through ``get()``."""

    @classmethod
    def measures(cls) -> list[str]: ...

    def get(self, measure: str) -> Any: ...
