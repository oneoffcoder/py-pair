from __future__ import annotations

from pypair.biserial import Biserial
from pypair.contingency import BinaryTable, CategoricalTable, ConfusionMatrix, AgreementTable
from pypair.continuous import Concordance, CorrelationRatio, Continuous
from pypair.typing import ArrayLike1D, DomainValues, MeasureValue, NumericArrayLike1D


def confusion(
    a: ArrayLike1D,
    b: ArrayLike1D,
    measure: str = "acc",
    a_0: object = 0,
    a_1: object = 1,
    b_0: object = 0,
    b_1: object = 1,
    pseudocount: bool = True,
) -> MeasureValue:
    """
    Gets the specified confusion matrix stats.

    :param a: Binary variable (iterable).
    :param b: Binary variable (iterable).
    :param measure: Measure. Default is `acc`.
    :param a_0: The a zero value. Default 0.
    :param a_1: The a one value. Default 1.
    :param b_0: The b zero value. Default 0.
    :param b_1: The b one value. Default 1.
    :return: Measure.
    """
    if measure not in ConfusionMatrix.measures():
        raise ValueError(f"{measure} is not a valid association measure.")
    return ConfusionMatrix(a, b, a_0=a_0, a_1=a_1, b_0=b_0, b_1=b_1, pseudocount=pseudocount).get(measure)


def binary_binary(
    a: ArrayLike1D,
    b: ArrayLike1D,
    measure: str = "chisq",
    a_0: object = 0,
    a_1: object = 1,
    b_0: object = 0,
    b_1: object = 1,
    pseudocount: bool = True,
) -> MeasureValue:
    """
    Gets the binary-binary association.

    :param a: Binary variable (iterable).
    :param b: Binary variable (iterable).
    :param measure: Measure. Default is `chisq`.
    :param a_0: The a zero value. Default 0.
    :param a_1: The a one value. Default 1.
    :param b_0: The b zero value. Default 0.
    :param b_1: The b one value. Default 1.
    :return: Measure.
    """
    if measure not in BinaryTable.measures():
        raise ValueError(f"{measure} is not a valid association measure.")
    return BinaryTable(a, b, a_0=a_0, a_1=a_1, b_0=b_0, b_1=b_1, pseudocount=pseudocount).get(measure)


def categorical_categorical(
    a: ArrayLike1D,
    b: ArrayLike1D,
    measure: str = "chisq",
    a_vals: DomainValues | None = None,
    b_vals: DomainValues | None = None,
    pseudocount: bool = True,
) -> MeasureValue:
    """
    Gets the categorical-categorical association.

    :param a: Categorical variable (iterable).
    :param b: Categorical variable (iterable).
    :param measure: Measure. Default is `chisq`.
    :param a_vals: The unique values in `a`.
    :param b_vals: The unique values in `b`.
    :return: Measure.
    """
    if measure not in CategoricalTable.measures():
        raise ValueError(f"{measure} is not a valid association measure.")
    return CategoricalTable(a, b, a_vals=a_vals, b_vals=b_vals, pseudocount=pseudocount).get(measure)


def agreement(
    a: ArrayLike1D,
    b: ArrayLike1D,
    measure: str = "chohen_k",
    a_vals: DomainValues | None = None,
    b_vals: DomainValues | None = None,
    pseudocount: bool = True,
) -> MeasureValue:
    """
    Gets the agreement association.

    :param a: Categorical variable (iterable).
    :param b: Categorical variable (iterable).
    :param measure: Measure. Default is `chohen_k`.
    :param a_vals: The unique values in `a`.
    :param b_vals: The unique values in `b`.
    :return: Measure.
    """
    if measure not in AgreementTable.measures():
        raise ValueError(f"{measure} is not a valid association measure.")
    return AgreementTable(a, b, a_vals=a_vals, b_vals=b_vals, pseudocount=pseudocount).get(measure)


def binary_continuous(
    b: ArrayLike1D,
    c: NumericArrayLike1D,
    measure: str = "biserial",
    b_0: object = 0,
    b_1: object = 1,
) -> MeasureValue:
    """
    Gets the binary-continuous association.

    :param b: Binary variable (iterable).
    :param c: Continuous variable (iterable).
    :param measure: Measure. Default is `biserial`.
    :param b_0: Value when `b` is zero. Default 0.
    :param b_1: Value when `b` is one. Default is 1.
    :return: Measure.
    """
    if measure not in Biserial.measures():
        raise ValueError(f"{measure} is not a valid association measure.")
    return Biserial(b, c, b_0=b_0, b_1=b_1).get(measure)


def categorical_continuous(x: ArrayLike1D, y: NumericArrayLike1D, measure: str = "eta") -> MeasureValue:
    """
    Gets the categorical-continuous association.

    :param x: Categorical variable (iterable).
    :param y: Continuous variable (iterable).
    :param measure: Measure. Default is `eta`.
    :return: Measure.
    """
    if measure not in CorrelationRatio.measures():
        raise ValueError(f"{measure} is not a valid association measure.")
    return CorrelationRatio(x, y).get(measure)


def concordance(x: NumericArrayLike1D, y: NumericArrayLike1D, measure: str = "kendall_tau") -> MeasureValue:
    """
    Gets the specified concordance between the two variables.

    :param x: Continuous or ordinal variable (iterable).
    :param y: Continuous or ordinal variable (iterable).
    :param measure: Measure. Default is `kendall_tau`.
    :return: Measure.
    """
    if measure not in Concordance.measures():
        raise ValueError(f"{measure} is not a valid association measure.")
    return Concordance(x, y).get(measure)


def continuous_continuous(
    x: NumericArrayLike1D,
    y: NumericArrayLike1D,
    measure: str = "pearson",
) -> MeasureValue:
    """
    Gets the continuous-continuous association.

    :param x: Continuous variable (iterable).
    :param y: Continuous variable (iterable).
    :param measure: Measure. Default is 'pearson'.
    :return: Measure.
    """
    if measure not in Continuous.measures():
        raise ValueError(f"{measure} is not a valid association measure.")
    return Continuous(x, y).get(measure)
