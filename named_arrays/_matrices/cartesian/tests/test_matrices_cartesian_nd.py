import pytest
import numpy as np
import named_arrays as na
import named_arrays.tests.test_core

_num_distribution = named_arrays.tests.test_core.num_distribution


def _matrix(n: int, uncertain: bool) -> na.CartesianNdMatrixArray:
    """
    A well-conditioned square matrix of the given size with no symmetry.

    Parameters
    ----------
    n
        The number of rows and columns.
    uncertain
        Whether each entry carries a distribution of samples, none of which
        is symmetric either.
    """
    rng = np.random.default_rng(0)
    values = rng.normal(size=(n, n)) + n * np.eye(n)
    rows = {}
    for i in range(n):
        components = {}
        for j in range(n):
            value = float(values[i, j])
            if uncertain:
                value = na.NormalUncertainScalarArray(
                    nominal=value,
                    width=0.05,
                    num_distribution=_num_distribution,
                )
            components[f"c{j}"] = value
        rows[f"r{i}"] = na.CartesianNdVectorArray(components)
    return na.CartesianNdMatrixArray(rows)


def _identity_error(product: na.AbstractCartesianNdMatrixArray) -> float:
    """
    The largest deviation of a matrix product from the identity, over the
    nominal values and, for uncertain entries, every sample of the
    distribution.
    """
    result = 0.0
    for r in product.rows:
        for c in product.rows[r].components:
            entry = na.as_named_array(product.rows[r].components[c])
            expected = 1.0 if r[1:] == c[1:] else 0.0
            values = [na.nominal(entry)]
            if isinstance(entry, na.AbstractUncertainScalarArray):
                values.append(entry.distribution)
            for value in values:
                error = np.abs(na.as_named_array(value) - expected).max()
                result = max(result, float(error.ndarray))
    return result


@pytest.mark.parametrize("n", [4, 5])
def test_inverse_of_a_matrix_with_no_symmetry(n: int):
    """
    The inverse of a matrix larger than 3x3, which is found numerically rather
    than from a closed form, is the inverse itself and not its transpose,
    which only a symmetric matrix would fail to tell apart.
    """
    matrix = _matrix(n, uncertain=False)
    inverse = matrix.inverse
    assert _identity_error(inverse @ matrix) < 1e-10
    assert _identity_error(matrix @ inverse) < 1e-10


@pytest.mark.parametrize("n", [2, 4, 5])
def test_inverse_of_a_matrix_with_uncertain_entries(n: int):
    """
    A matrix of uncertain entries is inverted for its nominal value and for
    every sample of its distribution, none of which is symmetric.
    """
    matrix = _matrix(n, uncertain=True)
    inverse = matrix.inverse
    assert _identity_error(inverse @ matrix) < 1e-10
    assert _identity_error(matrix @ inverse) < 1e-10


def test_matrix_inverse_of_an_uncertain_scalar():
    """
    An uncertain scalar is inverted as a matrix over two of its axes,
    separately for its nominal value and for every sample of its
    distribution.
    """
    n = 4
    rng = np.random.default_rng(0)
    nominal = na.ScalarArray(rng.normal(size=(n, n)) + n * np.eye(n), axes=("r", "c"))
    array = na.NormalUncertainScalarArray(
        nominal=nominal,
        width=0.05,
        num_distribution=_num_distribution,
    ).explicit

    inverse = array.matrix_inverse(axis_rows="r", axis_columns="c")

    # the rows of the inverse are labeled by the columns of the original and
    # vice versa, so bring each to the trailing axes accordingly
    def trailing(a: na.ScalarArray, axis_rows: str, axis_columns: str) -> np.ndarray:
        return np.moveaxis(
            a=a.ndarray,
            source=[a.axes.index(axis_rows), a.axes.index(axis_columns)],
            destination=[-2, -1],
        )

    for original, inverted in (
        (na.as_named_array(array.nominal), na.as_named_array(inverse.nominal)),
        (array.distribution, inverse.distribution),
    ):
        a = trailing(original, "r", "c")
        b = trailing(inverted, "c", "r")
        assert np.allclose(b @ a, np.eye(n))
        assert np.allclose(a @ b, np.eye(n))
