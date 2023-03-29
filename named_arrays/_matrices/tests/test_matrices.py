import numpy as np
import pytest
import astropy.units as u
import named_arrays as na
import named_arrays._vectors.tests.test_vectors

__all__ = [
    'AbstractTestAbstractMatrixArray',
    'AbstractTestAbstractExplicitMatrixArray',
    'AbstractTestAbstractExplicitMatrixArrayCreation',
    'AbstractTestAbstractImplicitMatrixArray',
    'AbstractTestAbstractMatrixRandomSample',
    'AbstractTestAbstractMatrixUniformRandomSample',
    'AbstractTestAbstractMatrixNormalRandomSample',
    'AbstractTestAbstractParameterizedMatrixArray',
]


class AbstractTestAbstractMatrixArray(
    named_arrays._vectors.tests.test_vectors.AbstractTestAbstractVectorArray,
):

    def test_rows(self, array: na.AbstractMatrixArray):
        assert np.all(array.rows == array.components)

    def test_is_consistent(self, array: na.AbstractMatrixArray):

        assert array.is_consistent

        array_2 = array.copy()
        array_2.x = 5
        assert not array_2.is_consistent

    def test_is_square(self, array: na.AbstractMatrixArray):
        assert array.is_square

    def test_matrix_transpose(self, array: na.AbstractMatrixArray):
        result = array.matrix_transpose

        assert isinstance(result, array.components[next(iter(array.components))].type_matrix)
        assert all(isinstance(result.components[r], array.type_vector) for r in result.components)

        assert np.all(result.matrix_transpose == array)
        assert np.all(result + result == (array + array).matrix_transpose)
        assert np.all(result @ result == (array @ array).matrix_transpose)
        assert np.all(2 * result == (2 * array).matrix_transpose)
        assert np.all(result.determinant == array.determinant)
        assert np.all(result.inverse == (array.inverse).matrix_transpose)

    def test_determinant(self, array: na.AbstractMatrixArray):
        result = array.determinant

        assert isinstance(result, (float, complex, u.Quantity, na.AbstractScalar))

        assert np.all((2**2) * result == (2 * array).determinant)
        assert np.all(result == array.matrix_transpose.determinant)
        assert np.allclose(result * result, (array @ array).determinant)
        assert np.allclose(result, 1 / array.inverse.determinant)

    def test_inverse(self, array: na.AbstractMatrixArray):
        result = array.inverse

        assert isinstance(result, na.AbstractMatrixArray)

        identity = na.Cartesian2dMatrixArray(
            x=na.Cartesian2dVectorArray(1, 0),
            y=na.Cartesian2dVectorArray(0, 1),
        )

        assert np.allclose(array @ result, identity)
        assert np.allclose(result.inverse, array)
        assert np.allclose((2 * array).inverse, result / 2)

    class TestMatmul(
        named_arrays._vectors.tests.test_vectors.AbstractTestAbstractVectorArray.TestMatmul
    ):

        def test_matmul(
                self,
                array: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray | na.AbstractMatrixArray,
                array_2: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray | na.AbstractMatrixArray,
        ):
            assert np.all(array @ (array_2 + array_2) == array @ array_2 + array @ array_2)
            assert np.all((array + array) @ array_2 == array @ array_2 + array @ array_2)
            assert np.all(2 * (array @ array_2) == (2 * array) @ array_2)
            assert np.all((array @ array_2) * 2 == array @ (array_2 * 2))


class AbstractTestAbstractExplicitMatrixArray(
    AbstractTestAbstractMatrixArray,
    named_arrays._vectors.tests.test_vectors.AbstractTestAbstractExplicitVectorArray,
):
    pass


class AbstractTestAbstractExplicitMatrixArrayCreation(
    named_arrays._vectors.tests.test_vectors.AbstractTestAbstractExplicitVectorArrayCreation,
):
    pass


class AbstractTestAbstractImplicitMatrixArray(
    AbstractTestAbstractMatrixArray,
    named_arrays._vectors.tests.test_vectors.AbstractTestAbstractImplicitVectorArray,
):
    pass


class AbstractTestAbstractMatrixRandomSample(
    AbstractTestAbstractImplicitMatrixArray,
    named_arrays._vectors.tests.test_vectors.AbstractTestAbstractVectorRandomSample,
):
    pass


class AbstractTestAbstractMatrixUniformRandomSample(
    AbstractTestAbstractMatrixRandomSample,
    named_arrays._vectors.tests.test_vectors.AbstractTestAbstractVectorUniformRandomSample,
):
    pass


class AbstractTestAbstractMatrixNormalRandomSample(
    AbstractTestAbstractMatrixRandomSample,
    named_arrays._vectors.tests.test_vectors.AbstractTestAbstractVectorNormalRandomSample,
):
    pass


class AbstractTestAbstractParameterizedMatrixArray(
    AbstractTestAbstractImplicitMatrixArray,
    named_arrays._vectors.tests.test_vectors.AbstractTestAbstractParameterizedVectorArray,
):
    pass
