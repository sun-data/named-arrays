import numpy as np
import pytest
import astropy.units as u
import named_arrays as na
import named_arrays._vectors.tests.test_vectors

__all__ = [
    'AbstractTestAbstractMatrixArray',
    'AbstractTestAbstractExplicitMatrixArray',
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

        array_2 = array.explicit.copy()
        array_2.x = 5
        assert not array_2.is_consistent

    def test_is_square(self, array: na.AbstractMatrixArray):
        assert array.is_square == (len(array.row_prototype.entries) == len(array.column_prototype.entries))

    def test_cartesian_nd(self, array: na.AbstractMatrixArray):
        cartesian_nd = array.cartesian_nd
        assert isinstance(cartesian_nd, na.AbstractCartesianNdMatrixArray)
        for c in cartesian_nd.components:
            component = cartesian_nd.components[c]
            assert isinstance(component, na.AbstractCartesianNdVectorArray)
            for c2 in component.components:
                assert isinstance(na.as_named_array(component.components[c2]), na.AbstractScalar)

    def test_from_cartesian_nd(self, array: na.AbstractMatrixArray):
        assert np.all(array.type_explicit.from_cartesian_nd(array.cartesian_nd, like=array) == array)

    def test_matrix_transpose(self, array: na.AbstractMatrixArray):
        result = array.matrix_transpose

        assert isinstance(result, array.components[next(iter(array.components))].type_matrix)
        assert all(isinstance(result.components[r], array.type_vector) for r in result.components)

        assert np.all(result.matrix_transpose == array)
        assert np.all(result + result == (array + array).matrix_transpose)
        if array.row_prototype.cartesian_nd.components.keys() == array.column_prototype.cartesian_nd.components.keys():
            assert np.all(result @ result == (array @ array).matrix_transpose)
        assert np.all(2 * result == (2 * array).matrix_transpose)
        if array.is_square:
            assert np.all(result.determinant == array.determinant)
            assert np.all(result.inverse == (array.inverse).matrix_transpose)

    def test_matrix(self, array: na.AbstractMatrixArray):
        assert np.all(array.matrix == array)

    def test_determinant(self, array: na.AbstractMatrixArray):
        if array.is_square:
            result = array.determinant

            assert isinstance(result, (int, float, complex, u.Quantity, na.AbstractScalar))

            assert np.all((2 ** 2) * result == (2 * array).determinant)
            assert np.all(result == array.matrix_transpose.determinant)
            assert np.allclose(result * result, (array @ array).determinant)
            assert np.allclose(result, 1 / array.inverse.determinant)

    def test_inverse(self, array: na.AbstractMatrixArray):
        if array.is_square:
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
            try:
                assert np.all(array @ (array_2 + array_2) == array @ array_2 + array @ array_2)
                assert np.all((array + array) @ array_2 == array @ array_2 + array @ array_2)
                assert np.all(2 * (array @ array_2) == (2 * array) @ array_2)
                assert np.all((array @ array_2) * 2 == array @ (array_2 * 2))

            except TypeError:
                with pytest.raises(TypeError):
                    array @ array_2
                return

    class TestNamedArrayFunctions(
        named_arrays._vectors.tests.test_vectors.AbstractTestAbstractVectorArray.TestNamedArrayFunctions
    ):

        @pytest.mark.skip
        class TestPltPlotLikeFunctions(
            named_arrays._vectors.tests.test_vectors.AbstractTestAbstractVectorArray.TestNamedArrayFunctions.TestPltPlotLikeFunctions
        ):
            pass

        @pytest.mark.skip
        class TestOptimizeRoot(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestNamedArrayFunctions.TestOptimizeRoot,
        ):
            pass

        @pytest.mark.skip
        class TestOptimizeMinimum(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestNamedArrayFunctions.TestOptimizeMinimum,
        ):
            pass


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
