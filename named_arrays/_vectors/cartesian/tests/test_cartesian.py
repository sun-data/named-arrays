import numpy as np
import pytest

import named_arrays as na
import named_arrays._vectors.tests.test_vectors

__all__ = [
    'AbstractTestAbstractCartesianVectorArray',
    'AbstractTestAbstractExplicitCartesianVectorArray',
    'AbstractTestAbstractExplicitCartesianVectorArrayCreation',
    'AbstractTestAbstractImplicitCartesianVectorArray',
    'AbstractTestAbstractCartesianVectorRandomSample',
    'AbstractTestAbstractCartesianVectorUniformRandomSample',
    'AbstractTestAbstractCartesianVectorNormalRandomSample',
    'AbstractTestAbstractParameterizedCartesianVectorArray',
    'AbstractTestAbstractCartesianVectorArrayRange',
    'AbstractTestAbstractCartesianVectorSpace',
    'AbstractTestAbstractCartesianVectorLinearSpace',
    'AbstractTestAbstractCartesianVectorStratifiedRandomSpace',
    'AbstractTestAbstractCartesianVectorLogarithmicSpace',
    'AbstractTestAbstractCartesianVectorGeometricSpace',
]


class AbstractTestAbstractCartesianVectorArray(
    named_arrays._vectors.tests.test_vectors.AbstractTestAbstractVectorArray,
):

    class TestUfuncUnary(
        named_arrays._vectors.tests.test_vectors.AbstractTestAbstractVectorArray.TestUfuncUnary,
    ):
        def test_ufunc_unary(
                self,
                ufunc: np.ufunc,
                array: na.AbstractVectorArray,
                # out: bool,
        ):
            out = False

            super().test_ufunc_unary(ufunc=ufunc, array=array)

            components = array.components
            components_expected = tuple(dict() for _ in range(ufunc.nout))
            try:
                for c in components:
                    result_c = ufunc(components[c])
                    if ufunc.nout == 1:
                        result_c = (result_c, )
                    for i in range(ufunc.nout):
                        components_expected[i][c] = result_c[i]

            except Exception as e:
                print(e)
                with pytest.raises(type(e)):
                    ufunc(array)
                return

            result_expected = tuple(array.type_array.from_components(components_expected[i]) for i in range(ufunc.nout))

            if out:
                out = tuple(0 * result_expected[i] for i in range(ufunc.nout))
                for i in range(ufunc.nout):
                    for c in components:
                        if not isinstance(out[i].components[c], na.AbstractArray):
                            out[i].components[c] = np.asanyarray(out[i].components[c])
            else:
                out = tuple(None for _ in range(ufunc.nout))

            result = ufunc(array, out=out[0] if ufunc.nout == 1 else out)
            if ufunc.nout == 1:
                result = (result, )

            for i in range(ufunc.nout):
                assert np.all(result[i] == result_expected[i])

                if out[i] is not None:
                    assert result[i] is out[i]

    class TestUfuncBinary(
        named_arrays._vectors.tests.test_vectors.AbstractTestAbstractVectorArray.TestUfuncBinary,
    ):

        def test_ufunc_binary(
                self,
                ufunc: np.ufunc,
                array: None | bool | int | float | complex | str | na.AbstractArray | na.AbstractCartesianVectorArray,
                array_2: None | bool | int | float | complex | str | na.AbstractArray | na.AbstractCartesianVectorArray,
                # out: bool,
        ):
            out = False

            super().test_ufunc_binary(ufunc, array, array_2)

            if isinstance(array, na.AbstractCartesianVectorArray):
                type_array = array.type_array
                array_normalized = array
                if isinstance(array_2, na.AbstractCartesianVectorArray):
                    array_2_normalized = array_2
                else:
                    array_2_normalized = type_array.from_scalar(array_2)

            else:
                if isinstance(array_2, na.AbstractCartesianVectorArray):
                    type_array = array_2.type_array
                    array_2_normalized = array_2
                    array_normalized = type_array.from_scalar(array)

            components = array_normalized.components
            components_2 = array_2_normalized.components

            components_expected = tuple(dict() for _ in range(ufunc.nout))
            try:
                for c in components:
                    result_c = ufunc(components[c], components_2[c])
                    if ufunc.nout == 1:
                        result_c = (result_c,)
                    for i in range(ufunc.nout):
                        components_expected[i][c] = result_c[i]
            except Exception as e:
                with pytest.raises(type(e)):
                    ufunc(array, array_2)
                return

            result_expected = tuple(type_array.from_components(components_expected[i]) for i in range(ufunc.nout))

            if out:
                out = tuple(0 * result_expected[i] for i in range(ufunc.nout))
                # for i in range(ufunc.nout):
                #     for c in components:
                #         if not isinstance(out[i].components[c], na.AbstractArray):
                #             out[i].components[c] = np.asanyarray(out[i].components[c])
                #         elif isinstance(out[i].components[c], na.AbstractScalarArray):
                #             out[i].components[c].ndarray = np.asanyarray(out[i].components[c].ndarray)

            else:
                out = tuple(None for _ in range(ufunc.nout))

            result = ufunc(array, array_2, out=out[0] if ufunc.nout == 1 else out)
            if ufunc.nout == 1:
                result = (result, )

            for i in range(ufunc.nout):
                assert np.all(result[i] == result_expected[i])

                if out[i] is not None:
                    assert result[i] is out[i]


class AbstractTestAbstractExplicitCartesianVectorArray(
    AbstractTestAbstractCartesianVectorArray,
    named_arrays._vectors.tests.test_vectors.AbstractTestAbstractExplicitVectorArray,
):
    pass


class AbstractTestAbstractExplicitCartesianVectorArrayCreation(
    named_arrays._vectors.tests.test_vectors.AbstractTestAbstractExplicitVectorArrayCreation,
):
    pass


class AbstractTestAbstractImplicitCartesianVectorArray(
    AbstractTestAbstractCartesianVectorArray,
    named_arrays._vectors.tests.test_vectors.AbstractTestAbstractImplicitVectorArray,
):
    pass


class AbstractTestAbstractCartesianVectorRandomSample(
    AbstractTestAbstractImplicitCartesianVectorArray,
    named_arrays._vectors.tests.test_vectors.AbstractTestAbstractVectorRandomSample,
):
    pass


class AbstractTestAbstractCartesianVectorUniformRandomSample(
    AbstractTestAbstractCartesianVectorRandomSample,
    named_arrays._vectors.tests.test_vectors.AbstractTestAbstractVectorUniformRandomSample,
):
    pass


class AbstractTestAbstractCartesianVectorNormalRandomSample(
    AbstractTestAbstractCartesianVectorRandomSample,
    named_arrays._vectors.tests.test_vectors.AbstractTestAbstractVectorNormalRandomSample,
):
    pass


class AbstractTestAbstractParameterizedCartesianVectorArray(
    AbstractTestAbstractImplicitCartesianVectorArray,
    named_arrays._vectors.tests.test_vectors.AbstractTestAbstractParameterizedVectorArray,
):
    pass


class AbstractTestAbstractCartesianVectorArrayRange(
    AbstractTestAbstractParameterizedCartesianVectorArray,
    named_arrays._vectors.tests.test_vectors.AbstractTestAbstractVectorArrayRange,
):
    pass


class AbstractTestAbstractCartesianVectorSpace(
    AbstractTestAbstractParameterizedCartesianVectorArray,
    named_arrays._vectors.tests.test_vectors.AbstractTestAbstractVectorSpace,
):
    pass


class AbstractTestAbstractCartesianVectorLinearSpace(
    AbstractTestAbstractCartesianVectorSpace,
    named_arrays._vectors.tests.test_vectors.AbstractTestAbstractVectorLinearSpace,
):
    pass


class AbstractTestAbstractCartesianVectorStratifiedRandomSpace(
    AbstractTestAbstractCartesianVectorLinearSpace,
    named_arrays._vectors.tests.test_vectors.AbstractTestAbstractVectorStratifiedRandomSpace,
):
    pass


class AbstractTestAbstractCartesianVectorLogarithmicSpace(
    AbstractTestAbstractCartesianVectorSpace,
    named_arrays._vectors.tests.test_vectors.AbstractTestAbstractVectorLogarithmicSpace,
):
    pass


class AbstractTestAbstractCartesianVectorGeometricSpace(
    AbstractTestAbstractCartesianVectorSpace,
    named_arrays._vectors.tests.test_vectors.AbstractTestAbstractVectorGeometricSpace,
):
    pass
