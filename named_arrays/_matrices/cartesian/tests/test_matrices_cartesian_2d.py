from typing import Mapping
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import named_arrays.tests.test_core
import named_arrays._vectors.cartesian.tests.test_vectors_cartesian_2d
from . import test_matrices_cartesian

__all__ = [
    'AbstractTestAbstractCartesian2dMatrixArray',
    'TestCartesian2dMatrixArray',
]

_num_x = named_arrays.tests.test_core.num_x
_num_y = named_arrays.tests.test_core.num_y
_num_distribution = named_arrays.tests.test_core.num_distribution



def _cartesian_2d_matrices():
    xx_1d = na.ScalarUniformRandomSample(-4, 4, shape_random=dict(y=_num_y)).explicit
    yx_2d = na.ScalarUniformRandomSample(-5, 5, shape_random=dict(x=_num_x, y=_num_y)).explicit
    yx_uncertain = na.UniformUncertainScalarArray(
        nominal=na.ScalarUniformRandomSample(-5, 5, shape_random=dict(x=_num_x, y=_num_y)),
        width=1,
        num_distribution=_num_distribution
    ).explicit

    def matrix(xx, yx):
        return na.Cartesian2dMatrixArray(
            x=na.Cartesian2dVectorArray(x=xx, y=1),
            y=na.Cartesian2dVectorArray(x=yx, y=0),
        )

    matrices = [
        matrix(4, 5.),
        matrix(4, 5.) * u.mm,
        matrix(xx_1d, yx_2d) * u.mm,
        matrix(4, yx_uncertain),
        matrix(xx_1d, yx_uncertain) * u.mm,
    ]

    matrices.append(
        na.Cartesian2dMatrixArray(
            x=na.Cartesian2dMatrixArray(
                x=na.Cartesian2dMatrixArray(
                    x=na.Cartesian2dVectorArray(x=1, y=2),
                    y=na.Cartesian2dVectorArray(x=1, y=2)),
                y=na.Cartesian2dMatrixArray(
                    x=na.Cartesian2dVectorArray(x=1, y=2),
                    y=na.Cartesian2dVectorArray(x=1, y=2)),
            ),
            y=na.Cartesian2dMatrixArray(
                x=na.Cartesian2dMatrixArray(
                    x=na.Cartesian2dVectorArray(x=1, y=2),
                    y=na.Cartesian2dVectorArray(x=1, y=2)),
                y=na.Cartesian2dMatrixArray(
                    x=na.Cartesian2dVectorArray(x=1, y=2),
                    y=na.Cartesian2dVectorArray(x=1, y=2)),
            ),
        )
    )

    return matrices



def _cartesian_2d_matrices_2():
    xx_1d = na.ScalarUniformRandomSample(-6, 6, shape_random=dict(y=_num_y)).explicit
    yy_2d = na.ScalarUniformRandomSample(-7, 7, shape_random=dict(x=_num_x, y=_num_y)).explicit
    yy_uncertain = na.UniformUncertainScalarArray(
        nominal=na.ScalarUniformRandomSample(-7, 7, shape_random=dict(x=_num_x, y=_num_y)),
        width=1,
        num_distribution=_num_distribution
    ).explicit

    def matrix(xx, yy):
        return na.Cartesian2dMatrixArray(
            x=na.Cartesian2dVectorArray(x=xx, y=1),
            y=na.Cartesian2dVectorArray(x=0, y=yy),
        )

    scalars = [
        7.,
        yy_2d * u.mm,
        yy_uncertain,
    ]
    matrices = [
        matrix(6, 7.),
        matrix(6, yy_2d) * u.mm,
        matrix(xx_1d, yy_uncertain),
        matrix(xx_1d, 7.) * u.mm,
    ]

    matrices.append(
        na.Cartesian2dMatrixArray(
            x=na.Cartesian2dMatrixArray(
                x=na.Cartesian2dMatrixArray(
                    x=na.Cartesian2dVectorArray(x=1, y=2),
                    y=na.Cartesian2dVectorArray(x=1, y=2)),
                y=na.Cartesian2dMatrixArray(
                    x=na.Cartesian2dVectorArray(x=1, y=2),
                    y=na.Cartesian2dVectorArray(x=1, y=2)),
            ),
            y=na.Cartesian2dMatrixArray(
                x=na.Cartesian2dMatrixArray(
                    x=na.Cartesian2dVectorArray(x=1, y=2),
                    y=na.Cartesian2dVectorArray(x=1, y=2)),
                y=na.Cartesian2dMatrixArray(
                    x=na.Cartesian2dVectorArray(x=1, y=2),
                    y=na.Cartesian2dVectorArray(x=1, y=2)),
            ),
        )
    )

    return scalars + matrices


class AbstractTestAbstractCartesian2dMatrixArray(
    test_matrices_cartesian.AbstractTestAbstractCartesianMatrixArray,
    named_arrays._vectors.cartesian.tests.test_vectors_cartesian_2d.AbstractTestAbstractCartesian2dVectorArray,
):
    @pytest.mark.parametrize(
        argnames='item',
        argvalues=[
            dict(y=0),
            dict(y=slice(0, 1)),
            dict(y=na.ScalarArrayRange(0, 2, axis='y')),
            dict(
                y=na.Cartesian2dMatrixArray(
                    x=na.Cartesian2dVectorArray(
                        x=na.ScalarArrayRange(0, 2, axis='y'),
                        y=na.ScalarArrayRange(0, 2, axis='y'),
                    ),
                    y=na.Cartesian2dVectorArray(
                        x=na.ScalarArrayRange(0, 2, axis='y'),
                        y=na.ScalarArrayRange(0, 2, axis='y'),
                    )
                ),
            ),
            dict(
                y=na.UncertainScalarArray(
                    nominal=na.ScalarArray(np.array([0, 1]), axes=('y',)),
                    distribution=na.ScalarArray(
                        ndarray=np.array([[0, ], [1, ]]),
                        axes=('y', na.UncertainScalarArray.axis_distribution),
                    )
                ),
                _distribution=na.UncertainScalarArray(
                    nominal=None,
                    distribution=na.ScalarArray(
                        ndarray=np.array([[0], [0]]),
                        axes=('y', na.UncertainScalarArray.axis_distribution),
                    )
                )
            ),
            na.ScalarLinearSpace(0, 1, axis='y', num=_num_y) > 0.5,
            na.UniformUncertainScalarArray(
                nominal=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y),
                width=0.1,
                num_distribution=_num_distribution,
            ) > 0.5,
            na.Cartesian2dMatrixArray(
                x=na.Cartesian2dVectorArray(
                    x=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y) > 0.3,
                    y=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y) > 0.4,
                ),
                y=na.Cartesian2dVectorArray(
                    x=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y) > 0.5,
                    y=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y) > 0.6,
                ),
            ),
            ]
    )
    def test__getitem__(
            self,
            array: na.AbstractCartesian2dVectorArray,
            item: Mapping[str, int | slice | na.AbstractArray] | na.AbstractArray
    ):
        super().test__getitem__(array=array, item=item)

    @pytest.mark.parametrize('array_2', _cartesian_2d_matrices_2())
    class TestUfuncBinary(
        test_matrices_cartesian.AbstractTestAbstractCartesianMatrixArray.TestUfuncBinary
    ):
        pass

    class TestArrayFunctions(
        named_arrays._vectors.cartesian.tests.test_vectors_cartesian_2d
        .AbstractTestAbstractCartesian2dVectorArray.TestArrayFunctions,
    ):
        @pytest.mark.parametrize('array_2', _cartesian_2d_matrices_2())
        class TestStackLikeFunctions(
            named_arrays.tests.test_core.AbstractTestAbstractArray.TestArrayFunctions.TestStackLikeFunctions,
        ):
            pass

    @pytest.mark.parametrize(
        argnames='array_2',
        argvalues=_cartesian_2d_matrices_2() + [na.Cartesian2dVectorArray(1, 2)]
    )
    class TestMatmul(
        test_matrices_cartesian.AbstractTestAbstractCartesianMatrixArray.TestMatmul
    ):
        pass

    @pytest.mark.parametrize("exponent", [1, 2, 5])
    def test_power(
        self,
        array: na.AbstractCartesian2dMatrixArray,
        exponent: int
    ):
        if na.unit(array.length) is not None:
            return

        if len(array.cartesian_nd.entries) != 4:
            return

        result = array.power(exponent)

        result_expected = na.Cartesian2dIdentityMatrixArray()
        for i in range(exponent):
            result_expected = result_expected @ array

        assert np.allclose(result, result_expected)


@pytest.mark.parametrize(
    argnames='array',
    argvalues=_cartesian_2d_matrices()
)
class TestCartesian2dMatrixArray(
    AbstractTestAbstractCartesian2dMatrixArray,
    test_matrices_cartesian.AbstractTestAbstractCartesianMatrixArray,
):
    pass


@pytest.mark.parametrize("type_array", [na.Cartesian2dMatrixArray])
class TestCartesian2dMatrixArrayCreation(
    test_matrices_cartesian.AbstractTestAbstractExplicitCartesianMatrixArrayCreation,
):

    @pytest.mark.parametrize("like", [None] + _cartesian_2d_matrices())
    class TestFromScalarArray(
        test_matrices_cartesian.AbstractTestAbstractExplicitCartesianMatrixArrayCreation.TestFromScalarArray,
    ):
        pass


@pytest.mark.parametrize(
    argnames="array",
    argvalues=[
        na.Cartesian2dIdentityMatrixArray(),
    ],
)
class TestCartesian2dIdentityMatrixArray(
    test_matrices_cartesian.AbstractTestAbstractImplicitCartesianMatrixArray,
    AbstractTestAbstractCartesian2dMatrixArray,
):
    pass
