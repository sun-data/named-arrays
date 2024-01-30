import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
from ..cartesian.tests import test_vectors_cartesian

_num_x = test_vectors_cartesian._num_x
_num_y = test_vectors_cartesian._num_y
_num_z = test_vectors_cartesian._num_z
_num_distribution = test_vectors_cartesian._num_distribution


def _spectral_positional_arrays() -> list[na.SpectralPositionalVectorArray]:
    return [
        na.SpectralPositionalVectorArray(
            wavelength=500 * u.nm,
            position=na.Cartesian2dVectorArray(1, 2) * u.mm,
        ),
        na.SpectralPositionalVectorArray(
            wavelength=na.linspace(400, 600, axis="y", num=_num_y) * u.nm,
            position=na.Cartesian2dVectorLinearSpace(1, 2, axis="y", num=_num_y).explicit * u.mm,
        ),
    ]


def _spectral_positional_arrays_2() -> list[na.SpectralPositionalVectorArray]:
    return [
        na.SpectralPositionalVectorArray(
            wavelength=400 * u.nm,
            position=na.Cartesian2dVectorArray(3, 4) * u.m,
        ),
        na.SpectralPositionalVectorArray(
            wavelength=na.NormalUncertainScalarArray(400 * u.nm, width=1 * u.nm),
            position=na.Cartesian2dVectorArray(
                x=na.NormalUncertainScalarArray(3, width=1) * u.m,
                y=na.NormalUncertainScalarArray(4, width=1) * u.m,
            )
        )
    ]


def _spectral_positional_items() -> list[na.AbstractArray | dict[str, int | slice | na.AbstractArray]]:
    return [
        dict(y=0),
        dict(y=slice(0, 1)),
        dict(y=na.ScalarArrayRange(0, 2, axis='y')),
    ]


class AbstractTestAbstractSpectralPositionalVectorArray(
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray,
):

    @pytest.mark.parametrize(
        argnames='item',
        argvalues=_spectral_positional_items()
    )
    def test__getitem__(
            self,
            array: na.AbstractSpectralVectorArray,
            item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray
    ):
        super().test__getitem__(array=array, item=item)

    @pytest.mark.parametrize('array_2', _spectral_positional_arrays_2())
    class TestUfuncBinary(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestUfuncBinary
    ):
        pass

    @pytest.mark.parametrize('array_2', _spectral_positional_arrays_2())
    class TestMatmul(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestMatmul,
    ):
        pass

    class TestArrayFunctions(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions,
    ):

        @pytest.mark.parametrize("array_2", _spectral_positional_arrays_2())
        class TestAsArrayLikeFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions.TestAsArrayLikeFunctions,
        ):
            pass

        @pytest.mark.parametrize(
            argnames='where',
            argvalues=[
                np._NoValue,
                True,
                na.ScalarArray(True),
            ]
        )
        class TestReductionFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions.TestReductionFunctions,
        ):
            pass

        @pytest.mark.parametrize(
            argnames='q',
            argvalues=[
                .25,
                25 * u.percent,
                na.ScalarLinearSpace(.25, .75, axis='q', num=3, endpoint=True),
            ]
        )
        class TestPercentileLikeFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions
            .TestPercentileLikeFunctions,
        ):
            pass

    class TestNamedArrayFunctions(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestNamedArrayFunctions,
    ):
        @pytest.mark.skip
        class TestPltPlotLikeFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestNamedArrayFunctions
            .TestPltPlotLikeFunctions,
        ):
            pass


@pytest.mark.parametrize("array", _spectral_positional_arrays())
class TestSpectralPositionalVectorArray(
    AbstractTestAbstractSpectralPositionalVectorArray,
    test_vectors_cartesian.AbstractTestAbstractExplicitCartesianVectorArray,
):
    @pytest.mark.parametrize(
        argnames="item",
        argvalues=[
            dict(y=0),
            dict(y=slice(None)),
        ],
    )
    @pytest.mark.parametrize(
        argnames="value",
        argvalues=[
            700 * u.nm,
        ]
    )
    def test__setitem__(
            self,
            array: na.ScalarArray,
            item: dict[str, int | slice | na.ScalarArray] | na.ScalarArray,
            value: float | na.ScalarArray
    ):
        super().test__setitem__(array=array, item=item, value=value)


class AbstractTestAbstractImplicitSpectralPositionalVectorArray(
    AbstractTestAbstractSpectralPositionalVectorArray,
    test_vectors_cartesian.AbstractTestAbstractImplicitCartesianVectorArray,
):
    pass


class AbstractTestAbstractParameterizedSpectralPositionalVectorArray(
    AbstractTestAbstractImplicitSpectralPositionalVectorArray,
    test_vectors_cartesian.AbstractTestAbstractParameterizedCartesianVectorArray,
):
    pass


class AbstractTestAbstractSpectralPositionalVectorSpace(
    AbstractTestAbstractParameterizedSpectralPositionalVectorArray,
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorSpace,
):
    pass


def _spectral_positional_linear_spaces() -> list[na.SpectralPositionalVectorLinearSpace]:
    return [
        na.SpectralPositionalVectorLinearSpace(
            start=400 * u.nm,
            stop=600 * u.nm,
            axis="y",
            num=_num_y,
        )
    ]


@pytest.mark.parametrize("array", _spectral_positional_linear_spaces())
class TestSpectralPositionalVectorLinearSpace(
    AbstractTestAbstractSpectralPositionalVectorSpace,
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorLinearSpace,
):
    pass
