import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
from . import test_vectors
from ..cartesian.tests import test_vectors_cartesian

_num_x = test_vectors._num_x
_num_y = test_vectors._num_y
_num_z = test_vectors._num_z
_num_distribution = test_vectors._num_distribution


def _spectral_arrays() -> list[na.SpectralVectorArray]:
    return [
        na.SpectralVectorArray(wavelength=500 * u.nm),
        na.SpectralVectorArray(wavelength=na.linspace(400, 600, axis="y", num=_num_y) * u.nm),
    ]


def _spectral_arrays_2() -> list[na.SpectralVectorArray]:
    return [
        na.SpectralVectorArray(wavelength=400 * u.nm),
        na.SpectralVectorArray(wavelength=na.NormalUncertainScalarArray(400 * u.nm, width=1 * u.nm))
    ]


def _spectral_items() -> list[na.AbstractSpectralVectorArray | dict[str, int | slice | na.AbstractSpectralVectorArray]]:
    return [
        dict(y=0),
        dict(y=slice(0, 1)),
        dict(y=na.ScalarArrayRange(0, 2, axis='y')),
    ]


class AbstractTestAbstractSpectralVectorArray(
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray,
):
    def test_wavelength(self, array: na.AbstractSpectralVectorArray):
        assert isinstance(na.as_named_array(array.wavelength), na.AbstractScalar)
        assert np.all(array.wavelength > 0)

    @pytest.mark.parametrize(
        argnames='item',
        argvalues=_spectral_items()
    )
    def test__getitem__(
            self,
            array: na.AbstractSpectralVectorArray,
            item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray
    ):
        super().test__getitem__(array=array, item=item)

    @pytest.mark.parametrize('array_2', _spectral_arrays_2())
    class TestUfuncBinary(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestUfuncBinary
    ):
        pass

    @pytest.mark.parametrize('array_2', _spectral_arrays_2())
    class TestMatmul(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestMatmul
    ):
        pass

    class TestArrayFunctions(
        test_vectors.AbstractTestAbstractVectorArray.TestArrayFunctions
    ):

        @pytest.mark.parametrize("array_2", _spectral_arrays_2())
        class TestAsArrayLikeFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions.TestAsArrayLikeFunctions
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
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestNamedArrayFunctions.TestPltPlotLikeFunctions,
        ):
            pass


@pytest.mark.parametrize("array", _spectral_arrays())
class TestSpectralVectorArray(
    AbstractTestAbstractSpectralVectorArray,
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


class AbstractTestAbstractImplicitSpectralVectorArray(
    AbstractTestAbstractSpectralVectorArray,
    test_vectors_cartesian.AbstractTestAbstractImplicitCartesianVectorArray,
):
    pass


class AbstractTestAbstractParameterizedSpectralVectorArray(
    AbstractTestAbstractImplicitSpectralVectorArray,
    test_vectors_cartesian.AbstractTestAbstractParameterizedCartesianVectorArray,
):
    pass


class AbstractTestAbstractSpectralVectorSpace(
    AbstractTestAbstractParameterizedSpectralVectorArray,
    test_vectors_cartesian.AbstractTestAbstractParameterizedCartesianVectorArray,
):
    pass


def _spectral_linear_spaces() -> list[na.SpectralVectorLinearSpace]:
    return [
        na.SpectralVectorLinearSpace(400 * u.nm, 600 * u.nm, axis="y", num=_num_y)
    ]


@pytest.mark.parametrize("array", _spectral_linear_spaces())
class TestSpectralVectorLinearSpace(
    AbstractTestAbstractSpectralVectorSpace,
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorLinearSpace,
):
    pass
