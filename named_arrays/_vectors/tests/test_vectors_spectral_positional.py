from typing import Mapping
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
        # a separable 3d grid: wavelength along "y", position along "x" and "z".
        na.SpectralPositionalVectorArray(
            wavelength=na.linspace(400, 600, axis="y", num=_num_y) * u.nm,
            position=na.Cartesian2dVectorArray(
                x=na.linspace(1, 2, axis="x", num=_num_x),
                y=na.linspace(3, 4, axis="z", num=_num_z),
            ) * u.mm,
        ),
        # a 3d grid whose position also varies with wavelength.
        na.SpectralPositionalVectorArray(
            wavelength=na.linspace(400, 600, axis="y", num=_num_y) * u.nm,
            position=na.Cartesian2dVectorArray(
                x=na.linspace(1, 2, axis="x", num=_num_x)
                + na.linspace(0, 0.1, axis="y", num=_num_y),
                y=na.linspace(3, 4, axis="z", num=_num_z),
            ) * u.mm,
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


def _spectral_positional_items() -> list[na.AbstractArray | Mapping[str, int | slice | na.AbstractArray]]:
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
            item: Mapping[str, int | slice | na.AbstractArray] | na.AbstractArray
    ):
        super().test__getitem__(array=array, item=item)

    def test_volume_cell(self, array: na.AbstractSpectralPositionalVectorArray):
        axis = tuple(na.shape(array))
        result = array.volume_cell(axis)

        # the volume is a scalar with one fewer cell along each grid axis
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        for ax in na.shape(array):
            assert na.shape(result)[ax] == na.shape(array)[ax] - 1
        assert np.all(result > 0 * na.unit_normalized(result))

        # the wavelength and position axes are inferred from the array, so the
        # order in which they are given does not matter
        assert np.all(result == array.volume_cell(tuple(reversed(axis))))

    def test_spectral_positional(
        self, array: na.AbstractSpectralPositionalVectorArray
    ):
        result = array.spectral_positional
        assert isinstance(result, na.SpectralPositionalVectorArray)
        assert np.all(result.wavelength == array.wavelength)
        assert np.all(result.position == array.position)

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
        class TestStackLikeFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions.TestStackLikeFunctions,
        ):
            pass

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
            start=na.SpectralPositionalVectorArray(
                wavelength=400 * u.nm,
                position=na.Cartesian2dVectorArray(1, 3) * u.mm,
            ),
            stop=na.SpectralPositionalVectorArray(
                wavelength=600 * u.nm,
                position=na.Cartesian2dVectorArray(2, 4) * u.mm,
            ),
            axis=na.SpectralPositionalVectorArray(
                wavelength="y",
                position=na.Cartesian2dVectorArray("x", "z"),
            ),
            num=na.SpectralPositionalVectorArray(
                wavelength=_num_y,
                position=na.Cartesian2dVectorArray(_num_x, _num_z),
            ),
        )
    ]


@pytest.mark.parametrize("array", _spectral_positional_linear_spaces())
class TestSpectralPositionalVectorLinearSpace(
    AbstractTestAbstractSpectralPositionalVectorSpace,
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorLinearSpace,
):
    pass
