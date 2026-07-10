import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
from . import test_vectors, test_vectors_spectral
from ..cartesian.tests import test_vectors_cartesian

_num_x = test_vectors._num_x
_num_y = test_vectors._num_y
_num_z = test_vectors._num_z
_num_distribution = test_vectors._num_distribution


def _doppler_arrays() -> list[na.DopplerVectorArray]:
    return [
        na.DopplerVectorArray(
            wavelength=500 * u.nm,
            wavelength_rest=500 * u.nm,
        ),
        na.DopplerVectorArray.from_velocity(
            velocity=na.linspace(-10, 10, axis="y", num=_num_y) * u.km / u.s,
            wavelength_rest=400 * u.nm,
        ),
    ]


def _doppler_arrays_2() -> list[na.DopplerVectorArray]:
    return [
        na.DopplerVectorArray(
            wavelength=400 * u.nm,
            wavelength_rest=400 * u.nm,
        ),
    ]


def _doppler_items() -> list[na.AbstractDopplerVectorArray | dict[str, int | slice | na.AbstractDopplerVectorArray]]:
    return [
        dict(y=0),
        dict(y=slice(0, 1)),
        dict(y=na.ScalarArrayRange(0, 2, axis='y')),
    ]


class AbstractTestAbstractDopplerVectorArray(
    test_vectors_spectral.AbstractTestAbstractSpectralVectorArray,
):
    def test_wavelength_rest(self, array: na.AbstractDopplerVectorArray):
        assert isinstance(na.as_named_array(array.wavelength), na.AbstractScalar)
        assert np.all(array.wavelength > 0)

    def test_velocity(self, array: na.AbstractDopplerVectorArray):
        result = array.velocity
        kms = dict(
            unit=u.km / u.s,
            equivalencies=u.doppler_optical(array.wavelength_rest),
        )
        assert np.allclose(result, array.wavelength.to(**kms))

        b = array.type_explicit.from_velocity(
            velocity=array.velocity,
            wavelength_rest=array.wavelength_rest,
        )

        assert np.allclose(b.wavelength, array.wavelength)


    @pytest.mark.parametrize(
        argnames='item',
        argvalues=_doppler_items()
    )
    def test__getitem__(
            self,
            array: na.AbstractDopplerVectorArray,
            item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray
    ):
        super().test__getitem__(array=array, item=item)

    @pytest.mark.parametrize('array_2', _doppler_arrays_2())
    class TestUfuncBinary(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestUfuncBinary
    ):
        pass

    @pytest.mark.parametrize('array_2', _doppler_arrays_2())
    class TestMatmul(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestMatmul
    ):
        pass

    class TestArrayFunctions(
        test_vectors.AbstractTestAbstractVectorArray.TestArrayFunctions
    ):

        @pytest.mark.parametrize("array_2", _doppler_arrays_2())
        class TestStackLikeFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions.TestStackLikeFunctions,
        ):
            pass

        @pytest.mark.parametrize("array_2", _doppler_arrays_2())
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


@pytest.mark.parametrize("array", _doppler_arrays())
class TestDopplerVectorArray(
    AbstractTestAbstractDopplerVectorArray,
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


class AbstractTestAbstractImplicitDopplerVectorArray(
    AbstractTestAbstractDopplerVectorArray,
    test_vectors_spectral.AbstractTestAbstractImplicitSpectralVectorArray,
):
    pass
