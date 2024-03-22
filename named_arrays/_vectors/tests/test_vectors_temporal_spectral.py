import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
from ..tests import test_vectors
from ..cartesian.tests import test_vectors_cartesian

_num_x = test_vectors_cartesian._num_x
_num_y = test_vectors_cartesian._num_y
_num_z = test_vectors_cartesian._num_z
_num_distribution = test_vectors_cartesian._num_distribution


def _temporal_spectral_arrays() -> (
    list[na.TemporalSpectralVectorArray]
):
    return [
        na.TemporalSpectralVectorArray(
            time=10 * u.s,
            wavelength=500 * u.nm,
        ),
        na.TemporalSpectralVectorArray(
            time=na.linspace(0, 10, axis="y", num=_num_y) * u.s,
            wavelength=na.linspace(400, 600, axis="y", num=_num_y) * u.nm,
        ),
    ]


def _temporal_spectral_arrays_2() -> (
    list[na.TemporalSpectralVectorArray]
):
    return [
        na.TemporalSpectralVectorArray(
            time=20 * u.s,
            wavelength=400 * u.nm,
        ),
        na.TemporalSpectralVectorArray(
            time=na.NormalUncertainScalarArray(10 * u.s, width=1 * u.s),
            wavelength=na.NormalUncertainScalarArray(400 * u.nm, width=1 * u.nm),
        ),
    ]


def _temporal_spectral_items() -> (
    list[na.AbstractArray | dict[str, int | slice | na.AbstractArray]]
):
    return [
        dict(y=0),
        dict(y=slice(0, 1)),
        dict(y=na.ScalarArrayRange(0, 2, axis="y")),
    ]


class AbstractTestAbstractTemporalSpectralVectorArray(
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray,
):

    @pytest.mark.parametrize(
        argnames="item", argvalues=_temporal_spectral_items()
    )
    def test__getitem__(
        self,
        array: na.AbstractSpectralVectorArray,
        item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray,
    ):
        super().test__getitem__(array=array, item=item)

    @pytest.mark.parametrize("array_2", _temporal_spectral_arrays_2())
    class TestUfuncBinary(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestUfuncBinary
    ):
        pass

    @pytest.mark.parametrize("array_2", _temporal_spectral_arrays_2())
    class TestMatmul(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestMatmul,
    ):
        pass

    class TestArrayFunctions(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions,
    ):

        @pytest.mark.parametrize("array_2", _temporal_spectral_arrays_2())
        class TestAsArrayLikeFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions.TestAsArrayLikeFunctions,
        ):
            pass

        @pytest.mark.parametrize(
            argnames="where",
            argvalues=[
                np._NoValue,
                True,
                na.ScalarArray(True),
            ],
        )
        class TestReductionFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions.TestReductionFunctions,
        ):
            pass

        @pytest.mark.parametrize(
            argnames="q",
            argvalues=[
                0.25,
                25 * u.percent,
                na.ScalarLinearSpace(0.25, 0.75, axis="q", num=3, endpoint=True),
            ],
        )
        class TestPercentileLikeFunctions(
            test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions.TestPercentileLikeFunctions,
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


@pytest.mark.parametrize("array", _temporal_spectral_arrays())
class TestTemporalSpectralVectorArray(
    AbstractTestAbstractTemporalSpectralVectorArray,
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
        ],
    )
    def test__setitem__(
        self,
        array: na.ScalarArray,
        item: dict[str, int | slice | na.ScalarArray] | na.ScalarArray,
        value: float | na.ScalarArray,
    ):
        super().test__setitem__(array=array, item=item, value=value)


class AbstractTestAbstractImplicitTemporalSpectralVectorArray(
    AbstractTestAbstractTemporalSpectralVectorArray,
    test_vectors_cartesian.AbstractTestAbstractImplicitCartesianVectorArray,
):
    pass


class AbstractTestAbstractParameterizedTemporalSpectralVectorArray(
    AbstractTestAbstractImplicitTemporalSpectralVectorArray,
    test_vectors_cartesian.AbstractTestAbstractParameterizedCartesianVectorArray,
):
    pass


class AbstractTestAbstractTemporalSpectralVectorSpace(
    AbstractTestAbstractParameterizedTemporalSpectralVectorArray,
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorSpace,
):
    pass


def _temporal_spectral_linear_spaces() -> (
    list[na.TemporalSpectralVectorLinearSpace]
):
    return [
        na.TemporalSpectralVectorLinearSpace(
            start=400 * u.nm,
            stop=600 * u.nm,
            axis="y",
            num=_num_y,
        )
    ]


@pytest.mark.parametrize("array", _temporal_spectral_linear_spaces())
class TestTemporalSpectralVectorLinearSpace(
    AbstractTestAbstractTemporalSpectralVectorSpace,
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorLinearSpace,
):
    pass


@pytest.mark.parametrize(
    argnames="array",
    argvalues=[
        na.ExplicitTemporalWcsSpectralVectorArray(
            time=10 * u.s,
            crval=na.SpectralVectorArray(
                wavelength=500 * u.nm,
            ),
            crpix=na.CartesianNdVectorArray(
                dict(
                    wavelength=1,
                    x=2,
                    y=3,
                )
            ),
            cdelt=na.SpectralVectorArray(
                wavelength=1 * u.nm,
            ),
            pc=na.SpectralMatrixArray(
                wavelength=na.CartesianNdVectorArray(dict(wavelength=1, x=0, y=0)),
            ),
            shape_wcs=dict(wavelength=5, x=_num_x, y=_num_y),
        ),
    ],
)
class TestExplicitTemporalWcsSpectralVectorArray(
    AbstractTestAbstractImplicitTemporalSpectralVectorArray,
    test_vectors.AbstractTestAbstractWcsVector,
):
    pass
