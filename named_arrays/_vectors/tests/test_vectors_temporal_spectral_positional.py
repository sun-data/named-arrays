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


def _temporal_spectral_positional_arrays() -> (
    list[na.TemporalSpectralPositionalVectorArray]
):
    return [
        na.TemporalSpectralPositionalVectorArray(
            time=10 * u.s,
            wavelength=500 * u.nm,
            position=na.Cartesian2dVectorArray(1, 2) * u.mm,
        ),
        na.TemporalSpectralPositionalVectorArray(
            time=na.linspace(0, 10, axis="y", num=_num_y) * u.s,
            wavelength=na.linspace(400, 600, axis="y", num=_num_y) * u.nm,
            position=na.Cartesian2dVectorLinearSpace(
                1, 2, axis="y", num=_num_y
            ).explicit
            * u.mm,
        ),
    ]


def _temporal_spectral_positional_arrays_2() -> (
    list[na.TemporalSpectralPositionalVectorArray]
):
    return [
        na.TemporalSpectralPositionalVectorArray(
            time=20 * u.s,
            wavelength=400 * u.nm,
            position=na.Cartesian2dVectorArray(3, 4) * u.m,
        ),
        na.TemporalSpectralPositionalVectorArray(
            time=na.NormalUncertainScalarArray(10 * u.s, width=1 * u.s),
            wavelength=na.NormalUncertainScalarArray(400 * u.nm, width=1 * u.nm),
            position=na.Cartesian2dVectorArray(
                x=na.NormalUncertainScalarArray(3, width=1) * u.m,
                y=na.NormalUncertainScalarArray(4, width=1) * u.m,
            ),
        ),
    ]


def _temporal_spectral_positional_items() -> (
    list[na.AbstractArray | dict[str, int | slice | na.AbstractArray]]
):
    return [
        dict(y=0),
        dict(y=slice(0, 1)),
        dict(y=na.ScalarArrayRange(0, 2, axis="y")),
    ]


class AbstractTestAbstractTemporalSpectralPositionalVectorArray(
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray,
):

    @pytest.mark.parametrize(
        argnames="item", argvalues=_temporal_spectral_positional_items()
    )
    def test__getitem__(
        self,
        array: na.AbstractSpectralVectorArray,
        item: dict[str, int | slice | na.AbstractArray] | na.AbstractArray,
    ):
        super().test__getitem__(array=array, item=item)

    @pytest.mark.parametrize("array_2", _temporal_spectral_positional_arrays_2())
    class TestUfuncBinary(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestUfuncBinary
    ):
        pass

    @pytest.mark.parametrize("array_2", _temporal_spectral_positional_arrays_2())
    class TestMatmul(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestMatmul,
    ):
        pass

    class TestArrayFunctions(
        test_vectors_cartesian.AbstractTestAbstractCartesianVectorArray.TestArrayFunctions,
    ):

        @pytest.mark.parametrize("array_2", _temporal_spectral_positional_arrays_2())
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


@pytest.mark.parametrize("array", _temporal_spectral_positional_arrays())
class TestTemporalSpectralPositionalVectorArray(
    AbstractTestAbstractTemporalSpectralPositionalVectorArray,
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


class AbstractTestAbstractImplicitTemporalSpectralPositionalVectorArray(
    AbstractTestAbstractTemporalSpectralPositionalVectorArray,
    test_vectors_cartesian.AbstractTestAbstractImplicitCartesianVectorArray,
):
    pass


class AbstractTestAbstractParameterizedTemporalSpectralPositionalVectorArray(
    AbstractTestAbstractImplicitTemporalSpectralPositionalVectorArray,
    test_vectors_cartesian.AbstractTestAbstractParameterizedCartesianVectorArray,
):
    pass


class AbstractTestAbstractTemporalSpectralPositionalVectorSpace(
    AbstractTestAbstractParameterizedTemporalSpectralPositionalVectorArray,
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorSpace,
):
    pass


def _temporal_spectral_positional_linear_spaces() -> (
    list[na.TemporalSpectralPositionalVectorLinearSpace]
):
    return [
        na.TemporalSpectralPositionalVectorLinearSpace(
            start=400 * u.nm,
            stop=600 * u.nm,
            axis="y",
            num=_num_y,
        )
    ]


@pytest.mark.parametrize("array", _temporal_spectral_positional_linear_spaces())
class TestTemporalSpectralPositionalVectorLinearSpace(
    AbstractTestAbstractTemporalSpectralPositionalVectorSpace,
    test_vectors_cartesian.AbstractTestAbstractCartesianVectorLinearSpace,
):
    pass


@pytest.mark.parametrize(
    argnames="array",
    argvalues=[
        na.ExplicitTemporalSpectralWcsPositionalVectorArray(
            time=10 * u.s,
            wavelength=500 * u.nm,
            crval=na.PositionalVectorArray(
                position=na.Cartesian2dVectorArray(1, 1) * u.deg,
            ),
            crpix=na.CartesianNdVectorArray(
                dict(
                    x=2,
                    y=3,
                )
            ),
            cdelt=na.PositionalVectorArray(
                position=na.Cartesian2dVectorArray(1, 1) * u.arcsec,
            ),
            pc=na.PositionalMatrixArray(
                position=na.Cartesian2dMatrixArray(
                    x=na.CartesianNdVectorArray(dict(x=1, y=0)),
                    y=na.CartesianNdVectorArray(dict(x=0, y=1)),
                ),
            ),
            shape_wcs=dict(x=_num_x, y=_num_y),
        ),
    ],
)
class TestExplicitTemporalSpectralWcsPositionalVectorArray(
    AbstractTestAbstractImplicitTemporalSpectralPositionalVectorArray,
    test_vectors.AbstractTestAbstractWcsVector,
):
    pass


@pytest.mark.parametrize(
    argnames="array",
    argvalues=[
        na.ExplicitTemporalWcsSpectralPositionalVectorArray(
            time=10 * u.s,
            crval=na.SpectralPositionalVectorArray(
                wavelength=500 * u.nm,
                position=na.Cartesian2dVectorArray(1, 1) * u.deg,
            ),
            crpix=na.CartesianNdVectorArray(
                dict(
                    wavelength=1,
                    x=2,
                    y=3,
                )
            ),
            cdelt=na.SpectralPositionalVectorArray(
                wavelength=1 * u.nm,
                position=na.Cartesian2dVectorArray(1, 1) * u.arcsec,
            ),
            pc=na.SpectralPositionalMatrixArray(
                wavelength=na.CartesianNdVectorArray(dict(wavelength=1, x=0, y=0)),
                position=na.Cartesian2dMatrixArray(
                    x=na.CartesianNdVectorArray(dict(wavelength=0, x=1, y=0)),
                    y=na.CartesianNdVectorArray(dict(wavelength=0, x=0, y=1)),
                ),
            ),
            shape_wcs=dict(wavelength=5, x=_num_x, y=_num_y),
        ),
    ],
)
class TestExplicitTemporalWcsSpectralPositionalVectorArray(
    AbstractTestAbstractImplicitTemporalSpectralPositionalVectorArray,
    test_vectors.AbstractTestAbstractWcsVector,
):
    pass
