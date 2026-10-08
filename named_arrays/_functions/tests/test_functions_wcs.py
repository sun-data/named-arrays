import pytest
import numpy as np
import astropy.units as u
import named_arrays as na

_num = 16
_num_channel = 3
_num_wavelength = 5


def _same(value: na.ScalarArray) -> na.ScalarArray:
    return value


def _first(value: na.ScalarArray) -> na.ScalarArray:
    """The first element of `value` along `channel`, as a single element."""
    return value[dict(channel=slice(0, 1))]


def _function_arrays() -> list[na.FunctionArray]:

    shape_base = dict(channel=_num_channel)

    def full(value: float) -> na.ScalarArray:
        return na.ScalarArray.full(shape_base, value)

    def single(value: float) -> na.ScalarArray:
        return na.ScalarArray.full(dict(channel=1), value)

    wavelength = na.ScalarArray(np.array([171, 193, 211]) << u.AA, axes="channel")

    # A stack of images, each with its own slightly rotated WCS, with the
    # reference pixel at a vertex or, as for AIA, at the center of a pixel,
    # and with the time and the plate scale, or the whole WCS, either given
    # for each image or given once and broadcast against the images.
    images = [
        na.FunctionArray(
            inputs=na.ExplicitTemporalSpectralWcsPositionalVectorArray(
                time=constant(0) << u.s,
                wavelength=varying(wavelength),
                crval=na.PositionalVectorArray(
                    position=na.Cartesian2dVectorArray(varying(full(1)), varying(full(2))) << u.arcsec,
                ),
                crpix=na.CartesianNdVectorArray(dict(x=varying(full(crpix)), y=varying(full(crpix)))),
                cdelt=na.PositionalVectorArray(
                    position=na.Cartesian2dVectorArray(constant(0.5), constant(0.5)) << u.arcsec,
                ),
                pc=na.PositionalMatrixArray(
                    position=na.Cartesian2dMatrixArray(
                        x=na.CartesianNdVectorArray(dict(x=varying(full(1)), y=varying(full(0.125)))),
                        y=na.CartesianNdVectorArray(dict(x=varying(full(-0.125)), y=varying(full(1)))),
                    ),
                ),
                shape_wcs=dict(x=_num + 1, y=_num + 1),
            ),
            outputs=na.random.uniform(
                low=0,
                high=1,
                shape_random=shape_base | dict(x=_num, y=_num),
                seed=42,
            ),
        )
        for crpix, constant, varying in [
            (_num / 2, full, _same),
            (_num / 2 - 0.5, full, _same),
            (_num / 2 - 0.5, single, _same),
            (_num / 2 - 0.5, single, _first),
        ]
    ]

    # A spectrograph raster, where the time of each exposure is an explicit
    # component which varies along one of the WCS axes.
    raster = na.FunctionArray(
        inputs=na.ExplicitTemporalWcsSpectralPositionalVectorArray(
            time=na.linspace(0, 10, axis="x", num=_num + 1) << u.s,
            crval=na.SpectralPositionalVectorArray(
                wavelength=1400 * u.AA,
                position=na.Cartesian2dVectorArray(1, 2) * u.arcsec,
            ),
            crpix=na.CartesianNdVectorArray(dict(wavelength=2, x=3, y=4)),
            cdelt=na.SpectralPositionalVectorArray(
                wavelength=0.25 * u.AA,
                position=na.Cartesian2dVectorArray(0.5, 0.25) * u.arcsec,
            ),
            pc=na.SpectralPositionalMatrixArray(
                wavelength=na.CartesianNdVectorArray(dict(wavelength=1, x=0, y=0)),
                position=na.Cartesian2dMatrixArray(
                    x=na.CartesianNdVectorArray(dict(wavelength=0, x=1, y=0)),
                    y=na.CartesianNdVectorArray(dict(wavelength=0, x=0, y=1)),
                ),
            ),
            shape_wcs=dict(wavelength=_num_wavelength + 1, x=_num + 1, y=_num + 1),
        ),
        outputs=na.random.uniform(
            low=0,
            high=1,
            shape_random=dict(wavelength=_num_wavelength, x=_num, y=_num),
            seed=43,
        ),
    )

    return images + [raster]


@pytest.mark.parametrize("array", _function_arrays())
@pytest.mark.parametrize(
    argnames="item,lazy",
    argvalues=[
        (dict(x=slice(2, 5), y=slice(3, 7)), True),
        (dict(x=3), True),
        (dict(x=slice(2, 5), y=0), True),
        (dict(x=slice(8, 12)), True),
        (dict(x=-1), True),
        (dict(x=slice(-5, None)), True),
        (dict(x=slice(2, -3), y=slice(-4, -1)), True),
        (dict(x=slice(4, None), wavelength=slice(1, 3)), True),
        (dict(channel=1, x=slice(2, 5)), True),
        (dict(channel=slice(1, None), y=slice(None, 4)), True),
        (dict(channel=na.ScalarArray(np.array([0, 2]), axes="channel")), True),
        (dict(x=slice(None, None, -1)), False),
        # no cells keep a single vertex, which is computed explicitly
        (dict(x=slice(5, 2)), False),
    ],
)
def test__getitem__(
    monkeypatch: pytest.MonkeyPatch,
    array: na.FunctionArray,
    item: dict[str, int | slice | na.AbstractArray],
    lazy: bool,
) -> None:
    """
    Indexing a function array whose inputs are a WCS vector leaves the inputs
    a WCS vector, without computing the coordinates of more than two pixels
    along each axis, and gives the same function as indexing the explicit
    array.
    """
    # Record the shape of every grid of pixels the WCS is evaluated on.
    shapes = []
    components_wcs = na.AbstractWcsVector._components_wcs

    def record(self: na.AbstractWcsVector) -> dict[str, na.ArrayLike]:
        shapes.append(self.shape_wcs)
        return components_wcs.fget(self)

    monkeypatch.setattr(na.AbstractWcsVector, "_components_wcs", property(record))

    result = array[item]

    if lazy:
        assert type(result.inputs) is type(array.inputs)
        assert all(n <= 2 for shape in shapes for n in shape.values())
        assert result.inputs.shape == result.inputs.explicit.shape
    else:
        assert isinstance(result.inputs, na.AbstractExplicitVectorArray)

    expected = array.explicit[item]
    assert result.shape == expected.shape
    assert result.inputs.shape == expected.inputs.shape
    assert np.all(result.inputs.explicit == expected.inputs)
    assert np.all(result.inputs.explicit == array.broadcasted[item].inputs)
    assert np.all(result.outputs == expected.outputs)
