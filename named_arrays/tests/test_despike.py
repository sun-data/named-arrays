from typing import Any
import pytest
import numpy as np
import astropy.units as u
import astroscrappy
import named_arrays as na

shape = dict(x=101, y=101)

img = na.random.normal(0, 6.5, shape)
spikes = 1000 * na.random.poisson(0.001, shape)

_arrays = [
    img + 100 * spikes,
    (img + 100 * spikes) * u.DN,
    img + na.NormalUncertainScalarArray(100, 5) * spikes,
    na.NormalUncertainScalarArray(img + 100 * spikes, 5),
    na.FunctionArray(
        inputs=na.ScalarUniformRandomSample(-5, 5, shape_random=shape),
        outputs=img + 100 * spikes,
    ),
]


@pytest.mark.parametrize(
    argnames="array",
    argvalues=_arrays,
)
@pytest.mark.parametrize(
    argnames="axis",
    argvalues=[
        tuple(shape),
    ],
)
@pytest.mark.parametrize("where", [None, na.random.uniform(0, 1, shape) > 0.01])
@pytest.mark.parametrize("inbkg", [None, na.ScalarArray.zeros(shape)])
@pytest.mark.parametrize("invar", [None, 6.5**2])
@pytest.mark.parametrize("psfk", [
    None,
    na.ScalarArray.ones(dict(x=5, y=5)) / 25,
])
def test_despike(
    array: na.AbstractArray,
    axis: tuple[str, str],
    where: None | bool | na.AbstractArray,
    inbkg: None | na.AbstractArray,
    invar: None | float | na.AbstractArray,
    psfk: None | na.AbstractArray,
) -> None:
    result = na.despike(
        array=array,
        axis=axis,
        where=where,
        inbkg=inbkg,
        invar=invar,
        fsmode="median" if psfk is None else "convolve",
        psfk=psfk,
    )

    assert result.sum() != array.sum()


@pytest.mark.parametrize(
    argnames="array",
    argvalues=_arrays,
)
@pytest.mark.parametrize(
    argnames="axis",
    argvalues=[
        tuple(shape),
    ],
)
@pytest.mark.parametrize("where", [None, na.random.uniform(0, 1, shape) > 0.01])
@pytest.mark.parametrize("inbkg", [None, na.ScalarArray.zeros(shape)])
@pytest.mark.parametrize("invar", [None, 6.5**2])
@pytest.mark.parametrize("psfk", [
    None,
    na.ScalarArray.ones(dict(x=5, y=5)) / 25,
])
def test_despike_mask(
    array: na.AbstractArray,
    axis: tuple[str, str],
    where: None | bool | na.AbstractArray,
    inbkg: None | na.AbstractArray,
    invar: None | float | na.AbstractArray,
    psfk: None | na.AbstractArray,
) -> None:
    kwargs = dict(
        axis=axis,
        where=where,
        inbkg=inbkg,
        invar=invar,
        fsmode="median" if psfk is None else "convolve",
        psfk=psfk,
    )

    result = na.despike_mask(array, **kwargs)
    cleaned = na.despike(array, **kwargs)

    if isinstance(array, na.AbstractFunctionArray):
        assert isinstance(result, na.AbstractFunctionArray)
        array = array.outputs
        result = result.outputs
        cleaned = cleaned.outputs

    assert result.dtype == bool
    assert np.any(result)
    assert np.all((cleaned != array) == result)


def test_despike_mask_function_arguments() -> None:
    inputs = na.ScalarUniformRandomSample(-5, 5, shape_random=shape)
    array = na.FunctionArray(inputs=inputs, outputs=img + 100 * spikes)
    where = na.random.uniform(0, 1, shape) > 0.01
    inbkg = na.ScalarArray.zeros(shape)
    invar = na.ScalarArray.zeros(shape) + 6.5**2

    result = na.despike_mask(
        array=array,
        axis=tuple(shape),
        where=na.FunctionArray(inputs=inputs, outputs=where),
        inbkg=na.FunctionArray(inputs=inputs, outputs=inbkg),
        invar=na.FunctionArray(inputs=inputs, outputs=invar),
    )
    expected = na.despike_mask(
        array=array,
        axis=tuple(shape),
        where=where,
        inbkg=inbkg,
        invar=invar,
    )

    assert np.all(result.outputs == expected.outputs)


def test_despike_mask_function_where() -> None:
    inputs = na.ScalarUniformRandomSample(-5, 5, shape_random=shape)
    where = na.FunctionArray(
        inputs=inputs,
        outputs=na.random.uniform(0, 1, shape) > 0.01,
    )

    with pytest.raises(TypeError):
        na.despike_mask(img, axis=tuple(shape), where=where)

    array = na.FunctionArray(inputs=inputs + 1, outputs=img)
    with pytest.raises(ValueError, match="its inputs must match"):
        na.despike_mask(array, axis=tuple(shape), where=where)


@pytest.mark.parametrize("dtype", [bool, int])
def test_despike_mask_where(dtype: type) -> None:
    """
    Pixels where `where` is false keep their values for astroscrappy, which
    matters on an image with a background, and `where` is true wherever it
    is not zero.
    """
    array = na.random.poisson(1000, shape) + 300 * na.random.poisson(0.01, shape)
    where = na.random.uniform(0, 1, shape) > 0.1
    where[dict(x=slice(None, None, 5))] = False

    result = na.despike_mask(
        array=array,
        axis=tuple(shape),
        where=where.astype(dtype),
    )

    crmask, _ = astroscrappy.detect_cosmics(
        indat=array.ndarray,
        inmask=~where.ndarray,
    )
    expected = na.ScalarArray(crmask, axes=array.axes)

    assert np.any(expected)
    assert np.all(result == expected)


def test_despike_mask_units() -> None:
    line = na.ScalarArray.zeros(shape)
    line[dict(x=50)] = 5000

    array = (img + 100 * spikes + line) * u.DN
    inbkg = line * u.DN
    invar = (6.5**2 + line) * u.DN**2
    unit = u.Unit(1000 * u.DN)

    result = na.despike_mask(
        array=array,
        axis=tuple(shape),
        inbkg=inbkg.to(unit),
        invar=invar.to(unit**2),
    )
    expected = na.despike_mask(
        array=array,
        axis=tuple(shape),
        inbkg=inbkg,
        invar=invar,
    )

    assert np.all(result == expected)


@pytest.mark.parametrize(
    argnames="axes",
    argvalues=[
        ("x", "y"),
        ("y", "x"),
    ],
)
def test_despike_mask_psfk(
    monkeypatch: pytest.MonkeyPatch,
    axes: tuple[str, str],
) -> None:
    array = img + 100 * spikes
    array = na.ScalarArray(array.ndarray_aligned(axes), axes)
    psfk = na.ScalarArray(np.arange(15.).reshape(3, 5), ("x", "y"))
    psfk_transposed = na.ScalarArray(psfk.ndarray.T, ("y", "x"))

    kernels = []

    def detect_cosmics(
        indat: np.ndarray,
        psfk: np.ndarray,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        kernels.append(psfk)
        return np.zeros(indat.shape, dtype=bool), indat

    monkeypatch.setattr(astroscrappy, "detect_cosmics", detect_cosmics)

    for p in [psfk, psfk_transposed]:
        na.despike_mask(array, axis=("x", "y"), fsmode="convolve", psfk=p)

    assert len(kernels) == 2
    for kernel in kernels:
        assert kernel.dtype == np.float32
        assert np.all(kernel == psfk.ndarray_aligned(axes))


@pytest.mark.parametrize(
    argnames="nonfinite,arguments",
    argvalues=[
        ("array", ("array",)),
        ("array", ("array", "inbkg", "invar")),
        ("inbkg", ("array", "inbkg")),
        ("invar", ("array", "invar")),
    ],
)
def test_despike_nonfinite(
    nonfinite: str,
    arguments: tuple[str, ...],
) -> None:
    region = dict(x=slice(40, 50), y=slice(40, 50))
    spike_inside = dict(x=45, y=45)
    spike_near = dict(x=51, y=45)

    array = img.copy()
    array[spike_inside] = 1000
    array[spike_near] = 1000

    kwargs = dict(
        array=array,
        inbkg=na.ScalarArray.zeros(shape),
        invar=na.ScalarArray.zeros(shape) + 6.5**2,
    )
    kwargs = {name: kwargs[name] for name in arguments}
    kwargs[nonfinite][region] = np.nan

    mask = na.despike_mask(axis=tuple(shape), **kwargs)
    cleaned = na.despike(axis=tuple(shape), **kwargs)

    assert mask[spike_near]
    assert not np.any(mask[region])
    assert cleaned[spike_near] < 100
    assert np.all(np.isnan(cleaned) == np.isnan(array))
    assert np.all((cleaned == array) | mask | np.isnan(array))


def test_despike_nonfinite_image() -> None:
    array = (img + 100 * spikes) * na.ScalarArray(np.array([1, np.nan]), ("t",))

    mask = na.despike_mask(array, axis=tuple(shape))
    cleaned = na.despike(array, axis=tuple(shape))

    assert np.any(mask[dict(t=0)])
    assert not np.any(mask[dict(t=1)])
    assert np.all(np.isnan(cleaned[dict(t=1)]))
