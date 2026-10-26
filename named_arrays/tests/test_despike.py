import pytest
import numpy as np
import astropy.units as u
import named_arrays as na

shape = dict(x=101, y=101)

img = na.random.normal(0, 6.5, shape)
spikes = 1000 * na.random.poisson(0.001, shape)

_arrays = [
    img + 100 * spikes,
    (img + 100 * spikes) * u.DN,
    img + na.NormalUncertainScalarArray(100, 5) * spikes,
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
    na.ScalarArray.ones(dict(x=5, y=5)),
])
def test_despike(
    array: na.AbstractArray,
    axis: tuple[str, str],
    where: None | bool | na.AbstractArray,
    inbkg: None | na.AbstractArray,
    invar: None | float | na.AbstractArray,
    psfk: None | na.AbstractArray,
):
    result = na.despike(
        array=array,
        axis=axis,
        where=where,
        inbkg=inbkg,
        invar=invar,
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
    na.ScalarArray.ones(dict(x=5, y=5)),
])
def test_despike_mask(
    array: na.AbstractArray,
    axis: tuple[str, str],
    where: None | bool | na.AbstractArray,
    inbkg: None | na.AbstractArray,
    invar: None | float | na.AbstractArray,
    psfk: None | na.AbstractArray,
):
    kwargs = dict(
        axis=axis,
        where=where,
        inbkg=inbkg,
        invar=invar,
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


def test_despike_mask_function_arguments():
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


@pytest.mark.parametrize("nonfinite", ["array", "inbkg", "invar"])
def test_despike_nonfinite(nonfinite: str):
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
    kwargs[nonfinite][region] = np.nan

    mask = na.despike_mask(axis=tuple(shape), **kwargs)
    cleaned = na.despike(axis=tuple(shape), **kwargs)

    assert mask[spike_near]
    assert not np.any(mask[region])
    assert cleaned[spike_near] < 100
    assert np.all(np.isnan(cleaned) == np.isnan(array))
    assert np.all((cleaned == array) | mask | np.isnan(array))
