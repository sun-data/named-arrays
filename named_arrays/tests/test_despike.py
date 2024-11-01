import pytest
import named_arrays as na

shape = dict(x=101, y=101)

img = na.random.normal(0, 6.5, shape)
spikes = 1000 * na.random.poisson(0.001, shape)


@pytest.mark.parametrize(
    argnames="array",
    argvalues=[
        img + 100 * spikes,
        img + na.NormalUncertainScalarArray(100, 5) * spikes,
        na.FunctionArray(
            inputs=None,
            outputs=img + 100 * spikes,
        ),
    ],
)
@pytest.mark.parametrize(
    argnames="axis",
    argvalues=[
        tuple(shape),
    ],
)
@pytest.mark.parametrize("where", [None])
@pytest.mark.parametrize("inbkg", [None])
@pytest.mark.parametrize("invar", [None])
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
