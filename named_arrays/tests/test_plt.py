import pytest
import matplotlib.animation
import named_arrays as na

_num_t = 11
_num_x = 12
_num_y = 13


@pytest.mark.parametrize(
    argnames="TXY",
    argvalues=[
        (
            na.linspace(-1, 1, axis="t", num=_num_t),
            na.linspace(-2, 2, axis="x", num=_num_x),
            na.linspace(-1, 1, axis="y", num=_num_y),
        ),
    ],
)
@pytest.mark.parametrize(
    argnames="C",
    argvalues=[
        na.random.uniform(-1, 1, shape_random=dict(t=_num_t, x=_num_x, y=_num_y)),
    ],
)
@pytest.mark.parametrize(
    argnames="axis_time",
    argvalues=["t"],
)
def test_pcolormovie(
    TXY: tuple[
        na.AbstractScalarArray,
        na.AbstractScalarArray,
        na.AbstractScalarArray,
    ],
    C: na.AbstractScalarArray,
    axis_time: str,
):
    result = na.plt.pcolormovie(
        *TXY,
        C=C,
        axis_time=axis_time,
    )
    assert isinstance(result, matplotlib.animation.FuncAnimation)
    assert isinstance(result.to_jshtml(), str)
