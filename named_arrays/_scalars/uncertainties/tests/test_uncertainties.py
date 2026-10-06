import dataclasses
from typing import Mapping, Sequence, Callable
import numpy as np
import pytest
import astropy.units as u
import named_arrays as na
import named_arrays.tests.test_core
import named_arrays._scalars.tests.test_scalars

__all__ = [
    'TestNominalRecursive',
    'AbstractTestAbstractUncertainScalarArray',
    'TestUncertainScalarArray',
    'TestUncertainScalarArrayCreation',
    'AbstractTestAbstractImplicitUncertainScalarArray',
    'TestUniformUncertainScalarArray',
    'TestNormalUncertainScalarArray',
]

_num_x = named_arrays.tests.test_core.num_x
_num_y = named_arrays.tests.test_core.num_y
_num_distribution = named_arrays.tests.test_core.num_distribution


def _fill_unselected(dtype: np.dtype, dtype_other: np.dtype) -> None | float:
    """
    The value expected in the elements which a sample of an uncertain mask did
    not select, for a nominal value or distribution with data type `dtype`
    accompanied by one with data type `dtype_other`, or :obj:`None` if NaN
    cannot mark them.
    Boolean arrays are tested separately.
    """
    if np.issubdtype(dtype, np.inexact):
        return np.nan
    if np.issubdtype(dtype, np.integer) and np.issubdtype(dtype_other, np.inexact):
        return np.nan
    return None


def _equal_nan(a: na.AbstractArray, b: na.AbstractArray) -> bool:
    """Whether two arrays are equal, counting NaN as equal to NaN."""
    return bool(np.all((a == b) | ((a != a) & (b != b))))


def _item_varying() -> na.UncertainScalarArray:
    """
    An uncertain mask along the ``y`` axis whose nominal value and samples
    all select different elements.
    """
    index = na.ScalarArrayRange(0, _num_y, axis="y")
    index_distribution = na.ScalarArrayRange(0, _num_distribution, axis=na.UncertainScalarArray.axis_distribution)
    return na.UncertainScalarArray(
        nominal=index % 2 == 0,
        distribution=(index + index_distribution) % 2 == 0,
    )


def _index_varying() -> na.UncertainScalarArray:
    """
    Uncertain indices along the ``y`` axis which permute it differently in
    the nominal value and in every sample.
    """
    index = na.ScalarArrayRange(0, _num_y, axis="y")
    index_distribution = na.ScalarArrayRange(0, _num_distribution, axis=na.UncertainScalarArray.axis_distribution)
    return na.UncertainScalarArray(
        nominal=index,
        distribution=(index + index_distribution + 1) % _num_y,
    )


def _uncertain_scalar_arrays():
    nominal_2d = na.ScalarUniformRandomSample(-4, 4, shape_random=dict(x=_num_x, y=_num_y)).explicit
    distribution_0d = 4 + na.ScalarUniformRandomSample(-0.1, 0.1, shape_random=dict(_distribution=_num_distribution))
    distribution_2d = 4 + na.ScalarUniformRandomSample(
        start=-0.1,
        stop=0.1,
        shape_random=dict(x=_num_x, y=_num_y, _distribution=_num_distribution)
    )
    return [
        na.UncertainScalarArray(4., distribution_0d),
        na.UncertainScalarArray(4. * u.mm, distribution_2d * u.mm),
        na.UncertainScalarArray(na.ScalarArray(4.), distribution_2d),
        na.UncertainScalarArray(na.ScalarArray(4.) * u.mm, distribution_0d * u.mm),
        na.UncertainScalarArray(nominal_2d, distribution_2d),
        na.UncertainScalarArray(nominal_2d * u.mm, distribution_2d * u.mm),
    ]



def _uncertain_scalar_arrays_2():
    nominal_1d = na.ScalarUniformRandomSample(-5, 5, shape_random=dict(y=_num_y))
    distribution_0d = na.ScalarArray(5.1).add_axes(na.UncertainScalarArray.axis_distribution)
    distribution_2d = 5 + na.ScalarUniformRandomSample(
        start=-5.1,
        stop=5.1,
        shape_random=dict(x=_num_x, y=_num_y, _distribution=_num_distribution),
    )
    return [
        5,
        nominal_1d,
        na.UncertainScalarArray(5, distribution_0d),
        na.UncertainScalarArray(5 * u.mm, distribution_2d * u.mm),
        na.UncertainScalarArray(nominal_1d * u.mm, distribution_0d * u.mm),
        na.UncertainScalarArray(nominal_1d.explicit, distribution_2d),
    ]


@dataclasses.dataclass(eq=False)
class _NominalContainer:
    """A composite dataclass used to exercise the recursion of ``na.nominal``."""
    data: na.AbstractScalar
    label: str = "data"
    size: int = dataclasses.field(init=False, default=-1)


class TestNominalRecursive:
    """Tests for the recursion of :func:`named_arrays.nominal` into nested structures."""

    def _uncertain(self) -> na.UncertainScalarArray:
        return na.UniformUncertainScalarArray(5, 4, num_distribution=_num_distribution).explicit

    def _expected(self):
        # the nominal value is deterministic (only the distribution is random)
        return self._uncertain().nominal

    def test_nominal_scalar_passthrough(self):
        assert na.nominal(7) == 7

    def test_nominal_dict(self):
        result = na.nominal({"a": self._uncertain(), "b": 2})
        assert isinstance(result, dict)
        assert np.all(result["a"] == self._expected())
        assert result["b"] == 2

    def test_nominal_list(self):
        result = na.nominal([self._uncertain()])
        assert isinstance(result, list)
        assert np.all(result[0] == self._expected())

    def test_nominal_tuple(self):
        result = na.nominal((self._uncertain(), 7))
        assert isinstance(result, tuple)
        assert np.all(result[0] == self._expected())
        assert result[1] == 7

    def test_nominal_nested(self):
        result = na.nominal({"a": [self._uncertain()]})
        assert np.all(result["a"][0] == self._expected())

    def test_nominal_dataclass(self):
        result = na.nominal(_NominalContainer(data=self._uncertain()))
        assert isinstance(result, _NominalContainer)
        assert np.all(result.data == self._expected())
        # non-array fields are passed through unchanged
        assert result.label == "data"
        # ``init=False`` fields are skipped by ``dataclasses.replace``
        assert result.size == -1


class AbstractTestAbstractUncertainScalarArray(
    named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar,
):

    def test_nominal(self, array: na.AbstractUncertainScalarArray):
        assert np.sum(array.nominal) != 0
        assert array.axis_distribution not in na.shape(array.nominal)

    def test_distribution(self, array: na.AbstractUncertainScalarArray):
        assert np.sum(array.distribution) != 0

    def test_num_distribution(self, array: na.AbstractUncertainScalarArray):
        assert isinstance(array.num_distribution, int)
        assert array.num_distribution > 0

    def test_shape_distribution(self, array: na.AbstractUncertainScalarArray):
        assert isinstance(array.shape_distribution, dict)
        for ax in array.shape_distribution:
            assert isinstance(ax, str)
            assert isinstance(array.shape_distribution[ax], int)

        assert array.axis_distribution in array.shape_distribution

    @pytest.mark.parametrize(
        argnames='item',
        argvalues=[
            dict(y=0),
            dict(y=np.int64(0)),
            dict(y=slice(0, 1)),
            dict(y=na.ScalarArray(np.array([0, 1]), axes=('y', ))),
            dict(
                y=na.UncertainScalarArray(
                    nominal=na.ScalarArray(np.array([0, 1]), axes=('y', )),
                    distribution=na.ScalarArray(
                        ndarray=np.array([[0,], [1,]]),
                        axes=('y', na.UncertainScalarArray.axis_distribution),
                    )
                ),
                _distribution=na.UncertainScalarArray(
                    nominal=None,
                    distribution=na.ScalarArray(
                        ndarray=np.array([[0], [0]]),
                        axes=('y', na.UncertainScalarArray.axis_distribution),
                    )
                )
            ),
            na.ScalarLinearSpace(0, 1, axis='y', num=_num_y) > 0.5,
            na.UncertainScalarArray(
                nominal=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y),
                distribution=na.ScalarNormalRandomSample(
                    center=na.ScalarLinearSpace(0, 1, axis='y', num=_num_y),
                    width=0.1,
                    shape_random={na.UncertainScalarArray.axis_distribution: _num_distribution},
                )
            ) > 0.5,
        ]
    )
    def test__getitem__(
            self,
            array: na.AbstractUncertainScalarArray,
            item: Mapping[str, int | slice | na.AbstractArray] | na.AbstractArray
    ):
        super().test__getitem__(array=array, item=item)

        if isinstance(item, na.AbstractArray):

            if not set(item.shape).issubset(array.shape_distribution):
                with pytest.raises(ValueError):
                    array[item]
                return

            if isinstance(item, na.AbstractUncertainScalarArray):
                # Every element selected by the nominal value or by any sample
                # is kept, and is NaN, or False for boolean arrays, in the
                # realizations which did not select it.
                axis = na.UncertainScalarArray.axis_distribution
                union = item.nominal | np.any(item.distribution, axis=axis)
                array_broadcasted = array.broadcasted
                dtype_nominal = na.get_dtype(array_broadcasted.nominal)
                dtype_distribution = na.get_dtype(array_broadcasted.distribution)
                fill_nominal = _fill_unselected(dtype_nominal, dtype_distribution)
                fill_distribution = _fill_unselected(dtype_distribution, dtype_nominal)
                if fill_nominal is None or fill_distribution is None:
                    # Integer arrays are only parametrized here with masks
                    # whose samples agree, so no element is filled.
                    # `test__getitem__uncertain_item_integer` tests a mask
                    # which varies.
                    fill_nominal = fill_distribution = 0
                result = array[item]
                result_expected = na.UncertainScalarArray(
                    nominal=np.where(item.nominal, array_broadcasted.nominal, fill_nominal)[union],
                    distribution=np.where(item.distribution, array_broadcasted.distribution, fill_distribution)[union],
                )
                assert _equal_nan(result, result_expected)
                return
            else:
                item_nominal = item_distribution = item

        elif isinstance(item, dict):

            item_nominal = dict()
            item_distribution = dict()

            for ax in item:
                if isinstance(item[ax], na.AbstractArray):
                    if isinstance(item[ax], na.AbstractUncertainScalarArray):
                        item_nominal[ax] = item[ax].nominal
                        item_distribution[ax] = item[ax].distribution
                    else:
                        item_nominal[ax] = item_distribution[ax] = item[ax]
                else:
                    item_nominal[ax] = item_distribution[ax] = item[ax]

        result = array[item]
        result_expected = na.UncertainScalarArray(
            array.broadcasted.nominal[item_nominal],
            array.broadcasted.distribution[item_distribution],
        )

        assert np.all(result == result_expected)

    def test__getitem__uncertain_item_propagation(self, array: na.AbstractUncertainScalarArray):
        # Ignoring the NaN of the elements a sample did not select reproduces
        # the selection applied to the nominal value and each sample separately.
        # Integer arrays cannot hold NaN, which is tested separately below.
        array = array.astype(float)
        item = array > array.mean()
        result = np.nansum(array[item])
        result_expected = np.sum(np.where(item, array, 0))
        assert np.allclose(result, result_expected)

    def test__getitem__uncertain_item_certain(self, array: na.AbstractUncertainScalarArray):
        # A mask whose samples all agree with its nominal value selects
        # exactly like a plain mask, without any NaN
        item = na.ScalarLinearSpace(0, 1, axis="y", num=_num_y) > 0.5
        array = na.broadcast_to(array, na.broadcast_shapes(array.shape, item.shape))
        result = array[na.UncertainScalarArray(item, item)]
        assert np.all(result == array[item])

    def test__getitem__uncertain_item_integer(self, array: na.AbstractUncertainScalarArray):
        item = _item_varying()
        array = na.broadcast_to(array, na.broadcast_shapes(array.shape, item.shape)).astype(int)
        with pytest.raises(ValueError, match="`item` selects different elements"):
            array[item]

    def test__getitem__uncertain_item_bool(self, array: na.AbstractUncertainScalarArray):
        # Boolean arrays cannot hold NaN, so a sample which did not select an
        # element holds False there instead
        item = _item_varying()
        array = na.broadcast_to(array, na.broadcast_shapes(array.shape, item.shape))
        array = array > array.mean()
        union = item.nominal | np.any(item.distribution, axis=array.axis_distribution)
        result = array[item]
        assert na.get_dtype(result.nominal) == np.dtype(bool)
        assert na.get_dtype(result.distribution) == np.dtype(bool)
        assert np.all(result == np.where(item, array, False)[union])

    @pytest.mark.parametrize(
        argnames="dtype_nominal,dtype_distribution,dtype_nominal_expected,dtype_distribution_expected",
        argvalues=[
            (np.float32, np.float32, np.float32, np.float32),
            (np.float32, np.float64, np.float32, np.float64),
            (np.int64, np.float32, np.float64, np.float32),
        ],
    )
    def test__getitem__uncertain_item_dtype(
        self,
        array: na.AbstractUncertainScalarArray,
        dtype_nominal: type,
        dtype_distribution: type,
        dtype_nominal_expected: type,
        dtype_distribution_expected: type,
    ):
        # The NaN which fill the elements a sample did not select keep the
        # precision of the nominal value and of the distribution, and an
        # integer nominal value becomes floating-point
        item = _item_varying()
        array = na.broadcast_to(array, na.broadcast_shapes(array.shape, item.shape))
        array = na.UncertainScalarArray(
            nominal=array.nominal.astype(dtype_nominal),
            distribution=array.distribution.astype(dtype_distribution),
        )
        result = array[item]
        assert na.get_dtype(result.nominal) == dtype_nominal_expected
        assert na.get_dtype(result.distribution) == dtype_distribution_expected

    @pytest.mark.parametrize(
        argnames="dtype",
        argvalues=[int, str],
    )
    def test__getitem__reversed_uncertain_item_certain(
        self,
        array: na.AbstractUncertainScalarArray,
        dtype: type,
    ):
        # An array without a distribution which can hold neither NaN nor
        # False, like the integer component of a vector, keeps every element
        # selected by any sample and stays certain
        item = _item_varying()
        array = na.broadcast_to(array, na.broadcast_shapes(array.shape, item.shape))
        union = item.nominal | np.any(item.distribution, axis=array.axis_distribution)
        companion = na.as_named_array(na.value(array.nominal)).astype(dtype)
        result = companion[item]
        assert isinstance(result, na.ScalarArray)
        assert np.all(result == companion[union])

    @pytest.mark.parametrize(
        argnames="dtype",
        argvalues=[float, bool],
    )
    def test__getitem__reversed_uncertain_item_filled(
        self,
        array: na.AbstractUncertainScalarArray,
        dtype: type,
    ):
        # An array without a distribution which can hold NaN or False is
        # filled like an uncertain one, so that reductions select each sample
        item = _item_varying()
        array = na.broadcast_to(array, na.broadcast_shapes(array.shape, item.shape))
        companion = na.as_named_array(na.value(array.nominal)).astype(dtype)
        result = companion[item]
        result_expected = na.UncertainScalarArray(companion, companion)[item]
        assert isinstance(result, na.UncertainScalarArray)
        assert _equal_nan(result, result_expected)

    @pytest.mark.parametrize(
        argnames="dtype",
        argvalues=[float, int, bool, str],
    )
    def test__getitem__reversed_uncertain_item_agrees(
        self,
        array: na.AbstractUncertainScalarArray,
        dtype: type,
    ):
        # A mask whose samples all agree with its nominal value leaves an
        # array without a distribution certain, so the selection can be
        # assigned back through the same mask
        mask = _item_varying().nominal
        shape_distribution = {**mask.shape, array.axis_distribution: _num_distribution}
        item = na.UncertainScalarArray(mask, na.broadcast_to(mask, shape_distribution))
        array = na.broadcast_to(array, na.broadcast_shapes(array.shape, item.shape))
        companion = na.as_named_array(na.value(array.nominal)).astype(dtype).copy()
        result = companion[item]
        assert isinstance(result, na.ScalarArray)
        assert np.all(result == companion[mask])
        expected = companion.copy()
        companion[item] = result
        assert np.all(companion == expected)

    def test__getitem__reversed_uncertain_indices(self, array: na.AbstractUncertainScalarArray):
        # Indexing a plain array with indices which differ between samples,
        # like those returned by `np.argsort()`, gathers each sample separately
        array = na.broadcast_to(array, na.broadcast_shapes(array.shape, dict(x=_num_x)))
        indices = np.argsort(array, axis="x")
        result = na.ScalarArrayRange(0, _num_x, axis="x")[indices]
        assert np.all(result == indices["x"])

    def test__mul__(self, array: na.AbstractUncertainScalarArray):
        unit = u.mm
        result = array * unit
        result_nominal = array.nominal * unit
        result_distribution = array.distribution * unit
        assert np.all(result.nominal == result_nominal)
        assert np.all(result.distribution == result_distribution)

    def test__lshift__(self, array: na.AbstractUncertainScalarArray):
        unit = u.mm
        result = array << unit
        result_nominal = array.nominal << unit
        result_distribution = array.distribution << unit
        assert np.all(result.nominal == result_nominal)
        assert np.all(result.distribution == result_distribution)

    def test__truediv__(self, array: na.AbstractUncertainScalarArray):
        unit = u.mm
        result = array / unit
        result_nominal = array.nominal / unit
        result_distribution = array.distribution / unit
        assert np.all(result.nominal == result_nominal)
        assert np.all(result.distribution == result_distribution)

    class TestUfuncUnary(
        named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestUfuncUnary
    ):

        def test_ufunc_unary(
                self,
                ufunc: np.ufunc,
                array: na.AbstractUncertainScalarArray,
        ):
            super().test_ufunc_unary(ufunc, array)

            kwargs = dict()
            kwargs_nominal = dict()
            kwargs_distribution = dict()

            unit = array.unit_normalized
            if ufunc in [np.log, np.log2, np.log10, np.sqrt]:
                kwargs["where"] = array > 0
            elif ufunc in [np.log1p]:
                kwargs["where"] = array >= (-1 * unit)
            elif ufunc in [np.arcsin, np.arccos, np.arctanh]:
                kwargs["where"] = ((-1 * unit) < array) & (array < (1 * unit))
            elif ufunc in [np.arccosh]:
                kwargs["where"] = array >= (1 * unit)
            elif ufunc in [np.reciprocal]:
                kwargs["where"] = array != 0

            if "where" in kwargs:
                kwargs_nominal["where"] = kwargs["where"].nominal
                kwargs_distribution["where"] = kwargs["where"].distribution

            try:
                ufunc(array.nominal, **kwargs_nominal)
                ufunc(array.distribution, **kwargs_distribution)
            except (ValueError, TypeError) as e:
                with pytest.raises(type(e)):
                    ufunc(array, **kwargs)
                return

            result = ufunc(array, **kwargs)
            result_nominal = ufunc(array.nominal, **kwargs_nominal)
            result_distribution = ufunc(array.distribution, **kwargs_distribution)

            if ufunc.nout == 1:
                out = 0 * result
            else:
                out = tuple(0 * r for r in result)

            result_out = ufunc(array, out=out, **kwargs)

            if ufunc.nout == 1:
                out = (out, )
                result = (result, )
                result_nominal = (result_nominal, )
                result_distribution = (result_distribution, )
                result_out = (result_out, )

            for i in range(ufunc.nout):
                assert np.all(result[i].nominal == result_nominal[i], **kwargs_nominal)
                assert np.all(result[i].distribution == result_distribution[i], **kwargs_distribution)
                assert np.all(result[i] == result_out[i], **kwargs)
                assert result_out[i] is out[i]

    @pytest.mark.parametrize('array_2', _uncertain_scalar_arrays_2())
    class TestUfuncBinary(
        named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestUfuncBinary
    ):

        def check_ufunc_binary(
                self,
                ufunc: np.ufunc,
                array: None | bool | int | float | complex | na.AbstractUncertainScalarArray,
                array_2: None | bool | int | float | complex | na.AbstractUncertainScalarArray,
        ):
            super().check_ufunc_binary(ufunc=ufunc, array=array, array_2=array_2)

            if not isinstance(array, na.AbstractUncertainScalarArray):
                array_normalized = na.UncertainScalarArray(
                    nominal=array,
                    distribution=na.add_axes(array, na.UncertainScalarArray.axis_distribution),
                )
            else:
                array_normalized = array

            if not isinstance(array_2, na.AbstractUncertainScalarArray):
                array_2_normalized = na.UncertainScalarArray(
                    nominal=array_2,
                    distribution=na.add_axes(array_2, na.UncertainScalarArray.axis_distribution),
                )
            else:
                array_2_normalized = array_2

            kwargs = dict()
            kwargs_nominal = dict()
            kwargs_distribution = dict()

            unit_2 = na.unit_normalized(array_2)
            if ufunc in [np.power, np.float_power]:
                kwargs["where"] = (array_2_normalized >= (1 * unit_2)) & (array_normalized >= 0)
            elif ufunc in [np.divide, np.floor_divide, np.remainder, np.fmod, np.divmod]:
                kwargs["where"] = array_2_normalized != 0

            if "where" in kwargs:
                kwargs_nominal["where"] = kwargs["where"].nominal
                kwargs_distribution["where"] = kwargs["where"].distribution

            try:
                ufunc(array_normalized.nominal, array_2_normalized.nominal, **kwargs_nominal)
                ufunc(array_normalized.distribution, array_2_normalized.distribution, **kwargs_distribution)
            except (ValueError, TypeError) as e:
                with pytest.raises(type(e)):
                    ufunc(array, array_2, **kwargs)
                return

            result = ufunc(array, array_2, **kwargs)
            result_nominal = ufunc(array_normalized.nominal, array_2_normalized.nominal, **kwargs_nominal)
            result_distribution = ufunc(
                array_normalized.distribution, array_2_normalized.distribution, **kwargs_distribution)

            if ufunc.nout == 1:
                out = 0 * np.nan_to_num(result)
            else:
                out = tuple(0 * np.nan_to_num(r) for r in result)

            result_out = ufunc(array, array_2, out=out, **kwargs)

            if ufunc.nout == 1:
                out = (out, )
                result = (result, )
                result_nominal = (result_nominal, )
                result_distribution = (result_distribution, )
                result_out = (result_out, )

            for i in range(ufunc.nout):
                assert np.all(result[i].nominal == result_nominal[i], **kwargs_nominal)
                assert np.all(result[i].distribution == result_distribution[i], **kwargs_distribution)
                assert np.all(result[i] == result_out[i], **kwargs)
                assert result_out[i] is out[i]

    @pytest.mark.parametrize('array_2', _uncertain_scalar_arrays_2())
    class TestMatmul(
        named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestMatmul
    ):
        pass

    class TestArrayFunctions(
        named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestArrayFunctions,
    ):

        @pytest.mark.parametrize("array_2", _uncertain_scalar_arrays_2())
        class TestStackLikeFunctions(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestArrayFunctions
            .TestStackLikeFunctions
        ):
            pass

        @pytest.mark.parametrize("array_2", _uncertain_scalar_arrays_2())
        class TestAsArrayLikeFunctions(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestArrayFunctions
            .TestAsArrayLikeFunctions
        ):

            def test_asarray_like_functions(
                    self,
                    func: Callable,
                    array: None | float | u.Quantity | na.AbstractArray,
                    array_2: None | float | u.Quantity | na.AbstractArray,
            ):
                a = array
                like = array_2

                if a is None:
                    assert func(a, like=like) is None
                    return

                result = func(a, like=like)

                assert isinstance(result, na.UncertainScalarArray)
                assert isinstance(result.nominal, na.ScalarArray)
                assert isinstance(result.distribution, na.ScalarArray)
                assert isinstance(result.nominal.ndarray, np.ndarray)
                assert isinstance(result.distribution.ndarray, np.ndarray)

                assert np.all(result.value == na.value(a))

                super().test_asarray_like_functions(
                    func=func,
                    array=array,
                    array_2=array_2,
                )

        class TestSingleArgumentFunctions(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestArrayFunctions.TestSingleArgumentFunctions,
        ):
            def test_single_argument_functions(
                self,
                func: Callable,
                array: na.AbstractUncertainScalarArray,
            ):
                result = func(array)
                assert np.all(result.nominal == func(array.nominal))
                assert np.all(result.distribution == func(array.distribution))

        @pytest.mark.parametrize(
            argnames='where',
            argvalues=[
                np._NoValue,
                True,
                na.ScalarArray(True),
                (na.ScalarLinearSpace(-1, 1, 'x', _num_x) >= 0) | (na.ScalarLinearSpace(-1, 1, 'y', _num_y) >= 0),
                na.UncertainScalarArray(
                    nominal=(na.ScalarLinearSpace(-1, 1, 'x', _num_x) >= 0)
                            | (na.ScalarLinearSpace(-1, 1, 'y', _num_y) >= 0),
                    distribution=(na.ScalarLinearSpace(-1, 1, 'x', _num_x) >= 0)
                                 | (na.ScalarLinearSpace(-1, 1, 'y', _num_y) >= 0)
                                 | (na.ScalarLinearSpace(-1, 1, '_distribution', _num_distribution) >= 0)
                ),
            ]
        )
        class TestReductionFunctions(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestArrayFunctions.
            TestReductionFunctions,
        ):

            def test_reduction_functions(
                    self,
                    func: Callable,
                    array: na.AbstractUncertainScalarArray,
                    axis: None | str | Sequence[str],
                    dtype: None | type | np.dtype,
                    keepdims: bool,
                    where: bool | na.AbstractArray,
            ):
                super().test_reduction_functions(
                    func=func,
                    array=array,
                    axis=axis,
                    dtype=dtype,
                    keepdims=keepdims,
                    where=where,
                )

                kwargs = dict(
                    axis=axis,
                    dtype=dtype,
                    keepdims=keepdims,
                    where=where,
                )

                kwargs_nominal = kwargs.copy()
                kwargs_distribution = kwargs.copy()

                if axis is None:
                    axis_normalized = na.axis_normalized(array, axis)
                    kwargs_nominal["axis"] = axis_normalized
                    kwargs_distribution["axis"] = axis_normalized

                if isinstance(where, na.AbstractUncertainScalarArray):
                    kwargs_nominal["where"] = where.nominal
                    kwargs_distribution["where"] = where.distribution
                else:
                    kwargs_nominal["where"] = kwargs_distribution["where"] = where

                try:
                    result_nominal = func(array.broadcasted.nominal, **kwargs_nominal)
                    result_distribution = func(array.broadcasted.distribution, **kwargs_distribution)
                except (ValueError, TypeError, u.UnitsError) as e:
                    with pytest.raises(type(e)):
                        func(array, **kwargs)
                    return

                result = func(array, **kwargs)

                out = 0 * result
                result_out = func(array, out=out, **kwargs)

                assert np.all(result.nominal == result_nominal)
                assert np.all(result.distribution == result_distribution)
                assert np.allclose(result, result_out)
                assert result_out is out

        class TestCumlativeReductionFunctions(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestArrayFunctions.
            TestCumulativeReductionFunctions,
        ):

            def test_cumulative_reduction_functions(
                    self,
                    func: Callable,
                    array: na.AbstractUncertainScalarArray,
                    axis: None | str | Sequence[str],
                    dtype: None | type | np.dtype,
            ):
                super().test_cumulative_reduction_functions(
                    func=func,
                    array=array,
                    axis=axis,
                    dtype=dtype,
                )

                if not array.shape:
                    return

                kwargs = dict(
                    axis=axis,
                    dtype=dtype,
                )

                kwargs_nominal = kwargs.copy()
                kwargs_distribution = kwargs.copy()

                if axis is None:
                    axis_normalized = na.axis_normalized(array, axis)
                    kwargs_nominal["axis"] = axis_normalized
                    kwargs_distribution["axis"] = axis_normalized

                try:
                    result_nominal = func(array.broadcasted.nominal, **kwargs_nominal)
                    result_distribution = func(array.broadcasted.distribution, **kwargs_distribution)
                except (ValueError, TypeError, u.UnitsError) as e:
                    with pytest.raises(type(e)):
                        func(array, **kwargs)
                    return

                result = func(array, **kwargs)

                out = 0 * result
                result_out = func(array, out=out, **kwargs)

                assert np.all(result.nominal == result_nominal)
                assert np.all(result.distribution == result_distribution)
                assert np.allclose(result, result_out)
                assert result_out is out


        @pytest.mark.parametrize(
            argnames='q',
            argvalues=[
                .25,
                25 * u.percent,
                na.ScalarLinearSpace(.25, .75, axis='q', num=3, endpoint=True),
                na.UncertainScalarArray(
                    nominal=25 * u.percent,
                    distribution=na.ScalarNormalRandomSample(
                        center=25 * u.percent,
                        width=1 * u.percent,
                        shape_random=dict(_distribution=_num_distribution)
                    )
                )
            ]
        )
        class TestPercentileLikeFunctions(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestArrayFunctions.
            TestPercentileLikeFunctions
        ):

            def test_percentile_like_functions(
                    self,
                    func: Callable,
                    array: na.AbstractUncertainScalarArray,
                    q: float | u.Quantity | na.AbstractArray,
                    axis: None | str | Sequence[str],
                    keepdims: bool,
            ):
                super().test_percentile_like_functions(
                    func=func,
                    array=array,
                    q=q,
                    axis=axis,
                    keepdims=keepdims,
                )

                array_normalized = array
                if isinstance(array, na.AbstractArray):
                    if isinstance(array, na.AbstractScalar):
                        if isinstance(array, na.AbstractScalarArray):
                            array_normalized = na.UncertainScalarArray(array, array)

                kwargs = dict(
                    q=q,
                    axis=axis,
                    keepdims=keepdims,
                )

                kwargs_nominal = kwargs.copy()
                kwargs_distribution = kwargs.copy()

                if isinstance(q, na.AbstractUncertainScalarArray):
                    kwargs_nominal["q"] = q.nominal
                    kwargs_distribution["q"] = q.distribution
                else:
                    kwargs_nominal["q"] = kwargs_distribution["q"] = q

                if axis is None:
                    axis_normalized = na.axis_normalized(array, axis)
                    kwargs_nominal["axis"] = axis_normalized
                    kwargs_distribution["axis"] = axis_normalized

                try:
                    result_nominal = func(array_normalized.broadcasted.nominal, **kwargs_nominal)
                    result_distribution = func(array_normalized.broadcasted.distribution, **kwargs_distribution)
                except (ValueError, TypeError) as e:
                    with pytest.raises(type(e)):
                        func(array, **kwargs)
                    return

                result = func(array, **kwargs)

                out = 0 * result
                result_out = func(array, out=out, **kwargs)

                assert np.all(result.nominal == result_nominal)
                assert np.all(result.distribution == result_distribution)
                assert np.all(result == result_out)
                assert result_out is out

        class TestFFTLikeFunctions(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestArrayFunctions.TestFFTLikeFunctions,
        ):

            def test_fft_like_functions(
                    self,
                    func: Callable,
                    array: na.AbstractUncertainScalarArray,
                    axis: tuple[str, str],
            ):
                if axis[0] not in array.shape:
                    with pytest.raises(ValueError, match="`axis` .* not in array with shape .*"):
                        func(array, axis=axis)
                    return

                result = func(array, axis=axis)
                result_nominal = func(array.broadcasted.nominal, axis=axis)
                result_distribution = func(array.broadcasted.distribution, axis=axis)

                assert np.all(result.nominal == result_nominal)
                assert np.all(result.distribution == result_distribution)

        class TestFFTNLikeFunctions(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestArrayFunctions.TestFFTNLikeFunctions
        ):
            def test_fftn_like_functions(
                    self,
                    func: Callable,
                    array: na.AbstractUncertainScalarArray,
                    axes: dict[str, str],
                    s: None | dict[str, int],
            ):
                if not set(axes).issubset(array.shape):
                    with pytest.raises(ValueError, match="`axes`, .*, not a subset of array axes, .*"):
                        func(array, axes=axes, s=s)
                    return

                if s is not None and axes.keys() != s.keys():
                    with pytest.raises(ValueError):
                        func(a=array, axes=axes, s=s)
                    return

                result = func(array, axes=axes, s=s)
                result_nominal = func(array.broadcasted.nominal, axes=axes, s=s)
                result_distribution = func(array.broadcasted.distribution, axes=axes, s=s)

                assert np.all(result.nominal == result_nominal)
                assert np.all(result.distribution == result_distribution)

        class TestEmathFunctions(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestArrayFunctions.TestEmathFunctions,
        ):
            def test_emath_functions(
                self,
                func: Callable,
                array: na.AbstractUncertainScalarArray,
            ):
                result = func(array)
                assert np.all(result.nominal == func(array.nominal))
                assert np.all(result.distribution == func(array.distribution))

        @pytest.mark.parametrize('axis', [None, 'x', 'y', ('x', 'y'), ()])
        def test_sort(self, array: na.AbstractUncertainScalarArray, axis: None | str | Sequence[str]):

            axis_normalized = na.axis_normalized(array, axis)

            if axis is not None:
                if not axis:
                    with pytest.raises(ValueError, match="if `axis` is a sequence, it must not be empty, got .*"):
                        np.sort(array, axis=axis)
                    return

                if not set(axis_normalized).issubset(array.shape):
                    with pytest.raises(ValueError, match="`axis`, .* is not a subset of `a.axes`, .*"):
                        np.sort(array, axis=axis)
                    return

            result = np.sort(array, axis=axis)

            if not axis_normalized:
                assert np.all(result == array)
                return

            # The nominal value and every sample are sorted independently
            array_broadcasted = na.broadcast_to(array, array.shape)
            result_expected = na.UncertainScalarArray(
                nominal=np.sort(array_broadcasted.nominal, axis=axis_normalized),
                distribution=np.sort(array_broadcasted.distribution, axis=axis_normalized),
            )
            assert np.all(result == result_expected)

            # so the first element of every sorted sample is its minimum
            axis_flattened = na.flatten_axes(axis_normalized)
            assert np.all(result[{axis_flattened: 0}] == np.min(array, axis=axis_normalized))

        def test_nonzero(self, array: na.AbstractUncertainScalarArray):

            super().test_nonzero(array)

            mask = array > array.mean()

            # A single set of indices exists only if every sample selects the
            # same elements as the nominal value.
            if np.any(mask.distribution != mask.nominal):
                with pytest.raises(ValueError, match="the nonzero elements of `a` differ"):
                    np.nonzero(mask)

            mask_nominal = na.as_named_array(mask.nominal)
            result = np.nonzero(na.UncertainScalarArray(mask_nominal, mask_nominal))
            result_expected = np.nonzero(mask_nominal)
            assert result.keys() == result_expected.keys()
            for ax in result_expected:
                assert np.all(result[ax] == result_expected[ax])

        @pytest.mark.parametrize('copy', [False, True])
        def test_nan_to_num(
                self,
                array: na.AbstractUncertainScalarArray,
                copy: bool,
        ):

            super().test_nan_to_num(array=array, copy=copy)

            if not copy and not isinstance(array, na.AbstractExplicitArray):
                with pytest.raises(TypeError, match="can't write to an array .*"):
                    np.nan_to_num(array, copy=copy)
                return

            try:
                result_nominal = np.nan_to_num(array.nominal, copy=copy)
                result_distribution = np.nan_to_num(array.distribution, copy=copy)
            except ValueError as e:
                match = "Unable to avoid copy"
                if e.args[0].startswith(match):
                    with pytest.raises(ValueError, match=match):
                        np.nan_to_num(array, copy=copy)
                    return

            result = np.nan_to_num(array, copy=copy)

            assert np.all(result.nominal == result_nominal)
            assert np.all(result.distribution == result_distribution)

        @pytest.mark.parametrize('v', _uncertain_scalar_arrays_2())
        @pytest.mark.parametrize('mode', ['full', 'same', 'valid'])
        def test_convolve(self, array: na.AbstractArray, v: na.AbstractArray, mode: str):
            super().test_convolve(array=array, v=v, mode=mode)
            with pytest.raises(ValueError, match="`numpy.convolve` is not supported .*"):
                np.convolve(array, v=v, mode=mode)

    class TestNamedArrayFunctions(
        named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestNamedArrayFunctions
    ):

        def test_nominal(self, array: na.AbstractUncertainScalarArray):
            result = na.nominal(array)
            assert np.all(result == array.nominal)

        @pytest.mark.parametrize(
            argnames="array_2",
            argvalues=_uncertain_scalar_arrays_2(),
        )
        @pytest.mark.parametrize(
            argnames="where, alpha",
            argvalues=[
                (
                    np._NoValue,
                    np._NoValue,
                ),
                (
                    True,
                    na.linspace(0, 1, axis="x", num=_num_x),
                ),
                (
                    na.linspace(0, 1, axis="x", num=_num_x) > 0.5,
                    np._NoValue,
                ),
            ]
        )
        class TestPltPlotLikeFunctions(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestNamedArrayFunctions.TestPltPlotLikeFunctions,
        ):
            pass

        @pytest.mark.parametrize(
            argnames="array_2",
            argvalues=_uncertain_scalar_arrays_2(),
        )
        @pytest.mark.parametrize(
            argnames="s",
            argvalues=[
                None,
            ]
        )
        @pytest.mark.parametrize(
            argnames="c",
            argvalues=[
                None,
            ]
        )
        @pytest.mark.parametrize(
            argnames="where",
            argvalues=[
                True,
                na.linspace(0, 1, axis="x", num=_num_x) > 0.5,
            ]
        )
        class TestPltScatter(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestNamedArrayFunctions.TestPltScatter,
        ):
            pass

        @pytest.mark.skip
        class TestPltPcolormesh(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestNamedArrayFunctions.TestPltPcolormesh,
        ):
            pass

        @pytest.mark.parametrize(
            argnames="function",
            argvalues=[
                lambda x: a * x ** 3
                for a in [2, na.UniformUncertainScalarArray(2, width=0.5, num_distribution=_num_distribution)]
            ]
        )
        class TestJacobian(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestNamedArrayFunctions.TestJacobian,
        ):
            pass

        @pytest.mark.parametrize(
            argnames="func",
            argvalues=[
                na.optimize.root_secant,
                na.optimize.root_newton,
            ],
        )
        @pytest.mark.parametrize(
            argnames="function",
            argvalues=[
                lambda x: np.square(na.value(x) - shift_horizontal) + shift_vertical
                for shift_horizontal in [
                    20,
                    na.UniformUncertainScalarArray(20, width=1, num_distribution=_num_distribution),
                ]
                for shift_vertical in [-1]
            ]
        )
        class TestOptimizeRoot(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestNamedArrayFunctions.TestOptimizeRoot,
        ):
            pass

        @pytest.mark.skip
        class TestOptimizeMinimum(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestNamedArrayFunctions.TestOptimizeMinimum,
        ):
            pass

        @pytest.mark.parametrize(
            argnames="func",
            argvalues=[
                na.optimize.minimum_brent,
            ],
        )
        @pytest.mark.parametrize(
            argnames="function,expected",
            argvalues=[
                (
                    lambda x, p=profile, s=shift_horizontal: p(na.value(x) - s) + 1,
                    shift_horizontal,
                )
                for profile in [
                    np.square,
                    np.abs,
                ]
                for shift_horizontal in [
                    20,
                    na.linspace(19, 20, axis="c", num=6),
                    na.NormalUncertainScalarArray(20, width=1, num_distribution=_num_distribution),
                ]
            ]
        )
        class TestOptimizeMinimumBrent(
            named_arrays._scalars.tests.test_scalars.AbstractTestAbstractScalar.TestNamedArrayFunctions.TestOptimizeMinimumBrent,
        ):
            pass


@pytest.mark.parametrize('array', _uncertain_scalar_arrays())
class TestUncertainScalarArray(
    AbstractTestAbstractUncertainScalarArray,
    named_arrays.tests.test_core.AbstractTestAbstractExplicitArray
):

    @pytest.mark.parametrize(
        argnames="item",
        argvalues=[
            dict(y=0),
            dict(x=0, y=0),
            dict(y=slice(None)),
            dict(y=na.ScalarArrayRange(0, _num_y, axis='y')),
            dict(x=na.ScalarArrayRange(0, _num_x, axis='x'), y=na.ScalarArrayRange(0, _num_y, axis='y')),
            na.ScalarArray.ones(shape=dict(y=_num_y), dtype=bool),
        ],
    )
    @pytest.mark.parametrize(
        argnames="value",
        argvalues=[
            0,
            na.ScalarUniformRandomSample(-5, 5, dict(y=_num_y)),
            na.UncertainScalarUniformRandomSample(
                start=na.UniformUncertainScalarArray(-5, 0, num_distribution=_num_distribution),
                stop=5,
                shape_random=dict(y=_num_y),
            )
        ]
    )
    def test__setitem__(
            self,
            array: na.ScalarArray,
            item: dict[str, int | slice | na.ScalarArray] | na.ScalarArray,
            value: float | na.ScalarArray
    ):
        super().test__setitem__(array=array, item=item, value=value)

    @pytest.mark.parametrize(
        argnames="value",
        argvalues=[
            0,
            na.UncertainScalarArray(
                nominal=10,
                distribution=na.ScalarArrayRange(0, _num_distribution, axis=na.UncertainScalarArray.axis_distribution),
            ),
        ],
    )
    def test__setitem__uncertain_item(
            self,
            array: na.UncertainScalarArray,
            value: float | na.UncertainScalarArray,
    ):
        # Each realization only changes the elements it selected
        unit = na.unit(array)
        if unit is not None:
            value = value * unit
        item = array > array.mean()
        result = na.broadcast_to(array, array.shape).astype(float).copy()
        result_expected = np.where(item, value, result)
        result[item] = value
        assert np.all(result == result_expected)

    def test__setitem__uncertain_item_no_samples(self, array: na.UncertainScalarArray):
        # Each sample of a varying mask changes different elements, so a
        # distribution without a sample axis receives one
        nominal = na.broadcast_to(array, array.shape).astype(float).nominal
        result = na.UncertainScalarArray(nominal.copy(), nominal.copy())
        item = array > array.mean()
        result_expected = np.where(item, 0, result)
        result[item] = 0
        assert np.all(result == result_expected)

    def test__setitem__plain_uncertain_item(self, array: na.UncertainScalarArray):
        # An array without a distribution cannot store a different selection
        # for each sample, so it refuses an uncertain mask instead of ignoring it
        item = _item_varying()
        shape = na.broadcast_shapes(array.shape, item.shape)
        result = na.as_named_array(na.broadcast_to(array, shape).astype(float).nominal).copy()
        with pytest.raises(ValueError, match="convert this array to an uncertain array"):
            result[item] = 0
        with pytest.raises(ValueError, match="convert this array to an uncertain array"):
            result[dict(y=_index_varying())] = 0

    def test__setitem__plain_uncertain_item_certain(self, array: na.UncertainScalarArray):
        # An uncertain mask or index whose samples all agree with its nominal
        # value is applied like a plain one
        array = na.broadcast_to(array, na.broadcast_shapes(array.shape, dict(y=_num_y))).astype(float)
        nominal = na.as_named_array(array.nominal)
        mask = nominal > nominal.mean()
        index = np.argsort(nominal, axis="y")["y"]
        unit = na.unit(array)
        value = 0 if unit is None else 0 * unit

        result = nominal.copy()
        result[na.UncertainScalarArray(mask, mask)] = value
        assert np.all(result == np.where(mask, value, nominal))

        result = nominal.copy()
        result[dict(y=na.UncertainScalarArray(index, index))] = np.sort(nominal, axis="y")
        assert np.all(result == nominal)

    def test__setitem__uncertain_indices_no_samples(self, array: na.UncertainScalarArray):
        # Indices which differ between samples scatter each sample separately,
        # so a distribution without a sample axis receives one
        array = na.broadcast_to(array, na.broadcast_shapes(array.shape, dict(y=_num_y))).astype(float)
        item = dict(y=_index_varying())
        nominal = na.as_named_array(array.nominal)
        result = na.UncertainScalarArray(nominal.copy(), nominal.copy())
        result[item] = array
        assert array.axis_distribution in na.shape(result.distribution)
        assert np.all(result[item] == array)

    def test__setitem__uncertain_item_broadcasts(self, array: na.UncertainScalarArray):
        # A nominal value or a distribution which does not vary along the
        # axes of the item or of the value is broadcast to them first
        item = _item_varying()
        array = na.broadcast_to(array, na.broadcast_shapes(array.shape, item.shape)).astype(float)
        unit = na.unit(array)
        value = 0 if unit is None else 0 * unit

        result = na.UncertainScalarArray(np.mean(array.nominal), array.distribution.copy())
        result_expected = np.where(item, value, result)
        result[item] = value
        assert np.all(result == result_expected)

        item = na.UncertainScalarArray(item.nominal, item.nominal)
        result = na.UncertainScalarArray(array.nominal.copy(), np.mean(array.distribution, axis="y"))
        result_expected = np.where(item, value, result)
        result[item] = value
        assert np.all(result == result_expected)

    def test__setitem__uncertain_item_shared(self, array: na.UncertainScalarArray):
        # Assigning through an array which shares its nominal value and
        # distribution with another one never leaves the other one with an
        # updated nominal value next to a stale distribution, or vice versa
        item = _item_varying()
        array = na.broadcast_to(array, na.broadcast_shapes(array.shape, item.shape)).astype(float)
        nominal = na.as_named_array(array.nominal)
        unit = na.unit(array)
        value = 0 if unit is None else 0 * unit

        original = na.UncertainScalarArray(nominal.copy(), nominal.copy())
        copy = original.copy_shallow()
        copy[item] = value
        assert np.all(original == na.UncertainScalarArray(nominal, nominal))
        assert np.all(copy == np.where(item, value, original))

        original = array.copy()
        copy = original.copy_shallow()
        copy[item] = value
        assert np.all(original == copy)

    def test__setitem__argsort(self, array: na.UncertainScalarArray):
        # Assigning the sorted values of each sample through that sample's own
        # sorting indices puts every value back where it was
        shape = na.broadcast_shapes(array.shape, dict(y=_num_y))
        array = na.broadcast_to(array, shape).astype(float).copy()
        result = array.copy()
        result[np.argsort(result, axis="y")] = np.sort(result, axis="y")
        assert np.all(result == array)

    def test__setitem__uncertain_item_round_trip(self, array: na.UncertainScalarArray):
        # The NaN which `result[item]` puts in the elements a sample did not
        # select are never written back
        item = array > array.mean()
        result = na.broadcast_to(array, array.shape).astype(float).copy()
        result_expected = np.where(item, -result, result)
        result[item] = -result[item]
        assert np.all(result == result_expected)


@pytest.mark.parametrize("type_array", [na.UncertainScalarArray])
class TestUncertainScalarArrayCreation(
    named_arrays.tests.test_core.AbstractTestAbstractExplicitArrayCreation,
):

    @pytest.mark.parametrize("like", [None] + _uncertain_scalar_arrays())
    class TestFromScalarArray(
        named_arrays.tests.test_core.AbstractTestAbstractExplicitArrayCreation.TestFromScalarArray,
    ):
        pass


class AbstractTestAbstractImplicitUncertainScalarArray(
    AbstractTestAbstractUncertainScalarArray,
    named_arrays.tests.test_core.AbstractTestAbstractImplicitArray,
):
    pass


def _uniform_uncertain_scalar_arrays():
    arrays_exact = [
        4,
        na.ScalarUniformRandomSample(-4, 4, shape_random=dict(x=_num_x, y=_num_y)),
    ]
    widths = [
        1,
        na.ScalarLinearSpace(1, 2, axis='y', num=_num_y)
    ]
    units = [
        1,
    ]
    arrays = [
        na.UniformUncertainScalarArray(
            nominal=array_exact * unit,
            width=width * unit,
            num_distribution=_num_distribution
        )
        for array_exact in arrays_exact
        for width in widths
        for unit in units
    ]
    return arrays


@pytest.mark.parametrize('array', _uniform_uncertain_scalar_arrays())
class TestUniformUncertainScalarArray(
    AbstractTestAbstractImplicitUncertainScalarArray,
):
    pass


def _normal_uncertain_scalar_arrays():
    arrays_exact = [
        na.ScalarArray(4),
        na.ScalarUniformRandomSample(-4, 4, shape_random=dict(x=_num_x, y=_num_y)),
    ]
    widths = [
        1,
        na.ScalarLinearSpace(1, 2, axis='y', num=_num_y)
    ]
    units = [
        u.mm,
    ]
    arrays = [
        na.NormalUncertainScalarArray(
            nominal=array_exact * unit,
            width=width * unit,
            num_distribution=_num_distribution
        )
        for array_exact in arrays_exact
        for width in widths
        for unit in units
    ]
    return arrays


@pytest.mark.parametrize('array', _normal_uncertain_scalar_arrays())
class TestNormalUncertainScalarArray(
    AbstractTestAbstractImplicitUncertainScalarArray,
):
    pass


class AbstractTestAbstractUncertainScalarRandomSample(
    AbstractTestAbstractImplicitUncertainScalarArray,
    named_arrays.tests.test_core.AbstractTestAbstractRandomSample,
):
    pass


def _uncertain_scalar_uniform_random_samples() -> tuple[na.UncertainScalarUniformRandomSample, ...]:

    starts = (
        na.UniformUncertainScalarArray(
            na.ScalarLinearSpace(0, 1, axis='x', num=_num_x),
            width=0.2,
            num_distribution=_num_distribution,
        ),
    )
    stops = (
        2,
        na.UniformUncertainScalarArray(
            na.ScalarLinearSpace(2, 3, axis='x', num=_num_x),
            width=0.1,
            num_distribution=_num_distribution,
        ),
    )
    units = [
        u.mm,
    ]
    arrays = tuple(
        na.UncertainScalarUniformRandomSample(
            start=start * unit,
            stop=stop * unit,
            shape_random=dict(y=_num_y),
        )
        for start in starts
        for stop in stops
        for unit in units
    )
    return arrays


@pytest.mark.parametrize("array", _uncertain_scalar_uniform_random_samples())
class TestUncertainScalarUniformRandomSample(
    AbstractTestAbstractUncertainScalarRandomSample,
    named_arrays.tests.test_core.AbstractTestAbstractUniformRandomSample,
):
    pass


def _uncertain_scalar_normal_random_samples() -> tuple[na.UncertainScalarNormalRandomSample, ...]:

    centers = (
        na.UniformUncertainScalarArray(
            na.ScalarLinearSpace(0, 1, axis='x', num=_num_x),
            width=0.2,
            num_distribution=_num_distribution,
        ),
    )
    widths = (
        na.ScalarLinearSpace(2, 3, axis='x', num=_num_x),
        na.UniformUncertainScalarArray(
            na.ScalarLinearSpace(2, 3, axis='x', num=_num_x),
            width=0.1,
            num_distribution=_num_distribution,
        ),
    )
    units = [
        1,
    ]
    arrays = tuple(
        na.UncertainScalarNormalRandomSample(
            center=center * unit,
            width=width * unit,
            shape_random=dict(y=_num_y),
        )
        for center in centers
        for width in widths
        for unit in units
    )
    return arrays


@pytest.mark.parametrize("array", _uncertain_scalar_normal_random_samples())
class TestUncertainScalarNormalRandomSample(
    AbstractTestAbstractUncertainScalarRandomSample,
    named_arrays.tests.test_core.AbstractTestAbstractNormalRandomSample,
):
    pass


def _uncertain_scalar_poisson_random_samples() -> tuple[na.UncertainScalarPoissionRandomSample, ...]:

    centers = (
        na.UniformUncertainScalarArray(2, width=0.1, num_distribution=_num_distribution),
        na.UniformUncertainScalarArray(
            na.ScalarLinearSpace(2, 3, axis='x', num=_num_x),
            width=0.1,
            num_distribution=_num_distribution,
        ),
    )
    units = (1, u.mm)
    arrays = tuple(
        na.UncertainScalarPoissionRandomSample(
            center=center * unit,
            shape_random=dict(y=_num_y),
        )
        for center in centers
        for unit in units
    )
    return arrays


@pytest.mark.parametrize("array", _uncertain_scalar_poisson_random_samples())
class TestUncertainScalarPoissonRandomSample(
    AbstractTestAbstractUncertainScalarRandomSample,
    named_arrays.tests.test_core.AbstractTestAbstractPoissonRandomSample,
):
    pass


class AbstractTestAbstractParameterizedUncertainScalarArray(
    AbstractTestAbstractImplicitUncertainScalarArray,
    named_arrays.tests.test_core.AbstractTestAbstractParameterizedArray,
):
    pass


class AbstractTestAbstractScalarSpace(
    AbstractTestAbstractParameterizedUncertainScalarArray,
    named_arrays.tests.test_core.AbstractTestAbstractSpace,
):
    pass



def _uncertain_scalar_linear_spaces() -> tuple[na.UncertainScalarLinearSpace, ...]:
    start = na.UniformUncertainScalarArray(
        na.ScalarLinearSpace(0, 1, axis='x', num=_num_x),
        width=0.2,
        num_distribution=_num_distribution,
    )
    stop_uncertain = na.UniformUncertainScalarArray(
        na.ScalarLinearSpace(2, 3, axis='x', num=_num_x),
        width=0.1,
        num_distribution=_num_distribution,
    )
    return (
        na.UncertainScalarLinearSpace(start, 2, axis='y', num=_num_y),
        na.UncertainScalarLinearSpace(
            start=start * u.mm,
            stop=na.ScalarLinearSpace(2, 3, axis='x', num=_num_x) * u.mm,
            axis='y',
            num=_num_y,
        ),
        na.UncertainScalarLinearSpace(start, stop_uncertain, axis='y', num=_num_y),
        na.UncertainScalarLinearSpace(start * u.mm, stop_uncertain * u.mm, axis='y', num=_num_y),
    )


@pytest.mark.parametrize("array", _uncertain_scalar_linear_spaces())
class TestUncertainScalarLinearSpace(
    AbstractTestAbstractScalarSpace,
    named_arrays.tests.test_core.AbstractTestAbstractLinearSpace,
):
    @pytest.mark.parametrize(
        argnames="axis",
        argvalues=[
            None,
            "x",
            "y",
            "z",
        ]
    )
    def test_volume_cell(
            self,
            array: na.ScalarLinearSpace,
            axis: None | str | Sequence[str],
    ):
        super().test_volume_cell(array=array, axis=axis)

        axis_ = na.axis_normalized(array, axis)
        if len(axis_) != 1:
            with pytest.raises(ValueError):
                array.volume_cell(axis)
            return

        if not set(axis_).issubset(array.shape):
            with pytest.raises(ValueError):
                array.volume_cell(axis)
            return

        assert np.allclose(array.volume_cell(axis), array.explicit.volume_cell(axis))


def test_interp_axis_uncertain_xp():
    """
    The axis to interpolate along survives an uncertain ``xp``.

    Each of the nominal and the distribution is interpolated on its own, and
    the axis has to be carried into both. Without it the table is taken to
    have a single axis, and a table which has more than one raises.
    """
    axis = "wavelength"
    xp = na.linspace(0, 10, axis=axis, num=11)
    fp = 2 * xp
    x = na.linspace(0, 10, axis=axis, num=5)

    def uncertain(a):
        return na.NormalUncertainScalarArray(a, width=0.01)

    result = na.interp(x, uncertain(xp), uncertain(fp), axis=axis)

    assert na.shape(result) == {axis: 5}

    # the table is a line, so interpolating it gives the line back
    assert np.allclose(result.nominal, na.interp(x, xp, fp, axis=axis))


def test_interp_axis_uncertain_xp_extra_axis():
    """An uncertain table carrying a second axis needs the axis to be named."""
    axis = "wavelength"
    xp = na.linspace(0, 10, axis=axis, num=11) + na.linspace(0, 1, axis="channel", num=3)
    fp = 2 * xp
    x = na.linspace(0, 10, axis=axis, num=5)

    def uncertain(a):
        return na.NormalUncertainScalarArray(a, width=0.01)

    result = na.interp(x, uncertain(xp), uncertain(fp), axis=axis)

    assert na.shape(result) == {axis: 5, "channel": 3}


def _uncertain_coordinates(axis: str, num: int) -> na.UncertainScalarArray:
    """A coordinate whose distribution is ten times its nominal value."""
    nominal = na.ScalarArray(np.arange(num, dtype=float), axes=(axis,)) * u.nm
    return na.UncertainScalarArray(
        nominal=nominal,
        distribution=10 * nominal.add_axes("_distribution").broadcast_to(
            {"_distribution": _num_distribution, axis: num},
        ),
    )


def test_trapezoid_uncertain_coordinates():
    """
    The distribution is integrated against the distribution of the coordinates.

    A coordinate carries its own uncertainty, so the nominal value and each
    sample of the distribution are integrated over their own abscissa. Using
    the nominal coordinates for both would silently discard the uncertainty
    in the sample spacing.
    """
    axis = "x"
    num = 5

    x = _uncertain_coordinates(axis, num)
    y = na.UncertainScalarArray(
        nominal=na.ScalarArray(np.arange(num, dtype=float), axes=(axis,)) * u.ph,
        distribution=na.ScalarArray(np.arange(num, dtype=float), axes=(axis,)).add_axes(
            "_distribution"
        ).broadcast_to({"_distribution": _num_distribution, axis: num}) * u.ph,
    )

    result = np.trapezoid(y, x=x, axis=axis)

    # the nominal integrand and abscissa give the ordinary trapezoid sum
    assert np.allclose(result.nominal, 8 * u.nm * u.ph)

    # the distribution is spread over ten times the interval, so its integral
    # is ten times the nominal one rather than equal to it
    assert np.allclose(result.distribution, 80 * u.nm * u.ph)


def test_gradient_uncertain_coordinates():
    """
    The distribution is differentiated against the distribution of the coordinates.

    This is the counterpart to the trapezoid case: a coordinate with its own
    uncertainty gives a derivative whose distribution is divided by the
    spacing of that distribution.
    """
    axis = "x"
    num = 5

    x = _uncertain_coordinates(axis, num)
    y = na.UncertainScalarArray(
        nominal=na.ScalarArray(np.arange(num, dtype=float), axes=(axis,)) * u.ph,
        distribution=na.ScalarArray(np.arange(num, dtype=float), axes=(axis,)).add_axes(
            "_distribution"
        ).broadcast_to({"_distribution": _num_distribution, axis: num}) * u.ph,
    )

    result = np.gradient(y, x, axis=axis)

    # unit spacing in the nominal value, and ten times that in the
    # distribution, so its derivative is a tenth of the nominal one
    assert np.allclose(result.nominal, 1 * u.ph / u.nm)
    assert np.allclose(result.distribution, 0.1 * u.ph / u.nm)


def test_searchsorted_uncertain_grid():
    """
    Every sample of the distribution is searched in its own grid, so the index
    of one value can differ from sample to sample.

    The grid is shifted by an uncertain offset rather than jittered point by
    point, which keeps every sample sorted, as searching requires.
    """
    grid = na.linspace(0, 10, axis="w", num=11) * u.mm

    offset = na.NormalUncertainScalarArray(
        nominal=0 * u.mm,
        width=3 * u.mm,
        num_distribution=11,
        seed=7,
    )

    v = na.ScalarArray(
        ndarray=np.array([5.0]) * u.mm,
        axes=("line",),
    )

    result = na.searchsorted(grid + offset, v, axis="w")

    assert isinstance(result, na.AbstractUncertainScalarArray)

    # the nominal grid is the unshifted one
    assert np.all(result.nominal == na.searchsorted(grid, v, axis="w"))

    # and a shifted grid does not give the nominal answer for every sample
    assert np.any(result.distribution.ndarray != result.nominal.ndarray)

    # each sample counts the points of its own grid which fall below the value
    expected = np.sum((grid + offset) < v, axis="w")
    assert np.all(result == expected)


def test_searchsorted_uncertain_grid_and_vector_values():
    """
    An uncertain grid searched by vector values, which is the uncertain
    implementation declining the vector and the vector one taking it, with
    uncertain components.
    """
    grid = na.linspace(0, 10, axis="w", num=11) * u.mm

    grid = grid + na.NormalUncertainScalarArray(
        nominal=0 * u.mm,
        width=1 * u.mm,
        num_distribution=5,
        seed=3,
    )

    v = na.Cartesian2dVectorArray(
        x=na.ScalarArray(np.array([2.5]) * u.mm, axes=("line",)),
        y=na.ScalarArray(np.array([7.5]) * u.mm, axes=("line",)),
    )

    result = na.searchsorted(grid, v, axis="w")

    assert isinstance(result, na.AbstractCartesian2dVectorArray)
    assert isinstance(result.x, na.AbstractUncertainScalarArray)
    assert np.all(result.x == np.sum(grid < v.x, axis="w"))
    assert np.all(result.y == np.sum(grid < v.y, axis="w"))


def test_digitize_uncertain_bins_and_vector_values():
    """The same handover, for bins rather than a grid."""
    bins = na.linspace(0, 10, axis="w", num=11) * u.mm

    bins = bins + na.NormalUncertainScalarArray(
        nominal=0 * u.mm,
        width=1 * u.mm,
        num_distribution=5,
        seed=3,
    )

    x = na.Cartesian2dVectorArray(
        x=na.ScalarArray(np.array([2.5]) * u.mm, axes=("line",)),
        y=na.ScalarArray(np.array([7.5]) * u.mm, axes=("line",)),
    )

    result = na.digitize(x, bins, axis="w")

    assert isinstance(result.x, na.AbstractUncertainScalarArray)
    assert np.all(result.x == np.sum(bins <= x.x, axis="w"))
    assert np.all(result.y == np.sum(bins <= x.y, axis="w"))


def test_digitize_uncertain_values_and_vector_bins():
    """
    The handover the other way round: uncertain values binned by a vector of
    bins, where it is the uncertain implementation which declines.

    Which implementation is tried first follows the order of the arguments, so
    this is a different path through the dispatcher than
    :func:`test_digitize_uncertain_bins_and_vector_values` takes, not the same
    one written twice.
    """
    x = na.linspace(0, 10, axis="line", num=5) * u.mm

    x = x + na.NormalUncertainScalarArray(
        nominal=0 * u.mm,
        width=1 * u.mm,
        num_distribution=5,
        seed=4,
    )

    bins = na.Cartesian2dVectorArray(
        x=na.ScalarArray(np.array([0.0, 5.0, 10.0]) * u.mm, axes=("w",)),
        y=na.ScalarArray(np.array([0.0, 2.0, 4.0]) * u.mm, axes=("w",)),
    )

    result = na.digitize(x, bins, axis="w")

    assert isinstance(result, na.AbstractCartesian2dVectorArray)
    assert isinstance(result.x, na.AbstractUncertainScalarArray)
    assert np.all(result.x == np.sum(bins.x <= x, axis="w"))
    assert np.all(result.y == np.sum(bins.y <= x, axis="w"))


def _array_xy() -> na.UncertainScalarArray:
    """An uncertain array which varies along ``x``, ``y``, and its samples."""
    return na.UncertainScalarArray(
        nominal=na.ScalarUniformRandomSample(-4, 4, shape_random=dict(x=_num_x, y=_num_y)).explicit,
        distribution=na.ScalarUniformRandomSample(
            start=-4,
            stop=4,
            shape_random=dict(x=_num_x, y=_num_y, _distribution=_num_distribution),
        ).explicit,
    )


def test_vector_getitem_certain_integer_component():
    # The integer component of a vector keeps every element any sample
    # selected and stays certain, while the uncertain components are filled
    a = _array_xy()
    z = na.ScalarArray(np.arange(_num_x * _num_y).reshape(_num_x, _num_y), axes=("x", "y"))
    vector = na.Cartesian3dVectorArray(a, a, z)
    item = vector.x > 0
    union = item.nominal | np.any(item.distribution, axis=item.axis_distribution)
    result = vector[item]
    assert isinstance(result.z, na.ScalarArray)
    assert np.all(result.z == z[union])
    assert np.all((result.x == a[item]) | np.isnan(a[item]))


def test_function_getitem_certain_string_inputs():
    # The labels of a function's inputs are kept wherever any sample
    # selected the outputs
    a = _array_xy()
    inputs = na.ScalarArray(np.array([f"line {i}" for i in range(_num_x)]), axes="x")
    inputs = na.broadcast_to(inputs, a.shape)
    function = na.FunctionArray(inputs=inputs, outputs=a)
    item = function > 0
    union = item.outputs.nominal | np.any(item.outputs.distribution, axis=a.axis_distribution)
    result = function[item]
    assert isinstance(result.inputs, na.ScalarArray)
    assert np.all(result.inputs == inputs[union])


def test_vector_setitem_plain_component():
    # A component without a distribution which receives a selection that
    # differs between samples becomes uncertain, instead of the assignment
    # failing after the other components were already assigned
    a = _array_xy()
    nominal = na.as_named_array(a.nominal)
    vector = na.Cartesian2dVectorArray(a.copy(), nominal.copy())
    item = vector.x > 0
    vector[item] = na.Cartesian2dVectorArray(0, 0)
    assert isinstance(vector.y, na.UncertainScalarArray)
    assert np.all(vector.x == np.where(item, 0, a))
    assert np.all(vector.y == np.where(item, 0, nominal))


def test_function_setitem_plain_inputs_and_outputs():
    # Inputs and outputs without a distribution which receive a selection
    # that differs between samples become uncertain
    a = _array_xy()
    nominal = na.as_named_array(a.nominal)
    inputs = na.broadcast_to(na.ScalarLinearSpace(0, 1, axis="x", num=_num_x), a.shape).copy()
    function = na.FunctionArray(inputs=inputs.copy(), outputs=nominal.copy())
    item = na.FunctionArray(inputs=function.inputs, outputs=a > 0)
    function[item] = function[item]
    assert isinstance(function.inputs, na.UncertainScalarArray)
    assert isinstance(function.outputs, na.UncertainScalarArray)
    assert np.all(function.inputs == inputs)
    assert np.all(function.outputs == nominal)


def test_vector_setitem_uncertain_value():
    # A component without a distribution which receives an uncertain value
    # through a plain mask becomes uncertain, instead of the assignment
    # failing after the other components were already assigned
    a = _array_xy()
    nominal = na.as_named_array(a.nominal)
    vector = na.Cartesian2dVectorArray(a.copy(), nominal.copy())
    item = nominal > 0
    value = a[item] + 100
    vector[item] = na.Cartesian2dVectorArray(value, value)
    assert isinstance(vector.y, na.UncertainScalarArray)
    assert np.all(vector.x[item] == value)
    assert np.all(vector.y[item] == value)
    assert np.all(vector.y[~item] == nominal[~item])


@pytest.mark.parametrize("func", [np.all, np.any, np.sum])
def test_reduce_plain_uncertain_where(func: Callable):
    # Reducing an array without a distribution over an uncertain selection
    # reduces each sample over its own selection, which is how boolean
    # arrays should be reduced with `numpy.all()`
    a = _array_xy()
    b = na.as_named_array(a.nominal) > 0
    where = a > 0
    result = func(b, axis="x", where=where)
    assert isinstance(result, na.UncertainScalarArray)
    assert np.all(result == _reduce_per_sample(func, b, axis="x", where=where))


def _reduce_per_sample(
    func: Callable,
    a: na.ScalarArray,
    axis: str,
    where: na.UncertainScalarArray,
) -> na.UncertainScalarArray:
    """
    Reduce a plain array over the selection of the nominal value of an
    uncertain `where`, and over the selection of each of its samples, one at
    a time with a plain `where`.
    """
    axis_distribution = where.axis_distribution
    distribution = na.as_named_array(where.distribution)
    return na.UncertainScalarArray(
        nominal=func(a, axis=axis, where=na.as_named_array(where.nominal)),
        distribution=na.stack(
            arrays=[
                func(a, axis=axis, where=distribution[{axis_distribution: i}])
                for i in range(distribution.shape[axis_distribution])
            ],
            axis=axis_distribution,
        ),
    )


def test_reduce_one_sample_uncertain_where():
    # A distribution with a single sample is broadcast along the samples of
    # an uncertain `where`, like a distribution without a sample axis
    a = _array_xy()
    nominal = na.as_named_array(a.nominal)
    array = na.UncertainScalarArray(nominal, nominal.add_axes(a.axis_distribution))
    where = a > 0
    result = np.sum(array, axis="x", where=where)
    assert np.all(result == _reduce_per_sample(np.sum, nominal, axis="x", where=where))


def test_reduce_plain_agreeing_where():
    # An uncertain `where` whose samples all agree with its nominal value
    # leaves the reduction of an array without a distribution certain,
    # just like indexing with it
    a = _array_xy()
    nominal = na.as_named_array(a.nominal)
    mask = nominal > 0
    shape_distribution = {**mask.shape, a.axis_distribution: _num_distribution}
    where = na.UncertainScalarArray(mask, na.broadcast_to(mask, shape_distribution))
    result = np.sum(nominal, axis="x", where=where)
    assert isinstance(result, na.ScalarArray)
    assert np.all(result == np.sum(nominal, axis="x", where=mask))


def test_reduce_plain_uncertain_where_out():
    # A plain `out` cannot hold a result which differs between samples
    a = _array_xy()
    nominal = na.as_named_array(a.nominal)
    out = na.ScalarArray(np.zeros(_num_y), axes="y")
    with pytest.raises(ValueError, match="`out` must be an instance of `UncertainScalarArray`"):
        np.sum(nominal, axis="x", where=a > 0, out=out)


def test_getitem_plain_uncertain_indices_no_samples():
    # Indices which differ from their nominal value but have no sample axis
    # give a result without a sample axis
    array = na.ScalarArray(np.array([10., 20, 30, 40]), axes="t")
    index = na.UncertainScalarArray(
        nominal=na.ScalarArray(np.array([0, 1, 2, 3]), axes="t"),
        distribution=na.ScalarArray(np.array([3, 2, 1, 0]), axes="t"),
    )
    result = array[dict(t=index)]
    assert na.shape(result.distribution) == dict(t=4)
    assert np.all(result.nominal == array)
    assert np.all(result.distribution.ndarray == [40, 30, 20, 10])


def test_getitem_plain_agreeing_indices():
    # Uncertain indices whose samples all agree with their nominal value
    # leave an array without a distribution certain, so the selection can be
    # assigned back through them
    a = _array_xy()
    array = na.as_named_array(a.nominal).copy()
    index_nominal = np.argsort(array, axis="x")["x"]
    shape_distribution = {**index_nominal.shape, a.axis_distribution: _num_distribution}
    index = na.UncertainScalarArray(index_nominal, na.broadcast_to(index_nominal, shape_distribution))
    result = array[dict(x=index)]
    assert isinstance(result, na.ScalarArray)
    assert np.all(result == array[dict(x=index_nominal)])
    expected = array.copy()
    array[dict(x=index)] = result
    assert np.all(array == expected)


def test_vector_setitem_shared_component():
    # Components which are the same plain array are replaced by separate
    # uncertain copies, and the plain array itself is never written to
    x = na.ScalarArray(np.array([1., 2, 3]), axes="t")
    vector = na.Cartesian2dVectorArray(x, x)
    vector[dict(t=0)] = na.Cartesian2dVectorArray(
        x=na.UncertainScalarArray(5., 6.),
        y=na.UncertainScalarArray(7., 8.),
    )
    assert np.all(x.ndarray == [1, 2, 3])
    assert np.all(vector.x[dict(t=0)] == na.UncertainScalarArray(5., 6.))
    assert np.all(vector.y[dict(t=0)] == na.UncertainScalarArray(7., 8.))


def test_vector_setitem_failure_leaves_component():
    # An assignment which fails leaves a component which would have been
    # replaced by an uncertain copy untouched
    x = na.ScalarArray(np.array([1., 2, 3]), axes="t")
    vector = na.Cartesian2dVectorArray(x, 0)
    with pytest.raises(TypeError):
        vector[dict(t=0)] = na.UncertainScalarArray(5., 6.)
    assert vector.x is x
    assert np.all(x.ndarray == [1, 2, 3])


def test_function_setitem_failure():
    # An assignment which fails leaves the function as it was, even if it
    # would have replaced the outputs with an uncertain copy
    a = _array_xy()
    nominal = na.as_named_array(a.nominal)
    inputs = na.ScalarArray(np.array([f"line {i}" for i in range(_num_x)]), axes="x")
    outputs = nominal.copy()
    function = na.FunctionArray(inputs=inputs, outputs=outputs)
    item = na.FunctionArray(inputs=inputs, outputs=a > 0)
    value = na.FunctionArray(inputs=inputs[dict(x=slice(0, 1))], outputs=0)
    with pytest.raises(ValueError):
        function[item] = value
    assert function.inputs is inputs
    assert function.outputs is outputs
    assert np.all(function.outputs == nominal)
