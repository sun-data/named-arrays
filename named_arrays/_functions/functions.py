from __future__ import annotations
import functools
from typing import Mapping, TYPE_CHECKING, TypeVar, Generic, Type, ClassVar, Sequence, Callable, Collection, Any, Literal
from typing import Self
import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import named_arrays._core_array_functions as _core_array_functions
from named_arrays._functions import _fields
import itertools

if TYPE_CHECKING:
    import ndoptimize

__all__ = [
    "InputValueError",
    "AbstractFunctionArray",
    "FunctionArray",
    "AbstractPolynomialFunctionArray",
    "PolynomialFitFunctionArray",
    "AbstractModelFunctionArray",
    "ModelFitFunctionArray",
]

InputsT = TypeVar("InputsT", bound=na.AbstractArray, covariant=True)
OutputsT = TypeVar("OutputsT", bound=na.AbstractArray, covariant=True)


class InputValueError(ValueError):
    """
    Exception raised when the inputs of two functions do not match
    """


def _is_vertex(
    axis: str,
    shape_inputs: dict[str, int],
    shape_outputs: dict[str, int],
) -> bool:
    """
    Whether an axis of a function array is a vertex axis, along which the
    inputs are the edges of the cells of the outputs, rather than a center
    axis.

    Parameters
    ----------
    axis
        The name of the axis.
    shape_inputs
        The shape of the inputs of the function array.
    shape_outputs
        The shape of the outputs of the function array.
    """
    if axis in shape_inputs and axis in shape_outputs:
        num_inputs = shape_inputs[axis]
        num_outputs = shape_outputs[axis]
        if num_inputs == num_outputs + 1:
            return True
        if num_inputs != num_outputs and 1 not in (num_inputs, num_outputs):
            raise ValueError(
                f"output dimension, "
                f"self.outputs.shape[{axis}]={num_outputs},"
                f"must either match input dimension,"
                f"self.inputs.shape[{axis}]={num_inputs},"
                f"(representing bin centers) "
                "or exceed by one (representing bin vertices)."
            )
    return False


def _item_vertex(
    item: int | np.integer | slice,
    num: int,
    axis: str,
) -> tuple[slice, slice]:
    """
    Convert an index of the cells along a vertex axis of a function array
    into a slice of its outputs and a slice of its inputs.

    Parameters
    ----------
    item
        An integer or a slice of the cells along `axis`.
        Negative values count from the end, as they do for :mod:`numpy`.
        An integer selects one cell and keeps the axis, since the cell
        still has two vertices.
    num
        The number of cells along `axis`.
    axis
        The name of the vertex axis, for the error messages.

    Returns
    -------
    The slice of the outputs, the cells, and the slice of the inputs,
    the vertices which bound those cells.
    """
    if isinstance(item, slice):
        cells = range(num)[item]
        if len(cells) <= 1:
            # The step of a selection of at most one cell does not change
            # which cells it selects, only their direction.
            step = 1 if cells.step > 0 else -1
            cells = range(cells.start, cells.start + step * len(cells), step)
        elif cells.step not in (1, -1):
            raise ValueError(
                f"A slice along the vertex axis {axis!r} must have a step of "
                f"1 or -1, got {item}, since cells which are not adjacent "
                f"do not share vertices."
            )
    else:
        try:
            index = range(num)[item]
        except IndexError:
            raise IndexError(
                f"index {item} is out of bounds for the vertex axis {axis!r} "
                f"with {num} cells"
            ) from None
        cells = range(index, index + 1)

    # A selection of no cells keeps the vertex where it starts,
    # so that the axis stays a vertex axis.
    if cells.step == 1:
        start = cells.start
        stop = start + len(cells)
        return slice(start, stop), slice(start, stop + 1)
    else:
        # The cells `start, start - 1, ..., stop + 1` are bounded by the
        # vertices `start + 1, start, ..., stop + 1`.
        # A stop of -1 is written as `None`, since -1 would count from the end.
        start = cells.start
        stop = start - len(cells)
        stop = stop if stop >= 0 else None
        outputs = slice(start, stop, -1) if cells else slice(0, 0)
        return outputs, slice(start + 1, stop, -1)


def _equal(
    a: na.ArrayLike,
    b: na.ArrayLike,
) -> bool:
    """
    Whether two arrays are equal everywhere, after broadcasting them against
    each other, counting NaN as equal to NaN.

    Parameters
    ----------
    a
        The first array.
    b
        The second array.
    """
    if a is b:
        return True
    equal = a == b
    if np.all(equal):
        return True
    try:
        equal = equal | (np.isnan(a) & np.isnan(b))
    except TypeError:  # a type which has no NaN
        return False
    return bool(np.all(equal))


def _regrid(
    coordinates_input: na.AbstractScalar | na.AbstractVectorArray,
    coordinates_output: na.AbstractScalar | na.AbstractVectorArray,
    values_input: na.AbstractScalar | na.AbstractVectorArray,
    axis_input: tuple[str, ...],
    axis_output: tuple[str, ...],
    method: Literal['multilinear', 'conservative'],
    bounds: Literal["extrapolate", "nan", "raise"],
) -> na.AbstractArray:
    """
    Resample values from coordinates which are not uncertain to new ones
    using :func:`named_arrays.regridding.regrid`, as :meth:`FunctionArray.regrid`
    does, which calls this function once for the nominal values of uncertain
    coordinates and once for their samples.

    The ``multilinear`` method interpolates at points, so an axis which only
    `coordinates_output` varies along, like the one of the samples of
    uncertain points on a grid which is not uncertain, lists more points,
    rather than a batch of them which the values would be copied along.

    Parameters
    ----------
    coordinates_input
        The coordinates of the values.
    coordinates_output
        The new coordinates.
    values_input
        The values to resample.
    axis_input
        The axes of `coordinates_input` to resample.
    axis_output
        The axes of `coordinates_output` which take their place.
    method
        The resampling method to use.
    bounds
        How to treat new coordinates outside of `coordinates_input`.
    """
    if method == "multilinear":
        shape_input = na.shape_broadcasted(coordinates_input, values_input)
        axis_output = axis_output + tuple(
            a for a in na.shape(coordinates_output)
            if a not in axis_output and a not in shape_input
        )
    return na.regridding.regrid(
        coordinates_input=coordinates_input,
        coordinates_output=coordinates_output,
        values_input=values_input,
        axis_input=axis_input,
        axis_output=axis_output,
        method=method,
        bounds=bounds,
    )


@functools.cache
def _array_function_dispatch() -> tuple[dict[Callable, Callable], dict[Callable, Callable]]:
    """
    The handlers of the :mod:`numpy` functions which function arrays support,
    built on first use since :mod:`named_arrays._functions.function_array_functions`
    imports this module.

    Returns
    -------
    A dictionary mapping each function in a category of
    :mod:`~named_arrays._functions.function_array_functions` to the handler of
    that category, and the ``HANDLED_FUNCTIONS`` of that module.
    """
    from named_arrays._core import _array_function_handlers
    from . import function_array_functions as f
    handlers = _array_function_handlers([
        (f.DEFAULT_FUNCTIONS, f.array_function_default),
        (f.CUMULATIVE_REDUCE_FUNCTIONS, f.array_function_cumulative_reduce),
        (f.PERCENTILE_LIKE_FUNCTIONS, f.array_function_percentile_like),
        (f.ARG_REDUCE_FUNCTIONS, f.array_function_arg_reduce),
        (f.STACK_LIKE_FUNCTIONS, f.array_function_stack_like),
    ])
    return handlers, f.HANDLED_FUNCTIONS


@dataclasses.dataclass(eq=False, repr=False)
class AbstractFunctionArray(
    na.AbstractArray,
):

    @property
    @abc.abstractmethod
    def inputs(self) -> na.AbstractArray:
        """
        The arrays representing the inputs to this function.
        """

    @property
    @abc.abstractmethod
    def outputs(self) -> na.AbstractArray:
        """
        The arrays representing the outputs of this function.
        """

    __named_array_priority__: ClassVar[float] = (100 * na.AbstractVectorArray.__named_array_priority__)

    @property
    def axes_center(self) -> tuple(str):
        """
        Return keys corresponding to all input axes representing bin centers
        """
        shape_inputs = self.inputs.shape
        shape_outputs = self.outputs.shape
        return tuple(
            axis for axis in self.axes
            if not _is_vertex(axis, shape_inputs, shape_outputs)
        )

    @property
    def axes_vertex(self) -> tuple(str):
        """
        Return keys corresponding to all input axes representing bin vertices
        """
        axes_center = self.axes_center
        return tuple(axis for axis in self.axes if axis not in axes_center)

    @property
    def type_abstract(self) -> Type[AbstractFunctionArray]:
        return AbstractFunctionArray

    @property
    def type_explicit(self) -> Type[FunctionArray]:
        return type(self)

    @property
    def value(self) -> FunctionArray:
        exp = self.explicit
        return exp.replace(
            outputs=self.outputs.value,
        )

    @property
    def broadcasted(self) -> FunctionArray:

        exp = self.explicit

        broadcasted_shape_outputs = exp.shape
        broadcasted_shape_inputs = broadcasted_shape_outputs.copy()
        
        axes_vertex = exp.axes_vertex
        inputs = exp.inputs
        for key in broadcasted_shape_outputs:
            axes_vertex = axes_vertex
            if key in axes_vertex:  
                broadcasted_shape_inputs[key] = inputs.shape[key]

        return exp.replace(
            inputs=exp.inputs.broadcast_to(broadcasted_shape_inputs),
            outputs=exp.outputs.broadcast_to(broadcasted_shape_outputs),
        )

    def astype(
            self,
            dtype: str | np.dtype | type,
            order: str = 'K',
            casting='unsafe',
            subok: bool = True,
            copy: bool = True,
    ) -> FunctionArray:

        exp = self.explicit

        return exp.replace(
            inputs=exp.inputs,
            outputs=exp.outputs.astype(
                dtype=dtype,
                order=order,
                casting=casting,
                subok=subok,
                copy=copy,
            )
        )

    def to(
        self,
        unit: u.UnitBase,
        equivalencies: None | list[tuple[u.Unit, u.Unit]] = [],
        copy: bool = True,
    ) -> FunctionArray:
        exp = self.explicit
        return exp.replace(
            outputs=exp.outputs.to(
                unit=unit,
                equivalencies=equivalencies,
                copy=copy,
            ),
        )

    def to_value(
        self,
        unit: u.UnitBase,
        equivalencies: None | list[tuple[u.Unit, u.Unit]] = [],
    ) -> FunctionArray:
        exp = self.explicit
        return exp.replace(
            outputs=exp.outputs.to_value(
                unit=unit,
                equivalencies=equivalencies,
            ),
        )

    @property
    def length(self) -> FunctionArray:
        exp = self.explicit
        return exp.replace(
            outputs=exp.outputs.length,
        )

    def add_axes(self, axes: str | Sequence[str]) -> FunctionArray:
        exp = self.explicit
        return exp.replace(
            outputs=exp.outputs.add_axes(axes),
        )

    def combine_axes(
            self,
            axes: None | Sequence[str] = None,
            axis_new: None | str = None,
    ) -> FunctionArray:

        self = self.explicit

        axes = self.axes if axes is None else axes

        if not set(axes).issubset(set(self.axes)):
            raise ValueError(f"Axes {set(axes) - set(self.axes)} not in {self.axes}")

        for axis in axes:
            if axis in self.axes_vertex:
                raise ValueError(f"Axis {axis} describes input vertices and cannot be used in combine_axes.")

        inputs = self.inputs
        outputs = self.outputs
        inputs_shape_broadcasted = inputs.shape
        outputs_shape_broadcasted = outputs.shape

        shape = self.shape
        for axis in axes:
            if axis not in inputs_shape_broadcasted:
                inputs_shape_broadcasted[axis] = shape[axis]
            if axis not in outputs_shape_broadcasted:
                outputs_shape_broadcasted[axis] = shape[axis]

        inputs = na.broadcast_to(inputs, inputs_shape_broadcasted)
        outputs = na.broadcast_to(outputs, outputs_shape_broadcasted)

        def combine(v: na.AbstractArray) -> na.AbstractArray:
            v = na.broadcast_to(v, v.shape | {ax: shape[ax] for ax in axes})
            return v.combine_axes(axes=axes, axis_new=axis_new)

        return self.replace(
            inputs=inputs.combine_axes(axes=axes, axis_new=axis_new),
            outputs=outputs.combine_axes(axes=axes, axis_new=axis_new),
            **_fields.apply(self, combine, axes=axes),
        )

    def __call__(
        self,
        inputs: float | u.Quantity | na.AbstractArray,
        axis: None | str | Sequence[str] = None,
        method: Literal['multilinear', 'conservative'] = 'multilinear',
        bounds: Literal["extrapolate", "nan", "raise"] = "extrapolate",
        weights: None | tuple[na.AbstractScalar, dict[str, int], dict[str, int]] = None,
    ) -> Self:
        """
        Evaluate this function at a new set of inputs using :meth:`regrid`.

        Parameters
        ----------
        inputs
            The new inputs at which to evaluate this function.
        axis
            The axes of :attr:`inputs` to resample.
        method
            The resampling method to use.
        bounds
            How to treat new inputs outside of :attr:`inputs`.
        weights
            Optional weights computed in advance using :meth:`weights`.
        """
        return self.regrid(
            inputs=inputs,
            axis=axis,
            method=method,
            bounds=bounds,
            weights=weights,
        )

    def regrid(
        self,
        inputs: float | u.Quantity | na.AbstractArray,
        axis: None | str | Sequence[str] = None,
        method: Literal['multilinear', 'conservative'] = 'multilinear',
        bounds: Literal["extrapolate", "nan", "raise"] = "extrapolate",
        weights: None | tuple[na.AbstractScalar, dict[str, int], dict[str, int]] = None,
    ) -> Self:
        """
        Resample this function array onto a new set of input coordinates
        using :func:`named_arrays.regridding.regrid`.

        The new inputs can be another grid, or scattered points.
        The axes of this function which are not in `axis` are broadcast
        against the new inputs by name, as in any other operation, so the new
        inputs can share them, for example to give a separate set of points
        for each of several channels.
        Every other axis of the new inputs is an axis of the new grid, or an
        axis along which the points are listed, and takes the place of the
        axes in `axis` in the result.

        If the outputs of this function are uncertain, or if the components
        which are resampled are uncertain, either in its inputs or in the new
        inputs, the nominal values and the samples are resampled separately,
        and the outputs of the result are uncertain.

        Parameters
        ----------
        inputs
            The new input coordinates on which to resample the outputs.
            If :attr:`inputs` is a scalar, a single point can also be a
            plain number or quantity.
            A component of a vector which is not resampled should be
            :obj:`None`, or else equal everywhere to the component of
            :attr:`inputs`, counting NaN as equal to NaN, and the result
            keeps the component of :attr:`inputs`.
        axis
            The axes of :attr:`inputs` to resample.
            If :obj:`None`, every axis of :attr:`inputs` is resampled.
            Only the components of :attr:`inputs` which vary along at least
            one of these axes are resampled, and there should be as many of
            them as there are axes, so an axis along which :attr:`inputs` is
            a separate grid for each element should be left out.
        method
            The resampling method to use.
            The ``multilinear`` method interpolates at points, so the new
            inputs can be scattered points along any number of axes, or a
            single point.
            The ``conservative`` method resamples cells, so the new inputs
            must be a grid with as many axes as `axis`.
        bounds
            How to treat new inputs outside of :attr:`inputs`.
            Only applies when `method` is ``multilinear``.
            ``extrapolate`` (the default) extrapolates linearly from the
            nearest cell, ``nan`` sets the outputs there to :obj:`numpy.nan`,
            and ``raise`` raises a :class:`ValueError`.
            If the components which are resampled are uncertain, each of
            their samples is treated on its own, so ``nan`` sets only the
            samples outside to :obj:`numpy.nan`, and ``raise`` raises if any
            sample is outside.
            See :func:`regridding.weights` for the details.
        weights
            Optional weights which can be computed in advance using :meth:`weights`
            to greatly speed repeated resampling of the same `inputs`.
            If given, `method` and `bounds` are ignored, since the weights
            already encode them.
            They cannot be used if the inputs which are resampled are
            uncertain, since each of their samples has weights of its own,
            but they can be used if only the outputs are uncertain.

        See Also
        --------
        :meth:`weights`: If you need to resample the same coordinates more than once.

        Examples
        --------

        Evaluate a function sampled on a grid at a set of scattered points.

        .. jupyter-execute::

            import numpy as np
            import matplotlib.pyplot as plt
            import named_arrays as na

            # Sample a function on a grid
            inputs = na.Cartesian2dVectorLinearSpace(
                start=-1,
                stop=1,
                axis=na.Cartesian2dVectorArray("x", "y"),
                num=21,
            ).explicit
            outputs = np.sin(3 * inputs.x) * np.cos(2 * inputs.y)
            f = na.FunctionArray(inputs, outputs)

            # Scatter some points along a single axis
            points = na.Cartesian2dVectorArray(
                x=na.random.uniform(-1, 1, shape_random=dict(point=50), seed=1),
                y=na.random.uniform(-1, 1, shape_random=dict(point=50), seed=2),
            )

            # Evaluate the function at the points
            g = f(points, axis=("x", "y"))

            # Plot the samples of the function, and the points
            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.scatter(
                f.inputs.x,
                f.inputs.y,
                c=f.outputs,
                s=10,
                ax=ax,
                vmin=-1,
                vmax=1,
            )
            na.plt.scatter(
                g.inputs.x,
                g.inputs.y,
                c=g.outputs,
                s=60,
                ax=ax,
                vmin=-1,
                vmax=1,
                edgecolors="black",
            )
            ax.set_aspect("equal")
        """

        from named_arrays._scalars.uncertainties import uncertainties

        _self = self.explicit

        inputs_new, inputs_old, axis_input, axis_output = _self._normalize__regrid__args(
            inputs=inputs,
            axis=axis,
        )

        _fields.check_axes(
            _self,
            axis_input,
            operation="regrid resamples",
            resamples=True,
        )

        uncertain = uncertainties._is_uncertain(inputs_old) or uncertainties._is_uncertain(inputs_new)

        if uncertain:
            if weights is not None:
                raise ValueError(
                    "Weights computed in advance cannot be used to resample "
                    "uncertain inputs, since each of their samples has weights "
                    "of its own."
                )

            # The nominal values and the samples of the inputs have weights
            # of their own, so they are resampled separately, and each of
            # them should vary along every axis of the new inputs, even one
            # which only the other varies along.
            outputs_new = uncertainties._propagate(
                _regrid,
                arguments=dict(
                    coordinates_input=inputs_old,
                    coordinates_output=na.broadcast_to(inputs_new, na.shape(inputs_new)),
                    values_input=_self.outputs,
                ),
                axis_input=axis_input,
                axis_output=axis_output,
                method=method,
                bounds=bounds,
            )

        else:
            if weights is None:
                weights = na.regridding.weights(
                    coordinates_input=inputs_old,
                    coordinates_output=inputs_new,
                    axis_input=axis_input,
                    axis_output=axis_output,
                    method=method,
                    bounds=bounds,
                )

            _weights, shape_input, shape_output = weights

            # The weights do not depend on the outputs, so the nominal values
            # and the samples of uncertain outputs share them.
            outputs_new = uncertainties._propagate(
                na.regridding.regrid_from_weights,
                arguments=dict(values_input=_self.outputs),
                weights=_weights,
                shape_input=shape_input,
                shape_output=shape_output,
            )

        if isinstance(inputs, na.AbstractVectorArray) and isinstance(_self.inputs, na.AbstractVectorArray):

            # A component which is not resampled is equal to the one of this
            # function, so the result keeps the one of this function.
            components = _self.inputs.cartesian_nd.components | inputs_new.components

            return _self.replace(
                inputs=_self.inputs.type_explicit.from_cartesian_nd(
                    array=na.CartesianNdVectorArray(components),
                    like=_self.inputs
                ),
                outputs=outputs_new,
            )
        else:
            return _self.replace(
                inputs=inputs_new,
                outputs=outputs_new,
            )

    def weights(
        self,
        inputs: float | u.Quantity | na.AbstractArray,
        axis: None | str | Sequence[str] = None,
        method: Literal['multilinear', 'conservative'] = 'multilinear',
        bounds: Literal["extrapolate", "nan", "raise"] = "extrapolate",
    ) -> tuple[na.AbstractScalar, dict[str, int], dict[str, int]]:
        """
        Compute the resampling weights of this array using
        :func:`named_arrays.regridding.weights`.
        The output of this method is designed to be used by :meth:`regrid`.

        The inputs which are resampled cannot be uncertain, since each of
        their samples has weights of its own, but the outputs can be.

        Parameters
        ----------
        inputs
            The new input coordinates on which to resample the outputs.
        axis
            The axes of :attr:`inputs` to resample, as in :meth:`regrid`.
        method
            The resampling method to use.
        bounds
            How to treat new inputs outside of :attr:`inputs`,
            as in :meth:`regrid`.

        See Also
        --------
        :meth:`regrid`: A method designed to use these weights.
        """
        from named_arrays._scalars.uncertainties import uncertainties

        coordinates_new, coordinates_old, axis_input, axis_output = self._normalize__regrid__args(
            inputs=inputs,
            axis=axis,
        )

        if uncertainties._is_uncertain(coordinates_old) or uncertainties._is_uncertain(coordinates_new):
            raise ValueError(
                "The weights cannot be computed in advance for uncertain "
                "inputs, since each of their samples has weights of its own. "
                "`regrid()` resamples them without weights computed in advance."
            )

        return na.regridding.weights(
            coordinates_input=coordinates_old,
            coordinates_output=coordinates_new,
            axis_input=axis_input,
            axis_output=axis_output,
            method=method,
            bounds=bounds,
        )

    def _normalize__regrid__args(
        self,
        inputs: float | u.Quantity | na.AbstractArray,
        axis: None | str | Sequence[str],
    ) -> tuple[na.AbstractArray, na.AbstractArray, tuple[str, ...], tuple[str, ...]]:
        """
        Find the components of the inputs to resample, and the axes to
        resample them along.

        Parameters
        ----------
        inputs
            The new inputs, as given to :meth:`regrid`.
        axis
            The axes of :attr:`inputs` to resample, as given to :meth:`regrid`.

        Returns
        -------
        The components of the new inputs to resample, the same components of
        :attr:`inputs`, the axes of :attr:`inputs` to resample, and the axes
        of the new inputs which take their place.
        """

        inputs_old = self.inputs
        inputs_new = inputs

        if isinstance(inputs_new, na.AbstractVectorArray):
            new_input_components = inputs_new.cartesian_nd.components
        else:
            new_input_components = dict(_dummy=na.as_named_array(inputs_new).explicit)

        if isinstance(inputs_old, na.AbstractVectorArray):
            old_input_components = inputs_old.cartesian_nd.components
        else:
            old_input_components = dict(_dummy=inputs_old.explicit)

        # check physical(vector) dimensions of each input match
        if new_input_components.keys() != old_input_components.keys():
            raise ValueError('Physical axes of new and old inputs must match.')  # pragma: no cover

        shape_old = na.shape_broadcasted(*old_input_components.values())

        if axis is None:
            axis = tuple(shape_old)
        elif isinstance(axis, str):
            axis = (axis,)
        else:
            axis = tuple(axis)

        if not set(axis).issubset(shape_old):
            raise ValueError(
                f"The inputs of this function do not vary along "
                f"{tuple(a for a in axis if a not in shape_old)}, "
                f"so they cannot be resampled along `axis`, {axis}."
            )

        coordinates_new = {}
        coordinates_old = {}
        for c in old_input_components:
            component_new = new_input_components[c]
            component_old = old_input_components[c]
            # a component which does not vary along the resampled axes
            # is not resampled
            if not set(axis).isdisjoint(na.shape(component_old)):
                if component_new is None:
                    raise ValueError(
                        f"If a component is marked separable using `None`, its shape, "
                        f"{na.shape(component_old)}, should be disjoint from the "
                        f"interpolated axes, {axis}.",
                    )
                coordinates_new[c] = component_new
                coordinates_old[c] = component_old

        if len(coordinates_old) != len(axis):
            raise ValueError(
                f"The number of components of the inputs of this function "
                f"which vary along `axis`, {len(coordinates_old)}, should be "
                f"equal to the number of axes in `axis`, {axis}. `axis` "
                f"should name only the axes to interpolate along, and any "
                f"other axis is broadcast against the new inputs."
            )

        # The axes of this function which are not resampled are broadcast
        # against the new inputs by name, and every other axis of the new
        # inputs takes the place of the resampled axes.
        axis_orthogonal = set(self.axes) - set(axis)
        axis_new = tuple(
            a for a in na.shape_broadcasted(*coordinates_new.values())
            if a not in axis_orthogonal
        )

        # A component which is not resampled must be equal to the one of
        # this function, since the outputs are not resampled along it.
        for c in new_input_components:
            component_new = new_input_components[c]
            if c in coordinates_new or component_new is None:
                continue
            message = (
                f"The component {c!r} of the new inputs is not resampled, "
                f"since in the inputs of this function it does not vary "
                f"along `axis`, {axis}, so it should be `None` or equal to "
                f"the same component of the inputs of this function."
            )
            try:
                equal = _equal(component_new, old_input_components[c])
            except ValueError as e:  # incompatible units or shapes
                raise ValueError(message) from e
            if not equal:
                axes_varying = tuple(a for a in na.shape(component_new) if a in axis_new)
                if axes_varying:
                    raise ValueError(
                        f"The component {c!r} of the new inputs varies along "
                        f"{axes_varying}, but it is not resampled, since in "
                        f"the inputs of this function it does not vary along "
                        f"`axis`, {axis}. To resample it too, add the axes it "
                        f"varies along in the inputs of this function to `axis`."
                    )
                raise ValueError(message)

        if isinstance(inputs_new, na.AbstractVectorArray):
            coordinates_new = na.CartesianNdVectorArray.from_components(coordinates_new)
        else:
            coordinates_new = coordinates_new['_dummy']

        if isinstance(inputs_old, na.AbstractVectorArray):
            coordinates_old = na.CartesianNdVectorArray.from_components(coordinates_old)
        else:
            coordinates_old = coordinates_old['_dummy']

        return coordinates_new, coordinates_old, axis, axis_new

    def cell_centers(
        self,
        axis: None | str | Sequence[str] = None,
        random: bool = False,
        seed: None | int = None,
    ) -> na.AbstractExplicitArray:
        exp = self.explicit

        if axis is None:
            axes = tuple(na.shape(exp.outputs))
        elif isinstance(axis, str):
            axes = (axis,)
        else:
            axes = tuple(axis)

        # the fields are not sampled at the same random points as the outputs,
        # and the centers of a mask or a count would be fractions
        if random:
            _fields.check_axes(
                exp,
                axes,
                operation="cell_centers with random=True samples at random points",
                resamples=True,
                hint=" Or use random=False.",
            )
        _fields.check_inexact(exp, axes, operation="cell_centers")

        def centers(v: na.AbstractArray) -> na.AbstractArray:
            return v.cell_centers(axis=tuple(ax for ax in axes if ax in v.shape))

        return exp.replace(
            inputs=exp.inputs.cell_centers(axis, random=random, seed=seed),
            outputs=exp.outputs.cell_centers(axis, random=random, seed=seed),
            **_fields.apply(exp, centers, axes=axes),
        )

    def integrate(
        self,
        axis: str | Sequence[str],
        component: None | str = None,
    ) -> FunctionArray:
        """
        Integrate the outputs over the given input ``axis`` or axes.

        The differential measure is computed from the inputs via
        :meth:`AbstractArray.volume_cell`:

        * If ``component`` is :obj:`None`, the whole input is used as the
          integration variable and ``len(axis)`` must match the dimensionality
          of the inputs (one for a scalar, two for a
          :class:`Cartesian2dVectorArray`, etc.).
        * If ``component`` is a :class:`str`, that named sub-element of the
          inputs (a scalar or a sub-vector) supplies the measure, and
          ``len(axis)`` must match its dimensionality.

        A vertex axis (where the inputs represent bin edges) is integrated with
        a Riemann sum, while a center axis (where the inputs represent samples)
        is integrated with the trapezoidal rule; mixing the two in a single
        call is supported. The integrated axes are removed from the outputs and
        collapsed in the inputs by averaging.

        Parameters
        ----------
        axis
            The logical axis or axes to integrate over.
        component
            The named input sub-element supplying the integration variable.
            If :obj:`None`, the whole input is used.

        Examples
        --------

        Integrate a constant function over a 1D domain.

        .. jupyter-execute::

            import numpy as np
            import astropy.units as u
            import named_arrays as na

            f = na.FunctionArray(
                inputs=na.ScalarLinearSpace(0, 2, axis="x", num=101) * u.nm,
                outputs=na.ScalarArray(np.full(101, 3.0), axes=("x",)) * u.ph,
            )

            f.integrate("x")
        """

        self = self.explicit

        if isinstance(axis, str):
            axis = (axis,)
        axis = tuple(axis)

        if not set(axis).issubset(self.axes):
            raise ValueError(f"axes {set(axis) - set(self.axes)} not in {self.axes}")

        _fields.check_axes(self, axis, operation="integration removes")

        axes_vertex = self.axes_vertex
        inputs = self.inputs
        outputs = self.outputs

        # convert outputs to cell centers along the center axes
        # (corner-averaging across all center axes is the multi-D trapezoidal rule);
        # vertex axes are already at cell centers relative to their input edges.
        axes_center = tuple(ax for ax in axis if ax not in axes_vertex)
        if axes_center:
            outputs = outputs.cell_centers(axes_center)

        # compute the joint cell measure from the selected component.
        if component is None:
            measure = inputs.volume_cell(axis)
        else:
            measure = inputs.components[component].volume_cell(axis)

        outputs = (outputs * measure).sum(axis)
        inputs = inputs.mean(axis)

        return self.replace(
            inputs=inputs,
            outputs=outputs,
        )

    def gradient(
        self,
        axis: str,
        component: None | str = None,
        edge_order: int = 1,
    ) -> FunctionArray:
        """
        Differentiate the outputs with respect to the inputs along ``axis``.

        This is the counterpart to :meth:`integrate`: where that one sums the
        outputs against a measure computed from the inputs, this one divides
        the difference of the outputs by the difference of the inputs.

        The result has the same inputs as this array, and its outputs are the
        derivative, so the units of the result are those of the outputs
        divided by those of the differentiation variable.

        * If ``component`` is :obj:`None`, the inputs themselves are the
          differentiation variable, and they must be a scalar.
        * If ``component`` is a :class:`str`, that named sub-element of the
          inputs is the variable, and it must be a scalar.

        A center axis, where the inputs are samples, differentiates the
        outputs against those samples directly.
        A vertex axis, where the inputs are bin edges and so has one more
        input than output, differentiates against the cell centers, which is
        where the outputs live.

        Parameters
        ----------
        axis
            The logical axis to differentiate along.
        component
            The named input sub-element which is the differentiation variable.
            If :obj:`None`, the inputs themselves are the variable.
        edge_order
            The order of the one-sided differences used at the two ends of
            ``axis``, either 1 or 2.
            The interior always uses second-order central differences.

        Raises
        ------
        ValueError
            If ``axis`` is not an axis of this array, if the differentiation
            variable is not a scalar or does not vary along ``axis``, or if
            there are too few points along ``axis`` for the given
            ``edge_order``.

        See Also
        --------
        :meth:`integrate` : The counterpart of this method, which sums the
            outputs against the inputs rather than dividing by them.
        :func:`numpy.gradient` : The :mod:`numpy` function this method backs.

        Notes
        -----
        Unlike :func:`numpy.gradient`, the differentiation variable may vary
        along axes other than ``axis``, which is what a distorted grid gives,
        since the differences are taken elementwise.

        Examples
        --------

        Differentiate a parabola, whose derivative is a straight line.

        Second-order edges are asked for, since the first-order ones of the
        default are inexact wherever the function curves.

        .. jupyter-execute::

            import numpy as np
            import astropy.units as u
            import named_arrays as na

            x = na.linspace(0, 2, axis="x", num=5) * u.mm

            f = na.FunctionArray(
                inputs=x,
                outputs=np.square(x / u.mm) * u.ph,
            )

            f.gradient("x", edge_order=2)
        """

        self = self.explicit
        inputs = self.inputs
        outputs = self.outputs

        if not isinstance(axis, str):
            raise TypeError(
                f"`axis` must be the name of a single axis, got {axis!r}. "
                f"Differentiating along several axes gives one result per "
                f"axis, so use `numpy.gradient` for that."
            )

        if axis not in self.axes:
            raise ValueError(f"{axis=} must be a member of {self.axes}")

        if edge_order not in (1, 2):
            raise ValueError(f"{edge_order=} must be either 1 or 2")

        if component is None:
            x = inputs
        else:
            x = inputs.components[component]

        if not isinstance(x, na.AbstractScalar):
            raise ValueError(
                f"the differentiation variable must be a scalar, got "
                f"{x.type_abstract}. Name the component to differentiate "
                f"against using the `component` argument."
            )

        # where the inputs are bin edges there is one more of them than there
        # are outputs, so the variable is taken at the cell centers, which is
        # where the outputs live
        if axis in self.axes_vertex:
            x = x.cell_centers(axis)

        return self.replace(
            # a container of its own, so that the result is not the source's
            # inputs under another name, as every neighboring operation gives
            inputs=inputs.copy_shallow(),
            outputs=_core_array_functions._gradient(
                f=outputs,
                x=x,
                axis=axis,
                edge_order=edge_order,
            ),
        )

    def to_string_array(
        self,
        format_value: str = "%.2f",
        format_unit: str = "latex_inline",
        pad_unit: str = r"$\,$",
    ):
        exp = self.explicit
        kwargs = dict(
            format_value=format_value,
            format_unit=format_unit,
            pad_unit=pad_unit,
        )
        return exp.replace(
            inputs=na.as_named_array(exp.inputs).to_string_array(**kwargs),
            outputs=na.as_named_array(exp.outputs).to_string_array(**kwargs),
        )

    def _getitem(
            self,
            item: Mapping[str, int | slice | na.AbstractArray] | na.AbstractArray | na.AbstractFunctionArray,
    ) -> FunctionArray:

        # Implicit inputs and outputs are made explicit once, as they would be
        # by `explicit`, except inputs which are a WCS vector, which can be
        # indexed without computing every coordinate and so stay implicit.
        array = self
        inputs = array.inputs
        if not isinstance(inputs, na.AbstractWcsVector):
            inputs = na.explicit(inputs)
        outputs = na.explicit(array.outputs)

        if isinstance(item, na.AbstractArray):
            if isinstance(item, na.AbstractFunctionArray):
                if not np.all(item.inputs == array.inputs):
                    raise ValueError("item inputs do not match function inputs.")

                axes = set(inputs.axes) - set(array.axes_center)
                inputs = inputs.cell_centers(axis=axes)
                item_inputs = item.outputs
                item_outputs = item.outputs
            else:
                item_inputs = item_outputs = item

            shape_item_inputs = item_inputs.shape
            shape_item_outputs = item_outputs.shape

            inputs = na.broadcast_to(inputs, na.broadcast_shapes(inputs.shape, shape_item_inputs))
            outputs = na.broadcast_to(outputs, na.broadcast_shapes(outputs.shape, shape_item_outputs))

            # the fields are selected the same way as the outputs, so they
            # are broadcast against the item first
            fields = _fields.apply(
                array,
                lambda v: na.broadcast_to(v, na.broadcast_shapes(v.shape, shape_item_outputs))[item_outputs],
                axes=tuple(shape_item_outputs),
            )

        elif isinstance(item, dict):

            # the shapes are found once, since they may need to be computed
            shape_inputs = na.shape(inputs)
            shape_outputs = na.shape(outputs)

            # ignore axes that are not present in this function array, so that
            # indexing is consistent with ``ScalarArray`` and composes with
            # ``na.getitem`` over heterogeneous structures (e.g. a function
            # array that lacks the axis being selected is left untouched)
            item = {
                ax: item[ax] for ax in item
                if ax in shape_inputs or ax in shape_outputs
            }

            item_inputs = dict()
            item_outputs = dict()
            for ax in item:
                item_ax = item[ax]
                if isinstance(item_ax, na.AbstractFunctionArray):
                    item_inputs[ax] = item_ax.inputs
                    item_outputs[ax] = item_ax.outputs
                elif _is_vertex(ax, shape_inputs, shape_outputs):
                    if isinstance(item_ax, slice) or np.issubdtype(type(item_ax), np.integer):
                        item_outputs[ax], item_inputs[ax] = _item_vertex(
                            item=item_ax,
                            num=shape_outputs[ax],
                            axis=ax,
                        )
                    else:
                        return NotImplemented
                else:
                    #can't assume center ax is in both outputs and inputs
                    if ax in shape_inputs:
                        item_inputs[ax] = item_ax
                    if ax in shape_outputs:
                        item_outputs[ax] = item_ax

            # the fields are indexed like the outputs, and ignore the axes
            # which they do not have
            fields = dict()
            if _fields.names(array):
                item_fields = {ax: item_outputs.get(ax, item[ax]) for ax in item}
                fields = _fields.apply(array, lambda v: v[item_fields], axes=tuple(item_fields))

        else:
            return NotImplemented

        return array.replace(
            inputs=inputs[item_inputs],
            outputs=outputs[item_outputs],
            **fields,
        )

    def _getitem_reversed(
            self,
            array: na.AbstractArray,
            item: Mapping[str, int | slice | na.AbstractArray] | na.AbstractFunctionArray
    ):
        if isinstance(array, (na.AbstractScalar, na.AbstractVectorArray)):
            array = na.FunctionArray(array, array)
        else:
            return NotImplemented
        return array._getitem(item)

    def __bool__(self) -> bool:
        result = super().__bool__()
        return result and bool(self.outputs)


    def __mul__(self, other: na.ArrayLike | u.UnitBase) -> FunctionArray:
        if isinstance(other, u.UnitBase):
            exp = self.explicit
            return exp.replace(
                outputs=exp.outputs * other,
            )
        else:
            return super().__mul__(other)

    def __lshift__(self, other: na.ArrayLike | u.UnitBase) -> FunctionArray:
        if isinstance(other, u.UnitBase):
            exp = self.explicit
            return exp.replace(
                outputs=self.outputs << other,
            )
        else:
            return super().__lshift__(other)

    def __truediv__(self, other: na.ArrayLike | u.UnitBase) -> FunctionArray:
        if isinstance(other, u.UnitBase):
            exp = self.explicit
            return exp.replace(
                outputs=self.outputs / other,
            )
        else:
            return super().__truediv__(other)

    def __array_matmul__(
            self,
            x1: float | u.Quantity | FunctionArray,
            x2: float | u.Quantity | FunctionArray,
            out: None | na.FunctionArray | tuple[na.FunctionArray] = None,
            **kwargs,
    ) -> FunctionArray:

        # numpy passes the `out` argument of a ufunc as a tuple
        if isinstance(out, tuple):
            (out,) = out

        if isinstance(x1, na.AbstractArray):
            if isinstance(x1, AbstractFunctionArray):
                inputs_1 = x1.inputs
                outputs_1 = x1.outputs
            else:
                return NotImplemented
        else:
            inputs_1 = None
            outputs_1 = x1

        if isinstance(x2, na.AbstractArray):
            if isinstance(x2, AbstractFunctionArray):
                inputs_2 = x2.inputs
                outputs_2 = x2.outputs
            else:
                return NotImplemented
        else:
            inputs_2 = None
            outputs_2 = x2

        if inputs_1 is not None:
            inputs = inputs_1
            if inputs_2 is not None:
                if np.any(inputs_1 != inputs_2):
                    raise InputValueError("`x1.inputs` and `x2.inputs` must be equal")
            else:
                pass
        else:
            if inputs_2 is not None:
                inputs = inputs_2
            else:
                return NotImplemented

        if out is not None:
            if not np.all(inputs == out.inputs):
                raise InputValueError("`out.inputs` must be equal to `x1.inputs` and `x2.inputs`")
        result = self.explicit.replace(
            inputs=inputs,
            outputs=np.matmul(
                outputs_1,
                outputs_2,
                out=out.outputs if out is not None else out,
                **kwargs,
            ),
        )

        if out is not None:
            result = _fields.out(out, _fields.values(result))

        return result

    def __array_ufunc__(
            self,
            ufunc: np.ufunc,
            method: str,
            *inputs,
            **kwargs,
    ) -> None | na.AbstractArray | tuple[na.AbstractArray, ...]:

        result = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        if result is not NotImplemented:
            return result

        nout = ufunc.nout

        inputs_inputs = []
        inputs_outputs = []
        for inp in inputs:
            if isinstance(inp, na.AbstractArray):
                if isinstance(inp, AbstractFunctionArray):
                    inputs_inputs.append(inp.inputs)
                    inputs_outputs.append(inp.outputs)
                else:
                    inputs_outputs.append(inp)
            else:
                inputs_outputs.append(inp)

        for inputs_input in inputs_inputs[1:]:
            if np.any(inputs_input != inputs_inputs[0]):
                raise InputValueError(f"all inputs to {ufunc} must have the same coordinates")

        if "out" in kwargs:
            out = kwargs.pop("out")
            outputs_out = list()
            for o in out:
                if o is not None:
                    if isinstance(o, self.type_explicit):
                        outputs_o = o.outputs
                    else:
                        raise ValueError(
                            f"each element of `out` must be an instance of {self.type_explicit}, got {type(o)}"
                        )
                else:
                    outputs_o = None
                outputs_out.append(outputs_o)
            if nout == 1:
                outputs_out = outputs_out[0]
            else:
                outputs_out = tuple(outputs_out)

            kwargs["out"] = outputs_out
        else:
            out = (None,) * nout

        if "where" in kwargs:
            where = kwargs.pop("where")
            if isinstance(where, na.AbstractArray):
                if isinstance(where, AbstractFunctionArray):
                    if np.any(where.inputs != inputs_inputs[0]):  #pragma: nocover
                        raise InputValueError("`where.inputs` must match the rest of the inputs")
                    kwargs["where"] = where.outputs
                else:
                    return NotImplemented
            else:
                kwargs["where"] = where

        inputs_result = inputs_inputs[0]
        outputs_result = getattr(ufunc, method)(*inputs_outputs, **kwargs)

        if nout == 1:
            outputs_result = (outputs_result,)

        result = list(
            self.explicit.replace(
                inputs=inputs_result,
                outputs=outputs_result[i],
            )
            for i in range(nout)
        )

        for i in range(nout):
            if out[i] is not None:
                out[i].inputs = result[i].inputs
                out[i].outputs = result[i].outputs
                _fields.out(out[i], _fields.values(result[i]))
                result[i] = out[i]

        if nout == 1:
            result = result[0]
        else:
            result = tuple(result)

        return result

    def __array_function__(
            self,
            func: Callable,
            types: Collection,
            args: tuple,
            kwargs: dict[str, Any],
    ):
        result = super().__array_function__(func=func, types=types, args=args, kwargs=kwargs)
        if result is not NotImplemented:
            return result

        handlers, handled = _array_function_dispatch()

        if func in handlers:
            return handlers[func](func, *args, **kwargs)

        if func in handled:
            return handled[func](*args, **kwargs)

        return NotImplemented

    def __named_array_function__(self, func, *args, **kwargs):
        result = super().__named_array_function__(func, *args, **kwargs)
        if result is not NotImplemented:
            return result

        from . import function_named_array_functions

        if func in function_named_array_functions.ASARRAY_LIKE_FUNCTIONS:
            return function_named_array_functions.asarray_like(func=func, *args, **kwargs)

        if func in function_named_array_functions.NDFILTER_FUNCTIONS:
            return function_named_array_functions.ndfilter(func, *args, **kwargs)

        if func in function_named_array_functions.HANDLED_FUNCTIONS:
            return function_named_array_functions.HANDLED_FUNCTIONS[func](*args, **kwargs)

        return NotImplemented

    def interp_linear(
            self,
            item: dict[str, na.AbstractArray],
    ) -> FunctionArray:
        raise NotImplementedError


    def pcolormesh(
            self,
            axs: np.ndarray,
            input_component_x: str,
            input_component_y: str,
            input_component_row: None | str = None,
            input_component_column: None | str = None,
            index: None | dict[str, int] = None,
            output_component_color: None | str = None,
            **kwargs: Any,
    ):
        """
        Plot a :class:`FunctionArray` via :func:`matplotlib.pyplot.pcolormesh`.

        :func:`FunctionArray.pcolormesh` takes in an axes object, or array of axes objects, along with components to
        be plotted along the x and y plot axes (:attr:`input_component_x` and :attr:`input_component_y`). Additional
        components can be tiled along subplot row/column and are specified in :attr:`input_component_row` and
        :attr:`input_component_column`.

        .. jupyter-execute::

            import named_arrays as na
            import numpy as np
            import astropy.units as u
            import matplotlib.pyplot as plt

            position = na.Cartesian2dVectorLinearSpace(
                start=-10,
                stop=10,
                axis=na.Cartesian2dVectorArray(
                    x='position_x',
                    y='position_y',
                ),
                num=21,
            ) * u.m

            x_width = 5 * u.m
            y_width = 2 * u.m
            velocity = 1 * u.m/u.s
            time = na.ScalarLinearSpace(
                start=0 * u.s,
                stop=3 * u.s,
                num=4,
                axis='time'
            )

            intensity = np.exp(-(((position.x + velocity*time)/x_width) ** 2 + ((position.y + 2*velocity*time)/y_width)** 2))
            scene = na.FunctionArray(
                inputs=position,
                outputs=intensity,
            )

            fig, axs = plt.subplots(
                nrows=scene.outputs.shape['time'],
                squeeze=False,
                sharex=True,
                subplot_kw=dict(aspect='equal'),
            )
            scene.pcolormesh(
                axs=axs,
                input_component_x='x',
                input_component_y='y',
                input_component_row='time',
            )
        """
        import astropy.visualization

        if axs.ndim == 1:
            if input_component_row is not None:
                axs = axs[..., np.newaxis]
            if input_component_column is not None:
                axs = axs[np.newaxis, ...]

        axs = na.ScalarArray(
            ndarray=axs,
            axes=('row', 'column')
        )

        if index is None:
            index = dict()

        with astropy.visualization.quantity_support():
            for index_subplot in axs.ndindex():

                index_final = index.copy()
                if input_component_row is not None:
                    index_final[input_component_row] = index_subplot['row']
                if input_component_column is not None:
                    index_final[input_component_column] = index_subplot['column']

                array = self[index_final]

                inp = array.inputs.cartesian_nd

                inp_x = inp.components[input_component_x]
                inp_y = inp.components[input_component_y]

                out = array.outputs
                if output_component_color is not None:
                    out = out.components[output_component_color]

                # Broadcast the coordinates and the values to 2D arrays with
                # the same axis order so that matplotlib interprets them
                # consistently. Along axes where the coordinates are cell
                # vertices, the values have one fewer element.
                shape_inp = na.shape_broadcasted(inp_x, inp_y)
                shape_out = {
                    axis: shape_inp[axis] - 1 if axis in array.axes_vertex else shape_inp[axis]
                    for axis in shape_inp
                }
                inp_x = na.broadcast_to(inp_x, shape_inp).ndarray
                inp_y = na.broadcast_to(inp_y, shape_inp).ndarray
                out = na.broadcast_to(out, shape_out).ndarray

                ax = axs[index_subplot].ndarray
                ax.pcolormesh(
                    inp_x,
                    inp_y,
                    out,
                    shading='auto',
                    **kwargs,
                )

                if index_subplot['row'] == axs.shape['row'] - 1:
                    if isinstance(inp_x, u.Quantity):
                        ax.set_xlabel(f'{input_component_x} ({inp_x.unit})')
                    else:
                        ax.set_xlabel(f'{input_component_x}')
                else:
                    ax.set_xlabel(None)

                if index_subplot['column'] == 0:
                    if isinstance(inp_y, u.Quantity):
                        ax.set_ylabel(f'{input_component_y} ({inp_y.unit})')
                    else:
                        ax.set_ylabel(f'{input_component_y}')
                else:
                    ax.set_ylabel(None)

                if input_component_column is not None:
                    if input_component_column in inp.components:
                        if index_subplot['row'] == 0:
                            inp_column = inp.components[input_component_column]
                            ax.text(
                                x=0.5,
                                y=1.01,
                                s=f'{inp_column.mean().ndarray.value:0.03f} {inp_column.unit:latex_inline}',
                                transform=ax.transAxes,
                                ha='center',
                                va='bottom'
                            )

                if input_component_row is not None:
                    if input_component_row in inp.components:
                        if index_subplot['column'] == axs.shape['column'] - 1:
                            inp_row = inp.components[input_component_row]
                            ax.text(
                                x=1.01,
                                y=0.5,
                                s=f'{inp_row.mean().ndarray.value:0.03f} {inp_row.unit:latex_inline}',
                                transform=ax.transAxes,
                                va='center',
                                ha='left',
                                rotation=-90,
                            )


@dataclasses.dataclass(eq=False, repr=False)
class FunctionArray(
    AbstractFunctionArray,
    na.AbstractExplicitArray,
    Generic[InputsT, OutputsT],
):
    """
    A representation of a discrete function.

    A composition of two arrays: :attr:`inputs` and :attr:`outputs`.
    :attr:`inputs` represents the inputs (or independent variables) of the
    function, and :attr:`outputs` represents the outputs (or dependent variables) of
    the function.

    A subclass can add fields of its own, such as the exposure time of each
    image of an observation. Every such field whose value is a named array is
    treated as sampled along the axes of the function:

    * Indexing (with a :class:`dict`, :meth:`isel`, index arrays, or a boolean
      mask), :func:`numpy.stack`, :func:`numpy.concatenate`,
      :func:`numpy.moveaxis`, :func:`numpy.repeat`,
      :func:`numpy.take_along_axis`, :func:`numpy.reshape`,
      :meth:`combine_axes`, and :meth:`cell_centers` do to the field what they
      do to the outputs, along the axes which the field has.
      :func:`named_arrays.debroadcast` keeps an axis that the field varies
      along.
    * Assigning another array of the subclass with ``__setitem__``, or
      copying one with :func:`numpy.copyto`, writes its fields into those of
      this array, in place, like the outputs. A field which does not vary
      along an axis being assigned can only be given the value it already
      has, and so can a field which is not a named array, such as a string.
      Since most operations pass the fields of an array on without copying
      them, assigning into the result of, say, a ufunc also changes the
      fields of the array it was computed from. A field without a
      distribution is instead replaced by an uncertain copy of itself, like
      the outputs, when the value is uncertain or the mask selects different
      elements in different samples.
    * An operation which removes or resamples an axis that the field varies
      along raises an error: a reduction with ``keepdims=False``,
      :func:`numpy.percentile` (which removes the axes by default),
      :meth:`integrate`, :meth:`regrid`, or :meth:`cell_centers` with
      ``random=True``. So does a reduction with ``keepdims=True`` along an
      axis which the inputs do not have, since it keeps only the axes of the
      inputs. Whether the field should be summed, averaged, or dropped
      depends on what it is, so replace it first, for example with
      :func:`dataclasses.replace`.
    * :meth:`cell_centers` also raises an error for a field whose elements
      are not floating-point numbers, such as a mask, rather than turn them
      into fractions.
    * The other operations, including the reductions which keep the reduced
      axes (the default for a function array), pass the field on unchanged. A
      ufunc of several arrays of the subclass takes the fields of the first.
      An array given as the ``out`` argument of an operation takes the fields
      of the result, if its type has them.

    When the arrays combined by :func:`numpy.stack` or
    :func:`numpy.concatenate` have fields which are not named arrays, such as
    a string or a number, those must be the same in every array. Fields which
    cannot be compared, such as models of an instrument, are taken from the
    array whose type is a subclass of the types of all the others. Arrays
    with a field which that type does not have, such as arrays of two
    unrelated subclasses, cannot be combined.

    Every function array checks, when it is built, that each such field has
    the same number of elements as the outputs along the axes they share, so
    that an operation which changes the outputs without the field fails
    instead of leaving the field stale. The check has limits:

    * It allows outputs with a single element along an axis, as after a
      reduction which keeps its axes, and it cannot see an axis which an
      operation removed from the outputs but not from the field. Catching
      those would take the shape of the inputs, which can be expensive to
      compute.
    * It skips outputs and fields which are implicit, such as random samples,
      since finding their shape would compute them.
    * A subclass which defines its own ``__post_init__`` must call
      ``super().__post_init__()`` for the check to run.

    Fields whose values are not named arrays are otherwise passed on unchanged
    by every operation.

    Examples
    --------

    The exposure time of each of a sequence of images follows the images when
    they are indexed.

    .. jupyter-execute::

        import dataclasses
        import numpy as np
        import astropy.units as u
        import named_arrays as na

        @dataclasses.dataclass(eq=False, repr=False)
        class Images(na.FunctionArray):
            timedelta: na.AbstractScalar = 0 * u.s

        images = Images(
            inputs=na.linspace(0, 30, axis="time", num=4) * u.s,
            outputs=na.ScalarArray(np.arange(4.0), axes="time") * u.DN,
            timedelta=na.ScalarArray(np.array([1, 2, 3, 4]) * u.s, axes="time"),
        )

        images[dict(time=slice(1, 3))].timedelta

    Co-adding the images removes the axis of time, so the exposure times are
    added up first.

    .. jupyter-execute::

        total = np.sum(
            dataclasses.replace(images, timedelta=images.timedelta.sum("time")),
            axis="time",
            keepdims=False,
        )
        total.outputs / total.timedelta
    """
    inputs: InputsT = 0
    """The inputs of the function."""

    # The operators declared on `AbstractArray` can only promise the widest
    # array type. The result of an operation is the explicit array of the
    # highest family involved, so on an explicit array of this family the
    # result is `Self` unless a higher family absorbs it. Declarations only;
    # the implementation is inherited.
    if TYPE_CHECKING:  # pragma: nocover

        def __add__(self, other: na.ArrayLike) -> Self: ...

        def __sub__(self, other: na.ArrayLike) -> Self: ...

        def __floordiv__(self, other: na.ArrayLike) -> Self: ...

        def __mod__(self, other: na.ArrayLike) -> Self: ...

        def __pow__(self, other: na.ArrayLike) -> Self: ...

        def __radd__(self, other: na.ArrayLike) -> Self: ...

        def __rsub__(self, other: na.ArrayLike) -> Self: ...

        def __rmul__(self, other: na.ArrayLike) -> Self: ...

        def __rtruediv__(self, other: na.ArrayLike) -> Self: ...

        def __rfloordiv__(self, other: na.ArrayLike) -> Self: ...

        def __rmod__(self, other: na.ArrayLike) -> Self: ...

        def __rpow__(self, other: na.ArrayLike) -> Self: ...

        def __lt__(self, other: na.ArrayLike) -> Self: ...

        def __le__(self, other: na.ArrayLike) -> Self: ...

        def __gt__(self, other: na.ArrayLike) -> Self: ...

        def __ge__(self, other: na.ArrayLike) -> Self: ...

        def __mul__(self, other: na.ArrayLike | u.UnitBase) -> Self: ...

        def __truediv__(self, other: na.ArrayLike | u.UnitBase) -> Self: ...

        def __lshift__(self, other: na.ArrayLike | u.UnitBase) -> Self: ...

        def __neg__(self) -> Self: ...

        def __pos__(self) -> Self: ...

        def __abs__(self) -> Self: ...


    outputs: OutputsT = 0
    """The outputs of the function."""

    def __post_init__(self) -> None:
        _fields.check(self)

    @classmethod
    def from_scalar_array(
            cls: type[Self],
            a: float | u.Quantity | na.AbstractScalarArray,
            like: None | Self = None,
    ) -> Self:

        self = super().from_scalar_array(a=a, like=like)

        if like is None:
            self.inputs = a
            self.outputs = a
        else:
            if isinstance(like.inputs, na.AbstractArray):
                self.inputs = like.inputs.from_scalar_array(a=a, like=like.inputs)
            else:
                self.inputs = a

            if isinstance(like.outputs, na.AbstractArray):
                self.outputs = like.outputs.from_scalar_array(a=a, like=like.outputs)
            else:
                self.outputs = a

        return self

    @property
    def axes(self) -> tuple[str, ...]:
        return tuple(self.inputs.shape | self.outputs.shape)

    @property
    def shape(self) -> dict[str, int]:

        axes_center = self.axes_center
        axes_vertex = self.axes_vertex

        outputs_shape = self.outputs.shape
        inputs_shape = self.inputs.shape

        outputs_shape_center = {ax: outputs_shape[ax] for ax in outputs_shape if ax in axes_center}
        inputs_shape_center = {ax: inputs_shape[ax] for ax in inputs_shape if ax in axes_center}
        outputs_shape_vertex = {ax: outputs_shape[ax] for ax in outputs_shape if ax in axes_vertex}

        shape = na.broadcast_shapes(
            outputs_shape_center,
            outputs_shape_vertex,
            inputs_shape_center,
        )

        return shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.array(tuple(self.shape.values())).prod())

    @property
    def explicit(self) -> FunctionArray:
        return self.replace(
            inputs=na.explicit(self.inputs),
            outputs=na.explicit(self.outputs),
        )

    def __setitem__(
            self,
            item: Mapping[str, int | slice | na.AbstractArray] | na.AbstractArray,
            value: float | u.Quantity | na.FunctionArray,
    ):

        shape_inputs = na.shape(self.inputs)
        shape_outputs = na.shape(self.outputs)

        def is_vertex(axis: str) -> bool:
            return _is_vertex(axis, shape_inputs, shape_outputs)

        # Whether the inputs can be assigned the inputs of `value`.
        # Only the cells are assigned along a vertex axis, since `value`
        # cannot say where their vertices are.
        assign_inputs = True

        if isinstance(item, na.AbstractFunctionArray):
            if not np.all(item.inputs == self.inputs):
                raise ValueError("boolean advanced index does not have the same inputs as the array")

            item_inputs = item.outputs
            item_outputs = item.outputs

            # the mask selects cells, which are not the vertices
            if any(is_vertex(ax) for ax in shape_inputs):
                assign_inputs = False

        elif isinstance(item, dict):

            item_inputs = dict()
            item_outputs = dict()
            for ax in item:
                item_ax = item[ax]
                if isinstance(item_ax, na.AbstractFunctionArray):
                    item_inputs[ax] = item_ax.inputs
                    item_outputs[ax] = item_ax.outputs
                elif is_vertex(ax):
                    # An integer keeps the axis, as it does for indexing,
                    # so that ``a[item] = a[item]`` leaves `a` unchanged.
                    if np.issubdtype(type(item_ax), np.integer):
                        item_ax, _ = _item_vertex(
                            item=item_ax,
                            num=shape_outputs[ax],
                            axis=ax,
                        )
                    item_outputs[ax] = item_ax
                    assign_inputs = False
                else:
                    item_inputs[ax] = item_outputs[ax] = item_ax
        else:
            raise TypeError(
                f"`item` must be an instance of `{dict.__name__}`, or `{na.AbstractFunctionArray.__name__}`, "
                f"got `{type(item)}`"
            )

        if isinstance(value, na.AbstractArray):
            if isinstance(value, na.AbstractFunctionArray):
                value_inputs = value.inputs
                value_outputs = value.outputs

                # maybe this should only set to None vertex axes only
                for ax in na.shape(value_inputs):
                    if is_vertex(ax):
                        value_inputs = None

            else:
                if value.shape:
                    raise ValueError(
                        f"if `value` is an instance of `{na.AbstractArray.__name__}`, "
                        f"but not an instance of `{na.AbstractFunctionArray.__name__}`, "
                        f"`value.shape` should be empty, got {value.shape}"
                    )
                else:
                    value_inputs = None
                    value_outputs = value
        else:
            value_inputs = None
            value_outputs = value

        from named_arrays._scalars.uncertainties import uncertainties

        # every field is checked before anything is written, so that an error
        # does not leave this array half written
        writes = _fields.setitem(self, item_outputs, value)

        # Inputs or outputs without a distribution cannot store a selection
        # which differs between samples, or an uncertain value, so they are
        # replaced by an uncertain copy of themselves once both are assigned.
        inputs = self.inputs
        if value_inputs is not None and assign_inputs:
            inputs = uncertainties._as_uncertain(inputs, item_inputs, value_inputs)
            inputs[item_inputs] = value_inputs

        outputs = uncertainties._as_uncertain(self.outputs, item_outputs, value_outputs)
        outputs[item_outputs] = value_outputs

        self.inputs = inputs
        self.outputs = outputs

        for write in writes:
            write()


@dataclasses.dataclass(eq=False, repr=False)
class AbstractPolynomialFunctionArray(
    AbstractFunctionArray,
):
    @property
    @abc.abstractmethod
    def coefficients(self) -> na.AbstractVectorArray | na.AbstractMatrixArray:
        """
        A vector or matrix representing the coefficients of the polynomial.

        If this function is scalar-valued, :attr:`coefficients` should be a vector,
        and if this function is vector-valued, :attr`coefficients` should be a matrix.
        """

    @property
    @abc.abstractmethod
    def coefficient_names(self) -> Sequence[str]:
        """The names of the polynomial terms to fit (the design-matrix component keys)."""

    @property
    @abc.abstractmethod
    def axis_polynomial(self) -> None | str | Sequence[str]:
        """The logical axes along which this polynomial is distributed."""

    @property
    @abc.abstractmethod
    def center(self) -> None | InputsT:
        """
        The reference point which is subtracted from the inputs before fitting
        the polynomial.
        """

    @abc.abstractmethod
    def design_matrix(
        self,
        inputs: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
    ) -> na.AbstractVectorArray:
        """
        The `design matrix <https://en.wikipedia.org/wiki/Design_matrix>`_
        corresponding to the given inputs.

        Note that while this is `called` a matrix, this function returns a vector
        since the rows are independent observations, and this is concept is
        already captured by the logical axes in the array.

        Parameters
        ----------
        inputs
            the set of independent variables to convert into the design matrix
        """

    def __call__(
        self,
        inputs: float | u.Quantity | na.ScalarArray | na.AbstractVectorArray,
    ) -> na.AbstractFunctionArray:

        outputs = self.design_matrix(inputs) @ self.coefficients

        return na.FunctionArray(inputs, outputs)

    @property
    def predictions(self) -> OutputsT:
        """
        The outputs of the polynomial model given :attr:`inputs`.
        Equivalent to ``self(self.inputs).outputs``.
        """
        return self(self.inputs).outputs


@dataclasses.dataclass(eq=False, repr=False)
class PolynomialFitFunctionArray(
    FunctionArray[InputsT, OutputsT],
    AbstractPolynomialFunctionArray,
):
    """
    A :class:`named_arrays.PolynomialFitFunctionArray` carries the independent variables, inputs, and dependent variables, outputs,
    of a discrete function, and a linear least squares polynomial fit to that function.

    The set of polynomial terms to fit is given explicitly by :attr:`coefficient_names`.
    Use the :meth:`from_degree` classmethod to generate the standard term set for a
    given polynomial degree (a scalar total-degree cap, or a per-component vector of
    maximum exponents for a mixed-order fit).

    Parameters
    ----------
    inputs
        The set of independent variables.
    outputs
        The set of dependent variables.
    coefficient_names
        The names of the polynomial terms to fit (the design-matrix component keys).
    axis_polynomial
        The logical axes along which this polynomial is distributed.
    where_polynomial
        A boolean mask controlling which elements to use for fitting.

    Examples
    --------

    .. nblinkgallery::
        :caption: Relevant Tutorials
        :name: rst-link-gallery

        ../tutorials/PolynomialFunctionArray
    """

    center: None | InputsT = None
    """The reference point which is subtracted from the inputs before fitting."""

    coefficient_names: None | Sequence[str] = None
    """
    The names of the polynomial terms to fit (the design-matrix component keys).

    Each name is a ``*``-joined product of input component names (dotted for nested
    vectors, e.g. ``"position.x*wavelength"``); the empty string ``""`` is the
    constant term. Use :meth:`from_degree` to generate the standard set for a given
    degree, optionally pruning terms before constructing the fit.
    """

    axis_polynomial: None | str | Sequence[str] = None
    """The logical axes along which this polynomial is distributed."""

    where_polynomial: bool | na.ScalarArray = True
    """A boolean mask controlling which elements to use for fitting."""

    @functools.cached_property
    def coefficients(self) -> na.AbstractVectorArray | na.AbstractMatrixArray:
        d = self.design_matrix(self.inputs)
        axis = self.axis_polynomial
        where = self.where_polynomial
        dTd = self._outer(d, d, axis=axis, where=where)
        dTo = self._outer(d, self.outputs, axis=axis, where=where)

        return dTd.inverse @ dTo

    def design_matrix(
        self,
        inputs: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
    ) -> na.AbstractVectorArray:

        if self.coefficient_names is None:
            raise ValueError(
                "`coefficient_names` is required; use "
                "`PolynomialFitFunctionArray.from_degree` to generate it."
            )

        if self.center is not None:
            inputs = inputs - self.center

        inputs = na.as_named_array(inputs)
        if isinstance(inputs, na.AbstractScalar):
            inputs = na.CartesianNdVectorArray({"dummy": inputs})
        inputs = inputs.cartesian_nd.broadcasted.components

        design_matrix = {}
        for name in self.coefficient_names:
            term = 1
            if name:  # the empty string is the constant term
                for k in name.split("*"):
                    if k not in inputs:
                        raise ValueError(
                            f"coefficient name {name!r} references input component "
                            f"{k!r}, which is not one of the available components "
                            f"{tuple(inputs)}."
                        )
                    term = term * inputs[k]
            design_matrix[name] = term

        design_matrix = na.CartesianNdVectorArray(design_matrix)

        return design_matrix

    @classmethod
    def from_degree(
        cls,
        inputs: InputsT,
        outputs: OutputsT,
        degree: int,
        components: None | str | Sequence[str] = None,
        center: None | InputsT = None,
        axis_polynomial: None | str | Sequence[str] = None,
        where_polynomial: bool | na.ScalarArray = True,
    ) -> Self:
        """
        Construct a :class:`PolynomialFitFunctionArray` whose :attr:`coefficient_names`
        are the standard monomial terms for a given polynomial degree.

        Parameters
        ----------
        inputs
            The set of independent variables.
        outputs
            The set of dependent variables.
        degree
            The degree of the polynomial. Every monomial with total degree up to
            ``degree`` (over `components`) is included.
        components
            The components of the input that the polynomial depends on.
            Defaults to all components of `inputs`.
        center
            The reference point subtracted from the inputs before fitting.
        axis_polynomial
            The logical axes along which the polynomial is distributed.
        where_polynomial
            A boolean mask controlling which elements to use for fitting.
        """
        if components is None:
            inputs_nd = na.as_named_array(inputs)
            if isinstance(inputs_nd, na.AbstractScalar):
                # mirror `design_matrix`'s wrapping of scalar inputs
                inputs_nd = na.CartesianNdVectorArray({"dummy": inputs_nd})
            components = tuple(inputs_nd.cartesian_nd.broadcasted.components)
        elif isinstance(components, str):
            components = (components,)

        coefficient_names = []
        for i in range(degree + 1):
            for combination in itertools.combinations_with_replacement(components, i):
                coefficient_names.append("*".join(combination))

        return cls(
            inputs=inputs,
            outputs=outputs,
            coefficient_names=coefficient_names,
            center=center,
            axis_polynomial=axis_polynomial,
            where_polynomial=where_polynomial,
        )

    @classmethod
    def _outer(
        cls,
        v1: na.AbstractVectorArray,
        v2: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
        axis: None | str | Sequence[str] = None,
        where: bool | na.AbstractScalar = True,
    ):
        v1_T_v2_components = {}

        if isinstance(v1, na.AbstractVectorArray):
            v1_broadcasted = v1.broadcasted.components
            if isinstance(v2, na.AbstractVectorArray):
                for c1 in v1_broadcasted:
                    v2_broadcasted = v2.broadcasted.components
                    row_components = {}
                    for c2 in v2_broadcasted:
                        row_components[c2] = (
                                v1_broadcasted[c1] * v2_broadcasted[c2]
                        ).sum(axis=axis, where=where)
                    v1_T_v2_components[c1] = v2.type_explicit.from_components(row_components)
                v1_T_v2 = v1.type_matrix.from_components(v1_T_v2_components)

            else:
                for c1 in v1_broadcasted:
                    row_components = (v1_broadcasted[c1] * v2).sum(axis=axis, where=where)
                    v1_T_v2_components[c1] = row_components
                v1_T_v2 = v1.type_explicit.from_components(v1_T_v2_components)

        else:
            raise NotImplementedError

        return v1_T_v2


def _components(
    a: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
) -> dict[str, Any]:
    """
    The components of the inputs of a model by name, with those of nested
    vectors named with dots, or, if they are not a vector, the inputs
    themselves, under an empty name.
    """
    if isinstance(a, na.AbstractVectorArray):
        return a.cartesian_nd.components
    return {"": a}


@dataclasses.dataclass(eq=False, repr=False)
class AbstractModelFunctionArray(
    AbstractFunctionArray,
):
    """
    A function array with a model of its outputs, compiled with
    :func:`ndoptimize.model`, and the parameters of the model.
    """

    @property
    @abc.abstractmethod
    def model(self) -> ndoptimize.Model:
        """The model of the outputs, made with :func:`ndoptimize.model`."""

    @property
    @abc.abstractmethod
    def parameters(self) -> na.AbstractVectorArray:
        """
        The parameters of the model, a vector whose components are named
        after :attr:`ndoptimize.Model.parameters`.
        """

    @property
    @abc.abstractmethod
    def components_model(self) -> None | str | Sequence[str]:
        """
        The components of the inputs, in order, which are the inputs of the
        model, or :obj:`None` for every component.
        """

    @property
    @abc.abstractmethod
    def axis_model(self) -> None | str | Sequence[str]:
        """
        The logical axes along which the points of a single instance of the
        model vary, or :obj:`None` for every axis.
        """

    def _inputs_model(
        self,
        inputs: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
    ) -> float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray:
        """
        The inputs of the model for the given inputs of this function: the
        components named by :attr:`components_model`, with each one which
        varies along a vertex axis of :attr:`axis_model`, along which the
        inputs of this function are the edges of bins, replaced by the lower
        and upper edges of the bins.

        Raises a :class:`ValueError` if the model takes another number of
        inputs, as declared by :attr:`ndoptimize.Model.inputs`, or if,
        without :attr:`components_model`, the components of the inputs are
        the inputs of the model in another order.

        Parameters
        ----------
        inputs
            A set of independent variables of this function.
        """
        components = self.components_model
        if components is not None:

            if isinstance(components, str):
                components = (components,)

            if not isinstance(components, Sequence):
                raise TypeError(
                    f"`components_model` must be a name or a sequence of "
                    f"names, in the order of the inputs of the model, got "
                    f"{components!r}."
                )

            if len(set(components)) != len(components):
                raise ValueError(
                    f"Each component of `components_model` is one input of "
                    f"the model, so it must name each only once, got "
                    f"{components}."
                )

            if not isinstance(inputs, na.AbstractVectorArray):
                raise TypeError(
                    f"`components_model` selects components of vector inputs, "
                    f"got {type(inputs)}."
                )

            # the components of nested vectors are named with dots, as in
            # `PolynomialFitFunctionArray.coefficient_names`
            components_inputs = inputs.cartesian_nd.components
            missing = tuple(c for c in components if c not in components_inputs)
            if missing:
                raise ValueError(
                    f"The components {missing} of `components_model` are not "
                    f"components of the inputs, {tuple(components_inputs)}."
                )

            inputs = na.CartesianNdVectorArray(
                {c: components_inputs[c] for c in components}
            )

        axes_vertex = self.axes_vertex
        if axes_vertex:
            inputs = self._edges(inputs, axes_vertex)

        names = tuple(_components(inputs))
        if len(names) != len(self.model.inputs):
            message = (
                f"The model takes the inputs {self.model.inputs}, but this "
                f"function gives it {len(names)}"
            )
            if isinstance(inputs, na.AbstractVectorArray):
                message += f", {names}"
            message += "."
            if axes_vertex:
                message += (
                    f" Along {axes_vertex}, the inputs of this function are "
                    f"the edges of bins, so each input of the model which "
                    f"varies along one becomes the lower and upper edges of "
                    f"the bins, for a model of the average over each bin, "
                    f"like `ndoptimize.models.gaussian_binned`."
                )
            raise ValueError(message)

        # the model takes its inputs in order, whatever their names, so a
        # vector whose components are its inputs in another order would
        # swap them
        if (
            self.components_model is None
            and names != self.model.inputs
            and set(names) == set(self.model.inputs)
        ):
            raise ValueError(
                f"The model takes the inputs {self.model.inputs} in this "
                f"order, but the components of the inputs of this function "
                f"are in another order, {names}. Select them in the order of "
                f"the model with `components_model`."
            )

        return inputs

    def _edges(
        self,
        inputs: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
        axes_vertex: tuple[str, ...],
    ) -> na.CartesianNdVectorArray:
        """
        The inputs of the model with each one which varies along a vertex
        axis replaced by the lower and upper edges of the bins along it.

        Parameters
        ----------
        inputs
            The inputs of the model, selected with :attr:`components_model`.
        axes_vertex
            The vertex axes of this function.
        """
        axis_model = self.axis_model
        if isinstance(axis_model, str):
            axis_model = (axis_model,)

        result = dict()
        for name, value in _components(inputs).items():
            axes = tuple(ax for ax in axes_vertex if ax in na.shape(value))
            if not axes:
                result[name] = value
                continue
            if axis_model is not None:
                batch = tuple(ax for ax in axes if ax not in axis_model)
                if batch:
                    raise ValueError(
                        f"The input {name!r} of the model varies along {batch}, "
                        f"along which the inputs of this function are the "
                        f"edges of bins, but which are not in `axis_model`, "
                        f"{axis_model}. Select the inputs of the model with "
                        f"`components_model`, or use the centers of the bins."
                    )
            if len(axes) > 1:
                raise ValueError(
                    f"The input {name!r} of the model varies along {axes}, "
                    f"along each of which the inputs of this function are the "
                    f"edges of bins, so its bins have no single lower and "
                    f"upper edge."
                )
            ax = axes[0]
            prefix = f"{name}." if name else ""
            result[f"{prefix}lower"] = value[{ax: slice(None, -1)}]
            result[f"{prefix}upper"] = value[{ax: slice(1, None)}]

        return na.CartesianNdVectorArray(result)

    def __call__(
        self,
        inputs: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
    ) -> na.FunctionArray:
        """
        Evaluate the model at the given inputs, using
        :func:`ndoptimize.evaluate`.

        The model sees each input in the unit of the same input in
        :attr:`inputs`, and its values have the unit of :attr:`outputs`.
        If :attr:`parameters` or the inputs are uncertain, the model is
        evaluated for their nominal values and for their samples, and the
        outputs are uncertain.

        Parameters
        ----------
        inputs
            The inputs at which to evaluate the model, of the same kind as
            :attr:`inputs`.
            The axes of :attr:`axis_model` along which they vary are the
            points of each evaluation, and every other axis is a batch axis,
            like the axes along which :attr:`parameters` vary.
            If :attr:`axis_model` is :obj:`None`, every axis of the inputs
            is an axis of the points.
            Along an axis along which :attr:`inputs` are the edges of bins,
            these are the edges of bins too, and the outputs have one fewer
            element.
        """
        from named_arrays._scalars.uncertainties import uncertainties

        parameters = self.parameters

        inputs = na.as_named_array(inputs)

        x = self._inputs_model(inputs)
        components = _components(x)
        components_fit = _components(self._inputs_model(self.inputs))
        if tuple(components) != tuple(components_fit):
            raise ValueError(
                f"These inputs give the model the inputs {tuple(components)}, "
                f"but the inputs of this function give it {tuple(components_fit)}."
            )
        components = {
            c: na.as_named_array(components[c]).to(na.unit_normalized(components_fit[c]))
            for c in components
        }
        if isinstance(x, na.AbstractVectorArray):
            x = na.CartesianNdVectorArray(components)
        else:
            x = components[""]

        # the samples of uncertain arguments are evaluated separately, and the
        # shape of an uncertain array leaves out the axis of its samples
        axis = self.axis_model
        if axis is None:
            axis = tuple(na.shape(x))

        outputs = uncertainties._propagate(
            na.optimize._evaluate,
            arguments=dict(x=x, parameters=parameters),
            model=self.model,
            axis=axis,
            unit=na.unit(self.outputs),
        )
        return na.FunctionArray(inputs, outputs)

    @property
    def predictions(self) -> na.AbstractScalar:
        """
        The outputs of the model given :attr:`inputs`.
        Equivalent to ``self(self.inputs).outputs``.
        """
        return self(self.inputs).outputs


@dataclasses.dataclass(eq=False, repr=False)
class ModelFitFunctionArray(
    FunctionArray[InputsT, OutputsT],
    AbstractModelFunctionArray,
):
    r"""
    A :class:`named_arrays.ModelFitFunctionArray` carries the independent
    variables, inputs, and dependent variables, outputs, of a discrete
    function, and a nonlinear least-squares fit of a model to that function,
    using the Levenberg-Marquardt method of :func:`ndoptimize.least_squares`.

    The points of a single fit vary along :attr:`axis_model`, and every other
    axis is a batch axis, along which each set of outputs gets a fit of its
    own, so that one array can fit every spectrum of a spectral cube.
    The parameters are the components of a vector, named after
    :attr:`ndoptimize.Model.parameters`.
    They are fit the first time they, or any other result of the fit, are
    used, and fit again after any field is assigned to, including by
    assigning to elements of this array.
    Writing into the array of a field itself, like
    ``fit.outputs[index] = value``, or into one which shares its memory,
    like the outputs of an array indexed from this one, is not noticed, and
    leaves the old fit.

    Each fit minimizes

    .. math::

        \chi^2 = (y - f)^T C^{-1} (y - f)
            + \sum_k \left( \frac{p_k - \bar{p}_k}{\sigma_k} \right)^2,

    where :math:`y` is the outputs, :math:`f` is the model, :math:`C` is the
    covariance of the noise, :math:`p_k` are the parameters, and
    :math:`\bar{p}_k` and :math:`\sigma_k` are the means and standard
    deviations of their Gaussian priors.

    The model is a compiled function of plain numbers, so it sees the values
    of the inputs and outputs in their own units, and the value of each
    parameter in the unit of its initial guess.
    The guess must therefore be in the units which the model expects for the
    units of the inputs and outputs, like the unit of the outputs for the
    amplitude of a Gaussian, and the unit of the inputs for its center and
    width.
    The variance is converted to the unit of the outputs squared, and the
    bounds and priors of each parameter to the unit of its guess.
    Like any field of a function array, the guess, variance, bounds and
    priors are passed on unchanged by operations which change the inputs or
    outputs, so after one which changes their units or their scale, like
    :meth:`to` or multiplying this array by a quantity, replace them too,
    for example with :func:`dataclasses.replace`, before using the fit of
    the result.

    Any field may be uncertain, including a component of a vector.
    Then the nominal values and each of the samples are fit separately, and
    the results of the fit are uncertain arrays.

    Parameters
    ----------
    inputs
        The set of independent variables.
        If it is a vector, its components, in order, are the inputs of the
        model, unless `components_model` selects some of them.
        Components named after the inputs of the model, but in another order,
        raise an error, since the model takes its inputs in order, whatever
        their names, so select them in order with `components_model`.
        Along an axis of `axis_model` along which the inputs are the edges of
        bins, with one more element than the outputs, each input which varies
        along it becomes two inputs of the model, the lower and upper edges of
        the bins, for a model of the average over each bin, like
        :func:`ndoptimize.models.gaussian_binned`.
        The model must take as many inputs as it gets, as declared by
        :attr:`ndoptimize.Model.inputs`.
    outputs
        The set of dependent variables, which the model is fit to.
        Points which are NaN are left out of their fit.
    model
        The model to fit, made with :func:`ndoptimize.model`.
    guess
        The initial guess for the parameters, a vector whose components are
        the parameters named in :attr:`ndoptimize.Model.parameters`.
        It must not vary along `axis_model`, and neither may the bounds and
        priors.
        A guess which varies along an axis which the inputs and outputs do
        not, to start several fits from different points, adds that axis to
        the results, but not to the shape of this array, so this array cannot
        be indexed along it: index the guess or the results instead.
    variance
        The variance of the noise of each output.
        Points with a variance which is NaN or not positive are left out of
        their fit.
    components_model
        The components of the inputs, in order, which are the inputs of the
        model, with those of nested vectors named with dots, like
        ``"position.x"``.
        If :obj:`None` (the default), every component.
    axis_model
        The logical axes along which the points of a single fit vary.
        If :obj:`None` (the default), every axis of the inputs, outputs, and
        variance.
    where_model
        A boolean mask controlling which elements to use for fitting.
    lower
        The lower bounds of the parameters, a vector with a component for
        each parameter that has one.
        If :obj:`None` (the default), no parameter has a lower bound.
    upper
        The upper bounds of the parameters, in the same form as `lower`.
    prior_mean
        The means of the Gaussian priors of the parameters, a vector with a
        component for each parameter that has a prior.
    prior_sigma
        The standard deviations of the Gaussian priors of the parameters, a
        vector with the same components as `prior_mean`.
    correlation
        The correlation of the noise of two outputs by their separation along
        `axis_model`, in samples, starting with 1, as in
        :func:`ndoptimize.least_squares`.
        It requires a single axis in `axis_model`.
        If :obj:`None` (the default), the noise is uncorrelated.
    max_iterations
        The maximum number of steps of each fit.
    tolerance
        The convergence tolerance on the Newton decrement, as in
        :func:`ndoptimize.least_squares`.

    Examples
    --------

    Fit a Gaussian to each of a hundred noisy spectral lines, with the
    parameters in physical units.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import ndoptimize

        # the true parameters of the lines
        shape = dict(position=100)
        truth = na.CartesianNdVectorArray(dict(
            amplitude=na.random.uniform(5, 10, shape, seed=0) * u.DN,
            center=na.random.normal(0, 10, shape, seed=1) * u.km / u.s,
            width=na.random.uniform(10, 20, shape, seed=2) * u.km / u.s,
        ))

        # the noisy spectra
        velocity = na.linspace(-100, 100, axis="velocity", num=41) * u.km / u.s
        z = (velocity - truth.center) / truth.width
        sigma = 0.5 * u.DN
        intensity = na.random.normal(truth.amplitude * np.exp(-z * z / 2), sigma, seed=3)

        # fit every spectrum
        fit = na.ModelFitFunctionArray(
            inputs=velocity,
            outputs=intensity,
            model=ndoptimize.models.gaussian,
            guess=na.CartesianNdVectorArray(dict(
                amplitude=5 * u.DN,
                center=0 * u.km / u.s,
                width=15 * u.km / u.s,
            )),
            variance=np.square(sigma),
            axis_model="velocity",
        )

        # the best-fit model of the first three spectra, on a finer grid
        index = dict(position=slice(0, 3))
        best = fit(na.linspace(-100, 100, axis="velocity", num=201) * u.km / u.s)

        # compare the fitted centers to the truth
        error = fit.parameters.center - truth.center
        uncertainty = np.sqrt(fit.covariance.center.center)

        fig, ax = plt.subplots(ncols=2, figsize=(8, 3.5), constrained_layout=True)
        na.plt.scatter(fit.inputs, fit.outputs[index], ax=ax[0], color="gray");
        na.plt.plot(best.inputs, best.outputs[index], ax=ax[0], axis="velocity");
        ax[0].set_xlabel(f"velocity ({ax[0].get_xlabel()})");
        ax[0].set_ylabel(f"intensity ({ax[0].get_ylabel()})");
        na.plt.scatter(truth.center, error / uncertainty, ax=ax[1]);
        ax[1].set_xlabel(f"true center ({ax[1].get_xlabel()})");
        ax[1].set_ylabel("(fitted center - true center) / uncertainty");
    """

    model: None | ndoptimize.Model = None
    """The model to fit, made with :func:`ndoptimize.model`."""

    guess: None | na.AbstractVectorArray = None
    """The initial guess for the parameters."""

    variance: None | float | u.Quantity | na.AbstractScalar = None
    """The variance of the noise of each output."""

    components_model: None | str | Sequence[str] = None
    """The components of the inputs which are the inputs of the model."""

    axis_model: None | str | Sequence[str] = None
    """The logical axes along which the points of a single fit vary."""

    where_model: bool | na.AbstractScalar = True
    """A boolean mask controlling which elements to use for fitting."""

    lower: None | na.AbstractVectorArray = None
    """The lower bounds of the parameters."""

    upper: None | na.AbstractVectorArray = None
    """The upper bounds of the parameters."""

    prior_mean: None | na.AbstractVectorArray = None
    """The means of the Gaussian priors of the parameters."""

    prior_sigma: None | na.AbstractVectorArray = None
    """The standard deviations of the Gaussian priors of the parameters."""

    correlation: None | Sequence[float] = None
    """The correlation of the noise of two outputs by their separation."""

    max_iterations: int = 1000
    """The maximum number of steps of each fit."""

    tolerance: float = 1e-8
    """The convergence tolerance on the Newton decrement."""

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        # a fit of the old fields is out of date
        self.__dict__.pop("_fit", None)

    @functools.cached_property
    def _fit(self) -> na.optimize._LeastSquaresResult:
        """Every result of the fit, computed the first time one is used."""
        from named_arrays._scalars.uncertainties import uncertainties

        for name in ("model", "guess", "variance"):
            if getattr(self, name) is None:
                raise ValueError(f"`{name}` is required to fit the model.")

        x = self._inputs_model(self.inputs)

        outputs = self.outputs
        if self.where_model is not True:
            outputs = np.where(self.where_model, outputs, np.nan)

        # the samples of uncertain fields are fit separately, and the shape
        # of an uncertain array leaves out the axis of its samples, while an
        # axis which only the mask varies along is a batch axis, like one
        # which only the guess varies along
        axis = self.axis_model
        if axis is None:
            axis = tuple(na.shape_broadcasted(x, self.outputs, self.variance))

        return uncertainties._propagate(
            na.optimize._least_squares,
            arguments=dict(
                x=x,
                data=outputs,
                variance=self.variance,
                guess=self.guess,
                lower=self.lower,
                upper=self.upper,
                prior_mean=self.prior_mean,
                prior_sigma=self.prior_sigma,
            ),
            model=self.model,
            axis=axis,
            correlation=self.correlation,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
        )

    @property
    def parameters(self) -> na.AbstractVectorArray:
        """
        The best-fit parameters, a vector of the same type, components and
        units as :attr:`guess`, which varies along the batch axes.
        """
        return self._fit.parameters

    @property
    def covariance(self) -> na.AbstractMatrixArray:
        """
        The covariance of the best-fit parameters, a matrix whose rows and
        columns are named after the parameters, so that
        ``covariance.center.width`` is the covariance of ``center`` and
        ``width``.

        Each element has the product of the units of its two parameters.
        A parameter which has no effect on the model and no prior has an
        infinite variance, and the covariance of a fit whose parameters the
        outputs and the priors cannot constrain is NaN, as described in
        :attr:`ndoptimize.LeastSquaresResult.covariance`.
        """
        return self._fit.covariance

    @property
    def chi2(self) -> na.AbstractScalar:
        """
        The chi-squared of the outputs at the solution, without the
        contribution of the priors.
        """
        return self._fit.chi2

    @property
    def num_points(self) -> na.AbstractScalar:
        """The number of points used by each fit."""
        return self._fit.num_points

    @property
    def iterations(self) -> na.AbstractScalar:
        """The number of steps taken by each fit."""
        return self._fit.iterations

    @property
    def converged(self) -> na.AbstractScalar:
        """Whether each fit converged."""
        return self._fit.converged
