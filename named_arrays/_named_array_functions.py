from __future__ import annotations
from typing import Sequence, overload, Type, Any, Callable, TypeVar, Literal
import functools
import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    '_named_array_function',
    'asarray',
    'asanyarray',
    'arange',
    'step',
    'linspace',
    'logspace',
    'geomspace',
    'ndim',
    'shape',
    'unit',
    'unit_normalized',
    'broadcast_to',
    'stack',
    'concatenate',
    'add_axes',
    "interp",
    "histogram2d",
    'jacobian',
    'despike',
]

ArrayT = TypeVar("ArrayT")
LikeT = TypeVar("LikeT", bound="None | na.AbstractArray")
AxisT = TypeVar("AxisT", bound="str | na.AbstractArray")
NumT = TypeVar("NumT", bound="int | na.AbstractArray")
BaseT = TypeVar("BaseT", bound="int | na.AbstractArray")
InputT = TypeVar("InputT", bound="float | u.Quantity | na.AbstractScalarArray")
OutputT = TypeVar("OutputT", bound="float | u.Quantity | na.AbstractScalarArray")


def _is_subclass(a: Any, b: Any):
    if type(a) == type(b):
        return 0
    elif isinstance(a, type(b)):
        return 1
    elif isinstance(b, type(a)):
        return -1
    else:
        return 0


def _named_array_function(func: Callable, *args, **kwargs):
    arrays_args = tuple(arg for arg in args if isinstance(arg, na.AbstractArray))
    arrays_kwargs = tuple(kwargs[k] for k in kwargs if isinstance(kwargs[k], na.AbstractArray))
    arrays = arrays_args + arrays_kwargs

    arrays = sorted(arrays, key=functools.cmp_to_key(_is_subclass))

    for array in arrays:
        res = array.__named_array_function__(func, *args, **kwargs)
        if res is not NotImplemented:
            return res

    raise TypeError(
        f"all types, {[type(a).__qualname__ for a in arrays]}, returned `NotImplemented` "
        f"for function `{func.__name__}`"
    )


def _asarray_like(
        func: Callable,
        a: ArrayT,
        dtype: None | type | np.dtype | str = None,
        order: None | str = None,
        *,
        like: None | LikeT = None,
) -> ArrayT | LikeT:

    if a is None:
        return None

    if like is None:
        like = na.ScalarArray(None)
    elif not na.named_array_like(like):
        like = func(like)

    return _named_array_function(
        func=func,
        a=a,
        dtype=dtype,
        order=order,
        like=like,
    )


@overload
def asarray(
        a: ArrayT,
        dtype: None | type | np.dtype | str = ...,
        order: None | str = ...,
        *,
        like: None = ...
) -> ArrayT:
    ...


@overload
def asarray(
        a: ArrayT,
        dtype: None | type | np.dtype | str = ...,
        order: None | str = ...,
        *,
        like: LikeT = ...
) -> LikeT:
    ...


def asarray(
        a: ArrayT,
        dtype: None | type | np.dtype | str = None,
        order: None | str = None,
        *,
        like: None | LikeT = None,
) -> ArrayT | LikeT:
    """
    Converts the input to use only instances of :class:`numpy.ndarray` as the underlying data.

    This function does not convert an instance of :class:`named_arrays.AbstractArray` to an instance of
    :class:`numpy.ndarray` like you might expect from the documentation of :func:`numpy.asarray`.
    Instead, it recursively inspects the input, converting instances of :class:`named_arrays.AbstractImplicitArray`
    to `named_arrays.AbstractExplicitArray`, and calling :func:`numpy.asarray` on the underlying data.

    Parameters
    ----------
    a
        Input array to be converted
    dtype
        Data type of output, usually inferred from the input.
    order
        Memory layout. See the documentation of :func:`numpy.asarray` for more information.
    like
        Optional reference object.
        If provided, the result will be defined by this object.

    Returns
    -------
    out
        Standardized interpretation of ``a``, with all the underlying data expressed as instances of
        :class:`numpy.ndarray`.

    Examples
    --------

    Standardize a :class:`float`

    .. jupyter-execute::

        import named_arrays as na

        na.asarray(2)

    Standardize a :class:`named_arrays.ScalarArray` of :class:`float`

    .. jupyter-execute::

        na.asarray(na.ScalarArray(2))

    Standardize a :class:`named_arrays.Cartesian2dVectorArray` of :class:`float`

    .. jupyter-execute::

        na.asarray(na.Cartesian2dVectorArray(2, 3))

    See Also
    --------
    :func:`numpy.asarray` : Equivalent Numpy function
    :func:`named_arrays.asanyarray` : Similar to this function, but allows `numpy.ndarray` subclasses to pass through.
    """
    return _asarray_like(
        func=asarray,
        a=a,
        dtype=dtype,
        order=order,
        like=like,
    )


@overload
def asanyarray(
        a: ArrayT,
        dtype: None | type | np.dtype | str = ...,
        order: None | str = ...,
        *,
        like: None = ...
) -> ArrayT:
    ...


@overload
def asanyarray(
        a: ArrayT,
        dtype: None | type | np.dtype | str = ...,
        order: None | str = ...,
        *,
        like: LikeT = ...
) -> LikeT:
    ...


def asanyarray(
        a: ArrayT,
        dtype: None | type | np.dtype | str = None,
        order: None | str = None,
        *,
        like: None | LikeT = None,
) -> ArrayT | LikeT:
    """
    Converts the input to use only instances of :class:`numpy.ndarray` subclasses as the underlying data.

    This function does not convert an instance of :class:`named_arrays.AbstractArray` to an instance of a
    :class:`numpy.ndarray` subclass like you might expect from the documentation of :func:`numpy.asanyarray`.
    Instead, it recursively inspects the input, converting instances of :class:`named_arrays.AbstractImplicitArray`
    to `named_arrays.AbstractExplicitArray`, and calling :func:`numpy.asanyarray` on the underlying data.

    Parameters
    ----------
    a
        Input array to be converted
    dtype
        Data type of output, usually inferred from the input.
    order
        Memory layout. See the documentation of :func:`numpy.asanyarray` for more information.
    like
        Optional reference object.
        If provided, the result will be defined by this object.

    Returns
    -------
    out
        Standardized interpretation of ``a``, with all the underlying data expressed as instances of
        :class:`numpy.ndarray` subclasses.

    Examples
    --------

    Standardize an instance of :class:`astropy.units.Quantity`

    .. jupyter-execute::

        import astropy.units as u
        import named_arrays as na

        na.asanyarray(2 * u.mm)

    See Also
    --------
    :func:`numpy.asanyarray` : Equivalent Numpy function
    :func:`named_arrays.asarray`: Similar to this function but converts instances of :class:`numpy.ndarray` subclasses
        back to instances of :class:`numpy.ndarray`.
    """

    return _asarray_like(
        func=asanyarray,
        a=a,
        dtype=dtype,
        order=order,
        like=like,
    )


def arange(
        start: float | complex | u.Quantity | na.AbstractArray,
        stop: float | complex | u.Quantity | na.AbstractArray,
        axis: str | na.AbstractArray,
        step: int | na.AbstractArray = 1,
) -> na.AbstractExplicitArray:
    """
    Redefined version of :func:`numpy.arange` with an ``axis`` parameter.

    Parameters
    ----------
    start
        starting value of the sequence
    stop
        ending value of the sequence
    axis
        name of the new sequence axis
    step
        step size between consecutive elements of the sequence

    See Also
    --------
    :func:`numpy.arange` : Equivalent numpy function

    :class:`named_arrays.ScalarArrayRange` : Implicit array version of this function for scalars
    """
    return _named_array_function(
        func=arange,
        start=na.as_named_array(start),
        stop=na.as_named_array(stop),
        axis=axis,
        step=step,
    )


def step(
    start: na.StartT,
    stop: na.StopT,
    num: NumT,
    endpoint: bool = True,
    centers: bool = False,
) -> na.StartT | na.StopT | NumT:
    """
    Helper function to compute the step size for :func:`linspace`.

    Parameters
    ----------
    start
        The starting value of the sequence.
    stop
        The last value of the sequence, unless :attr:`endpoint` is :class:`False`.
    num
        The number of values in the sequence.
    endpoint
        Flag controlling whether :attr:`stop` should be included in the sequence.
    centers
        Flag controlling whether the returned values should be on cell centers.
        The default is to return values on cell edges.
    """
    if endpoint:
        num = num - 1
    if centers:
        num = num + 1
    result = (stop - start) / num
    return result


def linspace(
        start: na.StartT,
        stop: na.StopT,
        axis: AxisT,
        num: NumT = 50,
        endpoint: bool = True,
        dtype: None | type | np.dtype = None,
        centers: bool = False,
) -> na.StartT | na.StopT | AxisT | NumT:
    """
    Create an array of evenly-spaced numbers between :attr:`start` and :attr:`stop`

    Parameters
    ----------
    start
        The starting value of the sequence
    stop
        The last value of the sequence, unless :attr:`endpoint` is :class:`False`
    axis
        The name of the new axis corresponding to the sequence
    num
        Number of values in the sequence
    endpoint
        Flag controlling whether :attr:`stop` should be included in the sequence.
    dtype
        :class:`numpy.dtype` of the result
    centers
        Flag controlling whether the returned values should be on cell centers.
        The default is to return values on cell edges.

    See Also
    --------
    :func:`numpy.linspace` :  Corresponding numpy function.

    :class:`named_arrays.ScalarLinearSpace` : Corresponding implicit scalar array.

    :class:`named_arrays.UncertainScalarLinearSpace`: Corresponding implicit uncertain scalar array.

    Examples
    --------

    Compare the points returned using ``centers=False`` to those returned
    using ``centers=True``.

    .. jupyter-execute::

        import named_arrays as na
        import matplotlib.pyplot as plt

        # Define an array of cell edges
        edges_x = na.linspace(0, 1, axis="x", num=11)
        edges_y = na.linspace(0, 1, axis="y", num=11)

        # Define an array of cell centers
        centers_x = na.linspace(0, 1, axis="x", num=10, centers=True)
        centers_y = na.linspace(0, 1, axis="y", num=10, centers=True)

        # Plot the results as a scatterplot
        fig, ax = plt.subplots()
        na.plt.scatter(edges_x, edges_y, ax=ax, label="edges");
        na.plt.scatter(centers_x, centers_y, ax=ax, label="centers");
        ax.legend();
    """

    if centers:
        halfstep = step(
            start=start,
            stop=stop,
            num=num,
            endpoint=endpoint,
            centers=centers,
        ) / 2
        start = start + halfstep
        stop = stop - halfstep

    return np.linspace(
        start=na.as_named_array(start),
        stop=stop,
        axis=axis,
        num=num,
        endpoint=endpoint,
        dtype=dtype,
    )


def logspace(
        start: na.StartT,
        stop: na.StopT,
        axis: AxisT,
        num: NumT = 50,
        endpoint: bool = True,
        base: BaseT = 10,
        dtype: None | type | np.dtype = None,
) -> na.StartT | na.StopT | AxisT | NumT:
    """
    Create an array of evenly-spaced numbers on a log scale between :attr:`start` and :attr:`stop`

    Parameters
    ----------
    start
        The starting value of the sequence
    stop
        The last value of the sequence, unless :attr:`endpoint` is :class:`False`
    axis
        The name of the new axis corresponding to the sequence
    num
        Number of values in the sequence
    endpoint
        Flag controlling whether :attr:`stop` should be included in the sequence.
    base
        The base of the logarithm
    dtype
        :class:`numpy.dtype` of the result

    See Also
    --------
    :func:`numpy.logspace` :  Corresponding numpy function.

    :class:`named_arrays.ScalarLogarithmicSpace` : Corresponding implicit scalar array.

    :class:`named_arrays.UncertainScalarLogarithmicSpace`: Corresponding implicit uncertain scalar array.
    """
    return np.logspace(
        start=na.as_named_array(start),
        stop=stop,
        axis=axis,
        num=num,
        endpoint=endpoint,
        base=base,
        dtype=dtype,
    )


def geomspace(
        start: na.StartT,
        stop: na.StopT,
        axis: AxisT,
        num: NumT = 50,
        endpoint: bool = True,
        dtype: None | type | np.dtype = None,
) -> na.StartT | na.StopT | AxisT | NumT:
    """
    Create an array of a geometric progression of numbers between :attr:`start` and :attr:`stop`

    Parameters
    ----------
    start
        The starting value of the sequence
    stop
        The last value of the sequence, unless :attr:`endpoint` is :class:`False`
    axis
        The name of the new axis corresponding to the sequence
    num
        Number of values in the sequence
    endpoint
        Flag controlling whether :attr:`stop` should be included in the sequence.
    dtype
        :class:`numpy.dtype` of the result

    See Also
    --------
    :func:`numpy.geomspace` :  Corresponding numpy function.

    :class:`named_arrays.ScalarGeometricSpace` : Corresponding implicit scalar array.

    :class:`named_arrays.UncertainScalarGeometricSpace`: Corresponding implicit uncertain scalar array.
    """
    return np.geomspace(
        start=na.as_named_array(start),
        stop=stop,
        axis=axis,
        num=num,
        endpoint=endpoint,
        dtype=dtype,
    )


def ndim(a: na.AbstractArray) -> int:
    return np.ndim(a)


def shape(a: na.ArrayLike) -> dict[str, int]:
    if not isinstance(a, na.AbstractArray):
        a = na.ScalarArray(a)
    return np.shape(a)


def unit(
        a: Any,
        unit_dimensionless: None | float | u.UnitBase = None,
        squeeze: bool = True,
) -> None | u.UnitBase | na.AbstractArray:
    """
    Isolate the physical units associated with the given object.

    If the array has no physical units, this function returns :obj:`None`.

    Parameters
    ----------
    a
        object to isolate the units of
    unit_dimensionless
        The unit to use for dimensionless objects.
    squeeze
        If the result is an instance of :class:`named_arrays.AbstractVectorArray`,
        and all the components are the same, simplify the result into a single
        :class:`astropy.units.Unit` instance.

    See Also
    --------
    :func:`unit_normalized` : version of this function that by default returns
        :obj:`astropy.units.dimensionless_unscaled` instead of :obj:`None`
        if there is no unit associated with the given object.
    """
    if isinstance(a, u.UnitBase):
        return a
    elif isinstance(a, u.Quantity):
        return a.unit
    elif isinstance(a, na.AbstractArray):
        return na._named_array_function(
            func=unit,
            a=a,
            unit_dimensionless=unit_dimensionless,
            squeeze=squeeze,
        )
    else:
        return unit_dimensionless


def unit_normalized(
        a: Any,
        unit_dimensionless: float | u.UnitBase = u.dimensionless_unscaled,
        squeeze: bool = True,
) -> u.UnitBase | na.AbstractArray:
    """
    Isolate the physical units associated with a given object,`
    normalizing to the given dimensionless units if the object does not have
    associated units.


    Parameters
    ----------
    a
        object to isolate the units of
    unit_dimensionless
        The unit to use for dimensionless objects.
    squeeze
        If the result is an instance of :class:`named_arrays.AbstractVectorArray`,
        and all the components are the same, simplify the result into a single
        :class:`astropy.units.Unit` instance.

    See Also
    --------
    :func:`unit` : version of this function that by default returns :obj:`None`
        instead of :obj:`astropy.units.dimensionless_unscaled` if there is no
        unit associated with the given object.

    """
    if isinstance(a, u.UnitBase):
        return a
    elif isinstance(a, u.Quantity):
        return a.unit
    elif isinstance(a, na.AbstractArray):
        return na._named_array_function(
            func=unit_normalized,
            a=a,
            unit_dimensionless=unit_dimensionless,
            squeeze=squeeze,
        )
    else:
        return unit_dimensionless


@overload
def broadcast_to(array: float | complex | np.ndarray | u.Quantity, shape: dict[str, int]) -> na.ScalarArray:
    ...


@overload
def broadcast_to(array: na.AbstractScalarArray, shape: dict[str, int]) -> na.ScalarArray:
    ...


def broadcast_to(array, shape):
    if not isinstance(array, na.AbstractArray):
        array = na.ScalarArray(array)
    return np.broadcast_to(array=array, shape=shape)


def stack(
        arrays: Sequence[na.ArrayLike],
        axis: str,
        out: None | na.AbstractExplicitArray = None,
        *,
        dtype: str | np.dtype | Type = None,
        casting: None | str = "same_kind",
) -> na.AbstractArray:
    if not any(isinstance(a, na.AbstractArray) for a in arrays):
        arrays = list(arrays)
        arrays[0] = na.ScalarArray(arrays[0])
    return np.stack(
        arrays=arrays,
        axis=axis,
        out=out,
        dtype=dtype,
        casting=casting,
    )


def concatenate(
        arrays: Sequence[na.ArrayLike],
        axis: str,
        out: None | na.AbstractExplicitArray = None,
        *,
        dtype: str | np.dtype | Type = None,
        casting: None | str = "same_kind",
) -> na.AbstractArray:
    if not any(isinstance(a, na.AbstractArray) for a in arrays):
        arrays = list(arrays)
        arrays[0] = na.ScalarArray(arrays[0])
    return np.concatenate(
        arrays=arrays,
        axis=axis,
        out=out,
        dtype=dtype,
        casting=casting,
    )


def add_axes(array: na.ArrayLike, axes: str | Sequence[str]):
    if not isinstance(array, na.AbstractArray):
        array = na.ScalarArray(array)
    return array.add_axes(axes)


def interp(
        x: float | u.Quantity | na.AbstractArray,
        xp:  na.AbstractArray,
        fp: na.AbstractArray,
        axis: None | str = None,
        left: None | float | u.Quantity | na.AbstractArray = None,
        right: None | float | u.Quantity | na.AbstractArray = None,
        period: None | float | u.Quantity | na.AbstractArray = None,
) -> na.AbstractArray:
    """
    Thin wrapper around :func:`numpy.interp`.

    Performs 1D interpolation on monotonically-increasing sample points along
    the specified axes.

    This function adds an ``axis`` argument to allow for interpolating
    :math:`n`-dimensional arrays.

    Parameters
    ----------
    x
         The new :math:`x` coordinates where the interpolant will be evaluated.
    xp
        The :math:`x` coordinates of the data points.
    fp
        The :math:`y` coordinates of the data points.
    axis
        The logical axis along which to interpolate.
    left
        Value to return for points less than ``xp[{axis: 0}]``.
        Default is ``fp[{axis: 0}]``
    right
        Value to return for points larger than ``xp[{axis: ~0}]``
    period
        A period for the :math:`x` coordinates.
        This parameter allows for proper interpolation of angular coordinates
    """
    return _named_array_function(
        func=interp,
        x=x,
        xp=xp,
        fp=fp,
        axis=axis,
        left=left,
        right=right,
        period=period,
    )


def histogram2d(
    x: na.AbstractScalarArray,
    y: na.AbstractScalarArray,
    bins: dict[str, int] | na.AbstractCartesian2dVectorArray,
    axis: None | str | Sequence[str] = None,
    min: None | na.AbstractScalarArray | na.AbstractCartesian2dVectorArray = None,
    max: None | na.AbstractScalarArray | na.AbstractCartesian2dVectorArray = None,
    density: bool = False,
    weights: None | na.AbstractScalarArray = None,
) -> na.FunctionArray[na.Cartesian2dVectorArray, na.ScalarArray]:
    """
    A thin wrapper around :func:`numpy.histogram2d` which adds an `axis` argument.

    Parameters
    ----------
    x
        An array containing the x coordinates of the points to be sampled.
    y
        An array containing the y coordinates of the points to be sampled.
    bins
        The bin specification of the histogram:
         * If `bins` is a dictionary, the keys are interpreted as the axis names
           and the values are the number of bins along each axis.
         * If `bins` is a 2D Cartesian vector, each component of the vector
           represents the bin edges in each dimension.
    axis
        The logical axes along which to histogram the data points.
        If :obj:`None` (the default), the histogram will be computed along
        all the axes of `x` and `y`.
    min
        The lower boundary of the histogram along each dimension.
        If :obj:`None` (the default), the minimum of `x` and `y` is used.
    max
        The upper boundary of the histogram along each dimension.
        If :obj:`None` (the default), the maximum of `x` and `y` is used.
    density
        If :obj:`False` (the default), returns the number of samples in each bin.
        If :obj:`True`, returns the probability density in each bin.
    weights
        An optional array weighting each sample.

    Examples
    --------

    Construct a 2D histogram with constant bin width.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import named_arrays as na

        # Define the bin edges
        bins = dict(x=6, y=5)

        # Define random points to collect into a histogram
        x = na.random.normal(0, 2, shape_random=dict(h=101))
        y = na.random.normal(0, 3, shape_random=dict(h=101))

        # Compute the 2D histogram
        hist = na.histogram2d(x, y, bins=bins)

        # Plot the resulting histogram
        fig, ax = plt.subplots()
        na.plt.pcolormesh(C=hist);

    Construct a 2D histogram with variable bin width.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import named_arrays as na

        # Define the bin edges
        bins = na.Cartesian2dVectorArray(
            x=np.square(na.linspace(0, 2, axis="x", num=6)),
            y=np.square(na.linspace(0, 2, axis="y", num=5)),
        )

        # Define random points to collect into a histogram
        x = na.random.normal(0, 2, shape_random=dict(h=101))
        y = na.random.normal(0, 3, shape_random=dict(h=101))

        # Compute the 2D histogram
        hist = na.histogram2d(x, y, bins=bins)

        # Plot the resulting histogram
        fig, ax = plt.subplots()
        na.plt.pcolormesh(C=hist);
    """
    return _named_array_function(
        func=histogram2d,
        x=x,
        y=y,
        axis=axis,
        bins=bins,
        min=min,
        max=max,
        density=density,
        weights=weights,
    )


def jacobian(
        function: Callable[[InputT], OutputT],
        x: InputT,
        dx: None | InputT = None,
) -> na.AbstractMatrixArray:
    """
    Compute the Jacobian of the given function using the first-order finite difference method.

    Parameters
    ----------
    function
        The function to compute the Jacobian of
    x
        The point to evaluate the function
    dx
        The distance that ``x`` will be perturbed by to compute the slope
    """

    if dx is None:
        dx = 1e-10
        unit_x = na.unit(x)
        if unit_x is not None:
            dx = dx * unit_x
    dx = na.asanyarray(dx, like=x)

    f_x = function(x)

    return na._named_array_function(
        func=jacobian,
        function=function,
        x=x,
        dx=dx,
        like=f_x,
    )


def despike(
    array: ArrayT,
    axis: tuple[str, str],
    where: None | bool | na.AbstractArray = None,
    inbkg: None | na.AbstractArray = None,
    invar: None | float | ArrayT = None,
    sigclip: float = 4.5,
    sigfrac: float = 0.3,
    objlim: float = 5.0,
    gain: float = 1.0,
    readnoise: float = 6.5,
    satlevel: float = 65536.0,
    niter: int = 4,
    sepmed: bool = True,
    cleantype: Literal["median", "medmask", "meanmask", "idw"] = "meanmask",
    fsmode: Literal["median", "convolve"] = "median",
    psfmodel: Literal["gauss", "gaussx", "gaussy", "moffat"] = "gauss",
    psffwhm: float = 2.5,
    psfsize: int = 7,
    psfk: None | na.AbstractArray = None,
    psfbeta: float = 4.765,
    verbose: bool = False,
) -> ArrayT:
    """
    A thin wrapper around :func:`astroscrappy.detect_cosmics`
    :cite:t:`vanDokkum2001`, which removes cosmic ray spikes from a series of
    images.

    Parameters
    ----------
    array
        Input data array that will be used for cosmic ray detection. This
        should include the sky background (or a mean background level, added
        back in after sky subtraction), so that noise can be estimated
        correctly from the data values. This should be in units of "counts".
    axis
        The two axes defining the logical axes of each image.
    where
        A boolean array of which pixels to consider during the cleaning
        process. The inverse of `inmask` used in
        :func:`astroscrappy.detect_cosmics`.
    inbkg
        A pre-determined background image, to be subtracted from `array`
        before running the main detection algorithm.
        This is used primarily with spectroscopic data, to remove
        sky lines and the cross-section of an object continuum during
        iteration, "protecting" them from spurious rejection (see the above
        paper). This background is not removed from the final, cleaned output
        (`cleanarr`). This should be in units of "counts", the same units of `array`.
        This inbkg should be free from cosmic rays. When estimating the cosmic-ray
        free noise of the image, we will treat ``inbkg`` as a constant Poisson
        contribution to the variance.
    invar
        A pre-determined estimate of the data variance (ie. noise squared) in
        each pixel, generated by previous processing of `array`. If provided,
        this is used in place of an internal noise model based on `array`,
        ``gain`` and ``readnoise``. This still gets median filtered and cleaned
        internally, to estimate what the noise in each pixel *would* be in the
        absence of cosmic rays. This should be in units of "counts" squared.
    sigclip
        Laplacian-to-noise limit for cosmic ray detection. Lower values will
        flag more pixels as cosmic rays. Default: 4.5.
    sigfrac
        Fractional detection limit for neighboring pixels. For cosmic ray
        neighbor pixels, a lapacian-to-noise detection limit of
        sigfrac * sigclip will be used. Default: 0.3.
    objlim
        Minimum contrast between Laplacian image and the fine structure image.
        Increase this value if cores of bright stars are flagged as cosmic
        rays. Default: 5.0.
    gain
        Gain of the image (electrons / ADU). We always need to work in
        electrons for cosmic ray detection. Default: 1.0
    readnoise
        Read noise of the image (electrons). Used to generate the noise model
        of the image. Default: 6.5.
    satlevel
        Saturation of level of the image (electrons). This value is used to
        detect saturated stars and pixels at or above this level are added to
        the mask. Default: 65536.0.
    niter
        Number of iterations of the LA Cosmic algorithm to perform. Default: 4.
    sepmed
        Use the separable median filter instead of the full median filter.
        The separable median is not identical to the full median filter, but
        they are approximately the same and the separable median filter is
        significantly faster and still detects cosmic rays well. Default: True
    cleantype
        Set which clean algorithm is used:\n
        'median': An umasked 5x5 median filter\n
        'medmask': A masked 5x5 median filter\n
        'meanmask': A masked 5x5 mean filter\n
        'idw': A masked 5x5 inverse distance weighted interpolation\n
        Default: "meanmask".
    fsmode
        Method to build the fine structure image:\n
        'median': Use the median filter in the standard LA Cosmic algorithm
        'convolve': Convolve the image with the psf kernel to calculate the
        fine structure image using a matched filter technique.
        Default: 'median'.
    psfmodel
        Model to use to generate the psf kernel if fsmode == 'convolve' and
        psfk is None. The current choices are Gaussian and Moffat profiles.
        'gauss' and 'moffat' produce circular PSF kernels. The 'gaussx' and
        'gaussy' produce Gaussian kernels in the x and y directions
        respectively. Default: "gauss".
    psffwhm
        Full Width Half Maximum of the PSF to use to generate the kernel.
        Default: 2.5.
    psfsize
        Size of the kernel to calculate. Returned kernel will have size
        psfsize x psfsize. psfsize should be odd. Default: 7.
    psfk
        PSF kernel array to use for the fine structure image if
        fsmode == 'convolve'. If None and fsmode == 'convolve', we calculate
        the psf kernel using 'psfmodel'. Default: None.
    psfbeta
        Moffat beta parameter. Only used if fsmode=='convolve' and
        psfmodel=='moffat'. Default: 4.765.
    verbose
        Print to the screen or not. Default: False.
    """
    return na._named_array_function(
        despike,
        array=array,
        axis=axis,
        where=where,
        inbkg=inbkg,
        invar=invar,
        sigclip=sigclip,
        sigfrac=sigfrac,
        objlim=objlim,
        gain=gain,
        readnoise=readnoise,
        satlevel=satlevel,
        niter=niter,
        sepmed=sepmed,
        cleantype=cleantype,
        fsmode=fsmode,
        psfmodel=psfmodel,
        psffwhm=psffwhm,
        psfsize=psfsize,
        psfk=psfk,
        psfbeta=psfbeta,
        verbose=verbose,
    )
