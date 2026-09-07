named-arrays
============

:mod:`named_arrays` is an implementation of a
`named tensor <https://nlp.seas.harvard.edu/NamedTensor>`_
that includes first-class support for :mod:`astropy.units`.
Every axis of an array carries a name, and axes are referenced by that name
instead of by position, which allows for more readable code and better
modularity.

With a bare :class:`numpy.ndarray`, the meaning of each axis lives in the
programmer's head, and combining two arrays usually means inserting singleton
dimensions until their shapes line up.
Naming the axes removes both problems: arrays broadcast against each other by
matching names, so a singleton dimension is never needed, and an operation such
as a mean along the wavelength axis says exactly that.

:mod:`named_arrays` does `not` extend the :mod:`numpy` API like :mod:`xarray`.
Instead, it generalizes the :mod:`numpy` API to `only` use axis names instead of position.
This means that terms such as `shape`,
which referred to a :class:`tuple` of integers in the :mod:`numpy` API,
is now a :class:`dict`, where the keys are the axis names and the values
are the number of elements along that axis.
This forces consumers of this library to stick to the named axes,
and to not "cheat" by using positional indexing.

Many functions in the :mod:`numpy` API have been overridden if possible.
Other functions which are not expressible using the :mod:`numpy` API have
been redefined in the :mod:`named_arrays` namespace.

Installation
============
:mod:`named_arrays` is published on PyPI and can be installed using::

    pip install named-arrays


Features
========

The array types form a hierarchy, from a plain named tensor up to a discrete
function of several variables.

*   :class:`named_arrays.ScalarArray`, a named tensor with
    :mod:`astropy.units` support, the analogue of :class:`xarray.Variable`.
    Implicit variants such as :class:`named_arrays.ScalarLinearSpace` describe
    an array without materializing it, and remember the arguments used to
    define it.
*   :class:`named_arrays.UncertainScalarArray`, which carries a distribution
    alongside the nominal value and propagates uncertainty through every
    operation.
    :class:`named_arrays.NormalUncertainScalarArray` builds one from a width.
*   :class:`named_arrays.Cartesian2dVectorArray` and
    :class:`named_arrays.Cartesian3dVectorArray`, along with named-component
    variants such as :class:`named_arrays.SpectralPositionalVectorArray`, where
    each component is itself any of the array types above.
*   :class:`named_arrays.Cartesian2dMatrixArray` and its relatives, which are
    vectors of vectors, and support the usual matrix operations.
*   :class:`named_arrays.FunctionArray`, a discrete function pairing `inputs`
    with `outputs`, the analogue of :class:`xarray.DataArray`.

Several modules extend the array types to other libraries:
:mod:`named_arrays.plt` for :mod:`matplotlib`,
:mod:`named_arrays.random` and :mod:`named_arrays.stats` for sampling and
statistics,
:mod:`named_arrays.regridding` for resampling curvilinear grids,
:mod:`named_arrays.optimize` for root finding and minimization,
:mod:`named_arrays.transformations` for rotations and translations,
and :mod:`named_arrays.ndfilters`, :mod:`named_arrays.colorsynth`,
:mod:`named_arrays.geometry`, :mod:`named_arrays.pdf`, and
:mod:`named_arrays.numexpr`.


Key concepts
============

**The shape is a dictionary.**
:attr:`named_arrays.AbstractArray.shape` maps each axis name to its length,
and there is no positional equivalent.
Anywhere the :mod:`numpy` API takes an ``axis=0``, this library takes an
``axis="detector_x"``.

**Arrays broadcast by matching names.**
Two arrays combine along the axes whose names they share, and the axes unique
to either one are added to the result.
An array of shape ``{"x": 3}`` plus an array of shape ``{"y": 2}`` therefore
has shape ``{"x": 3, "y": 2}``, with no reshaping and no singleton dimensions.
Adding a new dimension to a calculation is a matter of giving an input an extra
named axis.

**Arrays are explicit or implicit.**
An explicit array such as :class:`named_arrays.ScalarArray` stores its values.
An implicit array such as :class:`named_arrays.ScalarLinearSpace` stores the
arguments that define it, so ``start``, ``stop``, and ``num`` remain available
long after the array is created.
Implicit arrays work in every operation an explicit array does, and
:attr:`named_arrays.AbstractArray.explicit` materializes one on demand.

**Units and uncertainties come along for the ride.**
The values inside an array can be an :class:`astropy.units.Quantity`, so a
dimensional error surfaces as an exception rather than a wrong number.
An :class:`named_arrays.UncertainScalarArray` carries a distribution which is
propagated through arithmetic, so an error bar at the end of a calculation
needs no separate bookkeeping.

**Most of the numpy API already works.**
These arrays implement the ``__array_function__`` and ``__array_ufunc__``
protocols, so :func:`numpy.mean`, :func:`numpy.sqrt`, and most of their
siblings accept them directly, using axis names.
Operations that :mod:`numpy` cannot express are defined in the
:mod:`named_arrays` namespace instead.


Examples
========

Arrays with different axis names broadcast against each other automatically,
and reductions take the name of the axis to remove.

.. jupyter-execute::

    import numpy as np
    import astropy.units as u
    import matplotlib.pyplot as plt
    import named_arrays as na

    a = na.ScalarArray(np.array([1, 2, 3]), axes=("x",))
    b = na.ScalarArray(np.array([4, 5]), axes=("y",))

    c = a + b
    c

.. jupyter-execute::

    c.mean("x")

|

Indexing uses a dictionary of axis names, so the meaning of an index does not
depend on the order of the axes.

.. jupyter-execute::

    c[dict(x=0)]

|

Since an extra named axis costs nothing, a family of curves is one array, and
one plotting call draws all of them.

.. jupyter-execute::

    # Define the independent variable
    x = na.linspace(0, 2 * np.pi, axis="x", num=101) * u.rad

    # Add an axis representing three different amplitudes
    amplitude = na.ScalarArray(np.array([1, 2, 3]), axes=("amplitude",))

    # The result has both axes, without any reshaping
    y = amplitude * np.sin(x)

    fig, ax = plt.subplots(constrained_layout=True);
    na.plt.plot(x, y, axis="x", ax=ax);
    ax.set_xlabel(f"angle ({x.unit:latex_inline})");
    ax.set_ylabel("amplitude");

|

Uncertainty is propagated through every operation, so the error bar at the end
of a calculation needs no separate bookkeeping.

.. jupyter-execute::

    # Define a radius known to about 5%
    radius = na.NormalUncertainScalarArray(
        nominal=10 * u.cm,
        width=0.5 * u.cm,
        num_distribution=11,
    )

    # Compute the area of the corresponding circle
    area = np.pi * np.square(radius)

    # The uncertainty in the radius is carried into the area
    area.nominal, np.std(area.distribution, axis="_distribution")

|


API Reference
=============

An in-depth description of the classes and functions defined in the this library.

.. autosummary::
    :toctree: _autosummary
    :template: module_custom.rst
    :recursive:

    named_arrays

Tutorials
=========

Jupyter notebook examples on how to use :mod:`named_arrays`.

.. toctree::
    :maxdepth: 1

    tutorials/indexing
    tutorials/PolynomialFunctionArray


References
==========

.. bibliography::

|


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
