# named-arrays

[![tests](https://github.com/sun-data/named-arrays/actions/workflows/tests.yml/badge.svg)](https://github.com/sun-data/named-arrays/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/sun-data/named-arrays/graph/badge.svg?token=1GhdcsgwO0)](https://codecov.io/gh/sun-data/named-arrays)
[![Ruff](https://github.com/sun-data/named-arrays/actions/workflows/ruff.yml/badge.svg?branch=main)](https://github.com/sun-data/named-arrays/actions/workflows/ruff.yml)
[![Documentation Status](https://readthedocs.org/projects/named-arrays/badge/?version=latest)](https://named-arrays.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/named-arrays.svg)](https://badge.fury.io/py/named-arrays)

`named-arrays` is an implementation of a [named tensor](https://nlp.seas.harvard.edu/NamedTensor), which assigns a name to each axis of an n-dimensional array such as a numpy array.

With a bare numpy array, the meaning of each axis lives in the programmer's head, and combining two arrays usually means inserting singleton dimensions until their shapes line up.
Naming the axes removes both problems: arrays broadcast against each other by matching names, so a singleton dimension is never needed, and an operation such as a mean along the wavelength axis says exactly that.

`named-arrays` provides a very unapologetic implementation of a named tensor, since axes can _only_ be accessed using their names,
unlike [`xarray`](https://github.com/pydata/xarray) which allows for both name and index.
Support for [`astropy.units`](https://docs.astropy.org/en/stable/units/index.html) is built in, so the values inside an array can carry a physical unit.

## Installation

`named-arrays` is available on PyPI and can be installed using pip
```bash
pip install named-arrays
```

## Features

The array types form a hierarchy, from a plain named tensor up to a discrete function of several variables.

- [`ScalarArray`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.ScalarArray.html), a named tensor with Astropy Quantity support. Analogue of [`xarray.Variable`](https://docs.xarray.dev/en/latest/generated/xarray.Variable.html). Implicit variants such as [`ScalarLinearSpace`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.ScalarLinearSpace.html) describe an array without materializing it.
- [`UncertainScalarArray`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.UncertainScalarArray.html), which carries a distribution alongside the nominal value and propagates uncertainty through every operation.
- [`Cartesian2dVectorArray`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.Cartesian2dVectorArray.html) and [`Cartesian3dVectorArray`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.Cartesian3dVectorArray.html), along with named-component variants such as [`SpectralPositionalVectorArray`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.SpectralPositionalVectorArray.html), where each component is itself any of the array types above.
- [`Cartesian2dMatrixArray`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.Cartesian2dMatrixArray.html) and its relatives, which are vectors of vectors supporting the usual matrix operations.
- [`FunctionArray`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.FunctionArray.html), a discrete function pairing `inputs` with `outputs`. Analogue of an [`xarray.DataArray`](https://docs.xarray.dev/en/latest/generated/xarray.DataArray.html#xarray.DataArray).

Several modules extend these types to other libraries: `na.plt` for matplotlib, `na.random` and `na.stats` for sampling and statistics, `na.regridding` for resampling curvilinear grids, `na.optimize` for root finding and minimization, and `na.transformations` for rotations and translations.

## Key concepts

**The shape is a dictionary.**
`shape` maps each axis name to its length, and there is no positional equivalent.
Anywhere the numpy API takes an `axis=0`, this library takes an `axis="detector_x"`.

**Arrays broadcast by matching names.**
Two arrays combine along the axes whose names they share, and the axes unique to either one are added to the result.
An array of shape `{"x": 3}` plus an array of shape `{"y": 2}` therefore has shape `{"x": 3, "y": 2}`, with no reshaping and no singleton dimensions.
Adding a new dimension to a calculation is a matter of giving an input an extra named axis.

**Arrays are explicit or implicit.**
An explicit array such as `ScalarArray` stores its values.
An implicit array such as `ScalarLinearSpace` stores the arguments that define it, so `start`, `stop`, and `num` remain available long after the array is created.
Implicit arrays work in every operation an explicit array does, and `.explicit` materializes one on demand.

**Most of the numpy API already works.**
These arrays implement the `__array_function__` and `__array_ufunc__` protocols, so `np.mean`, `np.sqrt`, and most of their siblings accept them directly, using axis names.
Operations that numpy cannot express are defined in the `named_arrays` namespace instead.

## Documentation

The full documentation, including the API reference, tutorials, and executable versions of the examples below, is hosted at [named-arrays.readthedocs.io](https://named-arrays.readthedocs.io/en/latest).

## Examples

### Broadcasting by name

The fundamental type is the [`ScalarArray`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.ScalarArray.html), a composition of a numpy ndarray-like object and a tuple of axis names, which must have the same length as the number of dimensions in the array.

```python
import numpy as np
import named_arrays as na

a = na.ScalarArray(np.array([1, 2, 3]), axes=("x",))
b = na.ScalarArray(np.array([4, 5]), axes=("y",))
```

Since the two arrays have different axis names, adding them together broadcasts them against each other automatically.

```python
c = a + b
```
```
ScalarArray(
    ndarray=[[5, 6],
             [6, 7],
             [7, 8]],
    axes=('x', 'y'),
)
```

The result is two-dimensional, and its shape is a dictionary.

```python
c.shape
```
```
{'x': 3, 'y': 2}
```

All the usual numpy reduction operations take the name of the axis to remove.

```python
c.mean("x")
```
```
ScalarArray(
    ndarray=[6., 7.],
    axes=('y',),
)
```

To index the array, use a dictionary with the axis names as the keys, so the meaning of an index does not depend on the order of the axes.

```python
c[dict(x=0)]
```
```
ScalarArray(
    ndarray=[5, 6],
    axes=('y',),
)
```

### Implicit arrays

We recommend that you rarely create instances of `ScalarArray` directly.
Instead, use the implicit array classes [`ScalarLinearSpace`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.ScalarLinearSpace.html), [`ScalarLogarithmicSpace`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.ScalarLogarithmicSpace.html), and [`ScalarGeometricSpace`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.ScalarGeometricSpace.html), which mirror [`numpy.linspace()`](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html), [`numpy.logspace()`](https://numpy.org/doc/stable/reference/generated/numpy.logspace.html), and [`numpy.geomspace()`](https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html), with the advantage of remembering the arguments used to define them.

```python
d = na.ScalarLinearSpace(0, 1, axis="z", num=4)
```
```
ScalarLinearSpace(start=0, stop=1, axis='z', num=4, endpoint=True, centers=False)
```

These implicit classes work just like a `ScalarArray` in any operation, and `.explicit` materializes one on demand.

```python
a + d
```
```
ScalarArray(
    ndarray=[[1.        , 1.33333333, 1.66666667, 2.        ],
             [2.        , 2.33333333, 2.66666667, 3.        ],
             [3.        , 3.33333333, 3.66666667, 4.        ]],
    axes=('x', 'z'),
)
```

### One extra axis, one plotting call

An extra named axis costs nothing, so a family of curves is a single array, and one plotting call draws all of them.

```python
import astropy.units as u
import matplotlib.pyplot as plt

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
```
![plot](https://named-arrays.readthedocs.io/en/latest/_images/index_3_0.png)

### Uncertainty propagation

An [`UncertainScalarArray`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.UncertainScalarArray.html) carries a distribution alongside its nominal value, and every operation propagates it, so the error bar at the end of a calculation needs no separate bookkeeping.

```python
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
```

## Development

Install the package in editable mode along with its test dependencies, and run the test suite using [pytest](https://docs.pytest.org):
```bash
pip install -e .[test]
pytest
```

The suite is large, so continuous integration splits it into five groups using [pytest-split](https://github.com/jerry-git/pytest-split). To run one group:
```bash
pytest --splits 5 --group 1
```

This project is linted using [ruff](https://docs.astral.sh/ruff), which is checked by continuous integration:
```bash
ruff check .
```

To build the documentation locally:
```bash
pip install -e .[doc]
sphinx-build docs docs/_build/html
```
