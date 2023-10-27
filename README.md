# named-arrays

[![tests](https://github.com/sun-data/named-arrays/actions/workflows/tests.yml/badge.svg)](https://github.com/sun-data/named-arrays/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/sun-data/named-arrays/graph/badge.svg?token=1GhdcsgwO0)](https://codecov.io/gh/sun-data/named-arrays)
[![Documentation Status](https://readthedocs.org/projects/named-arrays/badge/?version=latest)](https://named-arrays.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/named-arrays.svg)](https://badge.fury.io/py/named-arrays)

`named-arrays` is an implementation of a [named tensor](https://nlp.seas.harvard.edu/NamedTensor), which assigns names to each axis of an n-dimensional array such as a numpy array.

When using a Numpy n-dimensional array, the programmer must manually keep track of the physical meaning of each axis in the array.
Furthermore, it is often necessary to insert singleton dimensions at the end of the array to allow it to broadcastable against other arrays.
Named tensors solve this problem by giving each axis a name, which allows for automatic axis alignment without the need for inserting extra dimensions.
`named-arrays` provides a very unapologetic implementation of a named tensor, since axes can _only_ be accessed using their names,
unlike [`xarray`](https://github.com/pydata/xarray) which allows for both name and index.

## Features

 - [`ScalarArray`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.ScalarArray.html), a named tensor implementation with Astropy Quantity support. Analogue of [`xarray.Variable`](https://docs.xarray.dev/en/latest/generated/xarray.Variable.html)
 - [`UncertainScalarArray`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.UncertainScalarArray.html), a named tensor implementation with automatic uncertainty propagation.
 - [`Cartesian2dVectorArray`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.Cartesian2dVectorArray.html), representation of a 2D vector.
 - [`Cartesian3dVectorArray`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.Cartesian3dVectorArray.html), representation of a 3D vector.
 - [`FunctionArray`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.FunctionArray.html), representation of a discrete function. Analogue of an [`xarray.DataArray`](https://docs.xarray.dev/en/latest/generated/xarray.DataArray.html#xarray.DataArray)


## Installation
`named-arrays` is available on PyPi and can be installed using pip
```bash
pip install named-arrays
```

## Examples

### ScalarArray
The fundamental type of `named-arrays` is the [`ScalarArray`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.ScalarArray.html), which is a composition of a numpy ndarray-like object and a tuple of axis names which must have the same length as the number of dimensions in the array.


```python
import numpy as np
import named_arrays as na

a = na.ScalarArray(np.array([1, 2, 3]), axes=('x',))
```

If we create another array with a different axis name, it will be broadcasted automatically against the first array if we add them together


```python
b = na.ScalarArray(np.array([4, 5]), axes=('y',))
c = a + b
c
```




    ScalarArray(
        ndarray=[[5, 6],
                 [6, 7],
                 [7, 8]],
        axes=('x', 'y'),
    )



All the usual numpy reduction operations use the axis name instead of the axis index


```python
c.mean('x')
```




    ScalarArray(
        ndarray=[6., 7.],
        axes=('y',),
    )



To index the array we can use a dictionary with the axis names as the keys


```python
c[dict(x=0)]
```




    ScalarArray(
        ndarray=[5, 6],
        axes=('y',),
    )



### ScalarLinearSpace
We recommend that you rarely directly create instances of [`ScalarArray`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.ScalarArray.html) directly. Instead, you can use the implicit array classes: [`ScalarLinearSpace`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.ScalarLinearSpace.html), [`ScalarLogarithmicSpace`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.ScalarLogarithmicSpace.html), and [`ScalarGeometricSpace`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.ScalarGeometricSpace.html) to create arrays in a similar fashion to [`numpy.linspace()`](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html), [`numpy.logspace()`](https://numpy.org/doc/stable/reference/generated/numpy.logspace.html#numpy.logspace), and [`numpy.geomspace()`](https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html#numpy.geomspace) with the advantage of being able to access the inputs to these functions at a later point.


```python
d = na.ScalarLinearSpace(0, 1, axis='z', num=4)
d
```




    ScalarLinearSpace(start=0, stop=1, axis='z', num=4, endpoint=True)



Thses implicit array classes work just like `ScalarArray` and can be used with any of the usual array operations.


```python
a + d
```




    ScalarArray(
        ndarray=[[1.        , 1.33333333, 1.66666667, 2.        ],
                 [2.        , 2.33333333, 2.66666667, 3.        ],
                 [3.        , 3.33333333, 3.66666667, 4.        ]],
        axes=('x', 'z'),
    )




```python

```
