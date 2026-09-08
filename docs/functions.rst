Supported functions
===================

:mod:`named_arrays` implements the :mod:`numpy` protocols
``__array_function__`` and ``__array_ufunc__``, so most of the :mod:`numpy`
API accepts these arrays directly, using axis names where :mod:`numpy` would
take an axis number.
Coverage is not complete, and it is not uniform across the array types, since
each type implements each function separately.

The tables below are generated from the dispatch tables themselves, so they
describe the version of :mod:`named_arrays` which built this page rather than
a list someone remembered to update.
A check mark means the array type supports the function and an empty cell means
it does not, and a function which is absent is supported by no array type at
all.
If you need one which is missing, an issue or a pull request is welcome at
`the repository <https://github.com/sun-data/named-arrays>`_.

Every :mod:`numpy` universal function, such as :func:`numpy.sqrt` or
:func:`numpy.add`, is supported by every array type through
``__array_ufunc__``, and is not listed here.

A matrix is a vector whose components are themselves vectors, so it supports
whatever the ``Vector`` column supports.

Functions in the :mod:`numpy` namespace
---------------------------------------

.. support-table:: numpy

Functions in the :mod:`named_arrays` namespace
----------------------------------------------

.. support-table:: named_arrays
