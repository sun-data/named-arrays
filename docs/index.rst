Introduction
============
:mod:`named_arrays` is an implementation of a
`named tensor <https://nlp.seas.harvard.edu/NamedTensor>`_
that includes first-class support for :mod:`astropy.units`.
Named tensors allow the axes of an array to be referenced by name
instead of by position, which allows for more readable code and better
modularity.

:mod:`named_arrays` does `not` extend the :mod:`numpy` API like :mod:`xarray`.
Instead, it generalizes the :mod:`numpy` API to `only` use axis names instead of position.
This means that terms such as `shape`,
which referred to a :class:`tuple` of integers in the :mod:`numpy` API,
is now a :class:`dict`, where the keys are the axis names and the values
are the number of elements along that axis.
This forces consumers of this library to stick to the named axes,
and to not "cheat" by using positional indexing.

Most functions of the :mod:`numpy` API have been overloaded if possible.
Other functions which require

Installation
============
:mod:`named_arrays` is published on PyPI and can be installed using::

    pip install named-arrays



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
