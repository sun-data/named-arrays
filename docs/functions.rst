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
A function which is absent is not supported by any array type.
If you need one which is missing, an issue or a pull request is welcome at
`the repository <https://github.com/sun-data/named-arrays>`_.

Every :mod:`numpy` universal function, such as :func:`numpy.sqrt` or
:func:`numpy.add`, is supported by every array type through
``__array_ufunc__``, and is not listed here.

.. jupyter-execute::
    :hide-code:

    import importlib
    from IPython.display import HTML

    FAMILIES = {
        "Scalar": "_scalars.scalar_array_functions",
        "Uncertain": "_scalars.uncertainties.uncertainties_array_functions",
        "Vector": "_vectors.vector_array_functions",
        "Function": "_functions.function_array_functions",
    }
    FAMILIES_NAMED = {
        "Scalar": "_scalars.scalar_named_array_functions",
        "Uncertain": "_scalars.uncertainties.uncertainties_named_array_functions",
        "Vector": "_vectors.vector_named_array_functions",
        "Function": "_functions.function_named_array_functions",
    }


    def _supported(module: str) -> set:
        """Every callable named by a ``*_FUNCTIONS`` table in a dispatch module."""
        result = set()
        for name, value in vars(importlib.import_module(f"named_arrays.{module}")).items():
            if not name.endswith("_FUNCTIONS"):
                continue
            entries = value.keys() if isinstance(value, dict) else value
            result |= {f for f in entries if callable(f)}
        return result


    def _name(func) -> str:
        module = (getattr(func, "__module__", "") or "").replace("numpy._core.", "numpy.")
        if module.startswith("numpy"):
            module = module.split("_")[0].rstrip(".")
            return f"{module}.{func.__name__}"
        return func.__name__


    def _table(families: dict, shared: set) -> str:
        supported = {k: _supported(v) | shared for k, v in families.items()}
        functions = sorted(set().union(*supported.values()), key=_name)
        head = "".join(f"<th>{k}</th>" for k in supported)
        rows = ""
        for func in functions:
            cells = "".join(
                f"<td style='text-align:center'>{'&#10003;' if func in supported[k] else '&ndash;'}</td>"
                for k in supported
            )
            rows += f"<tr><td><code>{_name(func)}</code></td>{cells}</tr>"
        return f"<table><thead><tr><th>Function</th>{head}</tr></thead><tbody>{rows}</tbody></table>"


    HTML(
        "<h3>Functions in the <code>numpy</code> namespace</h3>"
        + _table(FAMILIES, _supported("_core_array_functions"))
        + "<h3>Functions in the <code>named_arrays</code> namespace</h3>"
        + _table(FAMILIES_NAMED, set())
    )

A matrix is a vector whose components are themselves vectors, so it supports
whatever the ``Vector`` column supports.
