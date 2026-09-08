"""
A Sphinx directive which tabulates the functions each array type supports.

The table is built from the dispatch tables of :mod:`named_arrays` itself, so
it describes the version which built the page rather than a list someone has to
remember to update.

The rows are emitted as a ``list-table`` and parsed, rather than written as
HTML, so that the result is an ordinary table as far as Sphinx is concerned and
the theme styles it in both its light and its dark mode.
"""

import importlib
from typing import Callable

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList

__all__ = [
    "SupportTableDirective",
    "setup",
]

#: The dispatch module of each array type, for functions of the ``numpy``
#: namespace, which are reached through ``__array_function__``.
MODULES_NUMPY = {
    "Scalar": "_scalars.scalar_array_functions",
    "Uncertain": "_scalars.uncertainties.uncertainties_array_functions",
    "Vector": "_vectors.vector_array_functions",
    "Function": "_functions.function_array_functions",
}

#: The dispatch module of each array type, for functions of the
#: ``named_arrays`` namespace.
MODULES_NAMED = {
    "Scalar": "_scalars.scalar_named_array_functions",
    "Uncertain": "_scalars.uncertainties.uncertainties_named_array_functions",
    "Vector": "_vectors.vector_named_array_functions",
    "Function": "_functions.function_named_array_functions",
}

#: The module whose table every array type consults, for functions of the
#: ``numpy`` namespace.
MODULE_SHARED = "_core_array_functions"

SUPPORTED = "\N{CHECK MARK}"
UNSUPPORTED = ""


def _supported(module: str) -> set[Callable]:
    """Every callable named by a ``*_FUNCTIONS`` table in a dispatch module."""
    result = set()
    module = importlib.import_module(f"named_arrays.{module}")
    for name, value in vars(module).items():
        if not name.endswith("_FUNCTIONS"):
            continue
        entries = value.keys() if isinstance(value, dict) else value
        result |= {f for f in entries if callable(f)}
    return result


def _names_named_arrays() -> dict[Callable, str]:
    """
    Where each function of the :mod:`named_arrays` namespace is exposed.

    A function is defined in a private module but exposed either at the top
    level or in one of the public submodules, and it is the latter name which
    the reader would write and which the reference documents, so it is found by
    searching those namespaces rather than by reading ``__module__``.
    """
    import named_arrays

    result = dict()
    namespaces = [("named_arrays", named_arrays)]
    for name in getattr(named_arrays, "__all__", ()):
        member = getattr(named_arrays, name, None)
        if isinstance(member, type(named_arrays)):
            namespaces.append((f"named_arrays.{name}", member))

    for prefix, namespace in namespaces:
        for name in dir(namespace):
            if name.startswith("_"):
                continue
            member = getattr(namespace, name, None)
            if callable(member) and not isinstance(member, type):
                result.setdefault(member, f"{prefix}.{name}")

    return result


_NAMES_NAMED_ARRAYS = None


def _name(function: Callable) -> str:
    """The qualified name of a function, as someone would write it."""
    module = getattr(function, "__module__", "") or ""
    if module.startswith("numpy"):
        return f"{module}.{function.__name__}"

    global _NAMES_NAMED_ARRAYS
    if _NAMES_NAMED_ARRAYS is None:
        _NAMES_NAMED_ARRAYS = _names_named_arrays()

    return _NAMES_NAMED_ARRAYS.get(function, function.__name__)


class SupportTableDirective(Directive):
    """
    Tabulate the functions of a namespace against the array types.

    Takes one argument, either ``numpy`` or ``named_arrays``, naming the
    namespace to tabulate.
    """

    required_arguments = 1

    def run(self) -> list[nodes.Node]:
        namespace = self.arguments[0]

        if namespace == "numpy":
            modules, shared = MODULES_NUMPY, _supported(MODULE_SHARED)
        elif namespace == "named_arrays":
            modules, shared = MODULES_NAMED, set()
        else:  # pragma: nocover
            raise self.error(
                f"the namespace must be `numpy` or `named_arrays`, "
                f"got {namespace}"
            )

        supported = {k: _supported(v) | shared for k, v in modules.items()}
        functions = sorted(set().union(*supported.values()), key=_name)

        lines = [
            ".. list-table::",
            "    :header-rows: 1",
            "    :widths: auto",
            "",
            "    * - Function",
        ]
        lines += [f"      - {k}" for k in supported]
        for function in functions:
            lines.append(f"    * - :func:`{_name(function)}`")
            lines += [
                f"      - {SUPPORTED if function in supported[k] else UNSUPPORTED}"
                for k in supported
            ]

        node = nodes.Element()
        self.state.nested_parse(
            StringList(lines, source=""),
            self.content_offset,
            node,
        )
        return node.children


def setup(app):
    app.add_directive("support-table", SupportTableDirective)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
