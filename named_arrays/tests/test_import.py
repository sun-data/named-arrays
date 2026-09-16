"""
Guard against slow imports of :mod:`named_arrays`.

Importing :mod:`named_arrays` should not import optional heavy dependencies
such as :mod:`matplotlib` or :mod:`numba`, which are only needed by a few
functions and should be imported lazily by those functions instead.
"""

import subprocess
import sys
import pytest

modules_heavy = [
    "astropy.time",
    "astropy.visualization",
    "colorsynth",
    "matplotlib",
    "mpl_toolkits",
    "ndfilters",
    "numba",
    "pandas",
    "regridding",
    "scipy",
    "xarray",
]
"""Modules that should not be imported by :mod:`named_arrays`."""


def _modules_loaded(code: str) -> set[str]:
    """
    Run `code` in a fresh interpreter and return the modules it imported.
    """
    code = f"{code}\nimport sys\nprint(*sys.modules)"
    result = subprocess.run(
        args=[sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(result.stdout.split())


@pytest.mark.parametrize(
    argnames="code",
    argvalues=[
        "import named_arrays",
        # Named-array functions load the per-type dispatch modules on first use,
        # so make sure those modules do not import the heavy dependencies either.
        "\n".join([
            "import named_arrays as na",
            "a = na.linspace(0, 1, axis='x', num=5)",
            "b = na.broadcast_to(a, dict(x=5, y=2))",
            "c = na.stack([a, a], axis='y')",
            "d = na.Cartesian2dVectorArray(a, a) + 1",
            "e = na.NormalUncertainScalarArray(a, width=0.1) + 1",
        ]),
    ],
)
def test_import_is_lazy(code: str):
    loaded = _modules_loaded(code)
    heavy = {
        module
        for module in loaded
        if module.split(".")[0] in modules_heavy or module in modules_heavy
    }
    assert not heavy
