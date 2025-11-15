"""
A wrapper around the :mod:`numexpr` package.
"""
from typing import Literal
import sys
import numexpr
import named_arrays as na

__all__ = [
    "evaluate",
]

numexpr.set_num_threads(numexpr.detect_number_of_cores())


def _getArguments(names, local_dict=None, global_dict=None, _frame_depth: int=2):
    """
    A copy of :func:`numexpr.necompiler.getArguments` which does not
    use :func:`numpy.asarray`.

    This was necessary since :func:`numpy.asarray` was prematurely stripping the units
    from the arguments.
    """
    call_frame = sys._getframe(_frame_depth)

    clear_local_dict = False
    if local_dict is None:
        local_dict = call_frame.f_locals
        clear_local_dict = True
    try:
        frame_globals = call_frame.f_globals
        if global_dict is None:
            global_dict = frame_globals

        # If `call_frame` is the top frame of the interpreter we can't clear its
        # `local_dict`, because it is actually the `global_dict`.
        clear_local_dict = clear_local_dict and not frame_globals is local_dict

        arguments = []
        for name in names:
            try:
                a = local_dict[name]
            except KeyError:
                a = global_dict[name]
            arguments.append(a)
    finally:
        # If we generated local_dict via an explicit reference to f_locals,
        # clear the dict to prevent creating extra ref counts in the caller's scope
        # See https://github.com/pydata/numexpr/issues/310
        if clear_local_dict and hasattr(local_dict, 'clear'):
            local_dict.clear()

    return arguments


def evaluate(
    ex: str,
    local_dict: None | dict = None,
    global_dict: None | dict = None,
    order: str = 'K',
    casting: str = 'same_kind',
    sanitize: None | bool = None,
    optimization: Literal["none", "moderate", "aggressive"] = "aggressive",
    truediv: bool | Literal["auto"] = "auto",
):
    r"""
    A wrapper around :func:`numexpr.evaluate`.

    Evaluates a mathematical expression element-wise using the virtual machine.

    Parameters
    ----------
    ex
        a string forming an expression, like "2*a+3*b". The values for "a"
        and "b" will by default be taken from the calling function's frame
        (through use of sys._getframe()). Alternatively, they can be specified
        using the 'local_dict' or 'global_dict' arguments.

    local_dict
        A dictionary that replaces the local operands in current frame.

    global_dict
        A dictionary that replaces the global operands in current frame.

    order: {'C', 'F', 'A', or 'K'}, optional
        Controls the iteration order for operands. 'C' means C order, 'F'
        means Fortran order, 'A' means 'F' order if all the arrays are
        Fortran contiguous, 'C' order otherwise, and 'K' means as close to
        the order the array elements appear in memory as possible.  For
        efficient computations, typically 'K'eep order (the default) is
        desired.

    casting: {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur when making a copy or
        buffering.  Setting this to 'unsafe' is not recommended, as it can
        adversely affect accumulations.

          * 'no' means the data types should not be cast at all.
          * 'equiv' means only byte-order changes are allowed.
          * 'safe' means only casts which can preserve values are allowed.
          * 'same_kind' means only safe casts or casts within a kind,
            like float64 to float32, are allowed.
          * 'unsafe' means any data conversions may be done.

    sanitize: bool
        `validate` (and by extension `evaluate`) call `eval(ex)`, which is
        potentially dangerous on non-sanitized inputs. As such, NumExpr by default
        performs simple sanitization, banning the characters ':;[', the
        dunder '__[\w+]__', and attribute access to all but '.real' and '.imag'.

        Using `None` defaults to `True` unless the environment variable
        `NUMEXPR_SANITIZE=0` is set, in which case the default is `False`.
        Nominally this can be set via `os.environ` before `import numexpr`.

    optimization
        The optimization level of the compiler

    truediv
        Whether to use integer or floating-point division.
    """

    kwargs = dict(
        optimization=optimization,
        truediv=truediv,
    )

    context = numexpr.necompiler.getContext(kwargs)

    names = numexpr.necompiler.getExprNames(
        text=ex,
        context=context,
        sanitize=sanitize,
    )[0]

    args = _getArguments(
        names=names,
        local_dict=local_dict,
        global_dict=global_dict,
        _frame_depth=2,
    )

    arrays = {name: na.as_named_array(a) for name, a in zip(names, args)}

    return na._named_array_function(
        func=evaluate,
        ex=ex,
        order=order,
        casting=casting,
        sanitize=sanitize,
        optimization=optimization,
        truediv=truediv,
        **arrays,
    )
