"""
A :mod:`named_arrays` wrapper around the :mod:`numexpr` package.
"""
from typing import Literal
import numexpr
import named_arrays as na

__all__ = [
    "evaluate",
]

def evaluate(
    ex: str,
    local_dict: None | dict = None,
    global_dict: None | dict = None,
    # out: numpy.ndarray = None,
    order: str = 'K',
    casting: str = 'same_kind',
    sanitize: None | bool = None,
    optimization: Literal["none", "moderate", "aggressive"] = "aggressive",
    truediv: bool | Literal["auto"] = "auto",
):

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

    args = numexpr.necompiler.getArguments(
        names=names,
        local_dict=local_dict,
        global_dict=global_dict,
        _frame_depth=2,
    )

    arrays = {name: a.item() for name, a in zip(names, args)}

    return na._named_array_function(
        func=evaluate,
        ex=ex,
        # local_dict=local_dict,
        # global_dict=global_dict,
        order=order,
        casting=casting,
        sanitize=sanitize,
        optimization=optimization,
        truediv=truediv,
        **arrays,
    )
