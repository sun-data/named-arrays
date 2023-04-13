from typing import Callable, Sequence
import numpy as np
import astrpy.units as u
import named_arrays as na

def array_function_default(
        func: Callable,
        a: na.AbstractFunctionArray,
        axis: None | str | Sequence[str] = None,
        dtype: None | type | np.dtype = np._NoValue,
        out: None | na.AbstractFunctionArray = None,
        keepdims: bool = False,
        initial: None | bool | int | float | complex | u.Quantity = np._NoValue,
        where: na.AbstractFunctionArray = np._NoValue,
) -> na.FunctionArray:

    a = a.explicit

    if where is not np._NoValue:
        if np.any(a.inputs != where.inputs):
            raise na.InputValueError("`where.inputs` must match `a.inputs`")



    # shape = na.shape_broadcasted(a, where)
    #
    # axis_normalized = tuple(shape) if axis is None else (axis,) if isinstance(axis, str) else axis