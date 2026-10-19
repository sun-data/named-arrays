"""
Minimization, root finding and least-squares fitting.

Analogous to :mod:`scipy.optimize`.
"""

from __future__ import annotations
from typing import TypeVar, Callable, Sequence, TYPE_CHECKING
import dataclasses
import astropy.units as u
import named_arrays as na

if TYPE_CHECKING:
    import numpy.typing as npt
    import ndoptimize

__all__ = [
    "root_secant",
    "root_newton",
    "minimum_gradient_descent",
    "minimum_brent",
    "LeastSquaresResult",
    "least_squares",
    "evaluate",
]

InputT = TypeVar("InputT", bound="float | u.Quantity | na.AbstractScalarArray")
OutputT = TypeVar("OutputT", bound="float | u.Quantity | na.AbstractScalarArray")


def root_newton(
        function: Callable[[InputT], OutputT],
        guess: InputT,
        jacobian: None | Callable[[InputT], OutputT] = None,
        max_abs_error: None | OutputT = None,
        max_iterations: int = 100,
        callback: None | Callable[[int, InputT, OutputT, na.AbstractArray], None] = None,
):
    """
    Find the root of a given function using
    `Newton's method <https://en.wikipedia.org/wiki/Newton%27s_method>`_

    Parameters
    ----------
    function
        a scalar-valued or vector-valued function to find the roots of
    guess
        an initial guess for the root
    jacobian
        The Jacobian of the function (optional). If :obj:`None`, :func:`named_arrays.jacobian`
        is used.
    max_abs_error
        Maximum absolute error between zero and the function value.
        If the function value at the current root guess is below this value,
        the root is considered converged.
    max_iterations
        Maxmimum number of iterations to carry out before a :class:`ValueError`
        is raised.
    callback
        Optional callback function that is called on every iteration as
        ``callback(i, x, f, converged)``, where ``i`` is the current iteration,
        ``x`` is the current root guess, ``f`` is the current function value,
        and ``converged`` is an array storing the convergence state for every
        root being computed.
    """
    if jacobian is None:
        def jacobian(x: float | u.Quantity | na.AbstractArray):
            return na.jacobian(
                function=function,
                x=x,
            )

    f_guess = function(guess)
    if max_abs_error is None:
        max_abs_error = 1e-10
        if na.unit(f_guess) is not None:
            max_abs_error = max_abs_error * na.unit(f_guess)
    max_abs_error = na.asanyarray(max_abs_error, like=f_guess)

    return na._named_array_function(
        func=root_newton,
        function=function,
        guess=guess,
        jacobian=jacobian,
        max_abs_error=max_abs_error,
        max_iterations=max_iterations,
        callback=callback,
    )


def root_secant(
        function: Callable[[InputT], OutputT],
        guess: InputT,
        min_step_size: None | InputT = None,
        max_abs_error: None | OutputT = None,
        max_iterations: int = 100,
        damping: None | float = None,
        callback: None | Callable[[int, InputT, OutputT, na.AbstractArray], None] = None,
) -> InputT:
    """
    Find the root of a given function using the
    `secant method <https://en.wikipedia.org/wiki/Secant_method>`_.

    Parameters
    ----------
    function
        a scalar-valued or vector-valued function to find the root of.
    guess
        an initial guess for the root
    min_step_size
        Stop iterating if the step size falls below this value
    max_abs_error
        Maximum absolute error between zero and the function value.
        If the function value at the current root guess is below this value,
        the root is considered converged.
    max_iterations
        Maxmimum number of iterations to carry out before a :class:`ValueError`
        is raised.
    damping
        If this parameter is equal to ``1``, it has no effect and is equivalent
        to the default, :obj:`None` (vanilla secant method).
        If this parameter is less than ``1``, the damped secant method is
        carried out.
        The damped secant method can address some of the failure modes of the
        secant method, particularly the problem of
        `entering an infinite cycle <https://en.wikipedia.org/wiki/Newton%27s_method#Starting_point_enters_a_cycle>`_
    callback
        Optional callback function that is called on every iteration as
        ``callback(i, x, f, converged)``, where ``i`` is the current iteration,
        ``x`` is the current root guess, ``f`` is the current function value,
        and ``converged`` is an array storing the convergence state for every
        root being computed.
    """

    if min_step_size is None:
        min_step_size = 1e-13
        if na.unit(guess) is not None:
            min_step_size = min_step_size * na.unit(guess)
    min_step_size = na.asanyarray(min_step_size, like=guess)

    f_guess = function(guess)
    if max_abs_error is None:
        max_abs_error = 1e-13
        if na.unit(f_guess) is not None:
            max_abs_error = max_abs_error * na.unit(f_guess)
    max_abs_error = na.asanyarray(max_abs_error, like=f_guess)

    return na._named_array_function(
        func=root_secant,
        function=function,
        guess=guess,
        min_step_size=min_step_size,
        max_abs_error=max_abs_error,
        max_iterations=max_iterations,
        damping=damping,
        callback=callback,
    )


def minimum_gradient_descent(
    function: Callable[[InputT], OutputT],
    guess: InputT,
    step_size: None | InputT = None,
    momentum: float | OutputT = 0,
    gradient: None | Callable[[InputT], InputT] = None,
    min_gradient: None | InputT = None,
    max_iterations: int = 1000,
    callback: None | Callable[[int, InputT, OutputT, na.AbstractArray], None] = None,
) -> InputT:
    r"""
    Find the local minimum of the given function using the
    `gradient descent <https://en.wikipedia.org/wiki/Gradient_descent>`_ method.

    Parameters
    ----------
    function
        The function to minimize.
    guess
        An initial guess for the local minimum.
    step_size
        The learning rate for the gradient descent algorithm.
        This should have the same units as ``x / gradient(x)``.
        If :obj:`None` (the default), this takes the value
        ``0.01 * na.unit(x / gradient(x))``.
    momentum
        The momentum constant, :math:`\beta` for the gradient descent algorithm.
        Should be a dimensionless number between zero and one.
        Defaults to zero, which equivalent to vanilla gradient descent with
        no momentum.
    gradient
        The gradient of `function`.
        If :obj:`None` (the default), the gradient is computed using
        :func:`named_arrays.jacobian`.
    min_gradient
        The convergence threshold for the local minimum.
        If the gradient is smaller than this value, this function will stop.
        This should have the same units as ``gradient(x)``.
        If :obj:`None` (the default), this takes the value
        ``1e-10 * na.unit(gradient(x))``.
    max_iterations
        The maximum number of steps to take before raising an error.
    callback
        Optional callback function that is called on every iteration as
        ``callback(i, x, f, converged)``, where ``i`` is the current iteration,
        ``x`` is the current guess, ``f`` is the current function value,
        and ``converged`` is an array storing the convergence state for every
        minimum being computed.

    Notes
    -----

    This function uses the update rules described in :cite:t:`Goh2017`,

    .. math::
        :label: momentum-equation

        z_{k + 1} = \beta z_k + \nabla f(x_k)

    .. math::
        :label: gradient-descent

        x_{k + 1} = x_k - \alpha z_k,

    where :math:`x_k` is the current guess for iteration :math:`k`,
    :math:`f` is the objective function,
    :math:`\alpha` is the learning rate,
    and :math:`\beta` is the momentum constant.
    """

    x = guess
    f = function(x)

    unit_f = na.unit(f, unit_dimensionless=1)
    unit_x = na.unit(x, unit_dimensionless=1, squeeze=False)

    unit_grad = unit_f / unit_x

    if step_size is None:
        step_size = 0.01 * (unit_x / unit_grad)

    if gradient is None:
        def gradient(x: float | na.AbstractScalar | na.AbstractVectorArray):
            return na.jacobian(
                function=function,
                x=x,
            )

    if min_gradient is None:
        min_gradient = 1e-10 * unit_grad
    min_gradient = na.asanyarray(min_gradient, like=f)

    return na._named_array_function(
        func=minimum_gradient_descent,
        function=function,
        guess=guess,
        step_size=step_size,
        momentum=momentum,
        gradient=gradient,
        min_gradient=min_gradient,
        max_iterations=max_iterations,
        callback=callback,
    )


def minimum_brent(
    function: Callable[[InputT], OutputT],
    a: InputT,
    b: InputT,
    min_step_size: None | InputT = None,
    max_iterations: int = 100,
    callback: None | Callable[[int, InputT, OutputT, na.AbstractArray], None] = None,
) -> InputT:
    r"""
    Find the minimum of a function of one variable inside a bracket using
    `Brent's method <https://en.wikipedia.org/wiki/Brent%27s_method>`_.

    This is the bounded form of the algorithm (golden-section search with
    parabolic interpolation, as in :func:`scipy.optimize.fminbound`),
    which needs neither a derivative nor an initial guess, only a bracket
    :math:`[a, b]` containing the minimum.

    The bracket, the tolerance, and the value of `function` are broadcast
    against each other, and every element of the result converges
    independently: `function` is evaluated exactly once per iteration on the
    whole array of trial points, so a function that is expensive but
    vectorized (a ray trace, for example) costs the same for one minimum as
    for a thousand.

    Parameters
    ----------
    function
        The function to minimize.
        It must accept an array with the broadcast shape of `a` and `b`
        and return an array of the same shape (or one which broadcasts
        with it).
    a
        The lower bound of the bracket.
    b
        The upper bound of the bracket.
    min_step_size
        The absolute tolerance on the location of the minimum, in the same
        units as `a` and `b`.
        If :obj:`None` (the default), this takes the value
        ``1e-5 * na.unit(a)``.
    max_iterations
        The maximum number of iterations to carry out before a
        :class:`ValueError` is raised.
    callback
        Optional callback function that is called on every iteration as
        ``callback(i, x, f, converged)``, where ``i`` is the current
        iteration, ``x`` is the current best estimate of the minimum,
        ``f`` is the function value there, and ``converged`` is an array
        storing the convergence state of every minimum being computed.

    Examples
    --------
    Find the minimum of a family of V-shaped functions, one per element of
    the array of offsets, in a single vectorized call.

    .. jupyter-execute::

        import numpy as np
        import named_arrays as na

        offset = na.linspace(-1, 1, axis="offset", num=5)

        def function(x):
            return np.abs(x - offset) + 1

        na.optimize.minimum_brent(
            function=function,
            a=-2,
            b=2,
        )
    """
    if min_step_size is None:
        min_step_size = 1e-5
        if na.unit(a) is not None:
            min_step_size = min_step_size * na.unit(a)
    min_step_size = na.asanyarray(min_step_size, like=a)

    return na._named_array_function(
        func=minimum_brent,
        function=function,
        a=a,
        b=b,
        min_step_size=min_step_size,
        max_iterations=max_iterations,
        callback=callback,
    )


@dataclasses.dataclass(frozen=True, eq=False)
class LeastSquaresResult:
    """
    The results of :func:`least_squares` for every fit of a batch, and what
    is needed to evaluate the best-fit model with :meth:`evaluate`.

    Each array varies along the batch axes of the fit.
    If any argument of the fit was uncertain, the nominal values of the
    arguments and each of their samples are fit separately, and the results
    are uncertain arrays.
    """

    parameters: na.AbstractVectorArray
    """
    The best-fit parameters, a vector of the same type, components and units
    as the initial guess.
    """

    covariance: na.AbstractMatrixArray
    """
    The covariance of the best-fit parameters, a matrix whose rows and
    columns are named after the parameters, so that
    ``covariance.center.width`` is the covariance of ``center`` and
    ``width``.

    A parameter which has no effect on the model and no prior has an infinite
    variance, and the covariance of a fit whose parameters the data and the
    priors cannot constrain is NaN, as described in
    :attr:`ndoptimize.LeastSquaresResult.covariance`.
    """

    chi2: na.AbstractScalarArray
    """
    The chi-squared of the data at the solution, without the contribution of
    the priors.
    """

    num_points: na.AbstractScalarArray
    """The number of data points used by each fit."""

    iterations: na.AbstractScalarArray
    """The number of steps taken by each fit."""

    converged: na.AbstractScalarArray
    """Whether each fit converged."""

    model: ndoptimize.Model
    """The model which was fit."""

    x: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray
    """The independent variable of the fit."""

    axis: tuple[str, ...]
    """The axes along which the data points of each fit vary."""

    unit: None | u.UnitBase
    """The unit of the data, and so of the model, or :obj:`None` if it has none."""

    def evaluate(
        self,
        x: None | float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray = None,
        axis: None | str | Sequence[str] = None,
    ) -> na.AbstractScalar:
        """
        Evaluate the best-fit model of every fit, using :func:`evaluate`.

        Parameters
        ----------
        x
            The points at which to evaluate the model.
            If :obj:`None` (the default), the independent variable of the
            fit, so that the result is the model of the data.
        axis
            The axis or axes along which the points of `x` vary.
            If :obj:`None` (the default), the axes of the fit.
        """
        return evaluate(
            model=self.model,
            x=self.x if x is None else x,
            parameters=self.parameters,
            axis=self.axis if axis is None else axis,
            unit=self.unit,
        )


def least_squares(
    model: ndoptimize.Model,
    x: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
    data: float | u.Quantity | na.AbstractScalar,
    variance: float | u.Quantity | na.AbstractScalar,
    guess: na.AbstractVectorArray,
    axis: str | Sequence[str],
    lower: None | na.AbstractVectorArray = None,
    upper: None | na.AbstractVectorArray = None,
    prior_mean: None | na.AbstractVectorArray = None,
    prior_sigma: None | na.AbstractVectorArray = None,
    correlation: None | npt.ArrayLike = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
) -> LeastSquaresResult:
    r"""
    Fit a model to many independent sets of data at once, using the
    Levenberg-Marquardt method of :func:`ndoptimize.least_squares`.

    The data points of a single fit vary along `axis`, and every other axis
    of the arguments is a batch axis, along which each set of data gets a fit
    of its own.
    The parameters are the components of a vector, named after
    :attr:`ndoptimize.Model.parameters`.

    Each fit minimizes

    .. math::

        \chi^2 = (y - f)^T C^{-1} (y - f)
            + \sum_k \left( \frac{p_k - \bar{p}_k}{\sigma_k} \right)^2,

    where :math:`y` is the data, :math:`f` is the model, :math:`C` is the
    covariance of the noise, :math:`p_k` are the parameters, and
    :math:`\bar{p}_k` and :math:`\sigma_k` are the means and standard
    deviations of their Gaussian priors.

    The model is a compiled function of plain numbers, so it sees the values
    of `x` and `data` in their own units, and the value of each parameter in
    the unit of its initial guess.
    The variance is converted to the unit of the data squared, and the bounds
    and priors of each parameter to the unit of its guess.

    Any argument may be uncertain, including a component of a vector.
    Then the nominal values of the arguments and each of their samples are
    fit separately, and the results are uncertain arrays.

    Parameters
    ----------
    model
        The model to fit, made with :func:`ndoptimize.model`.
    x
        The independent variable.
        If it is a vector, its components, in order, are the inputs of the
        model; otherwise it is the single input.
    data
        The data to fit.
        Points which are NaN are left out of their fit.
    variance
        The variance of the noise of each data point.
        Points with a variance which is NaN or not positive are left out of
        their fit.
    guess
        The initial guess for the parameters, a vector whose components are
        the parameters named in :attr:`ndoptimize.Model.parameters`.
        It must not vary along `axis`, and neither may the bounds and priors.
    axis
        The logical axis or axes along which the data points of a single fit
        vary.
    lower
        The lower bounds of the parameters, a vector with a component for
        each parameter that has one.
        If :obj:`None` (the default), no parameter has a lower bound.
    upper
        The upper bounds of the parameters, in the same form as `lower`.
    prior_mean
        The means of the Gaussian priors of the parameters, a vector with a
        component for each parameter that has a prior.
    prior_sigma
        The standard deviations of the Gaussian priors of the parameters, in
        the same form as `prior_mean`.
        A parameter has a prior only if it has a component here.
    correlation
        The correlation of the noise of two data points by their separation
        along `axis`, in samples, starting with 1, as in
        :func:`ndoptimize.least_squares`.
        It requires a single `axis`.
        If :obj:`None` (the default), the noise is uncorrelated.
    max_iterations
        The maximum number of steps for each fit.
    tolerance
        The convergence tolerance on the Newton decrement, as in
        :func:`ndoptimize.least_squares`.

    Examples
    --------

    Fit a Gaussian to each of a hundred noisy spectral lines, with the
    parameters in physical units.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import ndoptimize

        # the true parameters of the lines
        shape = dict(position=100)
        truth = na.CartesianNdVectorArray(dict(
            amplitude=na.random.uniform(5, 10, shape, seed=0) * u.DN,
            center=na.random.normal(0, 10, shape, seed=1) * u.km / u.s,
            width=na.random.uniform(10, 20, shape, seed=2) * u.km / u.s,
        ))

        # the noisy spectra
        velocity = na.linspace(-100, 100, axis="velocity", num=41) * u.km / u.s
        z = (velocity - truth.center) / truth.width
        sigma = 0.5 * u.DN
        data = na.random.normal(truth.amplitude * np.exp(-z * z / 2), sigma, seed=3)

        # fit every spectrum
        result = na.optimize.least_squares(
            model=ndoptimize.models.gaussian,
            x=velocity,
            data=data,
            variance=np.square(sigma),
            guess=na.CartesianNdVectorArray(dict(
                amplitude=5 * u.DN,
                center=0 * u.km / u.s,
                width=15 * u.km / u.s,
            )),
            axis="velocity",
        )

        # the best-fit model of the first three spectra, on a finer grid
        index = dict(position=slice(0, 3))
        velocity_fine = na.linspace(-100, 100, axis="velocity_fine", num=201) * u.km / u.s
        fit = result.evaluate(x=velocity_fine, axis="velocity_fine")

        # compare the fitted centers to the truth
        error = result.parameters.center - truth.center
        uncertainty = np.sqrt(result.covariance.center.center)

        fig, ax = plt.subplots(ncols=2, figsize=(8, 3.5), constrained_layout=True)
        na.plt.scatter(velocity, data[index], ax=ax[0], color="gray");
        na.plt.plot(velocity_fine, fit[index], ax=ax[0], axis="velocity_fine");
        ax[0].set_xlabel(f"velocity ({ax[0].get_xlabel()})");
        ax[0].set_ylabel(f"intensity ({ax[0].get_ylabel()})");
        na.plt.scatter(truth.center, error / uncertainty, ax=ax[1]);
        ax[1].set_xlabel(f"true center ({ax[1].get_xlabel()})");
        ax[1].set_ylabel("(fitted center - true center) / uncertainty");
    """
    return na._named_array_function(
        func=least_squares,
        model=model,
        x=x,
        data=data,
        variance=variance,
        guess=guess,
        axis=axis,
        lower=lower,
        upper=upper,
        prior_mean=prior_mean,
        prior_sigma=prior_sigma,
        correlation=correlation,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


def evaluate(
    model: ndoptimize.Model,
    x: float | u.Quantity | na.AbstractScalar | na.AbstractVectorArray,
    parameters: na.AbstractVectorArray,
    axis: str | Sequence[str],
    unit: None | u.UnitBase = None,
) -> na.AbstractScalar:
    """
    Evaluate a model for many sets of parameters at once, using
    :func:`ndoptimize.evaluate`.

    The points of a single evaluation vary along `axis`, and every other axis
    of the arguments is a batch axis.
    :meth:`LeastSquaresResult.evaluate` calls this function with the
    best-fit parameters of :func:`least_squares`.

    The model is a compiled function of plain numbers, so it sees the values
    of `x` and of each parameter in their own units, and its result is given
    the unit `unit`.

    Any argument may be uncertain, including a component of a vector.
    Then the model is evaluated for the nominal values of the arguments and
    for each of their samples separately, and the result is an uncertain
    array.

    Parameters
    ----------
    model
        The model to evaluate, made with :func:`ndoptimize.model`.
    x
        The points at which to evaluate the model.
        If it is a vector, its components, in order, are the inputs of the
        model; otherwise it is the single input.
    parameters
        The parameters of the model, a vector whose components are the
        parameters named in :attr:`ndoptimize.Model.parameters`.
        It must not vary along `axis`.
    axis
        The logical axis or axes along which the points of a single
        evaluation vary.
    unit
        The unit of the values of the model.
        If :obj:`None` (the default), the result has no unit.

    Examples
    --------

    Evaluate Gaussians of three different widths.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import named_arrays as na
        import ndoptimize

        velocity = na.linspace(-50, 50, axis="velocity", num=101) * u.km / u.s

        parameters = na.CartesianNdVectorArray(dict(
            amplitude=1 * u.DN,
            center=0 * u.km / u.s,
            width=na.ScalarArray(np.array([5, 10, 20]), axes="line") * u.km / u.s,
        ))

        f = na.optimize.evaluate(
            model=ndoptimize.models.gaussian,
            x=velocity,
            parameters=parameters,
            axis="velocity",
            unit=u.DN,
        )

        fig, ax = plt.subplots(constrained_layout=True)
        na.plt.plot(velocity, f, ax=ax, axis="velocity", label=parameters.width);
        ax.legend();
    """
    return na._named_array_function(
        func=evaluate,
        model=model,
        x=x,
        parameters=parameters,
        axis=axis,
        unit=unit,
    )
