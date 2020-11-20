#!/usr/bin/env python3

"""Module dedicated to the time simulation of reaction models.

Here are functions that calculate reaction rates as well, which is needed for
the time simulations.
"""

import logging

import numpy as np
from scipy.integrate import solve_ivp as _solve_ivp

from overreact import core as _core

logger = logging.getLogger(__name__)


def get_y(dydt, y0, t_span=None, method="Radau"):
    """Simulate a reaction scheme from its rate function.

    This uses scipy's ``solve_ivp`` under the hood.

    Parameters
    ----------
    dydt : callable
        Right-hand side of the system.
    y0 : array-like
        Initial state.
    t_span : array-like, optional
        Interval of integration (t0, tf). The solver starts with t=t0 and
        integrates until it reaches t=tf. If not given, a conservative value
        is chosen based on the system at hand (the method of choice works for
        any zeroth-, first- or second-order reactions).
    method : str, optional
        Integration method to use. See `scipy.integrade.solve_ivp` for details.
        If not sure, first try to run "RK45". If it makes unusually many
        iterations, diverges, or fails, your problem is likely to be stiff and
        you should use "BDF" or "Radau" (default).

    Returns
    -------
    y, r : callable
        Concentrations and reaction rates as functions of time. The y object
        is an OdeSolution and stores attributes t_min and t_max.


    Examples
    --------
    >>> import numpy as np
    >>> from overreact import core

    A toy simulation can be performed in just two lines:

    >>> scheme = core.parse_reactions("A <=> B")
    >>> y, r = get_y(get_dydt(scheme, [1, 1]), y0=[1, 0])

    The `y` object stores information about the simulation time, which can be
    used to produce a suitable vector of timepoints for, e.g., plotting:

    >>> y.t_min, y.t_max
    (0.0, 10.0)
    >>> t = np.linspace(y.t_min, y.t_max)
    >>> t
    array([ 0.        ,  0.20408163,  ...,  9.79591837, 10.        ])

    Both `y` and `r` can be used to check concentrations and rates in any
    point in time. In particular, both are vectorized:

    >>> y(t)
    array([[1.        , 0.83243215, ..., 0.50000008, 0.50000005],
           [0.        , 0.16756785, ..., 0.49999992, 0.49999995]])
    >>> r(t)
    array([[-1.00000000e+00, ..., -1.01639971e-07],
           [ 1.00000000e+00, ...,  1.01639971e-07]])
    """
    # TODO(schneiderfelipe): raise a meaningful error when y0 has the wrong shape.
    y0 = np.asanyarray(y0)

    if t_span is None:
        n_halflives = 10.0

        halflife_estimate = 1.0
        if hasattr(dydt, "k"):
            halflife_estimate = (
                np.max(
                    [
                        np.max(y0) / 2.0,  # zeroth-order half-life
                        np.log(2.0),  # first-order half-life
                        1.0 / np.min(y0[np.nonzero(y0)]),  # second-order half-life
                    ]
                )
                / np.min(dydt.k)
            )

        t_span = [
            0.0,
            n_halflives * halflife_estimate,
        ]
        logger.info(f"simulation time span = {t_span} s")

    res = _solve_ivp(dydt, t_span, y0, method=method, dense_output=True)
    y = res.sol

    def r(t):
        # TODO(schneiderfelipe): this is probably not the best way to
        # vectorize a function!
        try:
            return np.array([dydt(_t, _y) for _t, _y in zip(t, y(t).T)]).T
        except TypeError:
            return dydt(t, y(t))

    # TODO(schneiderfelipe): use a flag such as full_output to indicate we
    # want everything, not just y.
    return y, r


def get_dydt(scheme, k, ef=1.0e3):
    """Generate a rate function that models a reaction scheme.

    Parameters
    ----------
    scheme : Scheme
    k : array-like
        Reaction rate constant(s). Units match the concentration units given to
        the returned function ``dydt``.
    ef : float, optional

    Returns
    -------
    dydt : callable
        Reaction rate function. The actual reaction rate constants employed
        are stored in the attribute `k` of the returned function.

    Warns
    -----
    RuntimeWarning
        If the slowest half equilibrium is slower than the fastest non half
        equilibrium.

    Notes
    -----
    The returned function is suited to be used by ODE solvers such as
    `scipy.integrate.solve_ivp` or the older `scipy.integrate.ode` (see
    examples below). This is actually what the function `get_y` from the
    current module does.

    Examples
    --------
    >>> from overreact import core
    >>> scheme = core.parse_reactions("A <=> B")
    >>> dydt = get_dydt(scheme, [1, 1])
    >>> dydt(0.0, [1., 1.])
    array([0., 0.])

    The actually used reaction rate constants can be inspected with the `k`
    attribute of `dydt`:

    >>> dydt.k
    array([1, 1])

    """
    scheme = _core._check_scheme(scheme)
    is_half_equilibrium = np.asanyarray(scheme.is_half_equilibrium)
    k_adj = np.asanyarray(k).copy()
    A = np.asanyarray(scheme.A)

    # TODO(schneiderfelipe): this test for equilibria should go to get_k since
    # equilibria must obey the Collins-Kimball maximum reaction rate rule as
    # well.
    # TODO(schneiderfelipe): check whether we should filter RuntimeWarning.
    # TODO(schneiderfelipe): if there's only equilibria, should I want the
    # smallest one to be equal to one?
    if np.any(is_half_equilibrium) and np.any(~is_half_equilibrium):
        # TODO(schneiderfelipe): test those conditions
        k_adj[is_half_equilibrium] *= ef * (
            k_adj[~is_half_equilibrium].max() / k_adj[is_half_equilibrium].min()
        )

    def _dydt(t, y, k=k_adj, A=A):
        r = k * np.prod(np.power(y, np.where(A > 0, 0, -A).T), axis=1)
        return np.dot(A, r)

    _dydt.k = k_adj
    return _dydt
