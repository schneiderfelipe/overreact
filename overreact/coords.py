"""Module dedicated to classifying molecules into point groups."""

# TODO(schneiderfelipe): add types to this module
from __future__ import annotations

__all__ = ["find_point_group", "symmetry_number"]


import logging
import re

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree as KDTree
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.transform import Rotation

import overreact as rx
from overreact import _constants as constants
from overreact import _misc as _misc

logger = logging.getLogger(__name__)


# TODO(schneiderfelipe): alpha should depend on temperature?
def get_molecular_volume(
    atomnos,
    atomcoords,
    full_output=False,
    environment="water",
    method="garza",
    temperature=298.15,
    pressure=constants.atm,
    alpha=1.2,
    num=250,
    trials=3,
):
    """Calculate van der Waals volumes.

    Volume estimation is done through Quasi-Monte Carlo integration. As such,
    the computed volume is accurate to about two significant figures. This is
    sufficient to most applications.

    Parameters
    ----------
    atomnos : array-like
    atomcoords : array-like
    full_output : bool, optional
        If True, return an estimate of the cavity volume and an estimate on the
        error as well.
    method : str, optional
        This is a placeholder for future functionality.
        There are plans to implement more sophisticated methods for calculating
        entropies such as in
        [*Phys. Chem. Chem. Phys.*, **2019**, 21, 18920-18929](https://doi.org/10.1039/C9CP03226F)
        and
        [*J. Chem. Theory Comput.* **2019**, 15, 5, 3204-3214](https://doi.org/10.1021/acs.jctc.9b00214).
        Head over to the
        [discussions](https://github.com/geem-lab/overreact/discussions) if
        you're interested and would like to contribute.
        Leave this as "standard" for now.
    environment : str, optional
    temperature : array-like, optional
        Absolute temperature in Kelvin.
    pressure : array-like, optional
        Reference gas pressure.
    alpha : float, optional
    num, trials : int, optional

    Returns
    -------
    vdw_volume : float
    cav_volume, err : float, optional
        Volumes returned are in Å³ per molecule.

    Raises
    ------
    ValueError
        If `method` is not recognized.

    Notes
    -----
    For "izato", see equation 3 of doi:10.1039/C9CP03226F for the conceptual
    details. There is theoretical support for the equation in the work of
    Eyring (doi:10.1021/j150380a007).

    Examples
    --------
    >>> from overreact import _datasets as datasets

    >>> data = datasets.logfiles["symmetries"]["dihydrogen"]
    >>> get_molecular_volume(data.atomnos, data.atomcoords)
    8.4
    >>> get_molecular_volume(data.atomnos, data.atomcoords, method="izato",
    ...                      full_output=True)
    (8.4, 13.7, 0.1)
    >>> get_molecular_volume(data.atomnos, data.atomcoords, full_output=True)
    (8.4, 61., 0.1)

    >>> data = datasets.logfiles["symmetries"]["water"]
    >>> get_molecular_volume(data.atomnos, data.atomcoords)
    18.
    >>> get_molecular_volume(data.atomnos, data.atomcoords, method="izato",
    ...                      full_output=True)
    (18., 29., 0.1)
    >>> get_molecular_volume(data.atomnos, data.atomcoords, full_output=True)
    (18., 92., 0.1)

    >>> data = datasets.logfiles["symmetries"]["benzene"]
    >>> get_molecular_volume(data.atomnos, data.atomcoords)
    80.
    >>> get_molecular_volume(data.atomnos, data.atomcoords, method="izato",
    ...                      full_output=True)  # doctest: +SKIP
    (80., 115., 0.1)
    >>> get_molecular_volume(data.atomnos, data.atomcoords, full_output=True)  # doctest: +SKIP
    (80., 240., 0.1)
    >>> get_molecular_volume(data.atomnos, data.atomcoords, full_output=True,
    ...                      environment="benzene")
    (80., 593., 0.1)
    """
    atomnos = np.atleast_1d(atomnos)
    _, _, atomcoords = inertia(np.ones_like(atomnos), atomcoords)
    vdw_radii = constants.vdw_radius(atomnos)

    v1 = atomcoords.min(axis=0) - alpha * vdw_radii.max()
    v2 = atomcoords.max(axis=0) + alpha * vdw_radii.max()
    box_volume = np.prod(v2 - v1)
    n = int(num * box_volume)

    vdw_volumes = []
    if full_output and method == "izato":
        cav_volumes = []
    for _ in range(trials):
        points = rx._misc.halton(n, 3)
        points = v1 + points * (v2 - v1)
        tree = KDTree(points)

        within_vdw = set()
        if full_output and method == "izato":
            within_cav = set()
        for i, atomcoord in enumerate(atomcoords):
            within_vdw.update(tree.query_ball_point(atomcoord, vdw_radii[i]))
            if full_output and method == "izato":
                within_cav.update(
                    tree.query_ball_point(atomcoord, alpha * vdw_radii[i]),
                )

        vdw_volumes.append((len(within_vdw) / n) * box_volume)
        if full_output and method == "izato":
            cav_volumes.append((len(within_cav) / n) * box_volume)

    vdw_volume = np.mean(vdw_volumes)
    vdw_err = np.std(vdw_volumes)
    logger.info(f"van der Waals volume = {vdw_volume} ± {vdw_err} Å³")
    if full_output:
        if method == "izato":
            cav_volume = np.mean(cav_volumes)
            cav_err = np.std(cav_volumes)
            logger.debug(
                f"Izato cavity volume = {cav_volume} ± {cav_err} Å³",
            )
            return (vdw_volume, cav_volume, max(vdw_err, cav_err))
        elif method == "garza":
            # TODO(schneiderfelipe): test for the following solvents: water,
            # pentane, hexane, heptane and octane.

            cav_volume = _garza(
                vdw_volume,
                environment,
                temperature=temperature,
                pressure=pressure,
            )
            logger.debug(f"Garza cavity volume = {cav_volume} Å³")
            return (vdw_volume, cav_volume, vdw_err)
        else:
            msg = f"unavailable method: '{method}'"
            raise ValueError(msg)
    return vdw_volume


def _garza(
    vdw_volume,
    environment="water",
    full_output=False,
    temperature=298.15,
    pressure=constants.atm,
):
    """Calculate cavity attributes according to A. Garza.

    This is mainly a helper function for calculating solvation entropy
    according to doi:10.1021/acs.jctc.9b00214.

    Parameters
    ----------
    vdw_volume : float
    environment : str, optional
    full_output : bool, optional
        If True, return all model estimates.
    temperature : array-like, optional
        Absolute temperature in Kelvin.
    pressure : array-like, optional
        Reference gas pressure.

    Returns
    -------
    cav_volume : float
    N_cav : float, optional
    ratio : float, optional

    Examples
    --------
    >>> _garza(1.0)
    24.32
    >>> _garza(1.0, full_output=True)
    (24.32, 1.815, 0.3507458)
    >>> _garza(10.0)
    66.51
    >>> _garza(10.0, full_output=True)
    (66.51, 1.0, 0.7556590)
    >>> _garza(100.0)
    279.6
    >>> _garza(100.0, full_output=True)
    (279.6, 1.0, 1.628018)

    >>> _garza(1.0, environment="benzene")
    131.
    >>> _garza(1.0, full_output=True, environment="benzene")
    (131., 3.35, 0.2317882509934295)
    >>> _garza(10.0, environment="benzene")
    243.
    >>> _garza(10.0, full_output=True, environment="benzene")
    (243., 3.29, 0.499372648682062)
    >>> _garza(100.0, environment="benzene")
    665.
    >>> _garza(100.0, full_output=True, environment="benzene")
    (665., 1.0, 1.07586575757374)
    """
    solvent = rx._misc._get_chemical(environment, temperature, pressure)

    # TODO(schneiderfelipe): things to do:
    # 1. check correctness of this function,
    # 2. check it is called correctly everywhere,
    # 3. create a complete abstraction of the solvent/molecular properties:

    solvent_volume = solvent.Van_der_Waals_volume / (
        constants.angstrom**3 * constants.N_A
    )
    r_free = np.cbrt(
        solvent.Vm / (constants.angstrom**3 * constants.N_A) - solvent_volume,
    )
    r_M = np.cbrt(vdw_volume)

    cav_volume = (r_M + r_free) ** 3
    if not full_output:
        return cav_volume
    r_S = np.cbrt(solvent_volume)
    ratio = r_M / r_S

    area_free = r_free**2
    area_S_total = r_S**2 + area_free

    x = max(area_free - r_M**2, 0.0) / area_S_total
    if np.isclose(x, 0.0):
        return cav_volume, 1.0, ratio

    N_x = 4.0 * np.cbrt(cav_volume) ** 2 / area_S_total
    return cav_volume, 1.0 + N_x * x / (1.0 - x), ratio


def symmetry_number(point_group):
    """Return rotational symmetry number for point group.

    This function has a set of the most common point groups precomputed, but is
    able to calculate the symmetry number if it is not found in known tables.
    Most precomputed values are from
    [*Theor Chem Account* **2007** 118, 813-826](https://doi.org/10.1007/s00214-007-0328-0).

    Parameters
    ----------
    point_group : str
        Point group symbol.

    Returns
    -------
    int
        Rotational symmetry number.

    Raises
    ------
    ValueError
        If point group is not found in precomputed values.

    Examples
    --------
    >>> symmetry_number("C4")
    4
    >>> symmetry_number("C4") == symmetry_number("C4h")
    True
    >>> symmetry_number("C6")
    6
    >>> symmetry_number("C6") == symmetry_number("C6v")
    True
    >>> symmetry_number("C6") == symmetry_number("C6h")
    True
    >>> symmetry_number("D2") == symmetry_number("Vh")
    True
    >>> symmetry_number("D4")
    8
    >>> symmetry_number("D6")
    12
    >>> symmetry_number("D6") == symmetry_number("D6d")
    True
    >>> symmetry_number("S6")
    3
    >>> symmetry_number("T")
    12
    """
    point_group = point_group.strip().lower()

    if point_group in {"c1", "ci", "cs", "c∞v", "k", "r3"}:
        symmetry_number = 1
    elif point_group in {"c2", "c2v", "c2h", "d∞h", "s4"}:
        symmetry_number = 2
    elif point_group in {"c4", "c4v", "c4h", "d2", "d2d", "d2h", "s8", "vh"}:
        symmetry_number = 4
    elif point_group in {
        "c12",
        "c12v",
        "c12h",
        "d6",
        "d6d",
        "d6h",
        "s24",
        "t",
        "td",
    }:
        symmetry_number = 12
    elif point_group in {
        "c24",
        "c24v",
        "c24h",
        "d12",
        "d12d",
        "d12h",
        "s48",
        "oh",
    }:
        symmetry_number = 24
    elif point_group in {
        "c60",
        "c60v",
        "c60h",
        "d30",
        "d30d",
        "d30h",
        "s120",
        "ih",
    }:
        symmetry_number = 60
    else:
        pieces = re.match(
            r"(?P<letter>[^\s]+)(?P<number>\d+)(?P<type>[^\s]+)?",
            point_group,
        ).groupdict()

        if pieces["letter"] == "c":
            symmetry_number = int(pieces["number"])
        elif pieces["letter"] == "d":
            symmetry_number = 2 * int(pieces["number"])
        elif pieces["letter"] == "s":
            symmetry_number = int(pieces["number"]) // 2
        else:
            msg = f"unknown point group: '{point_group}'"
            raise ValueError(msg)

    logger.info(f"symmetry number = {symmetry_number}")
    return symmetry_number


def find_point_group(atommasses, atomcoords, proper_axes=None, rtol=0.0, atol=1.0e-2):
    """Determine point group of structure.

    Parameters
    ----------
    atommasses : array-like
        Atomic masses in atomic mass units (amu).
    atomcoords : array-like
        Atomic coordinates.
    proper_axes : sequence of tuples of int, array-like, optional
        Proper symmetry axes of rotation.
    rtol : float, optional
        The relative tolerance parameter (see `numpy.isclose`).
    atol : float, optional
        The absolute tolerance parameter (see `numpy.isclose`).

    Returns
    -------
    str
        Point group symbol.

    Examples
    --------
    >>> find_point_group([1], [[0, 0, 1]])
    'K'
    >>> find_point_group([1, 1], [[0, 0, 1], [0, 0, 0]])
    'D∞h'
    >>> find_point_group([1.008, 35.45], [[0, 0, 1], [0, 0, 0]])
    'C∞v'
    >>> find_point_group([16, 12, 16], [[1, 0, 1], [1, 0, 0], [1, 0, -1]])
    'D∞h'
    >>> find_point_group([16, 12, 32], [[1, 1, 1], [1, 1, 0], [1, 1, -1]])
    'C∞v'
    >>> find_point_group([12, 12, 12, 12], [[1, 0, 0],
    ...                                     [0, 1, 0],
    ...                                     [0, 0, 0],
    ...                                     [1, 1, 0]])
    'D4h'

    """
    if len(atommasses) == 1:  # atom
        point_group = "K"
    elif len(atommasses) == 2:  # diatomic molecule
        point_group = "D∞h" if atommasses[0] == atommasses[1] else "C∞v"
    else:
        groups = _equivalent_atoms(atommasses, atomcoords)
        moments, axes, atomcoords = inertia(atommasses, atomcoords)
        rotor_class = _classify_rotor(moments)

        if rotor_class[1] == "linear":
            point_group = _find_point_group_linear(
                atomcoords,
                groups,
                rtol=rtol,
                atol=atol,
            )
        else:
            if proper_axes is None:
                proper_axes = _get_proper_axes(
                    atomcoords,
                    groups,
                    axes,
                    rotor_class,
                    rtol=rtol,
                    atol=atol,
                )

            if rotor_class[0] == "asymmetric" or not proper_axes:
                point_group = _find_point_group_asymmetric(
                    atomcoords,
                    groups,
                    axes,
                    rotor_class,
                    proper_axes,
                    rtol=rtol,
                    atol=atol,
                )
            elif rotor_class[0] == "spheric":
                point_group = _find_point_group_spheric(
                    atomcoords,
                    groups,
                    axes,
                    rotor_class,
                    proper_axes,
                    rtol=rtol,
                    atol=atol,
                )
            else:  # symmetric
                point_group = _find_point_group_symmetric(
                    atomcoords,
                    groups,
                    axes,
                    rotor_class,
                    proper_axes,
                    rtol=rtol,
                    atol=atol,
                )

    logger.info(f"point group = {point_group}")
    return point_group


def _find_point_group_linear(atomcoords, groups, rtol=0.0, atol=1.0e-2):
    """Find point group for linear rotors.

    Point groups searched for are: D∞h, C∞v.

    See find_point_group for information on parameters and return values.
    """
    if _has_inversion_center(atomcoords, groups, rtol=rtol, atol=atol):
        return "D∞h"
    else:
        return "C∞v"


def _find_point_group_spheric(
    atomcoords,
    groups,
    axes,
    rotor_class,
    proper_axes=None,
    rtol=0.0,
    atol=1.0e-2,
):
    """Find point group for spheric tops.

    Point groups searched for are: Td, Oh, Ih.
    I might eventually search for T, Th, O in the future.

    See find_point_group for information on parameters and return values.
    """
    if not _has_inversion_center(atomcoords, groups, rtol=rtol, atol=atol):
        return "Td"

    if proper_axes is None:
        proper_axes = _get_proper_axes(
            atomcoords,
            groups,
            axes,
            rotor_class,
            rtol=rtol,
            atol=atol,
        )

    for n, _ in proper_axes:
        if n == 5:
            return "Ih"
        elif n < 5:
            break
    return "Oh"

    # see https://www.chem.uci.edu/~lawm/9-28.pdf for more about high
    # symmetry groups
    # see too
    # http://web.mit.edu/5.03/www/readings/point_groups/point_groups.pdf

    # the employed workflow is loosely inspired by some articles:
    # 1. doi:10.1016/0097-8485(76)80004-6


def _find_point_group_asymmetric(
    atomcoords,
    groups,
    axes,
    rotor_class,
    proper_axes=None,
    rtol=0.0,
    atol=1.0e-2,
):
    """Find point group for asymmetric tops.

    Point groups searched for here are: C1, Ci, Cs.
    Point groups delegated are: Cn, Cnh, Cnv, Dn, Dnh, Dnd, Sn.

    See find_point_group for information on parameters and return values.
    """
    if proper_axes is None:
        proper_axes = _get_proper_axes(
            atomcoords,
            groups,
            axes,
            rotor_class,
            rtol=rtol,
            atol=atol,
        )

    if proper_axes:
        return _find_point_group_symmetric(
            atomcoords,
            groups,
            axes,
            rotor_class,
            proper_axes,
            rtol=rtol,
            atol=atol,
        )
    elif rotor_class[1] in {
        "regular planar",
        "irregular planar",
    } or _get_mirror_planes(
        atomcoords,
        groups,
        axes,
        rotor_class,
        proper_axes,
        rtol=rtol,
        atol=atol,
    ):
        return "Cs"
    elif _has_inversion_center(atomcoords, groups, rtol=rtol, atol=atol):
        return "Ci"
    return "C1"


def _find_point_group_symmetric(
    atomcoords,
    groups,
    axes,
    rotor_class,
    proper_axes=None,
    rtol=0.0,
    atol=1.0e-2,
):
    """Find point group for symmetric tops.

    Point groups delegated are: Cn, Cnh, Cnv, Dn, Dnh, Dnd, Sn.

    See find_point_group for information on parameters and return values.
    """
    if proper_axes is None:
        proper_axes = _get_proper_axes(
            atomcoords,
            groups,
            axes,
            rotor_class,
            rtol=rtol,
            atol=atol,
        )
    n_principal = proper_axes[0][0]

    count_twofold = 0
    for n, _ in proper_axes:
        if n == 2:
            count_twofold += 1
        if n_principal == count_twofold:
            return _find_point_group_symmetric_dihedral(
                atomcoords,
                groups,
                axes,
                rotor_class,
                proper_axes,
                rtol=rtol,
                atol=atol,
            )
        if n < 2:
            break
    return _find_point_group_symmetric_nondihedral(
        atomcoords,
        groups,
        axes,
        rotor_class,
        proper_axes,
        rtol=rtol,
        atol=atol,
    )

    # the employed workflow is loosely inspired by some articles:
    # 1. doi:10.1016/0097-8485(76)80004-6


def _find_point_group_symmetric_dihedral(
    atomcoords,
    groups,
    axes,
    rotor_class,
    proper_axes=None,
    rtol=0.0,
    atol=1.0e-2,
):
    """Find a dihedral point group for symmetric tops.

    Point groups searched for are: Dn, Dnh, Dnd.

    See find_point_group for information on parameters and return values.
    """
    if proper_axes is None:
        proper_axes = _get_proper_axes(
            atomcoords,
            groups,
            axes,
            rotor_class,
            rtol=rtol,
            atol=atol,
        )
    mirror_axes = _get_mirror_planes(
        atomcoords,
        groups,
        axes,
        rotor_class,
        proper_axes,
        rtol=rtol,
        atol=atol,
    )

    if mirror_axes:
        if mirror_axes[0][0] == "h":
            return f"D{proper_axes[0][0]}h"
        elif len([v for c, v in mirror_axes if c == "v"]) == proper_axes[0][0]:
            # all vertical mirror planes are dihedral for Dnd point groups
            return f"D{proper_axes[0][0]}d"
    return f"D{proper_axes[0][0]}"


def _find_point_group_symmetric_nondihedral(
    atomcoords,
    groups,
    axes,
    rotor_class,
    proper_axes=None,
    rtol=0.0,
    atol=1.0e-2,
):
    """Find a nondihedral point group for symmetric tops.

    Point groups searched for are: Cn, Cnh, Cnv, Sn.

    See find_point_group for information on parameters and return values.
    """
    if proper_axes is None:
        proper_axes = _get_proper_axes(
            atomcoords,
            groups,
            axes,
            rotor_class,
            rtol=rtol,
            atol=atol,
        )
    mirror_axes = _get_mirror_planes(
        atomcoords,
        groups,
        axes,
        rotor_class,
        proper_axes,
        rtol=rtol,
        atol=atol,
    )

    if mirror_axes:
        if mirror_axes[0][0] == "h":
            return f"C{proper_axes[0][0]}h"
        elif len([v for c, v in mirror_axes if c == "v"]) == proper_axes[0][0]:
            return f"C{proper_axes[0][0]}v"

    improper_axes = _get_improper_axes(
        atomcoords,
        groups,
        axes,
        rotor_class,
        proper_axes,
        rtol=rtol,
        atol=atol,
    )
    if improper_axes:
        return f"S{improper_axes[0][0]}"
    return f"C{proper_axes[0][0]}"


def _update_proper_axes(
    ax,
    axes,  # found axes
    atomcoords,
    groups,
    orders,
    rtol,
    atol,
    nondeg_axes=None,
    normalize=False,
):
    """Update axes with ax, and return it with added order (or None).

    Helper function for _get_proper_axes.
    """
    if nondeg_axes is None:
        nondeg_axes = []

    if normalize:
        norm = np.linalg.norm(ax)
        if np.isclose(norm, 0.0, rtol=rtol, atol=atol):
            return axes, None
        ax = ax / norm

    if not all(
        np.isclose(ax @ v, 0.0, rtol=rtol, atol=atol) for v in nondeg_axes
    ) or any(np.isclose(np.abs(ax @ v), 1.0, rtol=rtol, atol=atol) for o, v in axes):
        return axes, None

    for order in orders[::-1]:
        if all(
            _is_symmetric(
                atomcoords[group],
                _operation("c", order=order, axis=ax),
                rtol=rtol,
                atol=atol,
            )
            for group in groups[::-1]
        ):
            axes.append((order, tuple(ax)))
            return axes, order

    return axes, None


def _get_proper_axes(
    atomcoords,
    groups,
    axes,
    rotor_class,
    rtol=0.0,
    atol=1.0e-2,
    slack=0.735,
):
    """Get proper symmetry axes and their orders.

    Parameters
    ----------
    atomcoords : array-like
        Atomic coordinates centered at the center of mass.
    groups : sequence of sequence of int
        Groups of symmetry equivalent atoms, in ascending order of size.
    axes : array-like
        Normalized principal axes of inertia.
    rotor_class : tuple of str
        Rigid rotor classification.
    rtol : float, optional
        The relative tolerance parameter (see `numpy.isclose`).
    atol : float, optional
        The absolute tolerance parameter (see `numpy.isclose`).
    slack : float, optional
        Number to multiply rtol and atol prior comparisons.

    Returns
    -------
    sequence of tuples of int, array-like
        Ordered sequence of tuples in the format ``(order, (x, y, z))``.

    Notes
    -----
    This function has some limitations. First, no C1 axis is never returned.
    Second, an empty list is returned if the structure has a single atom. And
    third, the largest symmetry axis found has order not greater than the
    maximum number of symmetry equivalent atoms, which particularly impacts
    linear rotors (for instance, the largest axis found for the hydrogen
    molecule is twofold, while no axis is found for hydrogen chloride).
    Furthermore, linear rotors are considered symmetric prolate tops, which
    means a single axis is returned (no perpendicular axes). Either way, this
    should be of little impact, as only cases whose groups can easily be
    inferred with no or little knowledge of symmetry axis are affected.

    Examples
    --------
    >>> from overreact import _datasets as datasets

    >>> data = datasets.logfiles["symmetries"]["diborane"]
    >>> groups = _equivalent_atoms(data.atommasses, data.atomcoords)
    >>> moments, axes, atomcoords = inertia(data.atommasses, data.atomcoords)
    >>> rotor_class = _classify_rotor(moments)
    >>> _get_proper_axes(atomcoords, groups, axes, rotor_class)
    [(2, (1.0, ...)),
     (2, (...)),
     (2, (...))]
    """
    rtol, atol = slack * rtol, slack * atol

    if rotor_class[1] == "atomic" or len(atomcoords) == 1:
        return []

    axes = np.asarray(axes)
    atomcoords = np.asarray(atomcoords)
    orders = _guess_orders(groups, rotor_class)

    found_axes = []
    nondeg_axes = []
    if rotor_class[0] == "symmetric prolate":
        nondeg_axes = [axes[:, 0]]
        found_axes, order = _update_proper_axes(
            axes[:, 0],
            found_axes,
            atomcoords=atomcoords,
            groups=groups,
            orders=orders,
            rtol=rtol,
            atol=atol,
        )
    elif rotor_class[0] == "symmetric oblate":
        nondeg_axes = [axes[:, 2]]
        found_axes, order = _update_proper_axes(
            axes[:, 2],
            found_axes,
            atomcoords=atomcoords,
            groups=groups,
            orders=orders,
            rtol=rtol,
            atol=atol,
        )
    elif rotor_class[0] == "asymmetric":
        for ax in axes.T:
            found_axes, order = _update_proper_axes(
                ax,
                found_axes,
                atomcoords=atomcoords,
                groups=groups,
                orders=orders,
                rtol=rtol,
                atol=atol,
            )
        return sorted(found_axes, reverse=True)

    for group in groups:
        for i, a in enumerate(group):
            through_ax = atomcoords[a]
            found_axes, order = _update_proper_axes(
                through_ax,
                found_axes,
                atomcoords=atomcoords,
                groups=groups,
                orders=orders,
                rtol=rtol,
                atol=atol,
                nondeg_axes=nondeg_axes,
                normalize=True,
            )
            if rotor_class[0] == "spheric" and order == 5:
                return sorted(found_axes, reverse=True)

            for b in group[:i]:
                midpoint_ax = atomcoords[a] + atomcoords[b]
                found_axes, order = _update_proper_axes(
                    midpoint_ax,
                    found_axes,
                    atomcoords=atomcoords,
                    groups=groups,
                    orders=orders,
                    rtol=rtol,
                    atol=atol,
                    nondeg_axes=nondeg_axes,
                    normalize=True,
                )
                if rotor_class[0] == "spheric" and order == 5:
                    return sorted(found_axes, reverse=True)

    if rotor_class[0] == "spheric":
        twofold_axes = [ax for o, ax in found_axes if o == 2]
        for i, ax_a in enumerate(twofold_axes):
            for ax_b in twofold_axes[:i]:
                ax = np.cross(ax_a, ax_b)
                found_axes, order = _update_proper_axes(
                    ax,
                    found_axes,
                    atomcoords=atomcoords,
                    groups=groups,
                    orders=orders,
                    rtol=rtol,
                    atol=atol,
                    nondeg_axes=nondeg_axes,
                    normalize=True,
                )
                if order == 5:
                    return sorted(found_axes, reverse=True)

    return sorted(found_axes, reverse=True)


def _guess_orders(groups, rotor_class):
    """Guess possible group orders based on groups.

    The guess consists of the numbers two to n, where n is the number of
    elements in the largest group. For cubic groups, n is taken to be at most
    five. The trivial order one is never guessed.

    Parameters
    ----------
    groups : sequence of sequence of int
        Groups of symmetry equivalent atoms, in ascending order of size.
    rotor_class : tuple of str
        Rigid rotor classification.

    Returns
    -------
    sequence of int

    Examples
    --------
    >>> _guess_orders([[0], [1, 2, 3, 4], [5, 6, 7, 8]],
    ...               ("symmetric prolate", "nonplanar"))
    range(2, 5)
    >>> _guess_orders([[0], [1, 2, 3, 4], [5, 6, 7, 8]],
    ...               ("spheric", "nonplanar"))
    range(2, 5)
    >>> _guess_orders([[0], [1, 2, 3, 4], [5, 6, 7, 8, 9, 10]],
    ...               ("symmetric prolate", "nonplanar"))
    range(2, 7)
    >>> _guess_orders([[0], [1, 2, 3, 4], [5, 6, 7, 8, 9, 10]],
    ...               ("spheric", "nonplanar"))
    range(2, 6)
    """
    max_order = len(groups[-1])
    if rotor_class[0] == "spheric":
        max_order = min(max_order, 5)
    return range(2, max_order + 1)


def _update_improper_axes(
    n,
    ax,
    axes,
    atomcoords,
    groups,
    rtol,
    atol,
    normalize=False,  # found axes
):
    """Update axes with ax and return it.

    Helper function for _get_improper_axes.
    """
    if normalize:
        norm = np.linalg.norm(ax)
        if np.isclose(norm, 0.0, rtol=rtol, atol=atol):
            return axes
        ax = ax / norm

    for order in [2 * n, n]:
        if all(
            _is_symmetric(
                atomcoords[group],
                _operation("s", order=order, axis=ax),
                rtol=rtol,
                atol=atol,
            )
            for group in groups[::-1]
        ):
            axes.append((order, tuple(ax)))
            break

    return axes


def _get_improper_axes(
    atomcoords,
    groups,
    axes,
    rotor_class,
    proper_axes=None,
    rtol=0.0,
    atol=1.0e-2,
    slack=1.888,
):
    """Get improper symmetry axes and their orders.

    Parameters
    ----------
    atomcoords : array-like
        Atomic coordinates centered at the center of mass.
    groups : sequence of sequence of int
        Groups of symmetry equivalent atoms, in ascending order of size.
    axes : array-like
        Normalized principal axes of inertia.
    rotor_class : tuple of str
        Rigid rotor classification.
    proper_axes : sequence of tuples of int, array-like, optional
        Proper symmetry axes of rotation.
    rtol : float, optional
        The relative tolerance parameter (see `numpy.isclose`).
    atol : float, optional
        The absolute tolerance parameter (see `numpy.isclose`).
    slack : float, optional
        Number to multiply rtol and atol prior comparisons.

    Returns
    -------
    sequence of tuples of int, array-like

    Examples
    --------
    >>> from overreact import _datasets as datasets

    >>> data = datasets.logfiles["tanaka1996"]["methane@UMP2/cc-pVTZ"]
    >>> groups = _equivalent_atoms(data.atommasses, data.atomcoords)
    >>> moments, axes, atomcoords = inertia(data.atommasses, data.atomcoords)
    >>> rotor_class = _classify_rotor(moments)
    >>> _get_improper_axes(atomcoords, groups, axes, rotor_class)
    [(4, (0.0, 0.0, -1.0)),
     (4, (0.0, -1.0, 0.0)),
     (4, (-1.0, 0.0, 0.0))]
    """
    rtol, atol = slack * rtol, slack * atol

    if rotor_class[1] == "atomic" or len(atomcoords) == 1:
        return []

    axes = np.asarray(axes)
    atomcoords = np.asarray(atomcoords)

    if proper_axes is None:
        proper_axes = _get_proper_axes(
            atomcoords,
            groups,
            axes,
            rotor_class,
            rtol=rtol,
            atol=atol,
        )

    found_axes = []
    for n, ax in proper_axes:
        found_axes = _update_improper_axes(
            n,
            ax,
            found_axes,
            atomcoords=atomcoords,
            groups=groups,
            rtol=rtol,
            atol=atol,
        )
    return sorted(found_axes, reverse=True)


def _update_mirror_axes(
    ax,
    axes,  # found axes
    atomcoords,
    groups,
    rtol,
    atol,
    proper_axes,
    nondeg_axes=None,
    normalize=False,
):
    """Update axes with ax and return it.

    Helper function for _get_mirror_planes.
    """
    if nondeg_axes is None:
        nondeg_axes = []

    if normalize:
        norm = np.linalg.norm(ax)
        if np.isclose(norm, 0.0, rtol=rtol, atol=atol):
            return axes
        ax = ax / norm

    if not all(
        np.isclose(ax @ v, 0.0, rtol=rtol, atol=atol) for v in nondeg_axes
    ) or any(np.isclose(np.abs(ax @ v), 1.0, rtol=rtol, atol=atol) for c, v in axes):
        return axes

    if all(
        _is_symmetric(
            atomcoords[group],
            _operation("sigma", axis=ax),
            rtol=rtol,
            atol=atol,
        )
        for group in groups[::-1]
    ):
        class_ = ""
        if any(
            np.isclose(np.abs(ax @ v), 1.0, rtol=rtol, atol=atol)
            for n, v in proper_axes
            if proper_axes[0][0] == n
        ):
            class_ = "h"
        elif any(
            np.isclose(ax @ v, 0.0, rtol=rtol, atol=atol)
            for n, v in proper_axes
            if proper_axes[0][0] == n
        ):
            class_ = "v"
        axes.append((class_, tuple(ax)))

    return axes


def _get_mirror_planes(
    atomcoords,
    groups,
    axes,
    rotor_class,
    proper_axes=None,
    rtol=0.0,
    atol=1.0e-2,
    slack=2.020,
):
    """Get (and attempt to classify) mirror plane normal axes.

    Parameters
    ----------
    atomcoords : array-like
        Atomic coordinates centered at the center of mass.
    groups : sequence of sequence of int
        Groups of symmetry equivalent atoms, in ascending order of size.
    axes : array-like
        Normalized principal axes of inertia.
    rotor_class : tuple of str
        Rigid rotor classification.
    proper_axes : sequence of tuples of int, array-like, optional
        Proper symmetry axes of rotation.
    rtol : float, optional
        The relative tolerance parameter (see `numpy.isclose`).
    atol : float, optional
        The absolute tolerance parameter (see `numpy.isclose`).
    slack : float, optional
        Number to multiply rtol and atol prior comparisons.

    Returns
    -------
    sequence of tuples of str, array-like

    Notes
    -----
    This function only classifies horizontal and vertical planes. Besides,
    horizontal classification always takes precedence. Unclassified planes are
    associated with an empty string. This is all that is needed for point group
    classification.

    This function has some limitations. First, an empty list is always returned
    if the structure has a single atom. Second, an empty list is returned for
    C∞v as well. And third, since classifications are made against proper
    symmetry axes, no classification can be made for point groups having no
    proper symmetry axes (e.g., the mirror plane in Cs is not classified).

    This should be of little impact, as only cases whose groups can easily be
    inferred with no or little knowledge of mirror planes are affected.

    Examples
    --------
    >>> from overreact import _datasets as datasets

    >>> data = datasets.logfiles["symmetries"]["1-iodo-2-chloroethylene"]
    >>> groups = _equivalent_atoms(data.atommasses, data.atomcoords)
    >>> moments, axes, atomcoords = inertia(data.atommasses, data.atomcoords)
    >>> rotor_class = _classify_rotor(moments)
    >>> _get_mirror_planes(atomcoords, groups, axes, rotor_class)
    [('', (0.0, 0.0, 1.0))]
    """
    rtol, atol = slack * rtol, slack * atol

    if rotor_class[1] == "atomic" or len(atomcoords) == 1:
        return []

    axes = np.asarray(axes)
    atomcoords = np.asarray(atomcoords)

    if proper_axes is None:
        proper_axes = _get_proper_axes(
            atomcoords,
            groups,
            axes,
            rotor_class,
            rtol=rtol,
            atol=atol,
        )

    def _kf(x):
        """Order function for returned list."""
        c, v = x
        if c:
            return -ord(c), v
        else:
            return 0, v

    found_axes = []
    nondeg_axes = []
    if rotor_class[0] == "symmetric prolate":
        nondeg_axes = [axes[:, 0]]
        found_axes = _update_mirror_axes(
            axes[:, 0],
            found_axes,
            atomcoords=atomcoords,
            groups=groups,
            rtol=rtol,
            atol=atol,
            proper_axes=proper_axes,
        )
    elif rotor_class[0] == "symmetric oblate":
        nondeg_axes = [axes[:, 2]]
        found_axes = _update_mirror_axes(
            axes[:, 2],
            found_axes,
            atomcoords=atomcoords,
            groups=groups,
            rtol=rtol,
            atol=atol,
            proper_axes=proper_axes,
        )
    elif rotor_class[0] == "asymmetric":
        for ax in axes.T:
            found_axes = _update_mirror_axes(
                ax,
                found_axes,
                atomcoords=atomcoords,
                groups=groups,
                rtol=rtol,
                atol=atol,
                proper_axes=proper_axes,
            )
        return sorted(found_axes, reverse=True, key=_kf)

    for group in groups:
        for i, a in enumerate(group):
            for b in group[:i]:
                ab_ax = atomcoords[b] - atomcoords[a]
                found_axes = _update_mirror_axes(
                    ab_ax,
                    found_axes,
                    atomcoords=atomcoords,
                    groups=groups,
                    rtol=rtol,
                    atol=atol,
                    proper_axes=proper_axes,
                    nondeg_axes=nondeg_axes,
                    normalize=True,
                )

    return sorted(found_axes, reverse=True, key=_kf)


def _has_inversion_center(atomcoords, groups, rtol=0.0, atol=1.0e-2, slack=1.888):
    """Check whether the molecule has an inversion center.

    Parameters
    ----------
    atomcoords : array-like
        Atomic coordinates centered at the center of mass.
    groups : sequence of sequence of int
        Groups of symmetry equivalent atoms, in ascending order of size.
    rtol : float, optional
        The relative tolerance parameter (see `numpy.isclose`).
    atol : float, optional
        The absolute tolerance parameter (see `numpy.isclose`).
    slack : float, optional
        Number to multiply rtol and atol prior comparisons.

    Returns
    -------
    bool

    Examples
    --------
    >>> _has_inversion_center([[0, 0, -1], [0, 0, 1]], [[0, 1]])
    True
    >>> _has_inversion_center([[0, 0, -1],
    ...                        [0, 0,  1],
    ...                        [1, 1,  1]], [[0, 1], [2]])
    False
    """
    rtol, atol = slack * rtol, slack * atol

    atomcoords = np.asarray(atomcoords)
    return all(
        _is_symmetric(atomcoords[group], _operation("i"), rtol=rtol, atol=atol)
        for group in groups[::-1]
    )


def _is_symmetric(atomcoords, op, rtol=0.0, atol=1.0e-2, slack=10.256):
    """Check if structure satisfies symmetry.

    Parameters
    ----------
    atomcoords : array-like
        Atomic coordinates centered at the center of mass.
    op : array-like
        Symmetry operator matrix.
    rtol : float, optional
        The relative tolerance parameter (see `numpy.isclose`).
    atol : float, optional
        The absolute tolerance parameter (see `numpy.isclose`).
    slack : float, optional
        Number to multiply rtol and atol prior comparisons.

    Returns
    -------
    bool

    Examples
    --------
    >>> _is_symmetric([[1, 0, 0],
    ...               [0, 1, 0],
    ...               [0, 0, 0]], _operation("c", order=4, axis=[0, 0, 1]))
    False
    >>> _is_symmetric([[1, 0, 0],
    ...               [0, 1, 0],
    ...               [0, 0, 0]], _operation("c", order=2, axis=[1, 1, 0]))
    True
    >>> _is_symmetric([[1, 0, 0],
    ...               [0, 1, 0],
    ...               [0, 0, 0]], _operation("sigma", axis=[0, 0, 1]))
    True
    >>> _is_symmetric([[1, 0, 0],
    ...               [0, 1, 0],
    ...               [0, 0, 0],
    ...               [-1, 0, 0]], _operation("c", order=4, axis=[0, 0, 1]))
    False
    >>> _is_symmetric([[1, 0, 0],
    ...               [0, 1, 0],
    ...               [0, 0, 0],
    ...               [-1, 0, 0]], _operation("sigma", axis=[1, 0, 0]))
    True
    >>> _is_symmetric([[1, 0, 0],
    ...               [0, 1, 0],
    ...               [0, 0, 0],
    ...               [-1, 0, 0],
    ...               [0, -1, 0]], _operation("c", order=4, axis=[0, 0, 1]))
    True
    >>> _is_symmetric([[1, 0, 0],
    ...               [0, 1, 0],
    ...               [0, 0, 0],
    ...               [-1, 0, 0],
    ...               [0, -1, 0]], _operation("i"))
    True
    """
    rtol, atol = slack * rtol, slack * atol
    inner_slack = 1.055

    tree = KDTree(atomcoords)
    d, i = tree.query(atomcoords @ op.T)

    return (
        set(i) == set(range(len(atomcoords)))
        and np.allclose(d.mean(), 0.0, rtol=rtol, atol=atol)
        and np.allclose(d.max(), 0.0, rtol=inner_slack * rtol, atol=inner_slack * atol)
    )


def _operation(name, order=2, axis=None):
    """Calculate a symmetry _operation.

    Parameters
    ----------
    name : str
        Operation symbol (see examples below).
    order : int, optional
        Operation order.
    axis : array-like, optional
        Operation axis.

    Returns
    -------
    array-like

    Raises
    ------
    ValueError
        If the operation is not recognized.

    Examples
    --------
    >>> _operation("e")
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    >>> _operation("i")
    array([[-1., -0., -0.],
           [-0., -1., -0.],
           [-0., -0., -1.]])
    >>> _operation("c", order=4, axis=[0, 0, 1])
    array([[ 0., -1., 0.],
           [ 1.,  0., 0.],
           [ 0.,  0., 1.]])
    >>> _operation("sigma", axis=[0, 0, 1])
    array([[ 1., 0.,  0.],
           [ 0., 1.,  0.],
           [ 0., 0., -1.]])
    >>> _operation("sigma", axis=[0, 1, 0])
    array([[ 1.,  0., 0.],
           [ 0., -1., 0.],
           [ 0.,  0., 1.]])
    >>> _operation("sigma", axis=[1, 0, 0])
    array([[-1., 0., 0.],
           [ 0., 1., 0.],
           [ 0., 0., 1.]])
    >>> _operation("s", order=4, axis=[0, 0, 1])
    array([[ 0., -1.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0., -1.]])
    >>> _operation("s", order=4, axis=[0, 1, 0])
    array([[ 0.,  0., 1.],
           [ 0., -1., 0.],
           [-1.,  0., 0.]])
    >>> _operation("s", order=4, axis=[1, 0, 0])
    array([[-1., 0.,  0.],
           [ 0., 0., -1.],
           [ 0., 1.,  0.]])
    """
    if axis is None:
        axis = np.array([0, 0, 1])

    if name == "i":
        return -np.eye(3)
    if name == "e":
        return np.eye(3)

    if name in {"c", "σ", "sigma", "s"}:
        # normalize axis
        axis = np.asarray(axis)
        axis = axis / np.linalg.norm(axis)

        if name in {"c", "s"}:
            rotation = Rotation.from_rotvec(2.0 * np.pi * axis / order).as_matrix()
        if name in {"σ", "sigma", "s"}:
            reflection = np.eye(3) - 2.0 * np.outer(axis, axis)

        if name == "c":
            return rotation
        if name in {"σ", "sigma"}:
            return reflection
        if name == "s":
            return rotation @ reflection

    msg = f"unknown operation: '{name}'"
    raise ValueError(msg)


def _classify_rotor(moments, rtol=0.0, atol=1.0e-2, slack=0.870):
    """Classify rotors based on moments of inertia.

    See doi:10.1002/jcc.23493.

    Parameters
    ----------
    moments : array-like
        Primary moments of inertia in ascending order. Units are in amu·Å².
    rtol : float, optional
        The relative tolerance parameter (see `numpy.isclose`).
    atol : float, optional
        The absolute tolerance parameter (see `numpy.isclose`).
    slack : float, optional
        Number to multiply rtol and atol prior comparisons.

    Return
    ------
    top, shape : str

    Notes
    -----
    Moments are actually compared by their ratios such that the "aspect ratio"
    of the molecule is what is compared.

    Examples
    --------
    Do examples from
    <https://www.tau.ac.il/~tsirel/dump/Static/knowino.org/wiki/Classification_of_rigid_rotors.html>.

    The idea behind this function is, among other things, to help in
    classifying point groups. For instance, there are the following
    possibilities for atoms or linear molecules:

    >>> _classify_rotor([0, 0, 0])
    ('spheric', 'atomic')
    >>> _classify_rotor([0, 1, 1])
    ('symmetric prolate', 'linear')

    Spheric tops can be any cubic group:

    >>> _classify_rotor([1, 1, 1])
    ('spheric', 'nonplanar')

    Asymmetric tops can be a lot of different groups, the ones lacking proper
    axis of symmetry (C1, Ci, Cs) being exclusively found in asymmetric tops.
    Furthermore, any group found for symmetric tops can be found for asymmetric
    tops as well, which complicates things a bit.

    >>> _classify_rotor([1, 2, 4])
    ('asymmetric', 'nonplanar')

    Symmetric tops can be found in a subset of the ones found in asymmetric
    tops and they always have a proper axis of symmetry. I further classify
    them here into smaller groups (I believe that this subclassification and
    its relationship with possible point groups can be further improved). For
    instance, planar symmetric tops are always oblate and have a plane of
    symmetry:

    >>> _classify_rotor([1, 1, 2])
    ('symmetric oblate', 'regular planar')

    >>> _classify_rotor([1, 3, 4])
    ('asymmetric', 'irregular planar')

    >>> _classify_rotor([1, 1, 3])
    ('symmetric oblate', 'nonplanar')

    >>> _classify_rotor([1, 2, 2])
    ('symmetric prolate', 'nonplanar')
    """
    rtol, atol = slack * rtol, slack * atol
    inner_slack = 2.130

    if np.isclose(moments[2], 0.0, rtol=inner_slack * rtol, atol=inner_slack * atol):
        return "spheric", "atomic"
    moments = np.asarray(moments) / moments[2]

    # basic tests for tops
    is_oblate = np.isclose(
        moments[0],
        moments[1],
        rtol=inner_slack * rtol,
        atol=inner_slack * atol,
    )
    is_spheric = np.isclose(
        moments[0],
        moments[2],
        rtol=inner_slack * rtol,
        atol=inner_slack * atol,
    )
    is_prolate = np.isclose(
        moments[1],
        moments[2],
        rtol=inner_slack * rtol,
        atol=inner_slack * atol,
    )

    # basic tests for shapes
    fits_line = np.isclose(
        moments[0],
        0.0,
        rtol=inner_slack * rtol,
        atol=inner_slack * atol,
    )
    fits_plane = np.isclose(moments[0] + moments[1], moments[2], rtol=rtol, atol=atol)

    is_spheric = is_spheric and is_oblate and is_prolate
    if is_spheric:
        top = "spheric"
    elif is_oblate:
        top = "symmetric oblate"
    elif is_prolate:
        top = "symmetric prolate"
    else:
        top = "asymmetric"

    fits_line = fits_line and is_prolate
    if fits_line:
        shape = "linear"
    elif fits_plane:
        shape = "regular planar" if is_oblate else "irregular planar"
    else:
        shape = "nonplanar"

    return top, shape


def gyradius(atommasses, atomcoords, method="iupac"):
    """Calculate the radius of gyration (or gyradius) of the molecule.

    Parameters
    ----------
    atommasses : array-like
        Atomic masses in atomic mass units (amu).
    atomcoords : array-like
        Atomic coordinates.
    method : str, optional

    Returns
    -------
    array-like

    Raises
    ------
    ValueError
        If `method` is not recognized.

    Examples
    --------
    >>> from overreact import _datasets as datasets

    >>> data = datasets.logfiles["tanaka1996"]["CH3·@UMP2/cc-pVTZ"]
    >>> gyradius(data.atommasses, data.atomcoords)
    0.481
    >>> gyradius(data.atommasses, data.atomcoords, method="mean")
    0.93

    >>> data = datasets.logfiles["symmetries"]["water"]
    >>> gyradius(data.atommasses, data.atomcoords)
    0.31915597673891866
    >>> gyradius(np.ones_like(data.atommasses), data.atomcoords)
    0.6833818299241241
    >>> gyradius(np.ones_like(data.atommasses), data.atomcoords, method="mean")
    0.6833818299241241
    >>> gyradius(data.atommasses, data.atomcoords, method="mean")
    0.7637734749747612
    """
    com = np.average(atomcoords, axis=0, weights=atommasses)
    atomcoords = atomcoords - com
    if method == "iupac":
        return np.sqrt(
            np.average(np.diag(atomcoords @ atomcoords.T), weights=atommasses),
        )
    elif method == "mean":
        return np.sqrt(np.mean(np.diag(atomcoords @ atomcoords.T)))
    else:
        msg = f"unavailable method: '{method}'"
        raise ValueError(msg)
    

@rx._misc.copy_unhashable()
def inertia(atommasses, atomcoords, align=True):
    r"""Calculate primary moments and axes from the inertia tensor.

    Parameters
    ----------
    atommasses : array-like
        Atomic masses in atomic mass units (amu).
    atomcoords : array-like
        Atomic coordinates.
    align : bool, optional
        If true, the returned coordinates are aligned to the primary axes of
        inertia. The returned axes correspond to the aligned coordinates as
        well.

    Returns
    -------
    moments, axes : array-like
        Primary moments of inertia in ascending order and associated normalized
        axes. Axes are column vectors and always correspond to returned atomic
        coordinates. Units are in amu·Å².
    atomcoords : array-like
        Coordinates centered at the center of mass and, if align was set to
        True, rotated to the primary axes of inertia.

    Examples
    --------
    >>> atommasses = np.array([12.011,  1.008,  1.008,  1.008])  # CH3·
    >>> atomcoords = np.array([[ 0.      ,  0.      , -1.      ],
    ...                        [ 1.07883 ,  0.      , -1.      ],
    ...                        [-0.539415,  0.934294, -1.      ],
    ...                        [-0.539415, -0.934294, -1.      ]])
    >>> moments, axes, atomcoords = inertia(atommasses, atomcoords)
    >>> moments
    array([1.75977704, 1.75977774, 3.51955478])
    >>> axes
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    >>> atomcoords
    array([[ 0.      ,  0.      ,  0.      ],
           [ 1.07883 ,  0.      ,  0.      ],
           [-0.539415,  0.934294,  0.      ],
           [-0.539415, -0.934294,  0.      ]])

    This allows one to calculate the rotational constants in cm-1:

    >>> constants.h * constants.centi \
    ...     / (8 * np.pi ** 2 * constants.c \
    ...     * moments * constants.atomic_mass * constants.angstrom ** 2)
    array([9.5794, 9.5794, 4.7897])
    """
    atommasses = np.atleast_1d(atommasses)
    com = np.average(atomcoords, axis=0, weights=atommasses)
    atomcoords = atomcoords - com

    w_coords = np.sqrt(atommasses)[:, np.newaxis] * atomcoords
    squared_w_coords = w_coords**2

    i_xx = np.sum(squared_w_coords[:, 1] + squared_w_coords[:, 2])
    i_yy = np.sum(squared_w_coords[:, 0] + squared_w_coords[:, 2])
    i_zz = np.sum(squared_w_coords[:, 0] + squared_w_coords[:, 1])

    i_xy = -np.sum(w_coords[:, 0] * w_coords[:, 1])
    i_xz = -np.sum(w_coords[:, 0] * w_coords[:, 2])
    i_yz = -np.sum(w_coords[:, 1] * w_coords[:, 2])

    inertia_tensor = np.array(
        [[i_xx, i_xy, i_xz], [i_xy, i_yy, i_yz], [i_xz, i_yz, i_zz]],
    )
    moments, axes = np.linalg.eigh(inertia_tensor)
    if align:
        return inertia(atommasses, atomcoords @ axes, align=False)
    logger.debug(f"moments = {moments} amu·Å²")
    return moments, axes, atomcoords


# TODO(schneiderfelipe): this needs rework, see
# https://chemistry.stackexchange.com/questions/74639/how-to-calculate-wavenumbers-of-normal-modes-from-the-eigenvalues-of-the-cartesi/74923#74923
# Ideally, the same Eckart transformation that make this work will also work
# in calc_vibfreqs, so one thing leads to the other.
def calc_hessian(atommasses, atomcoords, vibfreqs, vibdisps):
    """Compute the Hessian matrix from normal modes and frequencies.

    This function does the inverse of what is described in
    https://gaussian.com/vib/.

    Parameters
    ----------
    atommasses : array-like
        Atomic masses in atomic mass units (amu).
    atomcoords : array-like
        Atomic coordinates.
    vibfreqs : array-like
        Frequency magnitudes in cm-1.
    vibdisps : array-like
        Normal modes in cartesian coordinates.

    Returns
    -------
    array-like
        Complete Hessian matrix in cartesian coordinates.

    Notes
    -----
    This is a work in progress!

    Examples
    --------
    >>> from overreact import _datasets as datasets

    >>> data = datasets.logfiles["symmetries"]["water"]
    >>> H = calc_hessian(data.atommasses, data.atomcoords, data.vibfreqs, data.vibdisps)
    >>> calc_vibfreqs(H, data.atommasses)  # doctest: +SKIP
    array([1619.1, 3671.7, 3769.1])
    >>> H  # this is probably incorrect
    array([[ 0.25035519,  0.17924759, -0.21923846, -0.22425124, -0.15503866,
             0.20261806, -0.02610273, -0.02420995,  0.01662027],
           [ 0.17924759,  0.39097884,  0.13979548, -0.12659242, -0.11407083,
             0.08438083, -0.05265485, -0.27690678, -0.2241771 ],
           [-0.21923846,  0.13979548,  0.52730855,  0.23475659,  0.13230333,
            -0.2460061 , -0.01551983, -0.2720957 , -0.28130313],
           [-0.22425124, -0.12659242,  0.23475659,  0.22086973,  0.13025443,
            -0.22492564,  0.00337999, -0.00366091, -0.00983096],
           [-0.15503866, -0.11407083,  0.13230333,  0.13025443,  0.10037856,
            -0.10602169,  0.02478367,  0.01369292, -0.02628153],
           [ 0.20261806,  0.08438083, -0.2460061 , -0.22492564, -0.10602169,
             0.25914009,  0.02230951,  0.02163974, -0.01313383],
           [-0.02610273, -0.05265485, -0.01551983,  0.00337999,  0.02478367,
             0.02230951,  0.02272304,  0.0278711 , -0.00678954],
           [-0.02420995, -0.27690678, -0.2720957 , -0.00366091,  0.01369292,
             0.02163974,  0.0278711 ,  0.26321199,  0.25045664],
           [ 0.01662027, -0.2241771 , -0.28130313, -0.00983096, -0.02628153,
            -0.01313383, -0.00678954,  0.25045664,  0.29443748]])
    """
    dof = 3 * len(atommasses)
    L_cart = np.asarray(vibdisps).reshape((len(vibfreqs), dof)).T
    # this function is correct until here

    L_cart = np.linalg.qr(L_cart, mode="complete")[0]

    atommasses_sqrt = np.sqrt([mass for mass in atommasses for _ in range(3)])
    D = eckart_transform(atommasses, atomcoords)
    M = np.diag(1.0 / atommasses_sqrt)
    L = np.linalg.solve(M @ D, L_cart)

    assert np.allclose(M @ D @ L, L_cart), "L_cart is not orthogonal"

    # this function is correct from here
    nu = np.asarray(vibfreqs) * constants.c / constants.centi
    eigenvalues = (
        (2.0 * np.pi * nu) ** 2
        * (constants.atomic_mass * constants.bohr**2)
        / constants.hartree
    )
    eigenvalues = np.block([eigenvalues, np.zeros(dof - len(eigenvalues))])

    f_int = L @ np.diag(eigenvalues) @ L.T
    f_mwc = D @ f_int @ D.T
    return f_mwc * np.outer(atommasses_sqrt, atommasses_sqrt)


# TODO(schneiderfelipe): correct this function and project out translations
# and rotations, see
# https://chemistry.stackexchange.com/questions/74639/how-to-calculate-wavenumbers-of-normal-modes-from-the-eigenvalues-of-the-cartesi/74923#74923
def calc_vibfreqs(hessian, atommasses):
    """Calculate vibrational frequencies.

    This is described in https://gaussian.com/vib/.

    Parameters
    ----------
    hessian : array-like
    atommasses : array-like
        Atomic masses in atomic mass units (amu).
    atomcoords : array-like
        Atomic coordinates.

    Returns
    -------
    vibfreqs : array-like
        Frequency magnitudes in cm-1.

    Examples
    --------
    >>> from overreact import _datasets as datasets

    >>> data = datasets.logfiles["symmetries"]["water"]
    >>> calc_vibfreqs(data.hessian, data.atommasses)
    array([1619.1, 3671.7, 3769.1])
    """
    atommasses_sqrt = np.sqrt([mass for mass in atommasses for _ in range(3)])

    # mass-weighted Hessian
    hessian = np.asarray(hessian) / np.outer(atommasses_sqrt, atommasses_sqrt)

    eigenvalues = np.linalg.eigvals(hessian)

    # TODO(schneiderfelipe): the following probably misses some linear
    # molecules and transition states.
    eigenvalues = np.real(eigenvalues[eigenvalues > 0])[::-1]
    nu = np.sqrt(
        eigenvalues * constants.hartree / (constants.atomic_mass * constants.bohr**2),
    ) / (2.0 * np.pi)
    return nu * constants.centi / constants.c


# TODO(schneiderfelipe): ensure this is correct
# https://chemistry.stackexchange.com/questions/74639/how-to-calculate-wavenumbers-of-normal-modes-from-the-eigenvalues-of-the-cartesi/74923#74923
def eckart_transform(atommasses, atomcoords):
    """Compute the Eckart transform.

    This transform is described in https://gaussian.com/vib/.

    Parameters
    ----------
    atommasses : array-like
        Atomic masses in atomic mass units (amu).
    atomcoords : array-like
        Atomic coordinates.

    Returns
    -------
    array-like

    Examples
    --------
    >>> from overreact import _datasets as datasets

    >>> data = datasets.logfiles["tanaka1996"]["Cl·@UMP2/cc-pVTZ"]
    >>> eckart_transform(data.atommasses, data.atomcoords)
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    >>> data = datasets.logfiles["symmetries"]["dihydrogen"]
    >>> eckart_transform(data.atommasses, data.atomcoords)
    array([[...]])
    >>> data = datasets.logfiles["symmetries"]["water"]
    >>> eckart_transform(data.atommasses, data.atomcoords)
    array([[-9.42386999e-01,  0.00000000e+00,  0.00000000e+00,
             2.99716727e-01, -2.86166258e-06, -7.42376895e-02,
            -1.19022276e-02,  4.33736541e-03, -1.28081683e-01],
           [-0.00000000e+00, -9.42386999e-01,  0.00000000e+00,
             1.40934586e-02, -1.34562803e-07,  1.01850683e-01,
            -1.52466204e-01, -2.78628770e-01, -2.13218735e-02],
           [-0.00000000e+00, -0.00000000e+00, -9.42386999e-01,
            -1.47912143e-01,  1.41224899e-06, -1.40724409e-01,
            -3.86450545e-02, -1.77596105e-02, -2.61565554e-01],
           [-2.36544652e-01, -0.00000000e+00, -0.00000000e+00,
            -5.97037403e-01, -6.33525274e-01,  2.70812665e-02,
            -2.34354970e-01,  8.09905642e-02,  3.52169811e-01],
           [-0.00000000e+00, -2.36544652e-01, -0.00000000e+00,
            -2.80742485e-02, -2.97900030e-02, -6.93753868e-01,
             5.78451116e-01,  2.06337502e-01,  2.89647600e-01],
           [-0.00000000e+00, -0.00000000e+00, -2.36544652e-01,
             2.94641819e-01,  3.12648820e-01, -1.12274948e-02,
            -4.19760855e-01,  1.83772848e-01,  7.41205673e-01],
           [-2.36544652e-01, -0.00000000e+00, -0.00000000e+00,
            -5.97025305e-01,  6.33536675e-01,  2.68679525e-01,
             2.81773098e-01, -9.82705016e-02,  1.58103880e-01],
           [-0.00000000e+00, -2.36544652e-01, -0.00000000e+00,
            -2.80736797e-02,  2.97905391e-02,  2.87983715e-01,
             2.89697972e-02,  9.03711399e-01, -2.04701877e-01],
           [-0.00000000e+00, -0.00000000e+00, -2.36544652e-01,
             2.94635849e-01, -3.12654446e-01,  5.71869440e-01,
             5.73721626e-01, -1.13019078e-01,  3.00863871e-01]])
    """
    atommasses = np.asarray(atommasses)
    natom = len(atommasses)
    dof = 3 * natom

    moments, axes, atomcoords = inertia(atommasses, atomcoords, align=False)

    x = np.block(
        [
            np.ones(natom)[:, np.newaxis],
            np.zeros(natom)[:, np.newaxis],
            np.zeros(natom)[:, np.newaxis],
        ],
    )
    y = np.block(
        [
            np.zeros(natom)[:, np.newaxis],
            np.ones(natom)[:, np.newaxis],
            np.zeros(natom)[:, np.newaxis],
        ],
    )
    z = np.block(
        [
            np.zeros(natom)[:, np.newaxis],
            np.zeros(natom)[:, np.newaxis],
            np.ones(natom)[:, np.newaxis],
        ],
    )
    x *= np.sqrt(atommasses[:, np.newaxis])
    y *= np.sqrt(atommasses[:, np.newaxis])
    z *= np.sqrt(atommasses[:, np.newaxis])

    D_trans = np.block(
        [x.reshape(1, dof).T, y.reshape(1, dof).T, z.reshape(1, dof).T],
    )
    D_rot = np.array(
        [
            np.cross((atomcoords @ axes)[i], axes[:, j]) / np.sqrt(atommasses[i])
            for i in range(natom)
            for j in range(3)
        ],
    )
    D = np.block([D_trans, D_rot])
    return np.linalg.qr(D, mode="complete")[0]


# NOTE(schneiderfelipe): thresh was found to be reasonable
# when greater than or equal to 0.106.
def _equivalent_atoms(
    atommasses,
    atomcoords,
    method="cluster",
    thresh=0.106,
    plot=False,
):
    """Generate groups of symmetry equivalent atoms.

    Parameters
    ----------
    atommasses : array-like
        Atomic masses in atomic mass units (amu).
    atomcoords : array-like
        Atomic coordinates.
    method : str, optional
        Method of partitioning: "atommass" (same atoms same groups), "cluster".
    thresh : float, optional
        Threshold to consider atom clusters.

    Returns
    -------
    groups : sequence of sequence of int
        Groups of symmetry equivalent atoms, in ascending order of size. Each
        element in the list is a sequence of indices, one list for each group
        of equivalent atoms. See examples below.

    Raises
    ------
    ValueError
        If `method` is not recognized.

    Notes
    -----
    This function works for up to ten thousand randomly placed atoms and finds
    the equivalent groups in less than a second. As such, the function performs
    sufficiently well for the current use.

    Examples
    --------
    >>> atommasses = [12.011,  1.008,  1.008,  1.008]  # CH3·
    >>> atomcoords = np.array([[ 0.      ,  0.      , -1.      ],
    ...                        [ 1.07883 ,  0.      , -1.      ],
    ...                        [-0.539415,  0.934294, -1.      ],
    ...                        [-0.539415, -0.934294, -1.      ]])
    >>> for indices in _equivalent_atoms(atommasses, atomcoords):
    ...     indices
    [0]
    [1, 2, 3]

    >>> atommasses = np.array([14.007, 1.008, 1.008, 1.008])  # ammonia
    >>> atomcoords = np.array(
    ...     [
    ...         [0.0, 0.0, 0.07878],
    ...         [0.0, 0.98569, -0.18381],
    ...         [0.85363, -0.49284, -0.18381],
    ...         [-0.85363, -0.49284, -0.18381],
    ...     ]
    ... )
    >>> for indices in _equivalent_atoms(atommasses, atomcoords):
    ...     indices
    [0]
    [1, 2, 3]

    """
    if len(atommasses) == 1:  # atom
        return [[0]]
    elif len(atommasses) == 2:  # diatomic molecule
        if atommasses[0] == atommasses[1]:
            return [[0, 1]]
        return [[0], [1]]

    groups = []

    def _update_groups_with_condition(condition, groups):
        # condition is assumed to be an array-like of bool
        groups.append(sorted(np.nonzero(condition)[0]))
        return groups

    if method == "cluster":
        D = squareform(pdist(atomcoords))

        omega = np.mean(D, axis=0)
        sigma = np.std(D, axis=0)
        delta = np.sqrt(np.sum(D**2, axis=0))

        criteria = np.block([[omega], [sigma], [delta]]).T
        Z = linkage(pdist(criteria), method="single")
        clusters = fcluster(Z, thresh, criterion="distance")

        # TODO(schneiderfelipe): this was for debug and should eventually be removed.
        if plot:
            import matplotlib.pyplot as plt

            plt.clf()
            for cluster in np.unique(clusters):
                plt.scatter(
                    criteria[clusters == cluster, 0],
                    criteria[clusters == cluster, 1],
                )
            for i, (atommass, _) in enumerate(zip(atommasses, clusters)):
                plt.annotate(atommass, (criteria[i, 0], criteria[i, 1]))
            plt.xlabel("omega")
            plt.ylabel("sigma")
            plt.show()

        for mass in np.unique(atommasses):
            mass_condition = atommasses == mass
            for cluster in np.unique(clusters[mass_condition]):
                groups = _update_groups_with_condition(
                    mass_condition & (clusters == cluster),
                    groups,
                )
    elif method == "atommass":
        for mass in np.unique(atommasses):
            groups = _update_groups_with_condition(atommasses == mass, groups)
    else:
        msg = f"unavailable method: '{method}'"
        raise ValueError(msg)

    return sorted(groups, key=len)
