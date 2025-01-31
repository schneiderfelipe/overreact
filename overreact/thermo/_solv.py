"""Module dedicated to the calculation of thermodynamic properties in solvation."""

from __future__ import annotations

import logging

import numpy as np

import overreact as rx
from overreact import _constants as constants
from overreact._misc import _derivative as derivative
from overreact import coords

logger = logging.getLogger(__name__)


# TODO(schneiderfelipe): C-PCM already includes the whole Gibbs free energy
# associated with this (see doi:10.1021/jp9716997 and references therein, e.g.,
# doi:10.1016/0009-2614(96)00349-1, doi:10.1021/cr60304a002,
# doi:10.1002/jcc.540100504). As such, including this in the total free energy
# might overcount energy contributions. As an alternative, I might want to
# a. remove this contribution,
# b. keep this and remove the contribution from the cavity enthalpy,
#    either by
#    i.  doing a similar calculation as here and removing from the enthalpy or
#    ii. by actually implementing the original method of C-PCM and excluding
#        it from the enthalpy.
# c. implement something closer to the original and do one of the
#    cited in b. above.
#
# See doi:10.1021/cr60304a002.
def calc_cav_entropy(
    atomnos,
    atomcoords,
    environment="water",
    temperature=298.15,
    pressure=constants.atm,
    dx=3e-5,
    order=3,
):
    r"""Calculate the cavity entropy from scaled particle theory.

    This implements the method due to A. Garza, see doi:10.1021/acs.jctc.9b00214.

    Parameters
    ----------
    atomnos : array-like, optional
    atomcoords : array-like, optional
        Atomic coordinates.
    environment : str, optional
    temperature : array-like, optional
        Absolute temperature in Kelvin.
    pressure : array-like, optional
        Reference gas pressure.
    dx : float, optional
        Spacing.
    order : int, optional
        Number of points to use, must be odd.

    Returns
    -------
    float
        Cavity entropy in J/mol·K.

    Examples
    --------
    >>> from overreact import _datasets as datasets

    >>> data = datasets.logfiles["symmetries"]["dihydrogen"]
    >>> calc_cav_entropy(data.atomnos, data.atomcoords)
    -30.
    >>> data = datasets.logfiles["symmetries"]["water"]
    >>> calc_cav_entropy(data.atomnos, data.atomcoords)
    -40.
    >>> data = datasets.logfiles["tanaka1996"]["Cl·@UMP2/cc-pVTZ"]
    >>> calc_cav_entropy(data.atomnos, data.atomcoords)
    -43.
    >>> data = datasets.logfiles["symmetries"]["tetraphenylborate-"]
    >>> calc_cav_entropy(data.atomnos, data.atomcoords)
    -133.1
    """
    temperature = np.asarray(temperature)

    if np.isclose(temperature, 0.0):
        logger.warning("assuming cavity entropy zero at zero temperature")
        return 0.0

    assert atomnos is not None, "atomnos must be provided"
    assert atomcoords is not None, "atomcoords must be provided"
    vdw_volume = coords.get_molecular_volume(atomnos, atomcoords)

    def func(temperature, solvent):
        # TODO(schneiderfelipe): allow passing a "solvent" object everywhere so
        # that we don't repeat ourselves?
        _, _, ratio = coords._garza(
            vdw_volume,
            solvent,
            full_output=True,
            temperature=temperature,
            pressure=pressure,
        )

        solvent = rx._misc._get_chemical(
            environment,
            temperature,
            pressure,
        )

        permittivity = solvent.permittivity
        y = 3.0 * ((permittivity - 1.0) / (permittivity + 2.0)) / (4.0 * np.pi)
        omy = 1.0 - y
        yoomy = y / omy

        gamma = (
            -np.log(omy)
            # TODO(schneiderfelipe): the following term is probably wrong, the
            # next one is that actually shows up in the paper
            # + 3.0 * yoomy * ratio
            + 3.0 * ratio / omy
            + (3.0 + 4.5 * yoomy) * yoomy * ratio**2
            # TODO(schneiderfelipe): the following term shows up in an old
            # paper, but not in Garza's model
            + (y * solvent.Vm * pressure / (constants.R * temperature)) * ratio**3
        )
        return -constants.R * temperature * gamma
    
    cavity_entropy = derivative(
        func,
        x0=temperature,
        dx=dx,
        n=1,
        args=(environment,),
        order=order,
    )
    logger.info(f"cavity entropy = {cavity_entropy} J/mol·K")
    return cavity_entropy


# TODO(schneiderfelipe): the concept of free volume in polymer and membrane
# sciences are related to the difference between the specific volume (inverse
# of density) and the van der Waals volume (oftentimes multiplied by a factor,
# normally 1.3), see doi:10.1007/978-3-642-40872-4_279-5. This is very similar
# to the thing done here with "izato".
#
# Further theoretical support is given for the exact equation
# used in the work of Eyring (doi:10.1021/j150380a007), where it is also
# suggested that that the self-solvation (solvent molecule solvated by itself)
# outer volume (here called cavity volume) should match the specific volume of
# the solvent at that temperature and pressure.
#
# Further evidence that the free volume should change with temperature can be
# found in doi:10.1016/j.jct.2011.01.003. In fact, it is also shown there that
# the rotational entropy is almost the same as for the ideal gas for molecules
# that don't do hydrogen bonding at their boiling temperature. For molecules
# that do hydrogen bonding, the rotational entropy gain should be taken into
# account. All this can be used to improve this model.
#
# As such, I need to:
# 1. use the specific volumes of pure liquids at various temperatures to come
# up with an alpha that depends on temperature (and possibly pressure).
# 2. I need to validate this by checking Trouton's and Hildebrand's laws for
# apolar compounds, which should also give reasonable boiling temperatures and
# free volumes of around 1 Å³ (see again doi:10.1016/j.jct.2011.01.003).
# 3. Improve polar and hydrogen bonding molecules by adjusting their rotational
# entropies.
# 4. Some data evidence the possibility that the difference between gas and
# pure liquid translational entropies depend solely on the
# temperature/pressure, but not on the nature of the molecule. Changes in
# solvation interaction and rotational entropy might account for the difference
# need to validate Trouton's and Hildebrand's laws. This can be used as a
# guide.
#
# The remarks above are valid for "izato". "garza" incorporates most of the
# above in the usage of the mass density of the solvent.
def molar_free_volume(
    atomnos,
    atomcoords,
    environment="water",
    method="garza",
    temperature=298.15,
    pressure=constants.atm,
):
    r"""Calculate the molar free volume of a solute.

    The current implementation uses simple Quasi-Monte Carlo volume estimates
    through `coords.get_molecular_volume`.

    Parameters
    ----------
    atomnos : array-like
    atomcoords : array-like
        Atomic coordinates.
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

    Returns
    -------
    float
        Molar free volume in cubic meters per mole.

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
    >>> molar_free_volume(data.atomnos, data.atomcoords, method="izato") \
    ...     / (constants.angstrom ** 3 * constants.N_A)
    0.05
    >>> molar_free_volume(data.atomnos, data.atomcoords) \
    ...     / (constants.angstrom ** 3 * constants.N_A)
    61.
    >>> molar_free_volume(data.atomnos, data.atomcoords, environment="benzene") \
    ...     / (constants.angstrom ** 3 * constants.N_A)
    7.7e2

    >>> data = datasets.logfiles["symmetries"]["water"]
    >>> molar_free_volume(data.atomnos, data.atomcoords, method="izato") \
    ...     / (constants.angstrom ** 3 * constants.N_A)
    0.09
    >>> molar_free_volume(data.atomnos, data.atomcoords) \
    ...     / (constants.angstrom ** 3 * constants.N_A)
    92.
    >>> molar_free_volume(data.atomnos, data.atomcoords, environment="benzene") \
    ...     / (constants.angstrom ** 3 * constants.N_A)
    90e1

    >>> data = datasets.logfiles["symmetries"]["benzene"]
    >>> molar_free_volume(data.atomnos, data.atomcoords, method="izato") \
    ...     / (constants.angstrom ** 3 * constants.N_A)
    0.17
    >>> molar_free_volume(data.atomnos, data.atomcoords) \
    ...     / (constants.angstrom ** 3 * constants.N_A)
    240.
    >>> molar_free_volume(data.atomnos, data.atomcoords, environment="benzene") \
    ...     / (constants.angstrom ** 3 * constants.N_A)
    593.
    """
    if method == "izato":
        vdw_volume, cav_volume, _ = coords.get_molecular_volume(
            atomnos,
            atomcoords,
            method="izato",
            full_output=True,
        )
        r_M, r_cav = np.cbrt(vdw_volume), np.cbrt(cav_volume)
        molar_free_volume = (r_cav - r_M) ** 3 * constants.angstrom**3 * constants.N_A
    elif method == "garza":
        # TODO(schneiderfelipe): test for the following solvents: water,
        # pentane, hexane, heptane and octane.
        cav_volume, N_cav, _ = coords._garza(
            coords.get_molecular_volume(atomnos, atomcoords),
            environment,
            full_output=True,
            temperature=temperature,
            pressure=pressure,
        )
        molar_free_volume = N_cav * cav_volume * constants.angstrom**3 * constants.N_A
    else:
        msg = f"unrecognized method: '{method}'"
        raise ValueError(msg)
    logger.debug(f"molar free volume = {molar_free_volume} Å³")
    return molar_free_volume
