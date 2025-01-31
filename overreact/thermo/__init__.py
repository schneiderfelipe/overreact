"""Module dedicated to the calculation of thermodynamic properties."""

from __future__ import annotations

__all__ = ["equilibrium_constant", "change_reference_state"]


import logging

import numpy as np
from overreact._misc import _derivative as derivative
from scipy.special import factorial

import overreact as rx
from overreact import _constants as constants
from overreact.thermo._gas import (
    calc_elec_energy,
    calc_elec_entropy,
    calc_rot_energy,
    calc_rot_entropy,
    calc_trans_energy,
    calc_vib_energy,
    calc_vib_entropy,
    molar_volume,
)

logger = logging.getLogger(__name__)


def calc_trans_entropy(
    atommasses,
    atomnos=None,
    atomcoords=None,
    environment="gas",
    method="standard",
    temperature=298.15,
    pressure=constants.atm,
):
    r"""Calculate the translational entropy of an ideal gas.

    This implements the Sackur-Tetrode equation (equation 5-20 of Statistical
    Thermodynamics, McQuarrie). See also
    <https://en.wikipedia.org/wiki/Sackur%E2%80%93Tetrode_equation#Derivation_from_information_theoretic_perspective>.

    Parameters
    ----------
    atommasses : array-like
        Atomic masses or molecular mass in atomic mass units (amu).
    atomnos : array-like, optional
    atomcoords : array-like, optional
        Atomic coordinates.
    environment : str, optional
        Choose between "gas" and a solvent.
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
    temperature : array-like, optional
        Absolute temperature in Kelvin.
    pressure : array-like, optional
        Reference gas pressure.

    Returns
    -------
    float
        Translational entropy in J/mol·K.

    Raises
    ------
    ValueError
        If environment is "solid".

    Examples
    --------
    >>> calc_trans_entropy(35.45)  # Cl-
    153.246
    >>> calc_trans_entropy(35.45, pressure=constants.bar)
    153.356

    >>> calc_trans_entropy(35.45, [17], [[0, 0, 0]], environment="water")
    153.246

    As we can see, the "environment" parameter has only effect if set together
    with a proper "method":

    >>> calc_trans_entropy(35.45, [17], [[0, 0, 0]], environment="water",
    ...                    method="garza")
    103.7
    >>> calc_trans_entropy(35.45, [17], [[0, 0, 0]], environment="water",
    ...                    method="izato")
    51.

    >>> calc_trans_entropy(35.45, [17], [[0, 0, 0]], environment="benzene",
    ...                    method="garza")
    121.7
    """
    # TODO(schneiderfelipe): This is probably an ugly hack for zero temperature and
    # certainly wrong (https://physics.stackexchange.com/a/400431/77366).
    # See https://physics.stackexchange.com/a/468649/77366 and
    # https://physics.stackexchange.com/a/335828/77366 for further details on what we
    # should do (disclaimer: no Sackur-Tetrode at 0 K!).
    if np.isclose(temperature, 0.0):
        logger.warning("assuming translational entropy zero at zero temperature")
        return 0.0

    if environment in {"gas", None} or method == "standard":
        volume = molar_volume(temperature=temperature, pressure=pressure)
    elif environment == "solid":
        msg = f"environment not yet implemented: {environment}"
        raise ValueError(msg)
    else:
        assert atomnos is not None, "atomnos must be provided"
        assert atomcoords is not None, "atomcoords must be provided"
        volume = rx.thermo._solv.molar_free_volume(
            atomnos=atomnos,
            atomcoords=atomcoords,
            environment=environment,
            method=method,
            temperature=temperature,
            pressure=pressure,
        )

    translational_entropy = rx.thermo._gas._sackur_tetrode(
        atommasses,
        volume,
        temperature=temperature,
    )
    logger.info(
        f"translational entropy = {translational_entropy} J/mol·K",
    )
    return translational_entropy


# TODO(schneiderfelipe): "energy" has potentially two meanings here. Correct
# naming for the whole package?
def calc_internal_energy(
    energy=0.0,
    degeneracy=1,
    moments=None,
    vibfreqs=None,
    qrrho=True,
    temperature=298.15,
):
    """Calculate internal energy.

    Parameters
    ----------
    energy : array-like, optional
        Energies for different states, in J/mol.
    degeneracy : array-like, optional
        Degeneracies of the states of the molecule. This is normally the same
        as spin multiplicity, but might be total angular momentum degeneracy in
        some cases (e.g., fluorine).
    moments : array-like
        Primary moments of inertia in ascending order. Units are in amu * Å**2.
    vibfreqs : array-like
        Frequency magnitudes in cm-1.
    qrrho : bool, optional
        Apply the quasi-rigid rotor harmonic oscillator (QRRHO) approximation of
        M. Head-Gordon and others (see
        [*J. Phys. Chem. C* **2015**, 119, 4, 1840-1850](http://dx.doi.org/10.1021/jp509921r)) on top of the classical
        RRHO.
    temperature : array-like, optional
        Absolute temperature in Kelvin.

    Returns
    -------
    float
        Internal energy in J/mol.

    Examples
    --------
    >>> calc_internal_energy()  # F
    3718.

    The example above ignores the electronic energy. Taking electronic energy
    into account (data taken from the NIST atomic data
    (<https://physics.nist.gov/PhysRefData/Handbook/Tables/fluorinetable5.htm>);
    similar data can also be found in Table 5-1 of Statistical Thermodynamics,
    McQuarrie):

    >>> j = np.array([3/2, 1/2, 5/2, 3/2, 1/2, 3/2, 1/2])
    >>> degeneracy = 2 * j + 1
    >>> energy = np.array([0.000, 404.141, 102405.714, 102680.439,  # cm-1
    ...                    102840.378, 104731.048, 105056.283])
    >>> calc_internal_energy(
    ...     energy=energy * 100 * constants.h * constants.c * constants.N_A,
    ...     degeneracy=degeneracy)  # F
    4039.

    """
    internal_energy = (
        calc_trans_energy(temperature=temperature)
        + calc_elec_energy(energy, degeneracy, temperature=temperature)
        + calc_rot_energy(moments, temperature=temperature)
        + calc_vib_energy(vibfreqs, qrrho=qrrho, temperature=temperature)
    )
    logger.info(f"internal energy = {internal_energy} J/mol")
    return internal_energy


# TODO(schneiderfelipe): "energy" has potentially two meanings here. Correct
# naming for the whole package?
def calc_enthalpy(
    energy=0.0,
    degeneracy=1,
    moments=None,
    vibfreqs=None,
    qrrho=True,
    temperature=298.15,
):
    """Calculate enthalpy.

    This function uses `calc_internal_energy` and adds a volume work term to it.

    Parameters
    ----------
    energy : array-like, optional
        Energies for different states, in J/mol.
    degeneracy : array-like, optional
        Degeneracies of the states of the molecule. This is normally the same
        as spin multiplicity, but might be total angular momentum degeneracy in
        some cases (e.g., fluorine).
    moments : array-like
        Primary moments of inertia in ascending order. Units are in amu * Å**2.
    vibfreqs : array-like
        Frequency magnitudes in cm-1.
    qrrho : bool, optional
        Apply the quasi-rigid rotor harmonic oscillator (QRRHO) approximation of
        M. Head-Gordon and others (see
        [*J. Phys. Chem. C* **2015**, 119, 4, 1840-1850](http://dx.doi.org/10.1021/jp509921r)) on top of the classical
        RRHO.
    temperature : array-like, optional
        Absolute temperature in Kelvin.

    Returns
    -------
    float
        Enthalpy in J/mol.

    Examples
    --------
    >>> calc_enthalpy()  # F
    6197.

    The example above ignores the electronic energy. Taking electronic energy
    into account (data taken from the NIST atomic data
    (<https://physics.nist.gov/PhysRefData/Handbook/Tables/fluorinetable5.htm>);
    similar data can also be found in Table 5-1 of Statistical Thermodynamics,
    McQuarrie):

    >>> j = np.array([3/2, 1/2, 5/2, 3/2, 1/2, 3/2, 1/2])
    >>> degeneracy = 2 * j + 1
    >>> energy = np.array([0.000, 404.141, 102405.714, 102680.439,  # cm-1
    ...                    102840.378, 104731.048, 105056.283])
    >>> calc_enthalpy(energy=energy * 100 * constants.h * constants.c * constants.N_A,
    ...               degeneracy=degeneracy)  # F
    6518.

    """
    temperature = np.asarray(temperature)

    enthalpy = (
        calc_internal_energy(
            energy=energy,
            degeneracy=degeneracy,
            moments=moments,
            vibfreqs=vibfreqs,
            qrrho=qrrho,
            temperature=temperature,
        )
        + constants.R * temperature
    )
    logger.info(f"enthalpy = {enthalpy} J/mol")
    return enthalpy


# TODO(schneiderfelipe): "energy" has potentially two meanings here. Correct
# naming for the whole package?
def calc_entropy(
    atommasses,
    atomnos=None,
    atomcoords=None,
    energy=0.0,
    degeneracy=1,
    moments=None,
    symmetry_number=1,
    vibfreqs=None,
    environment="gas",
    method="standard",
    qrrho=True,
    temperature=298.15,
    pressure=constants.atm,
):
    """Calculate entropy.

    Either the classical gas phase or solvation entropies are available. For
    solvation entropies, the method of A. Garza (doi:10.1021/acs.jctc.9b00214)
    is available and recommended.

    Parameters
    ----------
    atommasses : array-like
        Atomic masses or molecular mass in atomic mass units (amu).
    atomnos : array-like, optional
    atomcoords : array-like, optional
        Atomic coordinates.
    energy : array-like, optional
        Energies for different states, in J/mol.
    degeneracy : array-like, optional
        Degeneracies of the states of the molecule. This is normally the same
        as spin multiplicity, but might be total angular momentum degeneracy in
        some cases (e.g., fluorine).
    moments : array-like
        Primary moments of inertia in ascending order. Units are in amu * Å**2.
    symmetry_number : int, optional
    vibfreqs : array-like
        Frequency magnitudes in cm-1.
    environment : str, optional
        Choose between "gas" and a solvent.
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
    qrrho : bool, optional
        Apply the quasi-rigid rotor harmonic oscillator (QRRHO) approximation of
        S. Grimme (see
        [*Theory. Chem. Eur. J.*, **2012**, 18: 9955-9964](https://doi.org/10.1002/chem.201200497)) on top of the classical
        RRHO.
    temperature : array-like, optional
        Absolute temperature in Kelvin.
    pressure : array-like, optional
        Reference gas pressure.

    Notes
    -----
    When solvation entropies are requested, the returned values include the
    reference state transformation from the gas phase to the 1 M liquid
    reference state.

    Returns
    -------
    float
        Entropy in J/mol·K.

    Raises
    ------
    ValueError
        If environment is "solid".

    Notes
    -----
    The improved solvation entropy model is a work in progress!

    Examples
    --------
    >>> calc_entropy(18.998)  # F
    145.467

    The example above ignores the electronic entropy. Taking electronic entropy
    into account (data taken from the NIST atomic data
    (<https://physics.nist.gov/PhysRefData/Handbook/Tables/fluorinetable5.htm>);
    similar data can also be found in Table 5-1 of Statistical Thermodynamics,
    McQuarrie):

    >>> j = np.array([3/2, 1/2, 5/2, 3/2, 1/2, 3/2, 1/2])
    >>> degeneracy = 2 * j + 1
    >>> energy = np.array([0.000, 404.141, 102405.714, 102680.439,  # cm-1
    ...                    102840.378, 104731.048, 105056.283])
    >>> calc_entropy(18.998,
    ...     energy=energy * 100 * constants.h * constants.c * constants.N_A,
    ...     degeneracy=degeneracy)  # F
    158.641

    The following calculates the solvation entropy of a hydrogen atom in water:

    >>> calc_entropy(1.008, 1, [[0, 0, 0]], environment="water")  # doctest: +SKIP
    10.5
    """
    entropy = (
        calc_trans_entropy(
            atommasses=atommasses,
            atomnos=atomnos,
            atomcoords=atomcoords,
            environment=environment,
            method=method,
            temperature=temperature,
            pressure=pressure,
        )
        + calc_elec_entropy(
            energy=energy,
            degeneracy=degeneracy,
            temperature=temperature,
        )
        + calc_rot_entropy(
            atommasses=atommasses,
            atomnos=atomnos,
            atomcoords=atomcoords,
            moments=moments,
            symmetry_number=symmetry_number,
            environment=environment,
            method=method,
            temperature=temperature,
            pressure=pressure,
        )
        + calc_vib_entropy(vibfreqs=vibfreqs, qrrho=qrrho, temperature=temperature)
    )

    if environment in {"gas", None}:
        pass
    elif environment == "solid":
        msg = f"environment not yet implemented: {environment}"
        raise ValueError(msg)
    else:
        concentration_correction = -change_reference_state(
            temperature=temperature,
            pressure=pressure,
        )
        logger.debug(
            f"concentration correction = {concentration_correction} J/mol·K",
        )
        entropy = entropy + concentration_correction
        if method == "standard":
            pass
        else:
            assert atomnos is not None, "atomnos must be provided"
            assert atomcoords is not None, "atomcoords must be provided"
            # TODO(schneiderfelipe): this includes "izato", "garza" and
            # possibly future methods for extra entropy terms such as cavity.
            entropy = entropy + rx.thermo._solv.calc_cav_entropy(
                atomnos=atomnos,
                atomcoords=atomcoords,
                environment=environment,
                temperature=temperature,
                pressure=pressure,
            )  # TODO(schneiderfelipe): check extra options for calc_cav_entropy.
    logger.info(f"entropy = {entropy} J/mol·K")
    return entropy


def calc_heat_capacity(
    energy=0.0,
    degeneracy=1,
    moments=None,
    vibfreqs=None,
    qrrho=True,
    temperature=298.15,
    dx=3e-5,
    order=3,
):
    """Calculate heat capacity by finite differences on energy.

    Parameters
    ----------
    energy : array-like, optional
        Energies for different states, in J/mol.
    degeneracy : array-like, optional
        Degeneracies of the states of the molecule. This is normally the same
        as spin multiplicity, but might be total angular momentum degeneracy in
        some cases (e.g., fluorine).
    moments : array-like
        Primary moments of inertia in ascending order. Units are in amu * Å**2.
    vibfreqs : array-like
        Frequency magnitudes in cm-1.
    qrrho : bool, optional
        Apply the quasi-rigid rotor harmonic oscillator (QRRHO) approximation of
        M. Head-Gordon and others (see
        [*J. Phys. Chem. C* **2015**, 119, 4, 1840-1850](http://dx.doi.org/10.1021/jp509921r)) on top of the classical
        RRHO.
    temperature : array-like, optional
        Absolute temperature in Kelvin.
    dx : float, optional
        Spacing.
    order : int, optional
        Number of points to use, must be odd.

    Returns
    -------
    float
        Heat capacity in J/mol·K.

    Examples
    --------
    >>> calc_heat_capacity()  # F
    12.47

    The example above ignores the electronic energy. Taking electronic energy
    into account (data taken from the NIST atomic data
    (<https://physics.nist.gov/PhysRefData/Handbook/Tables/fluorinetable5.htm>);
    similar data can also be found in Table 5-1 of Statistical Thermodynamics,
    McQuarrie):

    >>> j = np.array([3/2, 1/2, 5/2, 3/2, 1/2, 3/2, 1/2])
    >>> degeneracy = 2 * j + 1
    >>> energy = np.array([0.000, 404.141, 102405.714, 102680.439,  # cm-1
    ...                    102840.378, 104731.048, 105056.283])
    >>> calc_heat_capacity(
    ...     energy=energy * 100 * constants.h * constants.c * constants.N_A,
    ...     degeneracy=degeneracy)  # F
    14.43

    """

    def func(temperature):
        return calc_internal_energy(
            energy=energy,
            degeneracy=degeneracy,
            moments=moments,
            vibfreqs=vibfreqs,
            qrrho=qrrho,
            temperature=temperature,
        )

    heat_capacity = derivative(func, x0=temperature, dx=dx, n=1, order=order)
    logger.info(f"heat capacity = {heat_capacity} J/mol·K")
    return heat_capacity


def get_molecularity(transform):
    """Calculate molecularity of a chemical transformation.

    The returned value is the reaction order, i.e., number of molecules that come
    together to react. This number is always at least one.

    Parameters
    ----------
    transform : array-like

    Returns
    -------
    array-like

    Examples
    --------
    >>> get_molecularity([-1, 1])
    array(1)
    >>> get_molecularity([-1, 0])
    array(1)
    >>> get_molecularity([-1, -1, 1])
    array(2)
    >>> get_molecularity([-1, -1, 0])
    array(2)
    >>> get_molecularity([[-1.,  1.,  1.],
    ...                   [-1.,  1.,  0.],
    ...                   [ 1., -1., -1.],
    ...                   [ 0.,  0.,  0.],
    ...                   [ 0.,  0.,  1.]])
    array([2, 1, 1])
    >>> get_molecularity([[0.], [0.]])
    array([1])
    """
    res = np.sum(np.asarray(transform) < 0, axis=0)
    return np.where(res > 0, res, 1)


def get_delta(transform, property):
    """Calculate deltas according to reactions.

    Delta properties are differences in a property between the final and
    initial state of a chemical transformation. They are calculated from
    matrices representing the transformation and the absolute properties.
    Transformation matrices are expected to have column-wise transformation
    defined.

    Very useful for the calculation of reaction and activation free energies
    from absolute free energies of compounds. Matrices ``A`` and ``B`` of a
    `Scheme` represent the transformations associated with reaction and
    activation free energies, respectively.

    Parameters
    ----------
    transform : array-like
    property : array-like

    Returns
    -------
    delta_property : array-like

    Examples
    --------
    >>> get_delta([-1, 1], [-10, 10])
    20

    You must ensure the transformation is properly defined, as no test is made
    to ensure, e.g., conservation of matter:

    >>> get_delta([-1, 0], [-10, 20])
    10

    Normally, transformations are given as columns in a matrix:

    >>> get_delta([[-1, -2],
    ...            [ 1,  3]], [-5, 12])
    array([17, 46])
    """
    return np.asarray(transform).T @ np.asarray(property)


def equilibrium_constant(
    delta_freeenergy: float | np.ndarray,
    delta_moles: int | np.ndarray | None = None,
    temperature: float | np.ndarray = 298.15,
    pressure: float = constants.atm,
    volume: float | None = None,
):
    r"""Calculate an equilibrium constant from a reaction [Gibbs free energy](https://en.wikipedia.org/wiki/Gibbs_free_energy).

    This function uses the usual `relationship between reaction Gibbs energy
    and equilibrium constant
    <https://en.wikipedia.org/wiki/Equilibrium_constant>`_:

    .. math::
        K(T) = \exp\left(-\frac{\Delta_\text{r} G^\circ}{R T}\right)

    If `delta_moles` is given, the above will be multiplied by a term
    :math:`\left( \frac{R T}{p} \right)^{-\Delta n}`, which effectively
    transforms a :math:`K_p` equilibrium constant into a :math:`K_c`
    (see below).

    Parameters
    ----------
    delta_freeenergy : array-like
        Delta Gibbs reaction free energies. **This assumes values were already
        corrected for a one molar reference state.**
    delta_moles : array-like, optional
        Difference in moles between products and reactants. If set, this
        multiplies the end result by
        :math:`\left( \frac{R T}{p} \right)^{-\Delta n}`, which effectively
        calculates a solution equilibrium constant for gas phase data. You
        should set this to `None` if your free energies were already adjusted
        for solution Gibbs free energies.
    temperature : array-like, optional
        Absolute temperature in Kelvin.
    pressure : array-like, optional
        Reference gas pressure.
    volume : float, optional
        Molar volume. This substitutes :math:`\frac{R T}{p}` if given. See
        `delta_moles`.

    Returns
    -------
    K : array-like
        Equilibrium constant.

    Notes
    -----
    If you want to calculate a solution equilibrium constant from gas phase data, set
    `delta_moles` to the difference in moles between products and reactants.
    Alternatively, convert your energies to solution Gibbs free energies and
    set `delta_moles` to `None`.

    Examples
    --------
    The following is an example from the
    [LibreTexts Chemistry Library](https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Supplemental_Modules_(Physical_and_Theoretical_Chemistry)/Equilibria/Chemical_Equilibria/The_Equilibrium_Constant).
    Consider the following equilibrium:

    .. math::
        \ce{2 SO2(g) + O2(g) <=> 2 SO3(g)}

    with concentrations :math:`c_{\ce{SO2(g)}} = 0.2 M`,
    :math:`c_{\ce{O2(g)}} = 0.5 M` and :math:`c_{\ce{SO3(g)}} = 0.7 M` (room
    temperature). Its :math:`K_c` is given by:

    >>> Kc = 0.7**2 / (0.2**2 * 0.5)
    >>> Kc
    24.5

    You could use `equilibrium_constant` to reach the same result using the
    Gibbs reaction free energy (which can easily be obtained from
    :math:`K_c`):

    >>> temperature = 298.15
    >>> dG = -constants.R * temperature * np.log(Kc)
    >>> equilibrium_constant(dG)
    array([24.5])

    By giving a `delta_moles` value (in this case, :math:`2 - 2 - 1 = -1`),
    we can calculate the corresponding `K_p`:

    >>> equilibrium_constant(dG, delta_moles=-1)
    array([1.002])

    (As expected, it makes sense for gases to favor the most entropic side of
    the equilibrium.) The example above clearly used "solution-based" data
    (our :math:`K_c` was calculated using molar quantities, which means
    reference volumes of one liter). You could convert it to gas phase data
    to get the same result, by changing the reference state (in this case,
    from one molar to one atmosphere using `change_reference_state`):

    >>> dG += temperature * rx.change_reference_state()
    >>> equilibrium_constant(dG)
    array([1.002])

    Having gas phase information, the inverse path can be taken just by
    inverting the sign of `delta_moles`:

    >>> equilibrium_constant(dG, delta_moles=1)
    array([24.5])

    The following example is from
    [Wikipedia](https://en.wikipedia.org/wiki/Stability_constants_of_complexes#The_chelate_effect).
    The reactions are two copper complex forming equilibria with two
    different ligands. Since this reaction takes place in solution, it is the
    solution standard Gibbs reaction free energy that is given:

    >>> dG1 = -37.4e3
    >>> np.log10(equilibrium_constant(dG1))
    array([6.55])
    >>> dG2 = -60.67e3
    >>> np.log10(equilibrium_constant(dG2))
    array([10.62])

    The above are thus :math:`\log_{10}(K_c)`. Since we are talking about a
    mono- and a bidendate ligands, the `delta_moles` are -4 and -2,
    respectively, and we could obtain the :math:`\log_{10}(K_p)` the following
    way:

    >>> np.log10(equilibrium_constant(dG1, delta_moles=-4))
    array([0.998])
    >>> np.log10(equilibrium_constant(dG2, delta_moles=-2))
    array([7.85])

    You can easily check that the above values match the values given
    [here](https://en.wikipedia.org/wiki/Stability_constants_of_complexes#The_chelate_effect).
    """
    temperature = np.asarray(temperature)

    equilibrium_constant = np.exp(
        -np.atleast_1d(delta_freeenergy) / (constants.R * temperature),
    )

    if delta_moles is not None:
        if volume is None:
            volume = molar_volume(temperature, pressure) * constants.kilo

        equilibrium_constant *= volume**delta_moles

    logger.info(f"equilibrium constant = {equilibrium_constant}")
    return equilibrium_constant


def change_reference_state(
    new_reference: float = 1.0 / constants.liter,
    old_reference: float | None = None,
    sign: int = 1,
    temperature: float | np.ndarray = 298.15,
    pressure: float = constants.atm,
    volume: float | None = None,
):
    r"""Calculate an additive entropy correction to a change in reference states.

    .. math::
        \Delta G_\text{corr} =
            R T \ln \left( \frac{\chi_\text{new}}{\chi_\text{old}} \right)

    The value returned can be directly multiplied by temperature and summed to
    the old reference free energies to obtain free energies with respect to a
    new reference. See notes below.

    For instance, the concentration correction to Gibbs free energy for a
    gas-to-liquid standard state change is simply
    (:math:`c^\circ = \frac{\text{1 atm}}{R T}`),

    .. math::
        \Delta G_\text{conc} =
            R T \ln \left( \frac{\text{1 M}}{c^\circ} \right)

    Parameters
    ----------
    new_reference : array-like, optional
        New reference state. Default value corresponds to 1 mol/liter.
    old_reference : array-like, optional
        Old reference state. Default value corresponds to the concentration of
        an ideal gas at the given temperature and 1 atm.
    sign : float, optional
        Sign of the change in reference state. Default value is 1. This only
        multiplies the final result.
    temperature : array-like, optional
        Absolute temperature in Kelvin.
    pressure : array-like, optional
        Reference gas pressure.
    volume : float, optional
        Molar volume.

    Returns
    -------
    correction : array-like
        Entropy correction in J/mol·K.

    Notes
    -----
    This function can be used to add any entropy correction in the form above.
    The only drawback is that, sometimes, those corrections are written with a
    minus sign in front of them (this implies switching the roles of
    `old_reference` and `new_reference`). The easiest way to accomplish this is
    by using ``sign=-1`` or multiplying the result by ``-1``.

    Examples
    --------
    By default, the correction returns a change in concentration from the gas
    phase standard concentration to the solvated-state standard concentration:

    >>> -rx.change_reference_state() / constants.calorie
    -6.4
    >>> 298.15 * rx.change_reference_state() / constants.kcal
    1.89
    >>> 273.15 * rx.change_reference_state(temperature=273.15) / constants.kcal
    1.69

    But this function can also be used to adjust symmetry effects from C1
    calculations (symmetry number equals to one). For D7h, for instance, the
    symmetry number is 14:

    >>> -298.15 * rx.change_reference_state(14, 1) / constants.kcal
    -1.56

    >>> rx.change_reference_state(sign=-1) == -rx.change_reference_state()
    True

    """
    temperature = np.asarray(temperature)

    if old_reference is None:
        if volume is None:
            volume = molar_volume(temperature=temperature, pressure=pressure)
        old_reference = 1.0 / volume

    return sign * constants.R * np.log(new_reference / old_reference)


# TODO(schneiderfelipe): we need a concrete example of this for testing.
def get_reaction_entropies(transform, temperature=298.15, pressure=constants.atm):
    r"""Calculate entropy contributions from the overall reaction structure.

    This function currently implements the reaction translational entropy, a
    result of the indistinguishability of reactants or products, i.e., a
    difference in entropy of :math:`R ln 2!` for the reactions

    .. math::
        \ce{A + B -> C}

    and

    .. math::
        \ce{2A -> C}

    Parameters
    ----------
    transform : array-like
    temperature : array-like, optional
        Absolute temperature in Kelvin.
    pressure : array-like, optional
        Reference gas pressure.

    Returns
    -------
    delta_entropy : array-like

    Examples
    --------
    >>> import overreact as rx

    >>> scheme = rx.parse_reactions("A + B <=> C")
    >>> get_reaction_entropies(scheme.A)
    array([0.0, 0.0])
    >>> scheme = rx.parse_reactions("2A <=> C")
    >>> get_reaction_entropies(scheme.A)
    array([-5.763, 5.763])
    >>> get_reaction_entropies(scheme.A, temperature=400.0)
    array([-5.763, 5.763])

    >>> scheme = rx.parse_reactions("A <=> B + C")
    >>> get_reaction_entropies(scheme.A)
    array([0.0, 0.0])
    >>> scheme = rx.parse_reactions("A <=> 2B")
    >>> get_reaction_entropies(scheme.A)
    array([ 5.763, -5.763])

    >>> scheme = rx.parse_reactions("A + B -> E# -> C")
    >>> get_reaction_entropies(scheme.A)
    array([0.0])
    >>> get_reaction_entropies(scheme.B)
    array([0.0])
    >>> scheme = rx.parse_reactions("2A -> E# -> C")
    >>> get_reaction_entropies(scheme.A)
    array([-5.763])
    >>> get_reaction_entropies(scheme.B)
    array([-5.763])
    """
    sym = factorial(np.abs(np.asarray(transform)))
    return np.sum(
        np.sign(transform)
        * change_reference_state(sym, 1, temperature=temperature, pressure=pressure),
        axis=0,
    )
