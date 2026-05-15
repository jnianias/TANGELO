"""
Tools for computing and retrieving physical source properties from the megatable.
"""

import numpy as np
from astropy.table import Table
from astropy.cosmology import Planck18
import astropy.units as u

from .spectroscopy import muse_lsf_fwhm_poly
from .constants import wavedict, doublets


def flux_to_luminosity(flux: np.ndarray, flux_err: np.ndarray,
                       z: np.ndarray, mu: np.ndarray,
                       is_continuum: bool = False,
                       cosmo=Planck18) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert observed flux (or flux density) to intrinsic luminosity, correcting for
    gravitational lensing magnification.

    Parameters
    ----------
    flux : np.ndarray
        Observed flux in units of 1e-20 erg/s/cm². For continuum, flux density in
        units of 1e-20 erg/s/cm²/Å.
    flux_err : np.ndarray
        Uncertainty on ``flux``, in the same units.
    z : np.ndarray
        Source redshifts.
    mu : np.ndarray
        Lensing magnification values. The intrinsic flux is flux / mu.
    is_continuum : bool, optional
        If True, treat ``flux`` as a flux density (per Å) and apply a (1+z)^{-1}
        K-correction to convert from observed-frame to rest-frame bandwidth.
        Default is False (line flux, integrated over wavelength).
    cosmo : astropy cosmology, optional
        Cosmology used to compute the luminosity distance. Default is Planck18.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Luminosity and its propagated uncertainty, both in erg/s (line) or
        erg/s/Å (continuum).
    """
    # Luminosity distance in cm
    dl_cm = cosmo.luminosity_distance(z).to(u.cm).value  # shape matches z

    # Guard against unphysical magnification values (≤ 0 signals missing/bad data)
    mu = np.where(np.asarray(mu) > 0, mu, np.nan)

    # Negative flux values are unphysical (e.g. calibration artefacts in continuum
    # estimates); replace with NaN so they propagate cleanly rather than producing
    # negative luminosities or sign-flipped errors
    flux = np.array(flux, dtype=float)
    flux_err = np.array(flux_err, dtype=float)
    flux_err[flux < 0] = np.nan
    flux[flux < 0] = np.nan

    # Negative error values are unphysical — they are either sentinel flags or
    # sign artefacts. NaN them rather than taking abs, so that a sentinel like
    # -99 cannot become a catastrophically large error bar after scaling by 4πD_L².
    flux_err[flux_err <= 0] = np.nan

    # Lensing-corrected flux in physical units (erg/s/cm² or erg/s/cm²/Å)
    f_intrinsic     = flux     / mu * 1e-20
    f_intrinsic_err = flux_err / mu * 1e-20

    # Luminosity: L = 4π D_L² f
    factor = 4.0 * np.pi * dl_cm**2
    lum     = factor * f_intrinsic
    lum_err = factor * f_intrinsic_err

    # For continuum (flux density per observed-frame Å), divide by (1+z) to
    # convert to rest-frame bandwidth
    if is_continuum:
        lum     /= (1.0 + z)
        lum_err /= (1.0 + z)

    return lum, lum_err


def correct_inst_res(col: np.ndarray, col_err: np.ndarray,
                     lpeakr: np.ndarray, prop: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply instrumental resolution correction for FWHM and DISP using the MUSE LSF polynomial,
    with proper error propagation.

    Parameters
    ----------
    col : np.ndarray
        The column to correct (FWHM or DISP).
    col_err : np.ndarray
        The error on the column to correct.
    lpeakr : np.ndarray
        The observed wavelength of the red peak of Lyman alpha, used to determine the 
        instrumental resolution from the MUSE LSF polynomial.
    prop : str
        The property being corrected ('FWHM' or 'DISP').

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The instrumentally corrected column and its propagated error.
    """
    lsf_fwhm = muse_lsf_fwhm_poly(lpeakr)

    if prop == "DISP":
        lsf = lsf_fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to dispersion
    elif prop == "FWHM":
        lsf = lsf_fwhm
    else:
        raise ValueError("Invalid property for instrumental correction. Must be 'FWHM' or 'DISP'.")

    # Store original values for error propagation
    col_obs = col.copy()

    # Quadrature subtraction: corrected = sqrt(obs^2 - lsf^2)
    corrected_col = np.sqrt(np.maximum(col_obs**2 - lsf**2, 0))

    # Error propagation: d(corrected)/d(obs) = obs / corrected
    # Handle cases where corrected_col is near zero to avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        err_factor = col_obs / np.maximum(corrected_col, 1e-30)
        corrected_err = col_err * err_factor

    # Set error to large value where correction was forced to zero (obs < lsf)
    invalid_mask = (col_obs**2 - lsf**2) < 0
    corrected_err[invalid_mask] = np.inf

    return corrected_col, corrected_err


# ---------------------------------------------------------------------------
# Private helpers for get_line_property
# ---------------------------------------------------------------------------

def _line_abs_outvel(megatab: Table, line: str) -> tuple[np.ndarray, np.ndarray]:
    """Return outflow velocity of a stacked absorption composite relative to the systemic redshift.

    ``DV_{line}`` is the velocity of the absorption centroid relative to the
    Lyman-alpha red peak (negative = outflow / blueshift). ``DELTAV_LYA`` is
    the velocity of the Lya red peak relative to systemic (positive = Lya
    redshifted from systemic). The velocity relative to systemic is therefore:

        OUTVEL = DV_{line} + DELTAV_LYA
    """
    dv_col  = f"DV_{line}"
    dv_err_col = f"DV_{line}_ERR"
    required = [dv_col, dv_err_col, "DELTAV_LYA", "DELTAV_LYA_ERR"]
    if not all(c in megatab.colnames for c in required):
        raise ValueError(
            f"Required columns for OUTVEL of {line} not found in megatable "
            f"(need {dv_col}, {dv_err_col}, DELTAV_LYA, DELTAV_LYA_ERR)."
        )
    dv      = np.asarray(megatab[dv_col],          dtype=float)
    dv_err  = np.asarray(megatab[dv_err_col],      dtype=float)
    deltav     = np.asarray(megatab["DELTAV_LYA"],    dtype=float)
    deltav_err = np.asarray(megatab["DELTAV_LYA_ERR"], dtype=float)
    outvel     = dv + deltav
    outvel_err = np.sqrt(dv_err**2 + deltav_err**2)
    return -outvel, outvel_err


def _line_stacked_abs(megatab: Table, line: str, prop: str) -> tuple[np.ndarray, np.ndarray]:
    """Return property for a stacked absorption-line composite (LI_ABS, HI_ABS, TOT_ABS)."""
    if prop == "OUTVEL":
        return _line_abs_outvel(megatab, line)
    col_name = f"{prop}_{line}"
    err_name = f"{prop}_{line}_ERR"
    col = megatab[col_name] if col_name in megatab.colnames else None
    err = megatab[err_name] if err_name in megatab.colnames else None
    if col is None or err is None:
        raise ValueError(f"Required columns for {prop} of {line} not found in megatable.")
    # Flip EW sign so positive = stronger absorption
    return (-col, err) if prop == "EW" else (col, err)


def _line_doublet_flux(megatab: Table, line: str) -> tuple[np.ndarray, np.ndarray]:
    """Return combined flux and propagated error for a doublet line pair."""
    line1, line2 = doublets[line]
    flux1 = np.asarray(megatab[f"FLUX_{line1}"], dtype=float)
    flux2 = np.asarray(megatab[f"FLUX_{line2}"], dtype=float)
    err1  = np.asarray(megatab[f"FLUX_ERR_{line1}"], dtype=float)
    err2  = np.asarray(megatab[f"FLUX_ERR_{line2}"], dtype=float)
    flux_total = flux1 + flux2
    err_total  = np.sqrt(err1**2 + err2**2)
    # Propagate NaN from either component to the combined error
    err_total[np.isnan(flux1) | np.isnan(flux2)] = np.nan
    return flux_total, err_total


def _line_ew_from_flux_cont(megatab: Table, line: str,
                             rest_frame: bool, flip_sign: bool) -> tuple[np.ndarray, np.ndarray]:
    """Calculate EW from FLUX / CONT columns for a single line."""
    flux_col     = f"FLUX_{line}"
    flux_err_col = f"FLUX_ERR_{line}"
    cont_col     = f"CONT_{line}"
    cont_err_col = f"CONT_ERR_{line}"
    required = [flux_col, flux_err_col, cont_col, cont_err_col]
    if not all(c in megatab.colnames for c in required):
        raise ValueError(f"Required columns for calculating EW of {line} not found in megatable.")
    flux     = megatab[flux_col].copy()
    flux_err = megatab[flux_err_col].copy()
    cont     = megatab[cont_col].copy()
    cont_err = megatab[cont_err_col].copy()
    ew     = flux / cont
    ew_err = np.abs(ew * np.sqrt((flux_err / flux)**2 + (cont_err / cont)**2))
    if rest_frame:
        ew     /= (1 + megatab['z'])
        ew_err /= (1 + megatab['z'])
    if flip_sign:
        ew *= -1
    return ew, ew_err


def _line_fwhm(megatab: Table, line: str, err_name: str,
               rest_frame: bool, correct_inst: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return (optionally corrected and rest-framed) FWHM for a line."""
    fwhm_col = f"FWHM_{line}"
    if fwhm_col not in megatab.colnames or err_name not in megatab.colnames:
        raise ValueError(f"Required column for calculating FWHM of {line} not found in megatable.")
    fwhm     = megatab[fwhm_col].copy()
    fwhm_err = megatab[err_name].copy()
    if correct_inst:
        fwhm, fwhm_err = correct_inst_res(fwhm, fwhm_err, megatab[f'LPEAK_{line}'], "FWHM")
    if rest_frame:
        fwhm     /= (1 + megatab['z'])
        fwhm_err /= (1 + megatab['z'])
    return fwhm, fwhm_err


def _line_cvel(megatab: Table, line: str) -> tuple[np.ndarray, np.ndarray]:
    """Return velocity centroid of a line relative to the systemic redshift."""
    lya_z   = megatab['LPEAKR'] / 1215.67 - 1
    sys_z   = lya_z - megatab['DELTAV_LYA'] / 299792.458 * (1 + lya_z)
    rest_wave = wavedict[line]
    peak_rest     = megatab[f'LPEAK_{line}']     / (1 + sys_z)
    peak_rest_err = megatab[f'LPEAK_ERR_{line}'].copy() / (1 + sys_z)
    cvel     = (peak_rest - rest_wave) / rest_wave * 299792.458
    cvel_err = peak_rest_err           / rest_wave * 299792.458
    return cvel, cvel_err


def _line_luminosity(megatab: Table, line: str, is_continuum: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return line or continuum luminosity for a single line."""
    if is_continuum:
        flux_col, flux_err_col = f"CONT_{line}", f"CONT_ERR_{line}"
    else:
        flux_col, flux_err_col = f"FLUX_{line}", f"FLUX_ERR_{line}"
    if flux_col not in megatab.colnames:
        raise ValueError(f"Column {flux_col} not found in megatable.")
    return flux_to_luminosity(
        megatab[flux_col].copy(), megatab[flux_err_col].copy(),
        np.asarray(megatab['z']), np.asarray(megatab['MU']),
        is_continuum=is_continuum,
    )


def get_line_property(megatab: Table, line: str, prop: str,
                      rest_frame: bool = True, correct_inst: bool = True,
                      abs: bool = False,
                      combine_doublets: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieve a physical property for a spectral line from the megatable.

    Handles derived quantities that are not stored directly (EW computed from
    FLUX/CONT, LSF-corrected FWHM, velocity centroid, luminosity) and applies
    rest-frame corrections where appropriate. Absorption-line EWs are returned
    with a flipped sign so that positive values indicate stronger absorption.

    Parameters
    ----------
    megatab : astropy.table.Table
        The megatable containing the data.
    line : str
        The name of the line (e.g. 'CIV1548', 'SiII1260').
    prop : str
        The property to retrieve. Supported values: 'EW', 'FWHM', 'CVEL', 'FLUX',
        'LUM', 'CONT_LUM', or any column name following the ``{PROP}_{LINE}``
        convention present in ``megatab``.
    rest_frame : bool, optional
        Apply rest-frame correction for wavelength-based properties. Default True.
    correct_inst : bool, optional
        Apply instrumental resolution correction (MUSE LSF) to FWHM. Default True.
    abs : bool, optional
        Treat the line as an absorption line (flips EW sign). Default False.
    combine_doublets : bool, optional
        When True and ``line`` is a doublet key, sum the two components for
        additive properties (EW, FLUX, LUM). Default True.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The requested property column and its associated error column.
    """
    col_name = f"{prop}_{line}"
    err_name = f"{prop}_ERR_{line}"

    # Stacked absorption composites use a different column naming convention
    if line in ["LI_ABS", "HI_ABS", "TOT_ABS"]:
        return _line_stacked_abs(megatab, line, prop)

    # Doublet combining for additive properties
    if combine_doublets and line in doublets and prop in ("EW", "FLUX", "LUM"):
        flux_total, err_total = _line_doublet_flux(megatab, line)
        if prop == "FLUX":
            return flux_total, err_total
        elif prop == "LUM":
            return flux_to_luminosity(
                flux_total, err_total,
                np.asarray(megatab['z']), np.asarray(megatab['MU']),
                is_continuum=False,
            )
        else:  # EW
            line1 = doublets[line][0]
            cont_col     = f"CONT_{line1}"
            cont_err_col = f"CONT_ERR_{line1}"
            if cont_col not in megatab.colnames:
                raise ValueError(f"Required continuum column {cont_col} not found in megatable.")
            cont     = np.asarray(megatab[cont_col], dtype=float)
            cont_err = np.asarray(megatab[cont_err_col], dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                ew     = flux_total / cont
                ew_err = np.abs(ew) * np.sqrt((err_total / flux_total)**2
                                              + (cont_err / cont)**2)
            if rest_frame:
                ew     /= (1 + np.asarray(megatab['z']))
                ew_err /= (1 + np.asarray(megatab['z']))
            if abs:
                ew *= -1
            return ew, ew_err

    if prop == "EW" and col_name not in megatab.colnames:
        return _line_ew_from_flux_cont(megatab, line, rest_frame, flip_sign=abs)
    elif prop == "FWHM":
        return _line_fwhm(megatab, line, err_name, rest_frame, correct_inst)
    elif prop == "CVEL":
        return _line_cvel(megatab, line)
    elif prop == "LUM":
        return _line_luminosity(megatab, line, is_continuum=False)
    elif prop == "CONT_LUM":
        return _line_luminosity(megatab, line, is_continuum=True)
    elif col_name in megatab.colnames and err_name in megatab.colnames:
        return megatab[col_name].copy(), megatab[err_name].copy()
    else:
        raise ValueError(f"Column {col_name} not found in megatable.")


# ---------------------------------------------------------------------------
# Private helpers for get_lya_property
# ---------------------------------------------------------------------------

def _lya_ew(megatab: Table, rest_frame: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return Lyman-alpha EW, summing red and blue peak fluxes."""
    required = ["FLUXR", "FLUXR_ERR", "FLUXB", "FLUXB_ERR", "CONT", "CONT_ERR"]
    if not all(c in megatab.colnames for c in required):
        raise ValueError("Required columns for calculating Lya EW not found in megatable.")
    fluxr     = megatab["FLUXR"].copy()
    fluxr_err = megatab["FLUXR_ERR"].copy()
    # NaN blue-peak values are replaced with 0: sources without a significant blue
    # peak should still contribute to the total EW via their red peak alone.
    fluxb     = np.nan_to_num(megatab["FLUXB"].copy(), nan=0.0)
    fluxb_err = np.nan_to_num(megatab["FLUXB_ERR"].copy(), nan=0.0)
    flux_total     = fluxr + fluxb
    flux_total_err = np.sqrt(fluxr_err**2 + fluxb_err**2)
    cont     = megatab["CONT"].copy()
    cont_err = megatab["CONT_ERR"].copy()
    ew     = flux_total / cont
    ew_err = np.abs(ew * np.sqrt((flux_total_err / flux_total)**2 + (cont_err / cont)**2))
    if rest_frame:
        ew     /= (1 + megatab['z'])
        ew_err /= (1 + megatab['z'])
    return ew, ew_err


def _lya_brratio(megatab: Table) -> tuple[np.ndarray, np.ndarray]:
    """Return blue-to-red Lya flux ratio and its propagated error."""
    if "FLUXB" not in megatab.colnames or "FLUXR" not in megatab.colnames:
        raise ValueError("Required columns for calculating Lya blue-to-red flux ratio not found in megatable.")
    blue_flux = megatab["FLUXB"].copy()
    red_flux  = megatab["FLUXR"].copy()
    with np.errstate(divide='ignore', invalid='ignore'):
        br_ratio = blue_flux / red_flux
        br_ratio[red_flux == 0] = np.nan
    br_ratio_err = br_ratio * np.sqrt((megatab["FLUXB_ERR"].copy() / blue_flux)**2
                                      + (megatab["FLUXR_ERR"].copy() / red_flux)**2)
    return br_ratio, br_ratio_err


def _lya_brsep(megatab: Table, rest_frame: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return blue-red Lya peak separation and its propagated error."""
    if "LPEAKR" not in megatab.colnames or "LPEAKB" not in megatab.colnames:
        raise ValueError("Required columns for calculating Lya blue-red peak separation not found in megatable.")
    red_peak  = megatab["LPEAKR"].copy()
    blue_peak = megatab["LPEAKB"].copy()
    sep_err_raw = np.sqrt(megatab["LPEAKR_ERR"].copy()**2 + megatab["LPEAKB_ERR"].copy()**2)
    if rest_frame:
        br_sep     = (red_peak - blue_peak) / (1 + megatab['z'])
        br_sep_err = sep_err_raw            / (1 + megatab['z'])
    else:
        br_sep     = red_peak - blue_peak
        br_sep_err = sep_err_raw
    br_sep[(red_peak == 0) | (blue_peak == 0)] = np.nan
    return br_sep, br_sep_err


def _lya_width(megatab: Table, prop: str,
               rest_frame: bool, correct_inst: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return an LSF-corrected and rest-framed Lya width property (FWHM* or DISP*)."""
    err_name = f"{prop}_ERR"
    col     = megatab[prop].copy()
    col_err = megatab[err_name].copy()
    if correct_inst:
        if "LPEAKR" not in megatab.colnames:
            raise ValueError("Required column for instrumental resolution correction not found in megatable.")
        lpeakr   = megatab["LPEAKR"].copy()
        kind     = "DISP" if prop[:-1] == "DISP" else "FWHM"
        col, col_err = correct_inst_res(col, col_err, lpeakr, kind)
    if rest_frame:
        col     /= (1 + megatab['z'])
        col_err /= (1 + megatab['z'])
    return col, col_err


def _lya_zelda(megatab: Table, prop: str) -> tuple[np.ndarray, np.ndarray]:
    """Return a ZELDA model parameter, averaging the asymmetric error columns."""
    prop_base = prop.rsplit('_', 1)[0]
    errm_name = f"{prop_base}_ERRM_ZELDA"
    errp_name = f"{prop_base}_ERRP_ZELDA"
    if errm_name in megatab.colnames and errp_name in megatab.colnames:
        col_err = (megatab[errm_name].copy() + megatab[errp_name].copy()) / 2
    else:
        col_err = None
    return megatab[prop].copy(), col_err


def _lya_total_luminosity(megatab: Table) -> tuple[np.ndarray, np.ndarray]:
    """Return total Lya luminosity (red + blue peak)."""
    fluxr     = megatab['FLUXR'].copy()
    fluxr_err = megatab['FLUXR_ERR'].copy()
    fluxb     = np.nan_to_num(megatab['FLUXB'].copy(), nan=0.0)
    fluxb_err = np.nan_to_num(megatab['FLUXB_ERR'].copy(), nan=0.0)
    flux_total     = fluxr + fluxb
    flux_total_err = np.sqrt(fluxr_err**2 + fluxb_err**2)
    return flux_to_luminosity(
        flux_total, flux_total_err,
        np.asarray(megatab['z']), np.asarray(megatab['MU']),
        is_continuum=False,
    )


def get_lya_property(megatab: Table, prop: str, rest_frame: bool = True,
                     correct_inst: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieve a Lyman-alpha property from the megatable.

    Handles derived quantities not stored directly (EW, blue-to-red ratio, peak
    separation, luminosity) and applies rest-frame and instrumental-resolution
    corrections where appropriate.

    Parameters
    ----------
    megatab : astropy.table.Table
        The megatable containing the data.
    prop : str
        The Lya property to retrieve. Supported values: 'EW_LYA' / 'EW',
        'BRRATIO', 'BRSEP', 'LUM_LYA', 'CONT_LUM_LYA', any FWHM*/DISP* column,
        any ZELDA parameter column, or any other column present in ``megatab``.
    rest_frame : bool, optional
        Apply rest-frame correction for wavelength-based properties. Default True.
    correct_inst : bool, optional
        Apply instrumental resolution correction (MUSE LSF) to FWHM/DISP
        properties. Default True.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The requested property column and its associated error column (or None
        when no error column exists).
    """
    if prop in ("EW_LYA", "EW"):
        return _lya_ew(megatab, rest_frame)
    elif prop == "BRRATIO":
        return _lya_brratio(megatab)
    elif prop == "BRSEP":
        return _lya_brsep(megatab, rest_frame)
    elif prop[:-1] in ("DISP", "FWHM", "FWHM_AB") and prop in megatab.colnames:
        return _lya_width(megatab, prop, rest_frame, correct_inst)
    elif "ZELDA" in prop and prop in megatab.colnames:
        return _lya_zelda(megatab, prop)
    elif prop == "LUM_LYA":
        return _lya_total_luminosity(megatab)
    elif prop == "CONT_LUM_LYA":
        return flux_to_luminosity(
            megatab['CONT'].copy(), megatab['CONT_ERR'].copy(),
            np.asarray(megatab['z']), np.asarray(megatab['MU']),
            is_continuum=True,
        )
    elif prop in megatab.colnames:
        err_name = f"{prop}_ERR"
        if err_name in megatab.colnames:
            return megatab[prop].copy(), megatab[err_name].copy()
        return megatab[prop].copy(), None
    else:
        raise ValueError(f"Column {prop} not found in megatable.")


_log_quantities = ["EW_LYA", "FWHMR", "DISPR", "FCEN_LYA", "MU",
                   "CONT", "EW", "FWHM", "DISP", "DISPR", 'W',
                   "VEXP_ZELDA", "BRSEP", "BRRATIO", "WINT_ZELDA",
                   "TDUST_ZELDA", "WINT_ZELDA", "FWHMB", "DISPB",
                   "LUM", "CONT_LUM", "LUM_LYA", "CONT_LUM_LYA"]

# All recognised line tokens: keys from wavedict plus the shorthand 'LYA'
_known_line_tokens: set[str] = set(wavedict.keys()) | {'LYA'} | set(doublets.keys())


def normalise_prop(prop: str) -> str:
    """
    Check whether a property string has the line name and property accidentally swapped
    (e.g. ``'LYA_EW'`` instead of ``'EW_LYA'``) and return the corrected form.

    The expected convention is always ``{property}_{line}``, for example:

    - ``'EW_LYA'``, ``'LUM_LYA'``, ``'CONT_LUM_LYA'``
    - ``'EW_CIV1548'``, ``'LUM_CIV1548'``, ``'CONT_LUM_CIV1548'``

    The function splits ``prop`` on ``'_'`` and tests whether the *leftmost* token (or
    pair of tokens for multi-part line names) is a known spectral line identifier. If so,
    it moves that token to the end and issues a :class:`UserWarning`.

    Parameters
    ----------
    prop : str
        The property string to validate.

    Returns
    -------
    str
        The (possibly corrected) property string, always in ``{property}_{line}`` order.

    Examples
    --------
    >>> normalise_prop('LYA_EW')
    UserWarning: ...
    'EW_LYA'
    >>> normalise_prop('EW_LYA')
    'EW_LYA'
    >>> normalise_prop('CIV1548_EW')
    UserWarning: ...
    'EW_CIV1548'
    """
    import warnings
    parts = prop.split('_')
    # Try progressively longer left-hand prefixes as candidate line tokens
    for split_idx in range(1, len(parts)):
        candidate_line = '_'.join(parts[:split_idx])
        candidate_prop = '_'.join(parts[split_idx:])
        if candidate_line in _known_line_tokens:
            corrected = f"{candidate_prop}_{candidate_line}"
            warnings.warn(
                f"Property '{prop}' appears to have the line name and property name "
                f"in the wrong order. Did you mean '{corrected}'? Correcting automatically.",
                UserWarning,
                stacklevel=2,
            )
            return corrected
    return prop
