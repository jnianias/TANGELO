import numpy as np
from astropy.io import fits, ascii
from astropy.table import Column, vstack
from scipy.optimize import curve_fit
from scipy import odr
from scipy.interpolate import interp1d
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.convolution import convolve_fft, Box1DKernel
import warnings
import glob
import os
import astropy.table as aptb
import error_propagation as ep
from mpdaf.MUSE import LSF
from pathlib import Path

# Import constants
from .constants import c, wavedict, doublets, skylines
from . import models as mdl
from . import io as io

# Import dictionaries for convenience
skylinedict = skylines
doubletdict = doublets

lsf = LSF(typ='qsim_v1')


def mask_skylines(wavelength):
    """
    Masks out regions around known sky lines to avoid contamination in fits.

    Parameters
    ----------
    wavelength : array_like
        Wavelength array.
    lambda_obs : float
        Observed wavelength of the line.
    continuum_buffer : float
        Buffer region around the line to include in the fit.

    Returns
    -------
    array_like
        Boolean mask array indicating regions to exclude.
    """
    sky_mask = np.ones_like(wavelength, dtype=bool)

    for skyline in skylinedict.values():
        sky_mask &= ~((wavelength > (skyline - 2.5)) & (wavelength < (skyline + 2.5)))

    return sky_mask

def mask_otherlines(wavelength, expected_wavelength, linename):
    """
    Mask out regions around other known lines to avoid contamination in fits.
    
    Parameters
    ----------
    wavelength : array_like
        Wavelength array.
    expected_wavelength : float
        Expected observed wavelength of the line being fitted.
    linename   : str
        Name of the line (needs to be in wavedict).

    Returns
    -------
    array_like
        Boolean mask array indicating regions to exclude.
    """
    # Get all other lines except the one being fitted and its doublet partner (if any)
    otherlines = [line for line in wavedict.keys() 
                  if (line != linename and line not in doubletdict.get(linename, ()) 
                      and linename not in doubletdict.get(line, ()))]
    
    # Initialize mask to all True
    line_mask = np.ones_like(wavelength, dtype=bool)

    # Get the expected redshift
    z_exp = (expected_wavelength / wavedict[linename]) - 1

    # Mask out +/- 2.5 Angstroms around each other line
    for line in otherlines:
        line_center = wavedict[line] * (1 + z_exp)
        line_mask &= ~((wavelength > (line_center - 2.5)) & (wavelength < (line_center + 2.5)))

    return line_mask

def generate_spec_mask(wavelength, spectrum, errors, lpeak_init, continuum_buffer, linename):
    """
    Generates a mask for the fitting region around a specified observed wavelength,
    excluding problematic areas (zeros, nans, infs), skylines, and unrelated spectral lines.

    Parameters
    ----------
    wavelength : array_like
        Wavelength array.
    spectrum  : array_like
        Flux density array.
    errors    : array_like
        Flux density uncertainties.
    lpeak_init : float
        Initial guess for the observed wavelength of the line.
    continuum_buffer : float
        Buffer region around the line to include in the fit.
    linename   : str
        Name of the line (needs to be in wavedict).

    Returns
    -------
    array_like
        Boolean mask array indicating the fitting region.
    """
    # Fitting region +/- continuum_buffer around the line
    fit_mask = (wavelength > (lpeak_init - continuum_buffer)) & (wavelength < (lpeak_init + continuum_buffer))

    # Mask out problematic data points
    fit_mask &= ~np.isnan(wavelength)
    fit_mask &= ~np.isnan(spectrum)
    fit_mask &= ~np.isnan(errors)
    fit_mask &= spectrum != 0
    fit_mask &= (errors > 0)
    fit_mask &= ~np.isinf(spectrum)
    fit_mask &= ~np.isinf(errors)

    # Mask out skylines and other known lines
    fit_mask &= mask_skylines(wavelength)
    fit_mask &= mask_otherlines(wavelength, lpeak_init, linename)

    return fit_mask

def get_lsf_fwhm(lsf_lpeak, step = 1.25):
    x0up = np.argwhere(lsf_lpeak > 0.5 * np.max(lsf_lpeak))[0][0]
    y0up = lsf_lpeak[x0up]
    x0down = x0up - 1
    y0down = lsf_lpeak[x0down]
    x1up = np.argwhere(lsf_lpeak > 0.5 * np.max(lsf_lpeak))[-1][0]
    y1up = lsf_lpeak[x1up]
    x1down = x1up + 1
    y1down = lsf_lpeak[x1down]
    x0 = np.interp(0.5 * np.max(lsf_lpeak), np.array([y0down, y0up]), np.array([x0down, x0up]))
    x1 = np.interp(0.5 * np.max(lsf_lpeak), np.array([y1down, y1up]), np.array([x1down, x1up]))
    return (x1 - x0) * step

def muse_lsf_fwhm_poly(lam):
    """
    Return the FWHM of the MUSE LSF at wavelength lam (in Angstroms), using the 
    polynomial model given in pyplatefit documentation: 
    https://pyplatefit.readthedocs.io/en/latest/tutorial.html

    Parameters
    ----------
    lam : float
        Wavelength in Angstroms.

    Returns
    -------
    float
        FWHM of the MUSE LSF at the given wavelength.
    """
    return 5.19939 - 7.56746 * 1e-4 * lam + 4.93397 * 1e-8 * lam**2.


#alternative version of wave2vel that uses relativistic relative velocity formula:
def wave2vel(observed_wavelength, rest_wavelength, redshift = 0):
    """
    Convert observed wavelength to velocity in the target object's rest frame,
    accounting for systemic redshift and using the full relativistic Doppler formula.

    Parameters
    ----------
    observed_wavelength : float
        Observed wavelength in the same units as rest_wavelength.
    rest_wavelength : float
        Rest wavelength of the spectral line.
    redshift : float
        Systemic redshift of the source.

    Returns
    -------
    float
        Velocity in the target object's rest frame, in km/s.
    """
    # Speed of light in km/s
    c = 299792.458

    # Calculate the velocity in the observer's frame (Earth's frame)
    wavelength_ratio = observed_wavelength / rest_wavelength
    beta_obs = (wavelength_ratio**2 - 1) / (wavelength_ratio**2 + 1)
    v_obs = beta_obs * c

    # Calculate the systemic velocity corresponding to the redshift
    beta_sys = ((1 + redshift)**2 - 1) / ((1 + redshift)**2 + 1)
    v_sys = beta_sys * c

    # Use the relativistic velocity addition formula to get the velocity in the target's frame
    v_target = (v_obs - v_sys) / (1 - (v_obs * v_sys) / c**2)

    return v_target

def vel2wave(vel, restLambda, z = 0.):
    """
    Convert velocity in the target object's rest frame to observed wavelength,
    accounting for systemic redshift and using the full relativistic Doppler formula.

    Parameters
    ----------
    vel : float
        Velocity in the target object's rest frame, in km/s.
    restLambda : float
        Rest wavelength of the spectral line.
    z : float
        Systemic redshift of the source.

    Returns
    -------
    float
        Observed wavelength in the same units as restLambda.
    """
    # Speed of light in km/s
    c = 299792.458

    # Calculate the systemic velocity corresponding to the redshift
    beta_sys = ((1 + z)**2 - 1) / ((1 + z)**2 + 1)
    v_sys = beta_sys * c

    # Use the relativistic velocity addition formula to get the velocity in the observer's frame
    v_obs = (vel + v_sys) / (1 + (vel * v_sys) / c**2)

    # Calculate the wavelength ratio from the observed velocity
    beta_obs = v_obs / c
    wavelength_ratio = np.sqrt((1 + beta_obs) / (1 - beta_obs))

    # Calculate the observed wavelength
    observed_wavelength = restLambda * wavelength_ratio

    return observed_wavelength

def is_reasonable_dpeak(popt, perr, z = None):
    """
    Check if the fitted parameters for a double peak Lyman-alpha profile are reasonable.
    Reasonable double peaked fits should be (1) statistically significant in both peaks,
    (2) be spectrally resolved, (3) have the blue peak at lower wavelength than the red peak,
    (4) have a velocity separation less than 1000 km/s.

    Parameters
    ----------
    popt : list
        List of fitted parameters from the lya_dpeak function
    perr : list
        List of parameter uncertainties from the lya_dpeak function
    z : float, optional
        Redshift of the source (used for velocity separation check)
    
    Returns
    -------
    bool
        True if the fit is reasonable, False otherwise.
    """
    # If z is not provided, get a rough estimate from the peak wavelength
    # of the red peak (popt[5])
    if z is None:
        z = popt[5] / 1215.67 - 1

    # Calculate 'SNR' values
    psnr = np.array(popt) / np.array(perr)
    
    # Calculate the right stddev of the blue and left stddev of the red lines to check 
    # that the lines are spectrally resolved
    fwhm_br = 2 * mdl.lya_swhm(ep.Complex(popt[2], perr[2]),
                                      ep.Complex(popt[3], perr[3]), +1)
    fwhm_rl = 2 * mdl.lya_swhm(ep.Complex(popt[6], perr[6]),
                                      ep.Complex(popt[7], perr[7]), -1)
    # Use the larger of the two widths as the minimum separation for the peaks
    wid = np.nanmax([np.abs(fwhm_br.value), np.abs(fwhm_rl.value)])
    # Catch cases where the blue peak has a poorly constrained width
    if fwhm_br.value < 3 * fwhm_br.error:
        wid = np.abs(fwhm_rl.value)
    
    # Calculate the velocity separation of the two peaks
    deltavel = np.abs(
        wave2vel(popt[1], 1215.67, redshift = z)
        - wave2vel(popt[5], 1215.67, redshift = z)
    )

    # Filter out bad results
    conditions = [
        (psnr[0] > 3.0 and psnr[4] > 3.0), # Both peaks are significant
        (psnr[0] < 5000 and psnr[4] < 5000), # Both peaks are not unphysically large
        popt[0] < popt[4],  # Blue peak amplitude < red peak amplitude
        np.abs(popt[5] - popt[1]) > wid, # Peaks are spectrally resolved
        popt[1] < popt[5], # Blue peak is at lower wavelength than red peak
        deltavel < 1000. # Velocity separation is less than 1000 km/s
    ]
    
    print(conditions)
    
    return all(conditions)

def get_line_spec(row, line, width, rest = True, spec_source = 'R21', spec_type = 'weight_skysub'):
    """
    Extract a spectrum around a given line from a row of a catalog.

    Parameters
    ----------
    row : astropy Table row
        Row of the catalog
    line : str
        Name of the line (e.g., 'LYA', 'CIV', etc.)
    width : float
        Width around the line to extract (in Angstroms)
    rest : bool
        If True, use rest-frame wavelength for the width; if False, use observed wavelength
    spec_source : str
        Source of the spectrum ('R21' for Richard et al. 2021, 'APER' for aperture spectra)
    spec_type : str
        Type of spectrum to load (either 'weight_skysub' or '2fwhm', etc.)

    Returns
    -------
    tuple of np.ndarray
        Returns wavelength array, flux array, and error array.
    """
    if line not in wavedict:
        raise ValueError(f"Line {line} not recognized. Available lines: {list(wavedict.keys())}")
    
    # Get the rest wavelength of the line
    rest_wavelength = wavedict[line]

    # See whether the line was detected in this object -- if so, use the line wavelength,
    # if not, use the LYMAN ALPHA redshift to get the expected wavelength
    snr_colname = f"SNR_{line}" if line != 'LYALPHA' else 'SNRR'
    lpeak_colname = f"LPEAK_{line}" if line != 'LYALPHA' else 'LPEAKR'
    if row[snr_colname] > 3.0 and not np.isnan(row[lpeak_colname]):
        line_wavelength = row[lpeak_colname]
    elif not np.isnan(row['LPEAKR']):
        line_wavelength = rest_wavelength * (row['LPEAKR'] / wavedict['LYALPHA'])
    else:
        print(f"No Lyman-alpha redshift available for object {row['iden']}. Cannot determine line wavelength.")
        return None, None, None
    
    # Load the spectrum
    spectab = None
    if spec_source == 'R21':
        spectab = io.load_r21_spec(row['CLUSTER'], row['iden'], row['idfrom'], spec_type)
    elif spec_source.upper() == 'APER':
        spectab = io.load_aper_spec(row['CLUSTER'], row['iden'], row['idfrom'], spec_type)

    if spectab is None:
        print(f"No spectrum found for {row['CLUSTER']} ID{row['iden']} in {spec_source} catalog.")
        return None, None, None
    
    # Determine the wavelength range to extract
    if rest:
        z = line_wavelength / rest_wavelength - 1
        obs_width = width * (1 + z)
    else:
        obs_width = width
    
    # Mask sky lines and bad data points
    mask = generate_spec_mask(spectab['wave'],
                                    spectab['spec'],
                                    spectab['spec_err'],
                                    lpeak_init = line_wavelength,
                                    continuum_buffer = obs_width,
                                    linename=line)

    return spectab['wave'][mask], spectab['spec'][mask], spectab['spec_err'][mask]


def find_partner(linename, linelist=None):
    """
    Find the doublet partner of a given spectral line.
    
    Parameters
    ----------
    linename : str
        Name of the spectral line (e.g., 'CIV1548')
    linelist : list, optional
        List of line names to search for the partner. If None, uses all lines in doublets dict.
    
    Returns
    -------
    str or None
        Name of the partner line, or None if no partner exists
    """
    if linelist is None:
        linelist = list(wavedict.keys())
    
    # Check each doublet
    for key, (line1, line2) in doubletdict.items():
        if linename == line1 and line2 in linelist:
            return line2
        elif linename == line2 and line1 in linelist:
            return line1
    
    return None


def flag_fitted_line(megatab, index, linename, spectab=None, 
                     fwhm_threshold=2.41, verbose=True):
    """
    Apply automatic quality flags to a fitted spectral line in a megatable.
    
    This function performs several tests on a fitted spectral line to identify
    potentially problematic measurements. Flags are applied as single characters:
    - 's': Sky line contamination
    - 't': Line too thin (FWHM below spectral resolution)
    - 'n': Negative flux domination (spectrum goes below zero in line region)
    - 'p': Peak-dominated (peak significance exceeds integrated flux SNR)
    
    The function modifies the table in place by updating the FLAG column for the line.
    
    Parameters
    -----------
    megatab : astropy.table.Table
        Results table containing fitted line parameters. Must have columns:
        - LPEAK_{linename}: Peak wavelength of the line
        - LPEAK_ERR_{linename}: Error on peak wavelength
        - FLUX_{linename}: Integrated flux of the line
        - SNR_{linename}: Signal-to-noise ratio of the integrated flux
        - FWHM_{linename}: Full width at half maximum
        - FLAG_{linename}: Flag column (modified in place)
        Optional columns for improved error propagation:
        - FLUX_ERR_{linename}: Error on integrated flux (falls back to FLUX/SNR)
        - FWHM_ERR_{linename}: Error on FWHM (falls back to 0 if not provided)
    index : int
        Row index in the table to flag
    linename : str
        Name of the spectral line (e.g., 'CIV1548', 'LYALPHA')
    spectab : astropy.table.Table, optional
        Spectrum table with columns 'wave', 'spec', 'spec_err'. If None, peak
        significance test will be skipped. Required for complete flagging.
    fwhm_threshold : float, optional
        Minimum FWHM threshold in Angstroms (default: 2.41, MUSE spectral resolution)
    verbose : bool, optional
        If True, print diagnostic information about each test
    
    Returns
    --------
    dict
        Dictionary of test results with keys:
        - 'sky': bool, True if sky line contamination detected
        - 'thin': bool, True if FWHM below threshold
        - 'negative': bool, True if spectrum dominated by negative flux
        - 'peakdominant': bool, True if peak dominates over integrated flux
        - 'contamination': bool, True if already flagged with 'c'
        - 'flags_applied': str, The flag string that was applied
    
    Notes:
    ------
    - If the line has a doublet partner that is also detected (SNR > 3), all flags
      are removed as the doublet provides confirmation of the detection.
    - Pre-existing 'c' (contamination) flags from other analyses are preserved.
    - Sky line contamination is checked against known sky line catalogs.
    - Peak amplitude uncertainty is computed using proper error propagation via the
      error_propagation package, accounting for uncertainties in both flux and FWHM.
      If error columns are not present, flux error is estimated from FLUX/SNR.
 
    """
    row = megatab[index]
    
    # Get column names for this line
    lpeak_col = f'LPEAK_{linename}'
    lpeak_err_col = f'LPEAK_ERR_{linename}'
    flux_col = f'FLUX_{linename}'
    snr_col = f'SNR_{linename}'
    fwhm_col = f'FWHM_{linename}'
    flag_col = f'FLAG_{linename}'
    
    # Initialize test results
    tests = {
        'sky': False,
        'thin': False,
        'peakdominant': False,
        'negative': False,
        'contamination': False,
        'flags_applied': ''
    }
    
    # Extract line parameters with errors where available
    lpeak = row[lpeak_col]
    lpeak_err = row[lpeak_err_col]
    flux = row[flux_col]
    snr = row[snr_col]
    fwhm = row[fwhm_col]
    current_flag = row[flag_col]
    
    # Get flux and FWHM errors if they exist
    flux_err_col = f'FLUX_ERR_{linename}' if f'FLUX_ERR_{linename}' in megatab.colnames else None
    fwhm_err_col = f'FWHM_ERR_{linename}' if f'FWHM_ERR_{linename}' in megatab.colnames else None
    
    flux_err = row[flux_err_col] if flux_err_col else np.abs(flux / snr)
    fwhm_err = row[fwhm_err_col] if fwhm_err_col else 0.0
    
    # Check for pre-existing contamination flag
    if current_flag == 'c':
        tests['contamination'] = True
    
    # Test 1: Sky line contamination
    # Check if the line peak coincides with any known sky lines
    tests['sky'] = check_sky_contamination(lpeak, lpeak_err, flux)
    
    # Test 2: Line too thin (below spectral resolution)
    tests['thin'] = (fwhm < fwhm_threshold)
    
    # Test 3: Negative flux domination
    # Check if the spectrum around the line is dominated by negative values
    if spectab is not None:
        # Define the region around the line (±2 FWHM or minimum 5 Å)
        half_width = np.max([2.0 * fwhm, 5.0])
        line_mask = (spectab['wave'] > lpeak - half_width) & (spectab['wave'] < lpeak + half_width)
        
        if np.sum(line_mask) > 0:
            line_flux = spectab['spec'][line_mask]
            # Check if more than 50% of pixels are negative
            frac_negative = np.sum(line_flux < 0) / len(line_flux)
            # Check if the median flux is negative
            median_negative = np.median(line_flux) < 0
            
            # Flag if either condition is met
            tests['negative'] = (frac_negative > 0.5) or median_negative
            
            if verbose and tests['negative']:
                print(f"Negative flux check: {frac_negative*100:.1f}% of pixels negative, median={np.median(line_flux):.2e}")
    
    # Test 4: Peak-dominated line
    # Calculate peak amplitude with proper error propagation
    if spectab is not None:
        # Use error_propagation to compute amplitude and its uncertainty
        # amp = flux / (sigma * sqrt(2*pi)), where sigma = fwhm / (2 * sqrt(2*ln(2)))
        flux_complex = ep.Complex(flux, flux_err)
        fwhm_complex = ep.Complex(fwhm, fwhm_err)
        
        # sigma = fwhm / 2.355
        sigma_complex = fwhm_complex / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        
        # amp = flux / (sigma * sqrt(2*pi))
        amp_complex = flux_complex / (sigma_complex * np.sqrt(2.0 * np.pi))
        
        # Get the continuum uncertainty at the line peak from the spectrum
        cont_err = np.interp(lpeak, spectab['wave'], spectab['spec_err'])
        
        # Peak significance is the ratio of amplitude to continuum uncertainty
        peak_snr = np.abs(amp_complex.value / cont_err)
        tests['peakdominant'] = (peak_snr > np.abs(snr))
        
        if verbose:
            print(f"Peak amplitude: {amp_complex.value:.2e} +/- {amp_complex.error:.2e}")
            print(f"Continuum error: {cont_err:.2e}")
            print(f"Peak SNR: {peak_snr:.2f}, Integrated SNR: {snr:.2f}")
    
    # Check for doublet partner
    partner = find_partner(linename, list(wavedict.keys()))
    if partner is not None:
        partner_snr_col = f'SNR_{partner}'
        partner_flag_col = f'FLAG_{partner}'
        # If both lines in doublet are detected, remove all flags
        if partner_snr_col in megatab.colnames:
            if np.abs(row[partner_snr_col]) > 3.0 and np.abs(snr) > 3.0:
                if verbose:
                    print(f"Doublet partner {partner} detected - removing flags")
                megatab[flag_col][index] = ''
                if partner_flag_col in megatab.colnames:
                    megatab[partner_flag_col][index] = ''
                tests['flags_applied'] = ''
                return tests
    
    # Apply flags (skip contamination as it's pre-existing)
    flag_string = ''
    if not tests['contamination']:
        # Clear any non-contamination flags
        megatab[flag_col][index] = ''
        
        # Apply new flags
        if tests['sky']:
            flag_string += 's'
            if verbose:
                print(f"  Applying flag 's': Sky line contamination detected")
        if tests['thin']:
            flag_string += 't'
            if verbose:
                print(f"  Applying flag 't': Line too thin (FWHM={fwhm:.2f} < {fwhm_threshold:.2f} Å)")
        if tests['negative']:
            flag_string += 'n'
            if verbose:
                print(f"  Applying flag 'n': Spectrum dominated by negative flux")
        if tests['peakdominant']:
            flag_string += 'p'
            if verbose:
                print(f"  Applying flag 'p': Peak-dominated line")
        
        megatab[flag_col][index] = flag_string
    
    tests['flags_applied'] = flag_string
    
    if verbose:
        print(f"\nFlagging results for {linename}:")
        if flag_string:
            print(f"  Flags applied: '{flag_string}'")
        else:
            print(f"  No flags raised")
    
    return tests


def check_sky_contamination(lpeak, lpeak_err, flux, sig=5):
    """
    Check if a spectral line coincides with known sky lines.
    
    Parameters
    ----------
    lpeak : float
        Peak wavelength of the fitted line
    lpeak_err : float
        Error on the peak wavelength
    flux : float
        Integrated flux of the line
    sig : float, optional
        Significance threshold for sky line contamination (default: 5)
    
    Returns
    -------
    bool
        True if sky line contamination is detected
    """
    # Check tolerance: use 3-sigma or minimum of 2.4 Angstroms
    tol = np.nanmax([lpeak_err * 3.0, 2.4])
    
    # Check strong sky lines first
    for skyline_wave in skylinedict.values():
        if (lpeak - tol < skyline_wave < lpeak + tol):
            return True
    
    # Note: Full sky line catalog checking would require loading an external catalog
    # For now, we only check against the strong lines defined in constants
    # Users can extend this by loading the full MUSE sky line catalog if needed
    
    return False


def avg_lines(row, lines, absorption=False, velbounds=[-2400, 2400], velstep=60., 
              lya=False, zelda_z=False, z=None, flags=False, spec_source='R21', spec_type='weight_skysub'):
    """
    Average multiple spectral lines onto a common velocity scale.
    
    This function loads a spectrum, transforms specified lines to velocity space,
    interpolates them onto a common velocity grid, and computes a weighted average.
    This is useful for stacking multiple emission or absorption lines to improve
    signal-to-noise or measure systemic velocities.
    
    Parameters
    ----------
    row : astropy.table.Row
        Catalog row containing source information. Must have 'CLUSTER', 'iden', 'idfrom',
        and optionally 'LPEAKR' (for default redshift), 'Z_ZELDA', and line
        parameters if `flags=True`.
    lines : list of str
        List of line names to average. Line names must be keys in the wavedict
        from constants module (e.g., ['CIII1907', 'CIII1909', 'HeII1640']).
    absorption : bool, optional
        If True, weight by average SNR of continuum (better for absorption features).
        If False, weight by inverse mean variance (better for emission). Default is False.
    velbounds : list of float, optional
        Velocity range [vmin, vmax] in km/s for the output spectrum. Default is [-2400, 2400].
    velstep : float, optional
        Velocity bin size in km/s for the output spectrum. Default is 60.
    lya : bool, optional
        Reserved for Lyman alpha specific handling (currently unused). Default is False.
    zelda_z : bool, optional
        If True, use 'Z_ZELDA' from the row instead of default redshift. 
        Returns None if Z_ZELDA <= 2.9. Default is False.
    z : float, optional
        Redshift to use for wavelength-to-velocity conversion. If None, uses
        row['LPEAKR'] / 1215.67 - 1 as default. Default is None.
    flags : bool, optional
        If True, skip lines with bad flags (FLAG_{line} != '') or invalid wavelengths.
        Default is False.
    spec_source : str, optional
        Source of the spectrum: 'R21' for Richard et al. (2021) spectra or 'APER' 
        for aperture-extracted spectra. Default is 'R21'.
    spec_type : str, optional
        Type of spectrum to load. For R21: 'weight_skysub', 'noweight', etc.
        For APER: '2fwhm', '1fwhm', etc. Default is 'weight_skysub'.
    
    Returns
    -------
    newvelax : numpy.ndarray
        Common velocity axis in km/s.
    line_avg_wtd : numpy.ndarray
        Weighted average flux density in units of 10^-20 erg s^-1 cm^-2 (km/s)^-1.
        Returns array of NaN if no valid lines found.
    line_avgerr_wtd : numpy.ndarray
        Uncertainty on the weighted average flux density (same units).
        Returns array of NaN if no valid lines found.
    
    Notes
    -----
    - Lines outside MUSE wavelength coverage (4750-9350 Å) are automatically skipped.
    - Lines named 'LYALPHA', 'SiII1527', or 'DUST' are always skipped.
    - Weighting strategy differs based on `absorption` parameter:
        * absorption=True: weight ∝ (median SNR)^2 to emphasize high-contrast regions
        * absorption=False: weight ∝ 1/(median variance) for optimal S/N in emission
    - The function converts flux density from f_λ to f_v accounting for (1+z).
    
    Examples
    --------
    >>> from tangiers import spectroscopy as auspec
    >>> from tangiers import constants as auconst
    >>> # Average optically-thin emission lines using R21 spectra
    >>> lines = ['CIII1907', 'CIII1909', 'HeII1640', 'OIII1666']
    >>> vel, flux, flux_err = auspec.avg_lines(
    ...     catalog_row, lines, absorption=False, velbounds=[-2000, 2000],
    ...     spec_source='R21', spec_type='weight_skysub'
    ... )
    >>> # Average absorption lines using aperture spectra with flag checking
    >>> abs_lines = ['SiII1260', 'CII1334', 'SiIV1394']
    >>> vel, flux, flux_err = auspec.avg_lines(
    ...     catalog_row, abs_lines, absorption=True, flags=True,
    ...     spec_source='APER', spec_type='2fwhm'
    ... )
    """
    newvelax = np.arange(velbounds[0], velbounds[1] + velstep, velstep)
    clus = row['CLUSTER']
    iden = row['iden']
    clid = f"{clus}.{iden}"
    
    # Load spectrum using the appropriate function
    idfrom = row['idfrom']
    spectab = io.load_spec(clus=clus, iden=iden, idfrom=idfrom, spec_source=spec_source, spec_type=spec_type)
    
    if spectab is None:
        return newvelax, np.zeros(np.size(newvelax)) * np.nan, np.zeros(np.size(newvelax)) * np.nan
    
    # Determine redshift
    if z is None:
        z = row['LPEAKR'] / 1215.67 - 1.
    if zelda_z:
        z = row['Z_ZELDA']
        if not (z > 2.9):
            return newvelax, None, None
    
    wave = spectab['wave'].data  # Generate wavelength axis
    spec = spectab['spec'].data
    spec_err = spectab['spec_err'].data
    
    interpd_specs = []
    interpd_specerrs = []
    weights = []
    
    # Velocity binning: we make this as coarse as possible to still get meaningful error bars
    for i, line in enumerate(lines):
        # Skip certain lines
        if line in ['LYALPHA', 'SiII1527', 'DUST']:
            pass
        elif flags and ((row[f"FLAG_{line}"] not in ['']) or not (row[f"LBDA_REST_{line}"] > 0.)):
            continue
        
        rest_wl = wavedict[line]
        
        # Ignore lines that are close to or beyond the edge of MUSE wavelength coverage
        if (rest_wl * (1 + z) < np.min(wave) + 10. or rest_wl * (1 + z) > np.max(wave) - 10.) and (line != 'LYALPHA'):
            print(f'Cannot process {line} for {clid} as outside of range {np.min(wave):.1f} to {np.max(wave):.1f} Å')
            continue
        
        # Transform wavelength axis into velocity axis
        vel = wave2vel(wave, rest_wl, redshift=z)
        velrange = np.logical_and(velbounds[0] - 200 < vel, vel < velbounds[1] + 200)
        
        vel_mini = vel[velrange]
        wave_mini = wave[velrange]
        wavebins = np.ediff1d(wave_mini, to_end=np.ediff1d(wave_mini)[-1:])
        velbins = np.ediff1d(vel_mini, to_end=np.ediff1d(vel_mini)[-1:])
        
        # Convert to flux units
        spec_flux = spec[velrange] * wavebins
        spec_err_flux = spec_err[velrange] * wavebins
        
        # Convert to flux density per km/s (includes (1+z) factor)
        spec_vel = spec_flux / velbins
        spec_err_vel = spec_err_flux / velbins
        
        # Interpolate to standardized velocity axis
        flux_interper = np.interp(newvelax, vel_mini, spec_vel, left=np.nan, right=np.nan)
        flux_err_interper = np.interp(newvelax, vel_mini, spec_err_vel, left=np.nan, right=np.nan)

        # Calculate weight
        if absorption:
            # Weight by average SNR of the CONTINUUM for absorption features
            weight = np.nanmedian(flux_interper / flux_err_interper) ** 2.
        else:
            # Weight by inverse mean variance for emission features
            weight = np.nanmedian(flux_err_interper) ** (-2.)
        
        weights.append(weight)
        interpd_specs.append(flux_interper)
        interpd_specerrs.append(flux_err_interper)
    
    # Return NaN arrays if no valid lines were processed
    if len(interpd_specs) == 0:
        return newvelax, np.zeros(np.size(newvelax)) * np.nan, np.zeros(np.size(newvelax)) * np.nan
    else:
        # Compute weighted average
        line_avg_wtd = np.nansum(np.array([w * s for w, s in zip(weights, interpd_specs)]), axis=0) / np.nansum(np.array(weights), axis=0)
        line_avgerr_wtd = np.sqrt(np.nansum(np.array([np.square(w * s) for w, s in zip(weights, interpd_specerrs)]), axis=0)) \
                            / np.nansum(np.array(weights), axis=0)
        return newvelax, line_avg_wtd, line_avgerr_wtd


def stack_spectra_across_sources(table, lines, velocity_frame='systemic', velbounds=[-2500, 2500], 
                                  velstep=75.0, weighting='inverse_variance', absorption=False,
                                  spec_source='R21', spec_type='weight_skysub', mask=None,
                                  systemic_column='DELTAV_LYA', lya_column='LPEAKR', sigclip_weights=None):
    """
    Stack spectral lines across multiple sources onto a common velocity grid.
    
    This function takes a table of sources, extracts and averages specified spectral
    lines for each source (using avg_lines), shifts them to a common velocity frame,
    and then stacks them across all sources using weighted averaging. This is useful
    for creating composite spectra from populations of galaxies.
    
    Parameters
    ----------
    table : astropy.table.Table
        Catalog table containing multiple sources. Each row represents one source.
        Must contain columns for source identification (CLUSTER, iden, idfrom) and
        velocity reference information.
    lines : list of str
        List of line names to average within each source before stacking. Line names
        must be keys in the wavedict (e.g., ['SiII1260', 'CII1334']).
    velocity_frame : str, optional
        Reference frame for velocity alignment. Options:
        - 'systemic': Align to systemic velocity using systemic_column (default)
        - 'lyalpha': Align to Lyman-alpha peak (no additional shift)
        Default is 'systemic'.
    velbounds : list of float, optional
        Velocity range [vmin, vmax] in km/s for the output spectrum.
        Default is [-2500, 2500].
    velstep : float, optional
        Velocity bin size in km/s for interpolation and output.
        Default is 75.0.
    weighting : str or float, optional
        Weighting scheme for stacking across sources. Options:
        - 'inverse_variance': Weight by inverse median variance (1/σ²) (default)
        - 'uniform': Equal weight for all sources
        - 'inverse_error': Weight by inverse median error (1/σ)
        - float: Custom exponent for weighting by σ -- -2 for inverse variance,
          -1 for inverse error, 0 for uniform, etc.
        - 'continuum': Weight by median SNR of continuum (better for absorption)
        Default is 'inverse_variance'.
    absorption : bool, optional
        Passed to avg_lines(). If True, weight lines within each source by SNR
        (better for absorption). If False, weight by inverse variance (better for
        emission). Default is False.
    spec_source : str, optional
        Source of spectra: 'R21' or 'APER'. Passed to avg_lines().
        Default is 'R21'.
    spec_type : str, optional
        Type of spectrum to load (e.g., 'weight_skysub', '2fwhm').
        Passed to avg_lines(). Default is 'weight_skysub'.
    mask : numpy.ndarray or None, optional
        Boolean mask to select subset of sources from table. If None, use all sources.
        Default is None.
    systemic_column : str, optional
        Column name containing systemic velocity offset in km/s. Used when
        velocity_frame='systemic'. Default is 'DELTAV_LYA'.
    lya_column : str, optional
        Column name containing Lyman-alpha peak wavelength for redshift calculation.
        Default is 'LPEAKR'.
    sigclip_weights : float or None, optional
        If provided, apply sigma clipping to weights across sources at this
        significance level before stacking. Default is None (no clipping).
    
    Returns
    -------
    velocity : numpy.ndarray
        Common velocity axis in km/s.
    stacked_flux : numpy.ndarray
        Weighted stacked flux density in units of 10^-20 erg s^-1 cm^-2 (km/s)^-1.
        Returns array of NaN if no valid sources.
    stacked_error : numpy.ndarray
        Uncertainty on stacked flux density (same units).
        Returns array of NaN if no valid sources.
    n_sources : int
        Number of sources successfully included in the stack.
    
    Notes
    -----
    - Sources are skipped if avg_lines() returns all NaN (e.g., missing spectrum).
    - For 'systemic' frame, velocity is shifted by the value in systemic_column.
    - Weighting strategies:
        * inverse_variance: w_i = 1 / median(σ_i²) - optimal for Gaussian errors
        * uniform: w_i = 1 for all sources - robust to outliers
        * inverse_error: w_i = 1 / median(σ_i) - intermediate option
    - Error propagation assumes independent measurements between sources.
    
    Examples
    --------
    >>> from tangiers import spectroscopy as auspec
    >>> # Stack low-ionization absorption lines in systemic frame
    >>> vel, flux, err, n = auspec.stack_spectra_across_sources(
    ...     table=catalog,
    ...     lines=['SiII1260', 'CII1334'],
    ...     velocity_frame='systemic',
    ...     weighting='inverse_variance',
    ...     absorption=True,
    ...     mask=catalog['DV_LI_ABS'] > -np.inf
    ... )
    >>> print(f"Stacked {n} sources")
    
    >>> # Stack emission lines with uniform weighting
    >>> vel, flux, err, n = auspec.stack_spectra_across_sources(
    ...     table=catalog,
    ...     lines=['CIII1907', 'CIII1909', 'HeII1640'],
    ...     velocity_frame='lyalpha',
    ...     weighting='uniform',
    ...     absorption=False
    ... )
    """
    
    # Apply mask if provided
    if mask is not None:
        sources = table[mask]
    else:
        sources = table
    
    # Initialize lists to collect spectra and errors
    spec_list = []
    err_list = []
    
    # Common velocity grid for interpolation
    common_vel = np.arange(velbounds[0], velbounds[1] + velstep, velstep)
    
    # Process each source
    for row in sources:
        # Average lines within this source
        vel, spec, spec_err = avg_lines(
            row, lines, 
            absorption=absorption,
            velbounds=velbounds,
            velstep=velstep,
            spec_source=spec_source,
            spec_type=spec_type
        )
        
        # Skip if spectrum is all NaN
        if np.all(np.isnan(spec)):
            print(f"Skipping source {row['CLUSTER']}.{row['iden']} - no valid spectrum")
            continue
        
        # Shift velocity frame if requested
        if velocity_frame == 'systemic':
            if systemic_column not in row.colnames:
                raise ValueError(f"Column '{systemic_column}' not found in table. "
                               f"Available columns: {row.colnames}")
            deltav = row[systemic_column]
            if np.isnan(deltav) or np.isinf(deltav):
                # Skip sources without valid systemic velocity
                continue
            vel_shifted = vel + deltav
        elif velocity_frame == 'lyalpha':
            vel_shifted = vel
        else:
            raise ValueError(f"velocity_frame must be 'systemic' or 'lyalpha', got '{velocity_frame}'")
        
        # Interpolate to common velocity grid
        spec_interp = np.interp(common_vel, vel_shifted, spec)
        err_interp = np.interp(common_vel, vel_shifted, spec_err)
        
        spec_list.append(spec_interp)
        err_list.append(err_interp)
    
    # Check if we have any valid spectra
    if len(spec_list) == 0:
        return common_vel, np.full_like(common_vel, np.nan), np.full_like(common_vel, np.nan), 0
    
    # Convert to arrays for easier manipulation
    spec_array = np.array(spec_list)  # Shape: (n_sources, n_velocity_bins)
    err_array = np.array(err_list)

    #TESTING
    print(spec_array.shape, err_array.shape)

    # Determine weight exponent based on string or float input
    weight_dict = {
        'inverse_variance': -2,
        'uniform': 0,
        'inverse_error': -1
    }
    if isinstance(weighting, str) and weighting != 'continuum':
        weight_exponent = weight_dict.get(weighting, None)
        if weight_exponent is None:
            raise ValueError(f"weighting must be 'inverse_variance', 'uniform', or 'inverse_error', "
                           f"got '{weighting}'")
    elif isinstance(weighting, (int, float)):
        weight_exponent = weighting
    elif weighting != 'continuum':
        raise ValueError(f"weighting must be a string or float, got '{type(weighting)}'")
    
    # Weight by inverse median variance: w_i = 1 / median(σ_i²)
    median_err = np.nanmedian(err_array, axis=1)
    if weighting == 'continuum':
        # Weight by median SNR of continuum for absorption features
        median_spec = np.nanmedian(spec_array, axis=1)
        median_snr = median_spec / median_err
        weights = np.where(np.isfinite(median_snr), median_snr, 0.0)
    else:
        # Handle sources with all-NaN errors or zero errors
        weights = np.where((median_err > 0) & np.isfinite(median_err), 
                        median_err ** (weight_exponent), 0.0)
        
    # Apply sigma clipping to weights if requested
    if sigclip_weights is not None:
        weight_scmean, weight_scmedian, weight_scstd = sigma_clipped_stats(weights, sigma=3, maxiters=5)
        upper_limit = weight_scmedian + sigclip_weights * weight_scstd
        weights[weights > upper_limit] = upper_limit
    
    # Reshape weights for broadcasting: (n_sources, 1)
    weights = weights[:, np.newaxis]
    
    # Compute weighted average across sources
    # stacked_flux = Σ(w_i * flux_i) / Σ(w_i)
    weighted_sum = np.nansum(weights * spec_array, axis=0)
    weight_sum = np.nansum(weights, axis=0)
    stacked_flux = weighted_sum / weight_sum
    
    # Propagate errors: σ_stacked = √(Σ(w_i² * σ_i²)) / Σ(w_i)
    weighted_err_squared = np.nansum((weights * err_array) ** 2, axis=0)
    stacked_error = np.sqrt(weighted_err_squared) / weight_sum
    
    n_sources = len(spec_list)
    
    return common_vel, stacked_flux, stacked_error, n_sources


def stack_entire_spectra(table, weighting = 'inverse variance', sigclip_weights=None, 
                         spec_source='R21', spec_type='weight_skysub', wave_bounds=None,
                         wave_step=1.25):
    """
    Stack entire spectra across multiple sources onto a common (rest frame) wavelength grid.
    
    This function takes a table of sources, loads their full spectra, and stacks
    them across all sources using weighted averaging. This is useful for creating
    composite spectra from populations of galaxies.
    
    Parameters
    ----------
    table : astropy.table.Table
        Catalog table containing multiple sources. Each row represents one source.
        Must contain columns for source identification (CLUSTER, iden, idfrom).
    weighting : str or float, optional
        Weighting scheme for stacking across sources. Options:
        - 'inverse variance': Weight by inverse median variance (1/σ²) (default)
        - 'uniform': Equal weight for all sources
        - 'inverse error': Weight by inverse median error (1/σ)
        - float: Custom exponent for weighting by σ -- -2 for inverse variance,
          -1 for inverse error, 0 for uniform, etc.
        Default is 'inverse variance'.
    sigclip_weights : float or None, optional
        If provided, apply sigma clipping to weights across sources at this
        significance level before stacking. Default is None (no clipping).
    spec_source : str, optional
        Source of spectra: 'R21' or 'APER'. Default is 'R21'.
    spec_type : str, optional
        Type of spectrum to use from the source. Default is 'weight_skysub'.
    wave_bounds : list of float or None, optional
        Wavelength range [λ_min, λ_max] in Angstroms for the output spectrum in the rest frame.
        If None, use full overlapping range across all sources. Default is None.
    wave_step : float, optional
        Wavelength bin size in Angstroms for interpolation and output.
        Default is 1.25.
    
    Returns
    -------
    common_wavelength : numpy.ndarray
        Common rest frame wavelength grid onto which spectra are stacked.
    stacked_flux : numpy.ndarray
        Weighted average flux across sources at each wavelength.
    stacked_error : numpy.ndarray
        Propagated error of the stacked flux.
    n_sources : int
        Number of sources stacked.

    Notes
    -----
    - Sources are skipped with a warning if their spectrum cannot be loaded.
    - Weighting strategies:
        * inverse_variance: w_i = 1 / median(σ_i²) - optimal for Gaussian errors
        * uniform: w_i = 1 for all sources - robust to outliers
        * inverse_error: w_i = 1 / median(σ_i) - intermediate option
    - Error propagation assumes independent measurements between sources.
    """
    spec_list = []
    err_list = []
    wavelength_list = []

    # Load spectra for each source
    for row in table:
        clus = row['CLUSTER']
        iden = row['iden']
        idfrom = row['idfrom']
        
        spectab = io.load_spec(clus=clus, iden=iden, idfrom=idfrom, spec_source=spec_source, spec_type=spec_type)
        
        if spectab is None:
            print(f"Warning: Could not load spectrum for {clus}.{iden}. Skipping.")
            continue

        # De-redshift to rest frame using Lyman-alpha peak as a proxy
        z = row['LPEAKR'] / 1215.67 - 1
        rest_wave = spectab['wave'].data / (1 + z)

        wavelength_list.append(rest_wave)
        spec_list.append(spectab['spec'].data)
        err_list.append(spectab['spec_err'].data)

    # Check if we have any valid spectra
    if len(spec_list) == 0:
        return None, None, None, 0
    
    # Determine common wavelength grid
    if wave_bounds is None:
        # Use full overlapping range across all sources
        wave_min = np.max([np.min(wave) for wave in wavelength_list])
        wave_max = np.min([np.max(wave) for wave in wavelength_list])
    else:
        # Constrain to provided bounds or overlapping range, whichever is smaller
        wave_min, wave_max = max(wave_bounds[0], np.max([np.min(wave) for wave in wavelength_list])), \
                             min(wave_bounds[1], np.min([np.max(wave) for wave in wavelength_list]))
    
    # Create common wavelength axis
    common_wavelength = np.arange(wave_min, wave_max, wave_step)

    # Interpolate each spectrum onto the common wavelength grid
    spec_array = []
    err_array = []

    for rest_wave, spec, err in zip(wavelength_list, spec_list, err_list):
        spec_interp = np.interp(common_wavelength, rest_wave, spec)
        err_interp = np.interp(common_wavelength, rest_wave, err)
        
        spec_array.append(spec_interp)
        err_array.append(err_interp)

    spec_array = np.array(spec_array)  # Shape: (n_sources, n_wavelength_bins)
    err_array = np.array(err_array)  # Shape: (n_sources, n_wavelength_bins)

    # Determine weight exponent based on string or float input
    weight_dict = {
        'inverse variance': -2,
        'uniform': 0,
        'inverse error': -1
    }
    if isinstance(weighting, str) and weighting != 'continuum':
        weight_exponent = weight_dict.get(weighting, None)
        if weight_exponent is None:
            raise ValueError(f"weighting must be 'inverse variance', 'uniform', or 'inverse error', "
                           f"got '{weighting}'")
    elif isinstance(weighting, (int, float)):
        weight_exponent = weighting
    elif weighting != 'continuum':
        raise ValueError(f"weighting must be a string or float, got '{type(weighting)}'")
    
    # Weight by inverse median variance: w_i = 1 / median(σ_i²)
    median_err = np.nanmedian(err_array, axis=1)
    if weighting == 'continuum':
        # Weight by median SNR of continuum for absorption features
        median_spec = np.nanmedian(spec_array, axis=1)
        median_snr = median_spec / median_err
        weights = np.where(np.isfinite(median_snr), median_snr, 0.0)
    else:
        # Handle sources with all-NaN errors or zero errors
        weights = np.where((median_err > 0) & np.isfinite(median_err), 
                        median_err ** (weight_exponent), 0.0)
        
    # Apply sigma clipping to weights if requested
    if sigclip_weights is not None:
        weight_scmean, weight_scmedian, weight_scstd = sigma_clipped_stats(weights, sigma=3, maxiters=5)
        upper_limit = weight_scmedian + sigclip_weights * weight_scstd
        weights[weights > upper_limit] = upper_limit

    # Reshape weights for broadcasting: (n_sources, 1)
    weights = weights[:, np.newaxis]

    # Compute weighted average across sources
    # stacked_flux = Σ(w_i * flux_i) / Σ(w_i)
    weighted_sum = np.nansum(weights * spec_array, axis=0)
    weight_sum = np.nansum(weights, axis=0)
    stacked_flux = weighted_sum / weight_sum

    # Propagate errors: σ_stacked = √(Σ(w_i² * σ_i²)) / Σ(w_i)
    weighted_err_squared = np.nansum((weights * err_array) ** 2, axis=0)
    stacked_error = np.sqrt(weighted_err_squared) / weight_sum
    n_sources = len(spec_list)

    return common_wavelength, stacked_flux, stacked_error, n_sources