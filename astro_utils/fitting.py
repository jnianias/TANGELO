"""
Fitting utilities for astronomical spectroscopy.
"""

from . import constants as const
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import median_abs_deviation, norm
import warnings
from . import models as mdl
from astropy.stats import sigma_clipped_stats
import traceback
from error_propagation import Complex
from . import spectroscopy as spectro
import matplotlib.pyplot as plt
import error_propagation as ep
from collections import namedtuple
from . import plotting as plot

# Import moved components for backwards compatibility
from .spectroscopy import generate_spec_mask, mask_skylines, mask_otherlines

# Import dictionary of line wavelengths and speed of light
wavedict    = const.wavedict
doubletdict = const.doublets
skylinedict = const.skylines
c           = const.c  # speed of light in km/s

# Define a namedtuple for fit results
FitResult = namedtuple('FitResult', ['parameters', 'errors', 'model', 'reduced_chisq'])

def which_fit_method(linename):
    """
    Determine the fitting method for a given line name.

    Parameters
    ----------
    linename : str
        Name of the spectral line (must be present in wavedict).

    Returns
    -------
    str
        'doublet' if the line is a principal doublet, 'single' otherwise.

    Notes
    -----
    - Principal doublet lines are defined in doubletdict.
    - Secondary doublet lines will trigger a warning and return 'single'.

    Examples
    --------
    >>> which_fit_method('CIV')
    'doublet'
    >>> which_fit_method('LYALPHA')
    'single'
    """
    doublet_principals  = list(doubletdict.keys())
    doublet_secondaries = [doubletdict[key][1] for key in doublet_principals 
                           if isinstance(doubletdict[key], tuple)]
    
    if linename in doublet_principals:
        return 'doublet'
    elif linename not in doublet_secondaries:
        return 'single'
    else:
        warnings.warn(f"Line {linename} is a secondary doublet line; consider skipping!")
        return 'single'
    

from numpy.polynomial import Polynomial as nppoly

def autocorr_length(wave, spec, yerr, max_lag=10, baseline_order=None):
    """
    Estimate the noise correlation length from spectral residuals using autocorrelation analysis.

    Parameters
    ----------
    wave : array-like
        Wavelength array (used for baseline fitting).
    spec : array-like
        Spectral residuals (observed - model) or flux values.
    yerr : array-like
        Error estimates for each pixel (for chi-squared calculation).
    max_lag : int, optional
        Maximum lag to search for correlations (default: 10).
    baseline_order : int or None, optional
        Polynomial order for baseline removal before ACF. Use None for mean removal.

    Returns
    -------
    tau : float
        Correlation length in pixels (>= 1.0). Returns 1.0 if no significant correlation detected at lag 1.
    inflation_factor : float
        Error inflation factor based on chi-squared (>= 1.0).

    Notes
    -----
    - Designed for short astronomical spectra (30-100 pixels).
    - Returns the e-folding length tau where ACF(tau) = 1/e.
    - Uses robust median-based diagnostics for error inflation.

    Examples
    --------
    >>> tau, inflation = autocorr_length(wave, spec, yerr)
    >>> print(f"Correlation length: {tau}, Error inflation: {inflation}")
    """
    # 1. Baseline removal with diagnostics
    if baseline_order is not None:
        p = nppoly.fit(wave, spec, baseline_order)
        residuals = spec - p(wave)
        print(f"Baseline: polynomial order {baseline_order}")
    else:
        residuals = spec - np.mean(spec)
        print(f"Baseline: mean removal only")

    # Check for systematic excess variance (Ly-α forest, sky lines, etc.)
    # Use robust estimator: median of |residuals|/yerr
    # This is less sensitive to outliers than χ²_red
    normalized_residuals = np.abs(residuals) / yerr
    median_normalized_residual = np.median(normalized_residuals)
    
    # Trigger conservatively using median to avoid outlier sensitivity
    # But correct using proper Gaussian calibration for accurate scaling
    threshold = 1.0
    if median_normalized_residual > threshold:
        # For Gaussian noise, median(|z|) ≈ 0.675, so scale to get true σ
        inflation_factor = median_normalized_residual / 0.675
        print(f"Systematic excess variance detected: inflating errors by {inflation_factor:.2f}x")
    else:
        inflation_factor = 1.0

    # 2. ACF calculation with unbiased estimator
    n = len(residuals)
    # Remove mean to ensure zero-mean signal
    residuals = residuals - np.mean(residuals)
    
    # Compute ACF using unbiased estimator
    # For each lag k, divide the sum by the actual number of pairs (n-k)
    max_acf_lag = n  # Can compute ACF up to lag n-1
    acf = np.zeros(max_acf_lag)
    
    for k in range(max_acf_lag):
        if n - k > 0:
            # Sum of products for lag k, divided by number of pairs
            acf[k] = np.sum(residuals[:n-k] * residuals[k:]) / (n - k)
        else:
            break
    
    if acf[0] <= 0:
        print("WARNING: Non-positive ACF(0); returning τ=1")
        return 1.0, inflation_factor  # Protection against bad normalization
    
    # Normalize so ACF(0) = 1
    acf = acf / acf[0]

    # 3. Simplified robust approach for short spectra
    # Don't try to estimate noise floor from unreliable tail - just use lag-1 correlation
    
    # Stage 1: Check if there's significant correlation at lag 1
    # Use 2σ threshold - more lenient to catch real correlations in noisy data
    # For short spectra (n < 100), ACF[1] standard error ~ 1/sqrt(n)
    acf_1_stderr = 1.0 / np.sqrt(n)
    detection_threshold = 2.0 * acf_1_stderr  # 2-sigma detection (more sensitive)
    
    print(f"\nACF[1] = {acf[1]:.3f} ± {acf_1_stderr:.3f}")
    print(f"Detection threshold (2σ): {detection_threshold:.3f}")
    
    if acf[1] < detection_threshold:
        print(f"No significant correlation detected at lag 1")
        print("Returning τ=1 (uncorrelated)")
        return 1.0, inflation_factor
    
    # Stage 2: Estimate τ from early lags using exponential fit
    # Use first few lags (3-7) where ACF is most reliable
    print(f"Correlation detected: ACF[1]={acf[1]:.3f} > {detection_threshold:.3f}")
    
    # Fit exponential through lags 1-5 (or fewer if spectrum is very short)
    max_fit_lag = min(7, max_lag, n // 4)  # Don't use more than 25% of data
    
    # Use fixed threshold for boundary detection (robust for short, noisy spectra)
    boundary_threshold = 0.05
    
    # 5. Find where ACF drops below boundary threshold (fallback method)
    crossing_lag = None
    for k in range(1, min(max_lag, len(acf)-1)):
        if acf[k] < boundary_threshold:
            crossing_lag = k
            break
    
    # 6. Fit exponential decay ACF(k) = exp(-k/τ) to estimate τ
    # For spectroscopy: focus on short lags (1-5 pixels) where noise mimics lines
    # Don't fit too far - long-range correlations from baseline structure aren't relevant
    
    # Restrict fitting to short lags relevant for line detection
    # Use crossing lag as guide, but cap at max_lag
    if crossing_lag is not None and 3 <= crossing_lag <= max_lag:
        # Found a clear crossing within search range
        fit_up_to = crossing_lag
    elif crossing_lag is not None and crossing_lag < 3:
        # Very short correlation - fit up to lag 5 to be sure
        fit_up_to = min(5, max_lag)
    else:
        # No crossing found - use max_lag but warn
        fit_up_to = max_lag
        if acf[max_lag] > boundary_threshold:
            print(f"WARNING: ACF({max_lag}) = {acf[max_lag]:.3f} still above boundary threshold")
            print(f"         Long-range correlation detected - may be baseline structure")
    
    # Fit exponential to early lags
    lags = np.arange(1, min(fit_up_to + 1, len(acf)))
    acf_to_fit = acf[1:min(fit_up_to + 1, len(acf))]
    
    # Only fit where ACF is positive and meaningful (> 0.05)
    min_acf_value = 0.05
    valid_mask = acf_to_fit > min_acf_value
    n_valid = np.sum(valid_mask)
    
    print(f"Fitting range: lags 1-{len(lags)}, {n_valid}/{len(lags)} valid points (ACF > {min_acf_value})")
    
    if n_valid >= 3:
        try:
            # Constrained exponential fit: ACF(k) = exp(-k/τ)
            # This enforces ACF(0) = 1 as required for autocorrelation
            # We fit: log(ACF) = -k/τ, forcing intercept = 0
            
            # Weighted least squares for log(ACF) vs lag (zero intercept)
            weights = 1.0 / np.sqrt(lags[valid_mask])
            
            # Manual weighted fit with zero intercept: slope = Σ(w*x*y) / Σ(w*x²)
            x = lags[valid_mask]
            y = np.log(acf_to_fit[valid_mask])
            w = weights**2
            slope = np.sum(w * x * y) / np.sum(w * x * x)
            tau_fit = -1.0 / slope
            
            # Sanity check: expect short correlations (0.5 to ~2*max_lag)
            # For spurious line detection, τ > max_lag means long-range structure
            if 0.5 <= tau_fit <= 2 * max_lag:
                print(f"Fitted τ = {tau_fit:.2f} pixels (exponential fit)")
                if crossing_lag is not None:
                    print(f"(Threshold crossing at lag {crossing_lag})")
                return max(1.0, tau_fit), inflation_factor
            else:
                print(f"Fitted τ = {tau_fit:.2f} outside expected range [0.5, {2*max_lag}]")
                # If too large, cap at max_lag (conservative)
                if tau_fit > 2 * max_lag:
                    print(f"Using τ = {max_lag} (capped - likely baseline structure)")
                    return float(max_lag), inflation_factor
        except (np.linalg.LinAlgError, RuntimeError, ValueError) as e:
            print(f"Exponential fit failed: {e}")
    else:
        print(f"Not enough valid points for fitting (need ≥3, have {n_valid})")
        # Fallback: estimate τ from ACF[1] directly
        # For exponential decay: ACF(1) = exp(-1/τ) → τ = -1/log(ACF[1])
        if acf[1] > 0.1:  # Reasonable correlation at lag 1
            tau_simple = -1.0 / np.log(acf[1])
            print(f"Fallback: estimating τ from ACF[1]={acf[1]:.3f} → τ={tau_simple:.2f}")
            return max(1.0, tau_simple), inflation_factor

    # 6. Final fallback to threshold crossing or max_lag
    if crossing_lag is not None:
        print(f"Using threshold crossing at lag {crossing_lag} (ACF={acf[crossing_lag]:.3f})")
        return float(crossing_lag), inflation_factor
    else:
        print(f"WARNING: No crossing found within max_lag={max_lag} - returning max_lag")
        if acf[min(max_lag, len(acf)-1)] > boundary_threshold:
            print(f"WARNING: ACF({max_lag}) = {acf[min(max_lag, len(acf)-1)]:.3f} still above boundary threshold")
        return float(max_lag), inflation_factor

def gen_corr_noise(yerr, corr_len, size=None):
    """
    Generate correlated noise using an AR(1) process for Monte Carlo error estimation.

    Parameters
    ----------
    yerr : array-like
        Array of error values.
    corr_len : float
        Correlation length tau in pixels (e-folding scale).
    size : int, optional
        Output size (defaults to len(yerr)).

    Returns
    -------
    noise : ndarray
        Correlated noise array with the same shape as yerr/size.

    Notes
    -----
    - For corr_len <= 1, returns uncorrelated Gaussian noise.
    - For corr_len > 1, uses AR(1) process: noise[i] = phi * noise[i-1] + sqrt(1-phi^2) * epsilon[i], phi = exp(-1/tau).

    Examples
    --------
    >>> noise = gen_corr_noise(yerr, corr_len=5)
    """
    if corr_len <= 1:
        return np.random.normal(scale=yerr, size=size)

    n = len(yerr) if size is None else size

    # AR(1) process: noise[i] = phi * noise[i-1] + sqrt(1-phi^2) * epsilon[i]
    noise = np.zeros(n)
    noise[0] = np.random.normal()

    # For correlation length τ (e-folding scale): phi = exp(-1/τ)
    phi = np.exp(-1.0 / corr_len)

    for i in range(1, n):
        noise[i] = phi * noise[i-1] + np.sqrt(1-phi**2) * np.random.normal()

    # Scale by errors while preserving correlation structure
    return noise * (yerr[:n] if yerr.ndim > 0 else yerr) 


def check_multiple_peaks(wave, residuals, err, fitted_peak_wave, fitted_amplitude, 
                         fitted_width, n_fit_params=3, min_separation=3, 
                         amplitude_ratio_threshold=0.5, width_ratio_range=(0.5, 2.0), 
                         detection_threshold=3.0, chi2_threshold=3.0):
    """
    Check for additional unexplained peaks in the residuals similar to the fitted line.

    Parameters
    ----------
    wave : array-like
        Wavelength array.
    residuals : array-like
        Residuals after subtracting the fitted model (observed - model).
    err : array-like
        Error array (for significance calculation).
    fitted_peak_wave : float
        Wavelength of the fitted peak.
    fitted_amplitude : float
        Amplitude (height) of the fitted peak.
    fitted_width : float
        FWHM or sigma of the fitted peak (in same units as wave).
    n_fit_params : int, optional
        Number of parameters in the fit (default: 3).
    min_separation : float, optional
        Minimum separation in units of fitted_width to consider peaks as distinct (default: 3).
    amplitude_ratio_threshold : float, optional
        Minimum amplitude ratio to flag comparable peaks (default: 0.5).
    width_ratio_range : tuple, optional
        Range of width ratios to flag comparable peaks (default: (0.5, 2.0)).
    detection_threshold : float, optional
        Minimum SNR for a peak in residuals to be considered significant (default: 3.0).
    chi2_threshold : float, optional
        Only search for peaks if reduced chi-square > this value (default: 3.0).

    Returns
    -------
    dict
        Dictionary with keys:
            - 'suspicious': bool
            - 'n_comparable_peaks': int
            - 'peak_info': list of dicts (wavelength, amplitude, snr, width, amplitude_ratio, width_ratio, separation)
            - 'flag': str
            - 'reduced_chi2': float
            - 'message': str

    Notes
    -----
    - Flags additional peaks in residuals that may indicate poor fit or blended lines.
    - Uses SNR and amplitude/width criteria to filter peaks.

    Examples
    --------
    >>> result = check_multiple_peaks(wave, residuals, err, 5000, 10, 3)
    >>> print(result['n_comparable_peaks'])
    """
    from scipy.signal import find_peaks
    
    # First check: Calculate reduced chi-square
    chi2 = np.sum((residuals / err) ** 2)
    n_data = len(residuals)
    dof = n_data - n_fit_params
    reduced_chi2 = chi2 / dof
    
    # If fit is good (low chi-square), no need to search for peaks
    if reduced_chi2 <= chi2_threshold:
        return {
            'suspicious': False,
            'n_comparable_peaks': 0,
            'peak_info': [],
            'flag': '',
            'reduced_chi2': reduced_chi2,
            'message': f'Good fit (χ²_red = {reduced_chi2:.2f} ≤ {chi2_threshold}), no peak search needed'
        }
    
    # Find peaks in the residuals (positive and negative)
    # Look at absolute value to catch both emission and absorption features
    abs_residuals = np.abs(residuals)
    significance = abs_residuals / err
    
    # Find peaks in the significance array
    # Use a minimum height of detection_threshold sigma
    peaks_idx, properties = find_peaks(significance, 
                                       height=detection_threshold,
                                       prominence=detection_threshold * 0.5)
    
    if len(peaks_idx) == 0:
        return {
            'suspicious': False,
            'n_comparable_peaks': 0,
            'peak_info': [],
            'flag': '',
            'reduced_chi2': reduced_chi2,
            'message': f'Poor fit (χ²_red = {reduced_chi2:.2f}) but no significant peaks found'
        }
    
    # Get peak properties
    peak_waves = wave[peaks_idx]
    peak_amplitudes = abs_residuals[peaks_idx]  # Use absolute residuals where peaks were found
    peak_snrs = significance[peaks_idx]
    
    # Estimate widths of peaks using scipy's peak_widths
    from scipy.signal import peak_widths
    widths_idx, width_heights, left_ips, right_ips = peak_widths(
        significance, peaks_idx, rel_height=0.5
    )
    
    # Convert width from indices to wavelength units
    delta_wave = np.median(np.diff(wave))
    peak_widths = widths_idx * delta_wave
    
    # Filter peaks:
    # 1. Exclude the fitted peak itself (within min_separation * fitted_width)
    # 2. Keep only peaks with comparable amplitude
    # 3. Keep only peaks with comparable width
    
    suspicious_peaks = []
    
    for i, (pw, pa, psnr, pwidth) in enumerate(zip(peak_waves, peak_amplitudes, 
                                                     peak_snrs, peak_widths)):
        # Check if this is the fitted peak (too close)
        separation = np.abs(pw - fitted_peak_wave)
        if separation < min_separation * fitted_width:
            continue  # This is probably the fitted line itself
        
        # Check amplitude ratio (pa is already absolute value from abs_residuals)
        amplitude_ratio = pa / np.abs(fitted_amplitude)
        if amplitude_ratio < amplitude_ratio_threshold:
            continue  # Too weak
        
        # Check width ratio
        width_ratio = pwidth / fitted_width
        if width_ratio < width_ratio_range[0] or width_ratio > width_ratio_range[1]:
            continue  # Too narrow or too wide
        
        # This peak is suspicious!
        suspicious_peaks.append({
            'wavelength': pw,
            'amplitude': pa,
            'snr': psnr,
            'width': pwidth,
            'amplitude_ratio': amplitude_ratio,
            'width_ratio': width_ratio,
            'separation': separation
        })
    
    n_suspicious = len(suspicious_peaks)
    flag = 'm' if n_suspicious > 0 else ''
    
    if n_suspicious > 0:
        message = f'Poor fit (χ²_red = {reduced_chi2:.2f}): found {n_suspicious} comparable peak(s)'
    else:
        message = f'Poor fit (χ²_red = {reduced_chi2:.2f}) but no comparable peaks (different amplitude/width)'
    
    return {
        'suspicious': n_suspicious > 0,
        'n_comparable_peaks': n_suspicious,
        'peak_info': suspicious_peaks,
        'flag': flag,
        'reduced_chi2': reduced_chi2,
        'message': message
    }


def sigma_to_percentile(sigma):
    """
    Convert a sigma level to a percentile for non-Gaussian uncertainties.

    Parameters
    ----------
    sigma : float
        Sigma level (e.g., 1, 2, 3 for 1σ, 2σ, 3σ).

    Returns
    -------
    float
        Percentile value representing the confidence level.

    Notes
    -----
    - Uses the cumulative distribution function (CDF) of the normal distribution.
    - Returns the two-tailed percentile.

    Examples
    --------
    >>> sigma_to_percentile(1)
    68.27
    >>> sigma_to_percentile(2)
    95.45
    """
    
    # Calculate the two-tailed percentile
    # For Gaussian: P(|X| < kσ) = erf(k/√2)
    confidence_level = norm.cdf(sigma) - norm.cdf(-sigma)
    
    # Convert to percentile (0-100 scale)
    percentile = confidence_level * 100
    
    return percentile


def avgfunc(poptl, errfunc, sig_clip = 7.0):
    """
    Compute the average and error of fit parameters from Monte Carlo samples.

    Parameters
    ----------
    poptl : array-like
        List or array of parameter samples from Monte Carlo realizations.
    errfunc : str
        Error estimation method. Options are:
            'stddev'    : standard deviation (sigma-clipped)
            'mad'       : median absolute deviation
            'mad_adj'   : adjusted MAD (scaled to match stddev for Gaussian)
            'stddev_X'  : sigma-clipped stddev with clipping value X
            '84-16'     : 68% confidence interval from percentiles
    sig_clip : float, optional
        Sigma clipping threshold for stddev (default is 7.0).

    Returns
    -------
    list
        [average, error] for each parameter.

    Notes
    -----
    Uses sigma-clipped statistics for 'stddev'.
    Uses median absolute deviation for 'mad'.

    Examples
    --------
    >>> avgfunc(samples, 'stddev')
    [mean, stddev]
    >>> avgfunc(samples, 'mad')
    [median, mad]
    """
    if errfunc == 'stddev':
        _scs = sigma_clipped_stats(np.array(poptl), axis=0, maxiters=1, sigma=sig_clip)
        return [_scs[0], _scs[2]]
    elif errfunc == 'mad':
        medpopt = np.nanmedian(poptl, axis=0)
        return [medpopt, np.nanmedian(np.abs(poptl - medpopt), axis=0)]
    elif errfunc == 'mad_adj':
        medpopt = np.nanmedian(poptl, axis=0)
        mad = np.nanmedian(np.abs(poptl - medpopt), axis=0)
        # Adjust MAD to match stddev for Gaussian
        adj_mad = mad * 1.4826
        return [medpopt, adj_mad]
    elif errfunc.startswith('stddev_'): # e.g., 'stddev_5.0'
        try:
            clip_val = float(errfunc.split('_')[1])
        except (IndexError, ValueError):
            clip_val = sig_clip
        _scs = sigma_clipped_stats(np.array(poptl), axis=0, maxiters=1, sigma=clip_val)
        return [_scs[0], _scs[2]]
    elif errfunc == '84-16': # 68% confidence interval
        p16 = np.nanpercentile(poptl, 16, axis=0)
        p84 = np.nanpercentile(poptl, 84, axis=0)
        medpopt = np.nanmedian(poptl, axis=0)
        return [medpopt, (p84 - p16) / 2.0]

def fit_mc(f, x, y, yerr, p0, bounds=None, niter=500, errfunc='mad',
           return_sample=False, chisq_thresh=np.inf, sig_clip=7.0,
           autocorrelation=False, max_lag=10, baseline_order=0, max_nfev=5000):
    """
    Monte Carlo fitting with correlated noise handling for spectroscopy.

    Fits a model to data and estimates parameter uncertainties via Monte Carlo
    resampling. Can account for correlated noise in residuals to avoid 
    underestimating errors when noise correlation mimics emission line structure.

    Parameters
    ----------
    f : callable
        Model function to fit: f(x, *params)
    x : array-like
        Independent variable (e.g., wavelength)
    y : array-like
        Dependent variable (e.g., flux)
    yerr : array-like
        Uncertainties in y
    p0 : array-like
        Initial guess for parameters
    bounds : tuple of arrays, optional
        Lower and upper bounds for parameters
    niter : int, optional
        Number of Monte Carlo iterations (default: 500)
    errfunc : str, optional
        Error estimation method: 'mad' or 'stddev' (default: 'mad')
    return_sample : bool, optional
        If True, return full MC sample in addition to mean/error (default: False)
    chisq_thresh : float, optional
        Chi-square threshold to filter bad fits (default: np.inf, no filter)
    sig_clip : float, optional
        Sigma clipping threshold for stddev (default: 7.0)
    autocorrelation : bool or int, optional
        False: assume uncorrelated noise (default)
        True: estimate correlation length from fit residuals
        int: use fixed correlation length in pixels
        For spectroscopy, use max_lag=5-10 to focus on line-scale correlations
    max_lag : int, optional
        Maximum lag for ACF estimation (default: 10 pixels)
        Relevant for detecting spurious lines from correlated noise
    baseline_order : int, optional
        Polynomial order for baseline removal in ACF (default: 0)
        Use 0 for residuals, 1-2 for raw spectra with trends
    max_nfev : int, optional
        Max function evaluations per fit (default: 5000)

    Returns
    -------
    params : array-like
        Best-fit parameters (mean or median of MC sample)
    errors : array-like
        Parameter uncertainties (stddev or MAD of MC sample)
    sample : array-like, optional
        Returned only if return_sample=True; shape (niter, nparams)

    Notes
    -----
    - If return_sample is True, returns ([params, errors], sample)
    - If return_sample is False, returns [params, errors]

    Examples
    --------
    >>> fit_mc(model, x, y, yerr, p0)
    [params, errors]
    >>> fit_mc(model, x, y, yerr, p0, return_sample=True)
    ([params, errors], sample)
    """

    try:
        # Initial fit using large max_nfev
        popt, _ = curve_fit(f, x, y, sigma=yerr, p0=p0, bounds=bounds,
                              absolute_sigma=True, max_nfev=100000)

        # Estimate correlation length from RESIDUALS
        if isinstance(autocorrelation, bool) and autocorrelation:
            residuals = y - f(x, *popt)
            correlation_length, error_inflation = autocorr_length(x, residuals, yerr, max_lag, baseline_order)
            print(f"Estimated correlation length: {correlation_length:.3f} pixels")
            if error_inflation > 1.0:
                print(f"→ Applying error inflation: {error_inflation:.2f}x (for Monte Carlo noise generation)")
                yerr_inflated = yerr * error_inflation
            else:
                yerr_inflated = yerr
                print(f"→ No error inflation needed (median residuals within expected range)")
        elif isinstance(autocorrelation, bool) and not autocorrelation:
            correlation_length = 1
            yerr_inflated = yerr
        elif isinstance(autocorrelation, int):
            correlation_length = autocorrelation
            yerr_inflated = yerr
            print(f"Using fixed correlation length: {correlation_length} pixels")
        else:
            correlation_length = 1
            yerr_inflated = yerr

        poptlist = []
        valid_iters = 0
        total_iters = 0
        max_total_iters = 2 * niter
        
        while valid_iters < niter and total_iters < max_total_iters:
            total_iters += 1
            
            # Generate perturbations (using inflated errors if systematic variance detected)
            yper = (gen_corr_noise(yerr_inflated, correlation_length) if correlation_length > 1
                   else np.random.normal(scale=np.abs(yerr_inflated)))

            try:
                popt_i, _ = curve_fit(f, x, y + yper, sigma=yerr, p0=popt,
                                    bounds=bounds, absolute_sigma=True, max_nfev=max_nfev)

                # Chi-square filtering
                if np.isfinite(chisq_thresh):
                    chisq = np.sum(((f(x, *popt_i) - (y + yper))/yerr)**2) / (len(x) - len(popt_i))
                    if chisq > chisq_thresh:
                        continue

                poptlist.append(popt_i)
                valid_iters += 1
            except RuntimeError:
                print(f"Iteration {valid_iters} failed to converge; skipping.")
                continue
        
        # Warn if we hit the max iteration limit
        if total_iters >= max_total_iters and valid_iters < niter:
            print(f"Warning: Reached maximum iteration limit ({max_total_iters}). "
                  f"Only {valid_iters}/{niter} successful fits obtained.")

        if not poptlist:
            raise RuntimeError("All MC iterations failed to converge")

        return (avgfunc(poptlist, errfunc), poptlist) if return_sample else avgfunc(poptlist, errfunc)

    except Exception as e:
        print(f"Error in fit_mc: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        return None
    

from typing import TypedDict, Optional, Union

class BootstrapParams(TypedDict, total=False):
    niter: int
    errfunc: str
    chisq_thresh: float
    sig_clip: float
    autocorrelation: Union[bool, int]
    max_lag: int
    baseline_order: Optional[int]
    max_nfev: int


def get_initial_guesses_from_catalog(linetab, linename, type='auto'):
    """
    Extract initial guesses for fitting parameters from a line table.

    Parameters
    ----------
    linetab : astropy Table
        Table containing line information with columns for initial guesses.
    linename : str
        Name of the line to extract initial guesses for.
    type : str, optional
        Type of line ('auto', 'abs', 'em')

    Returns
    -------
    dict
        Dictionary of initial guesses for parameters.

    Notes
    -----
    - Expects columns in linetab to be named as '{linename}_PARAM' for each parameter.
    - Parameters include LPEAK, FWHM, AMP, ASYM, KURT, etc.

    Examples
    --------
    >>> get_initial_guesses(linetab, 'LYALPHA')
    {'LPEAK': 1216.0, 'DISP': 300.0, 'AMP': 1.0}
    """
    # Get the row(s) of the table for this line
    linerows = linetab[linetab['LINE'] == linename]
    # If type has been specified, filter by type
    if type == 'abs':
        linerow = linerows[linerows['FAMILY'] == 'abs'][0]
    elif type == 'em':
        linerow = linerows[~(linerows['FAMILY'] == 'abs')][0]
    else: # If "auto", take the line with the greatest abs value of flux
        fluxes = linerows['FLUX']
        max_flux_idx = np.argmax(np.abs(fluxes))
        linerow = linerows[max_flux_idx]
    
    # If it's the primary line of a doublet, also get the secondary line info (if it's there)
    if linename in doubletdict:
        sec_linerow = None
        try:
            sec_linename = doubletdict[linename][1]
            sec_linerow = linetab[linetab['LINE'] == sec_linename][0]
        except IndexError: # No secondary line found
            # Check to see if this is due to the secondary being redshifted out of the MUSE wavelength range
            z_primary = (linerow['LBDA_OBS'] / wavedict[linename]) - 1
            sec_wave_rest = wavedict[sec_linename]
            sec_wave_obs = sec_wave_rest * (1 + z_primary)
            if sec_wave_obs > 9350: # MUSE red cutoff
                warnings.warn(f"Secondary line {sec_linename} of doublet {linename} "
                              f"redshifted out of MUSE range at z={z_primary:.3f}; "
                              f"proceeding without secondary line info.")
                sec_linerow = None
            else:
                warnings.warn(f"Secondary line {sec_linename} of doublet {linename} "
                              f"not found in line table; using primary line info only.")
                sec_linerow = None
    
    # Lyman alpha must be treated separately due to different parameters
    if linename == 'LYALPHA':
        default_peak_sep = 2.0 # default peak separation in rest frame
        z_est = (linerow['LBDA_OBS'] / wavedict['LYALPHA']) - 1
        initial_guesses = {
            'AMPB': linerow['PEAK_OBS'], # amplitude of blue peak
            'LPEAKB': linerow['LBDA_OBS'] - default_peak_sep * (1 + z_est), # peak wavelength for blue peak
            'DISPB': linerow['FWHM_OBS'] / 2.355, # approximate dispersion equivalent (n.b. not exact for asymmetric gaussian)
            'ASYMB': -0.1,                 # asymmetry -- generic guess
            'AMPR': linerow['PEAK_OBS'], # amplitude of red peak
            'LPEAKR': linerow['LBDA_OBS'], # peak wavelength for red peak
            'DISPR': linerow['FWHM_OBS'] / 2.355, # approximate dispersion equivalent (n.b. not exact for asymmetric gaussian)
            'ASYMR': 0.1,                  # asymmetry -- generic guess
            'CONT': linerow['CONT_OBS'], # continuum level
        }
    elif linename in doubletdict:
        initial_guesses = {
            'FLUX': linerow['FLUX'],  # integrated flux of primary line with correction factor
            'LPEAK': linerow['LBDA_OBS'], # peak wavelength
            'FWHM': linerow['FWHM_OBS'], # fwhm
            'FLUX2': sec_linerow['FLUX'] if sec_linerow is not None else linerow['FLUX'], 
            'CONT': linerow['CONT_OBS'], # continuum level
            'SLOPE': 0.0,               # slope -- generic guess
        }
    else:
        initial_guesses = {
            'FLUX': linerow['FLUX'],  # integrated flux with correction factor
            'LPEAK': linerow['LBDA_OBS'], # peak wavelength
            'FWHM': linerow['FWHM_OBS'], # fwhm
            'CONT': linerow['CONT_OBS'], # continuum level
            'SLOPE': 0.0,               # slope -- generic guess
        }

    return initial_guesses
        
def gen_bounds(initial_guesses, linename, input_bounds={}, force_sign=None):
    """
    Generate default bounds for fitting parameters based on initial guesses.

    Parameters
    ----------
    initial_guesses : dict
        Dictionary of initial guesses for parameters. Must be provided with all necessary keys in
        order to generate the bounds dictionary.
    linename : str
        Name of the line (must be present in wavedict).
    input_bounds : dict, optional
        Dictionary of user-specified bounds for parameters if desired. Any parameters
        not specified here will use default bounds.
    force_sign : str, optional
        If 'positive', forces amplitude/flux parameters to be positive.
        If 'negative', forces them to be negative.
        Otherwise, no sign constraint is applied.

    Returns
    -------
    dict
        Dictionary of bounds for parameters.

    Notes
    -----
    - Default bounds are set to ±6.25 Angstroms for LPEAK, and reasonable ranges for other parameters.

    Examples
    --------
    >>> gen_bounds({'LPEAK': 1216, 'FWHM': 300, 'AMP': 1}, 'LYALPHA')
    {'LPEAK': (1209.75, 1222.25), 'FWHM': (50, 1000), 'AMP': (0, 10)}
    """
    if linename not in wavedict:
        raise ValueError(f"Line {linename} not found in wavelength dictionary.")
    if linename == 'LYALPHA' and force_sign != 'positive':
        print("WARNING: Fitting Lyman alpha allowing negative solution! force_sign set to 'positive' recommended.")

    # Get central peak for Lya if applicable
    cen_r_init = np.nan
    lpeak_key = 'LPEAK'
    if linename == 'LYALPHA': # Gets the central peak for Lya double peak or single peak
        lpeak_key = 'LPEAKR' if 'LPEAKR' in initial_guesses else 'LPEAK' # This can vary based on input
        cen_r_init = initial_guesses[lpeak_key]
    
    rest_wave = wavedict[linename]
    z         = initial_guesses[lpeak_key] / rest_wave - 1

    default_flux_min = -100 * np.abs(initial_guesses.get('FLUX', 100)) if force_sign != 'positive' else 0
    default_flux_max = 100 * np.abs(initial_guesses.get('FLUX', 100)) if force_sign != 'negative' else 0
    default_lpeak_tol = 6.25 / 4  # Angstroms in rest-frame
    default_lpeakb_tol = [-3, -1] # What region relative to LPEAKR to allow for LPEAKB in rest frame
    default_lpeakr_tol = [-1, 3]  # How wide an area around LPEAKR to allow in rest frame

    defaults = {
        'LPEAK':  (initial_guesses.get('LPEAK', np.nan) - default_lpeak_tol * (1 + z), # if the user didn't provide LPEAK, set to nan so bounds will error out
                  initial_guesses.get('LPEAK', np.nan) + default_lpeak_tol * (1 + z)),
        'FWHM':   (2.0, np.abs(spectro.vel2wave(300, rest_wave, 0) - spectro.vel2wave(0, rest_wave, 0)) * (1 + z)),
        'FLUX':   (default_flux_min, default_flux_max),
        'FLUX2':  (default_flux_min, default_flux_max), # for doublets
        'CONT':   (-50, 2000), # appropriate for sky-subtracted MUSE LAE spectra
        'AMP':    (default_flux_min, 10000), # amplitude bounds -- note AMP is used for Lya
        'DISP':   (0.2 * (1 + z), 1 * (1 + z)), # dispersion bounds for Lya
        'ASYM':   (-0.5, 0.5), # asymmetry bounds for Lya
        'AMPB':   (default_flux_min, 10000), # amplitude of blue peak for Lya
        'LPEAKB': (initial_guesses.get('LPEAKR', np.nan) + default_lpeakb_tol[0] * (1 + z), 
                   initial_guesses.get('LPEAKR', np.nan) + default_lpeakb_tol[1] * (1 + z)),
        'DISPB':  (0.2 * (1 + z), 1 * (1 + z)), # dispersion of blue peak for Lya
        'ASYMB':  (-0.5, 0.5), # asymmetry of blue peak for Lya
        'AMPR':   (default_flux_min, 10000), # amplitude of red peak for Lya
        'LPEAKR': (initial_guesses.get('LPEAKR', np.nan) + default_lpeakr_tol[0] * (1 + z), 
                   initial_guesses.get('LPEAKR', np.nan) + default_lpeakr_tol[1] * (1 + z)),
        'DISPR':  (0.2 * (1 + z), 1 * (1 + z)), # dispersion of red peak for Lya
        'ASYMR':  (-0.5, 0.5), # asymmetry of red peak for Lya
        'SLOPE':  (-np.inf, np.inf), # slope for linear continuum
        'TAU':    (0, 200), # optical depth for Damped Lyman alpha profile
        'FWHM_ABS':   ((spectro.vel2wave(100, rest_wave, 0) - spectro.vel2wave(0, rest_wave, 0)) * (1 + z), 
                  (spectro.vel2wave(500, rest_wave, 0) - spectro.vel2wave(0, rest_wave, 0)) * (1 + z)), # FWHM bounds
        'LPEAK_ABS': (cen_r_init - 3.33 * (1 + z), cen_r_init + 3.33 * (1 + z))
    }

    bounds = {
        param: input_bounds.get(param, defaults[param]) for param in initial_guesses.keys()
    }

    return bounds

def prep_inputs(initial_guesses, linename, z_lya=None, bounds={}):
    """
    Prepare and validate input parameters for fitting routines using Lyman alpha redshift as a 
    reference point.

    Parameters
    ----------
    initial_guesses : dict
        Dictionary of initial guesses for parameters (LPEAK is required, others optional).
    linename : str
        Name of the line (must be present in wavedict).
    z_lya : float
        Redshift of Lyman-alpha line (used as sanity check for LPEAK).
    bounds : dict, optional
        Dictionary of bounds for parameters (keys matching those in initial_guesses).

    Returns
    -------
    initial_guesses : dict
        Validated initial guesses for parameters.
    bounds : dict
        Validated bounds for parameters.

    Notes
    -----
    - Resets LPEAK if initial guess is >500 km/s from expected value.
    - Adjusts initial guesses to be within bounds if necessary.

    Examples
    --------
    >>> prep_inputs({'LPEAK': 1216}, {'LPEAK': (1200, 1230)}, 'LYALPHA', 2.0)
    ({'LPEAK': 1216}, {'LPEAK': (1200, 1230)})
    """
    # If LPEAK is not provided, raise error
    if 'LPEAK' not in initial_guesses:
        raise ValueError("Initial guess for LPEAK is required.")
    
    # Get the approximate expected observed wavelength based on z_lya
    if linename not in wavedict:
        raise ValueError(f"Line {linename} not found in wavelength dictionary.")

    # Calculate the velocity buffer from the bounds if provided, else default to +/- 6.25 Angstroms
    wave_buffer = bounds.get('LPEAK', (initial_guesses['LPEAK'] - 6.25, initial_guesses['LPEAK'] + 6.25))
    wave_buffer = (wave_buffer[1] - wave_buffer[0]) / 2.0

    if z_lya is not None:
        # If the initial guess for LPEAK is more than 500 km/s away from expected, raise warning
        # and reset to generic offset of -200 km/s
        expected_wavelength = wavedict[linename] * (1 + z_lya)
        lpeak_init = initial_guesses['LPEAK']
        delta_v = spectro.wave2vel(lpeak_init, wavedict[linename], z_lya)
        if abs(delta_v) > 500:
            warnings.warn(f"Initial guess for LPEAK ({lpeak_init:.2f} Å) is {delta_v:.1f} km/s"
                        f" from Lyman alpha ({expected_wavelength:.2f} Å); resetting to -200 km/s offset.")
            initial_guesses['LPEAK'] = spectro.vel2wave(-200, wavedict[linename], z_lya)
            if 'LPEAK' in bounds:
                bounds['LPEAK'] = (initial_guesses['LPEAK'] - 6.25, initial_guesses['LPEAK'] + 6.25)

    # Check values of initial guesses and bounds for bad values
    for param, value in initial_guesses.items():
        if not np.isfinite(value):
            raise ValueError(f"Initial guess for {param} is not finite: {value}")
        if param in bounds:
            lower, upper = bounds[param]
            if lower >= upper:
                raise ValueError(f"Lower bound must be less than upper bound for {param}:"
                                 f" ({bounds[param][0]:.5f}, {bounds[param][1]:.5f})")
            if not (lower <= value <= upper):
                warnings.warn(f"Initial guess for {param} ({value:.5f}) is outside bounds"
                              f" ({bounds[param][0]:.5f}, {bounds[param][1]:.5f});"
                               "adjusting to midpoint.")
                initial_guesses[param] = (lower + upper) / 2.0

    return initial_guesses, bounds


def check_inputs(p0, bounds):
    """
    Ensure initial guesses are within bounds and not NaN. used as a final check before fitting.

    Parameters
    ----------
    p0 : list or array-like
        Initial guesses for parameters.
    bounds : tuple of lists or arrays
        Lower and upper bounds for each parameter.

    Returns
    -------
    p0_checked : list
        Initial guesses adjusted to be within bounds and not NaN.
    bounds : tuple
        Bounds for each parameter (unchanged).

    Notes
    -----
    - NaN values are replaced by the midpoint of bounds.
    - Out-of-bounds values are adjusted to the midpoint or slightly inside bounds.

    Examples
    --------
    >>> check_inputs([np.nan, 5], ([0, 0], [10, 10]))
    ([5.0, 5], ([0, 0], [10, 10]))
    """
    # Check that lower bounds are less than upper bounds
    for i in range(len(bounds[0])):
        if bounds[0][i] >= bounds[1][i]:
            raise ValueError(f"Lower bound {bounds[0][i]} is not less than upper bound {bounds[1][i]} for parameter {i}")

    p0_checked = []
    for i, val in enumerate(p0):
        lower, upper = bounds[0][i], bounds[1][i]
        
        # Check for NaN first
        if np.isnan(val):
            # Use midpoint of bounds for NaN values
            if np.isinf(lower) or np.isinf(upper):
                # Handle infinite bounds
                if np.isinf(lower) and np.isinf(upper):
                    val_new = 0.0  # Both infinite, use 0
                elif np.isinf(lower):
                    val_new = upper * 0.5 if upper > 0 else upper - 100
                else:  # upper is inf
                    val_new = lower * 0.5 if lower < 0 else lower + 100
            else:
                val_new = (lower + upper) / 2.0
            print(f"WARNING: Initial guess {i} is NaN; adjusting to {val_new:.3e}")
            val = val_new
        elif not (lower <= val <= upper):
            # Out of bounds - use midpoint or adjust slightly inside bounds
            original_val = val
            if np.isinf(lower) or np.isinf(upper):
                if np.isinf(lower):
                    val = upper * 0.9 if upper != 0 else upper - 0.1
                else:  # upper is inf
                    val = lower * 1.1 if lower != 0 else lower + 0.1
            else:
                val = (lower + upper) / 2.0
            print(f"WARNING: {i}-th initial guess {original_val} is outside bounds ({lower:.5f}, {upper:.5f}); adjusting to {val:.3e}")
            
        p0_checked.append(val)
    
    return p0_checked, bounds


def condition_initial_guesses(initial_guesses, wavelength, spectrum, errors, linename):    
    """
    Condition initial guesses for fitting parameters based on actual data.

    Parameters
    ----------
    initial_guesses : dict
        Dictionary of initial guesses for parameters.
    wavelength : array-like
        Wavelength array.
    spectrum : array-like
        Flux density array.
    errors : array-like
        Flux density uncertainties.
    linename : str
        Name of the line being fitted.

    Returns
    -------
    initial_guesses : dict
        Conditioned initial guesses.

    Notes
    -----
    - Adjusts initial guesses based on the actual data in the fitting region.
    - Ensures initial guesses are within reasonable bounds.

    Examples
    --------
    >>> condition_initial_guesses({'LPEAK': 5000, 'FLUX': 10}, wavelength, spectrum, errors, 'CIV')
    """
    # Get initial guess for LPEAK
    lpeak_init = initial_guesses['LPEAK']

    # Define a fitting region around LPEAK
    fit_region = (lpeak_init - 25 < wavelength) & (wavelength < lpeak_init + 25)
    if fit_region.sum() < 3:
        print(f"Not enough data points around initial LPEAK ({lpeak_init}) to condition guesses.")
        return initial_guesses

    wl_fit = wavelength[fit_region]
    spec_fit = spectrum[fit_region]
    err_fit = errors[fit_region]

    # Adjust continuum initial guess based on median of fitting region
    if 'CONT' in initial_guesses:
        cont_init = np.nanmedian(spec_fit)
        initial_guesses['CONT'] = cont_init

    # Adjust FLUX initial guess based on height of the tallest peak in the fitting region
    if 'FLUX' in initial_guesses:
        flux_init = initial_guesses['FLUX']
        if flux_init > 0: # Emission line
            peak_flux = np.max(spec_fit - np.nanmedian(spec_fit))
        else: # Absorption line
            peak_flux = np.min(spec_fit - np.nanmedian(spec_fit))
        # Estimate integrated flux as peak_flux * FWHM (assuming Gaussian shape)
        if 'FWHM' in initial_guesses:
            fwhm_init = initial_guesses['FWHM']
        else:
            fwhm_init = 3.0  # Default guess if FWHM not provided
        flux_estimate = peak_flux * fwhm_init * (np.sqrt(2 * np.pi) / 2.355)  # Convert peak to integrated flux
        initial_guesses['FLUX'] = flux_estimate

    return initial_guesses


def fit_line(wavelength, spectrum, errors, linename, initial_guesses,
             bounds = {}, continuum_buffer = 25., plot_result = True, 
             ax_in = None, bootstrap_params: Optional[BootstrapParams] = None, 
             save_plots=False, plot_dir=None, cluster='', full_iden='',
             spec_type='aper', lya_z=None):
    """
    Fit a single or double Gaussian profile to a spectral line (plus its doublet partner if present).
    If the line is Lyman alpha, raises an error (use specialized function).

    Parameters
    ----------
    wavelength : array-like
        Wavelength array.
    spectrum : array-like
        Flux density array.
    errors : array-like
        Flux density uncertainties.
    linename : str
        Name of the line (must be in wavedict).
    initial_guesses : dict
        Dictionary of initial guesses for parameters (LPEAK required, others optional).
    bounds : dict, optional
        Dictionary of parameter bounds.
    continuum_buffer : float, optional
        Buffer region around the line to include in the fit (default: 25).
    plot_result : bool, optional
        If True, plot the fitting result (default: True).
    ax_in : matplotlib axis, optional
        Axis to plot on (if plot_result is True).
    bootstrap_params : dict, optional
        Dictionary of parameters for Monte Carlo error estimation (niter, errfunc, chisq_thresh, 
        sig_clip, autocorrelation, max_lag, baseline_order, max_nfev). If None, uses standard curve_fit errors.
    save_plots : bool, optional
        If True, saves the plot to a file (default: False).
    plot_dir : str, optional
        Directory to save plots if save_plots is True.
    cluster : str, optional
        Cluster name for labelling and saving plots (default: '').
    full_iden : str, optional
        Full identifier for labelling and saving plots (default: '').
    spec_type : str, optional
        Type of spectrum being fitted (for labeling purposes, default: 'aper').
    lya_z : float, optional
        Redshift of Lyman alpha line (used for sanity check of LPEAK -- if None, no check performed).

    Returns
    -------
    fit_result : dict
        Dictionary containing fit parameters, errors, model, fit mask, reduced chi-squared, and peak check results.

    Notes
    -----
    - Uses curve_fit for fitting; can use Monte Carlo error estimation if bootstrap_params is provided.
    - Handles doublet lines automatically.

    Examples
    --------
    >>> fit_line(wavelength, spectrum, errors, 'CIV', {'LPEAK': 1549}, bounds={'LPEAK': (1540, 1560)})
    """

    # If the line is Lyman alpha, use the specialised Lyman alpha fitting function
    if linename == 'LYALPHA':
        raise ValueError("Use specialized Lyman-alpha fitting function.")

    # Get initial guess for flux (this is required to set up the fitting region)
    lpeak_init      = initial_guesses['LPEAK']

    # Determine whether to fit as a single line or doublet
    method = which_fit_method(linename)

    # Rest wavelength of the line
    rest_wavelength = wavedict[linename]
    rest_wavelength_second = np.nan
    if method == 'doublet':
        rest_wavelength_second = wavedict[doubletdict[linename][1]]

    # Define fitting region, masking any problematic areas (zeros, nans, infs)
    fit_mask = generate_spec_mask(wavelength, spectrum, errors, 
                                  lpeak_init, continuum_buffer, linename)
    # Extend it if its a doublet to include the second line
    if method == 'doublet':
        fit_mask += generate_spec_mask(wavelength, spectrum, errors,
                                       lpeak_init * (rest_wavelength_second / rest_wavelength),
                                       continuum_buffer, linename)
    
    # Make sure there are enough good points to fit
    if fit_mask.sum() < 10:
        print(f"Not enough good points to fit {linename}.")
        return {}
        
    # Make sure that the immediate region around the line is not all masked
    line_region = np.logical_and(lpeak_init - 2.5 < wavelength, wavelength < lpeak_init + 2.5)
    # Include second line region if doublet
    if method == 'doublet':
        line_region += np.logical_and(
            lpeak_init * (rest_wavelength_second / rest_wavelength) - 2.5 < wavelength,
            wavelength < lpeak_init * (rest_wavelength_second / rest_wavelength) + 2.5
        )
    if np.sum(fit_mask & line_region) < 3 and not method == 'doublet':
        print(f"Immediate region around {linename} is too heavily masked. Abandoning fit.")
        return {}

    # Confine data to fitting region
    wl_fit   = wavelength[fit_mask]
    spec_fit = spectrum[fit_mask]
    err_fit  = errors[fit_mask]

    # Condition initial guesses using the actual data
    initial_guesses = condition_initial_guesses(initial_guesses, wl_fit, spec_fit, err_fit, linename)
    # Generate bounds
    bounds = gen_bounds(initial_guesses, linename, input_bounds=bounds)

    # Prepare and validate inputs
    initial_guesses, bounds = prep_inputs(initial_guesses, linename,
                                          z_lya=lya_z,
                                          bounds=bounds)
    
    # Get initial guesses
    flux_init       = initial_guesses.get('FLUX', 10)
    fluxsecond_init = initial_guesses.get('FLUX2', flux_init) # this will be used if fitting a doublet
    fwhm_init       = initial_guesses.get('FWHM', 3.0)
    cont_init       = initial_guesses.get('CONT', 0)
    slope_init      = initial_guesses.get('SLOPE', 0)

    # Get bounds
    lpeak_bounds = bounds['LPEAK']
    flux_bounds = bounds['FLUX']
    fluxsecond_bounds = bounds.get('FLUX2', (np.nan, np.nan) if linename not in const.flines else flux_bounds) # this will be used if fitting a doublet
    fwhm_bounds = bounds['FWHM']
    cont_bounds = bounds['CONT']
    slope_bounds = bounds['SLOPE']


    # Initialise fit result dictionary that the function will return
    fit_result = {}
    model = None

    # Prepare for fitting
    if method == 'doublet':
        # Define primary and secondary lines of the doublet
        primary   = linename
        secondary = doubletdict[linename][1]

        # Get their wavelength ratio
        primary_rest   = wavedict[primary]
        secondary_rest = wavedict[secondary]
        rest_ratio     = secondary_rest / primary_rest

        model = mdl.gaussian_doublet(rest_ratio)

        # Generate list of initial guesses and bounds
        initg = [flux_init, lpeak_init, fwhm_init, fluxsecond_init, cont_init, slope_init]
        bounds = (
            [flux_bounds[0], lpeak_bounds[0], fwhm_bounds[0],
                fluxsecond_bounds[0], cont_bounds[0], slope_bounds[0]],
                [flux_bounds[1], lpeak_bounds[1], fwhm_bounds[1],
                fluxsecond_bounds[1], cont_bounds[1], slope_bounds[1]]
        )

        # Make sure that initial guesses are within bounds, raise warning if not and default to midpoint
        initg, bounds = check_inputs(initg, bounds)

        # Try fitting multiple times in case of failure
        max_retries = 3
        success = False
        poptg, pcovg = [np.nan for _ in initg], [np.nan for _ in initg] # Default to NaNs

        for attempt in range(max_retries):
            try:
                poptg, pcovg = curve_fit(
                    model, wl_fit, spec_fit,
                    sigma=err_fit, p0=initg, absolute_sigma=True,
                    max_nfev=100000, method='trf', bounds=bounds
                )
                success = True
                break
            except (RuntimeError, ValueError) as e:
                print(f"curve_fit attempt {attempt+1} failed: {e}")
                # Optionally: perturb p0 for next attempt
                # initg = np.array(initg) + np.random.normal(0, 0.1, size=len(initg))
        
        if not success: # If it keeps failing, print warning and return empty dict
            print("All curve_fit attempts failed for doublet fit.")
            return {}
        
        # If bootstrap_params provided, use fit_mc for error estimation
        if bootstrap_params is not None:
            print("Using Monte Carlo error estimation for doublet fit...")
            mc_result = fit_mc(
                model, wl_fit, spec_fit, err_fit, poptg, bounds=bounds,
                niter=bootstrap_params.get('niter', 500),
                errfunc=bootstrap_params.get('errfunc', 'std'),
                chisq_thresh=bootstrap_params.get('chisq_thresh', np.inf),
                sig_clip=bootstrap_params.get('sig_clip', 7.0),
                autocorrelation=bootstrap_params.get('autocorrelation', False),
                max_lag=bootstrap_params.get('max_lag', 5),
                baseline_order=bootstrap_params.get('baseline_order', 0),
                max_nfev=bootstrap_params.get('max_nfev', 5000)
            )
            if mc_result is None:
                print("Monte Carlo error estimation failed, using curve_fit errors")
                error_values = np.sqrt(np.diag(pcovg))
            else:
                poptg = mc_result[0]  # Best-fit parameters from MC
                error_values = mc_result[1]  # Errors from MC
        else:
            error_values = np.sqrt(np.diag(pcovg))
        
        # Populate the parameter dictionary
        param_list = ['FLUX', 'LPEAK', 'FWHM', 'FLUX2', 'CONT', 'SLOPE']
        param_dict = {param: val for param, val
                      in zip(param_list, poptg)}
        error_dict = {param: error_values[i] 
                      for i, param in enumerate(param_list)}
     
        # Calculate the reduced chi-squared value of the fit
        reduced_chisq = np.nansum(np.square((mdl.gaussian_doublet(rest_ratio)(wl_fit, *poptg) -
                                    spec_fit) / err_fit)) / (np.nansum(fit_mask) - len(initg))
        

    elif method == 'single':
        # Generate list of initial guesses and bounds
        initg = [flux_init, lpeak_init, fwhm_init, cont_init, slope_init]
        bounds = (
            [flux_bounds[0], lpeak_bounds[0], fwhm_bounds[0], cont_bounds[0], slope_bounds[0]],
            [flux_bounds[1], lpeak_bounds[1], fwhm_bounds[1], cont_bounds[1], slope_bounds[1]]
        )

        # Make sure that initial guesses are within bounds, raise warning if not and default to midpoint
        initg, bounds = check_inputs(initg, bounds)

        model = mdl.gaussian

        # Try fitting multiple times in case of failure
        max_retries = 3
        success = False
        poptg, pcovg = [np.nan for _ in initg], [np.nan for _ in initg] # Default to NaNs

        for attempt in range(max_retries):
            try:
                poptg, pcovg = curve_fit(
                    model, wl_fit, spec_fit,
                    sigma=err_fit, p0=initg, absolute_sigma=True,
                    max_nfev=100000, method='trf', bounds=bounds
                )
                success = True
                break
            except (RuntimeError, ValueError) as e:
                print(f"curve_fit attempt {attempt+1} failed: {e}")
                # Optionally: perturb p0 for next attempt
                # initg = np.array(initg) + np.random.normal(0, 0.1, size=len(initg))
        
        if not success: # If it keeps failing, print warning and return empty dict
            print("All curve_fit attempts failed for single line fit.")
            return {}
        
        # If bootstrap_params provided, use fit_mc for error estimation
        if bootstrap_params is not None:
            print("Using Monte Carlo error estimation for single line fit...")
            mc_result = fit_mc(
                model, wl_fit, spec_fit, err_fit, poptg, bounds=bounds,
                niter=bootstrap_params.get('niter', 500),
                errfunc=bootstrap_params.get('errfunc', 'std'),
                chisq_thresh=bootstrap_params.get('chisq_thresh', np.inf),
                sig_clip=bootstrap_params.get('sig_clip', 7.0),
                autocorrelation=bootstrap_params.get('autocorrelation', False),
                max_lag=bootstrap_params.get('max_lag', 5),
                baseline_order=bootstrap_params.get('baseline_order', 0),
                max_nfev=bootstrap_params.get('max_nfev', 5000)
            )
            if mc_result is None:
                print("Monte Carlo error estimation failed, using curve_fit errors")
                error_values = np.sqrt(np.diag(pcovg))
            else:
                poptg = mc_result[0]  # Best-fit parameters from MC
                error_values = mc_result[1]  # Errors from MC
        else:
            error_values = np.sqrt(np.diag(pcovg))
        
        # Populate the parameter dictionary
        param_list = ['FLUX', 'LPEAK', 'FWHM', 'CONT', 'SLOPE']
        param_dict = {param: val for param, val
                      in zip(param_list, poptg)}
        error_dict = {param: error_values[i] 
                      for i, param in enumerate(param_list)}
     
        # Calculate the reduced chi-squared value of the fit
        reduced_chisq = np.nansum(np.square((mdl.gaussian(wl_fit, *poptg) -
                                    spec_fit) / err_fit)) / (np.nansum(fit_mask) - len(initg))

    else:
        raise ValueError(f"Unknown fitting method: {method}")
        
    # If the reduced chi-squared is very high, print warning
    if reduced_chisq > 3.0:
        print(f'WARNING: HIGH REDUCED CHI SQUARED STATISTIC ({reduced_chisq})! REVIEW RESULT.')
    
    # Check for multiple comparable peaks in residuals
    residuals = spec_fit - model(wl_fit, *poptg)
    fitted_amplitude = poptg[0]  # First parameter is always flux/amplitude
    fitted_center = poptg[1]     # Second parameter is always center
    fitted_fwhm = poptg[2]       # Third parameter is always FWHM
    n_params = len(poptg)
    
    peak_check = check_multiple_peaks(
        wl_fit, residuals, err_fit,
        fitted_center, fitted_amplitude, fitted_fwhm,
        n_fit_params=n_params,
        chi2_threshold=3.0
    )
    
    if peak_check['suspicious']:
        print(f"WARNING: {peak_check['message']}")
        for i, peak in enumerate(peak_check['peak_info']):
            print(f"  Suspicious peak {i+1}: λ={peak['wavelength']:.2f}, "
                  f"SNR={peak['snr']:.1f}, amp_ratio={peak['amplitude_ratio']:.2f}")

    # Populate the fit_result dictionary
    fit_result['method'] = method
    fit_result['popt'] = poptg
    fit_result['pcov'] = pcovg
    fit_result['reduced_chisq'] = reduced_chisq
    fit_result['fit_mask'] = fit_mask
    fit_result['wl_fit'] = wl_fit
    fit_result['spec_fit'] = spec_fit
    fit_result['err_fit'] = err_fit
    fit_result['param_dict'] = param_dict
    fit_result['error_dict'] = error_dict
    fit_result['model'] = model
    fit_result['peak_check'] = peak_check  # Add multiple peaks check result
    fit_result['multipeak_flag'] = peak_check['flag']  # Quick access to flag

    # If requested, plot the fitting result
    if plot_result:
        # Use the plotting module's plot_line_fit function
        plot.plot_line_fit(
            wl_fit, spec_fit, err_fit, 
            poptg, model, linename,
            save_plots=save_plots,
            plot_dir=plot_dir if plot_dir is not None else './',
            cluster=cluster,
            full_iden=full_iden,
            spec_type=spec_type,
            ax_in=ax_in,
            method=method,
            initial_guesses=initg
        )

    return fit_result


def refit_other_line(wave, spec, spec_err, row, line_tab_row = None, width=25, 
                     ax_in=None, line_name = None, 
                     bootstrap_params: Optional[BootstrapParams] = None):
    """
    Refit a non-Lyman alpha emission line based on prior fitting results.

    Parses previous fit results from the row and delegates fitting to fit_line.
    If no significant fit was previously found, uses initial guesses from the 
    R21 catalogue. If a secondary line of a doublet is passed, the primary is 
    inferred and passed to fit_line instead.

    Parameters
    ----------
    wave : numpy.ndarray
        Wavelength array.
    spec : numpy.ndarray
        Flux density array.
    spec_err : numpy.ndarray
        Flux density error array.
    row : dict
        Row of the megatab containing the fitting results to use as priors.
    line_tab_row : dict, optional
        Row of the line table containing information about the line to fit from 
        the original catalogues. Only needs to be supplied if no significant 
        fit was found previously.
    width : float, optional
        Width (in Angstroms) around the line to use for fitting. Default is 25.
    ax_in : matplotlib.axes.Axes, optional
        Axis to plot the fit result on. If None, a new figure is created.
    line_name : str, optional
        Name of the line to fit. Required if line_tab_row is None.
    bootstrap_params : dict, optional
        Dictionary of parameters for Monte Carlo error estimation (niter, 
        errfunc, chisq_thresh, sig_clip, autocorrelation, max_lag, 
        baseline_order, max_nfev). If None, uses standard curve_fit errors.

    Returns
    -------
    fit_result : dict
        Dictionary containing fit parameters, errors, model, reduced chi-squared, and flag.
        (see fit_line for details)

    Notes
    -----
    - Uses single or double Gaussian models depending on the line type.
    - Initial guesses are taken from previous fit results or catalogue values.
    - Monte Carlo error estimation is used if bootstrap_params is provided.
    - Delegates actual fitting to fit_line function.

    Examples
    --------
    >>> param_dict, error_dict, model, reduced_chisq, flag = refit_other_line(wave, spec, spec_err, row)
    """
    # Validate inputs
    if line_tab_row is None and line_name is None:
        raise ValueError("Either line_tab_row or line_name must be provided.")
    if line_tab_row is not None:
        line_name = line_tab_row['LINE']

    # Determine the primary line (handle doublets)
    primary_line = line_name
    secondary_line = None
    doublet = np.any(const.flines == line_name) or np.any(const.slines == line_name)
    
    if doublet:
        if np.any(const.flines == line_name):
            # This is a primary doublet line
            secondary_line = const.doublets[line_name][1]
        elif np.any(const.slines == line_name):
            # This is a secondary doublet line - use the primary for fitting
            idx = np.where(const.slines == line_name)[0][0]
            primary_line = const.flines[idx]
            secondary_line = line_name

    # Was a significant fit found previously?
    significant_fit = np.abs(row['SNR_'+line_name]) > 3.0

    # Get observed wavelength from catalogue or previous fit
    if line_tab_row is None:
        r21_observed_wavelength = row[f'LPEAK_{primary_line}']
        r21_observed_flux = row[f'FLUX_{primary_line}']
    else:
        # Adjust wavelength if dealing with secondary line of a doublet
        wave_ratio = const.wavedict[primary_line] / const.wavedict[line_name]
        r21_observed_wavelength = line_tab_row['LBDA_OBS'] * wave_ratio
        r21_observed_flux = line_tab_row['FLUX']

    print(f"Refitting {primary_line} at {r21_observed_wavelength:.2f} Angstroms for {row['CLUSTER']} {row['iden']}...")
    
    # Build initial guesses dictionary
    initial_guesses = {
        'FLUX': row[f'FLUX_{primary_line}'] if significant_fit else r21_observed_flux,
        'LPEAK': row[f'LPEAK_{primary_line}'] if significant_fit else r21_observed_wavelength,
        'FWHM': row[f'FWHM_{primary_line}'] if significant_fit else 3.0,
        'CONT': row[f'CONT_{primary_line}'] if significant_fit else np.nanmedian(spec),
        'SLOPE': row[f'SLOPE_{primary_line}'] if significant_fit else 0.0,
    }
    
    # Add secondary flux for doublets
    if doublet and secondary_line is not None:
        secondary_fit = significant_fit and not np.isnan(row[f'FLUX_{secondary_line}'])
        initial_guesses['FLUX2'] = (row[f'FLUX_{secondary_line}'] if secondary_fit 
                                    else initial_guesses['FLUX'])
    
    # Build bounds dictionary
    cen_init = initial_guesses['LPEAK']
    bounds = {
        'FLUX': (-10000, 10000),
        'LPEAK': (cen_init - 6.25, cen_init + 6.25),
        'FWHM': (2.0, (300. / c) * cen_init),
        'CONT': (-50, 2000),
        'SLOPE': (-1000, 1000),
    }
    
    if doublet and secondary_line is not None:
        bounds['FLUX2'] = (-10000, 10000)
    
    # Call fit_line to do the actual fitting
    fit_result = fit_line(
        wave, spec, spec_err, 
        primary_line, 
        initial_guesses, 
        bounds=bounds,
        continuum_buffer=width, 
        plot_result=True, 
        ax_in=ax_in,
        bootstrap_params=bootstrap_params
    )
    
    # Handle fit failure
    if 'param_dict' not in fit_result:
        print(f"Fit failed for {primary_line} in {row['CLUSTER']} {row['iden']}.")
        nandict = {param: np.nan for param in initial_guesses.keys()}
        return nandict, nandict, None, np.nan, ''
    
    # Extract and return results
    return fit_result


def flatten_spectrum(spectrum, return_continuum=False):
    """
    Remove a linear continuum trend from a spectrum by fitting and subtracting a straight line.

    Fits a straight line to the spectrum and subtracts the slope, leaving only the continuum 
    level. Useful for absorption line analysis where large-scale continuum variations should 
    be removed.

    Parameters
    ----------
    spectrum : numpy.ndarray
        1D array of flux values to flatten.
    return_continuum : bool, optional
        If True, return both the flattened spectrum and the fitted continuum model. If False, 
        return only the flattened spectrum. Default is False.

    Returns
    -------
    flattened : numpy.ndarray
        Spectrum with linear trend removed, retaining only the continuum level.
    continuum : numpy.ndarray, optional
        The fitted linear continuum model (only if return_continuum=True).

    Notes
    -----
    - Uses scipy.optimize.curve_fit with a simple linear model: $f(x) = a x + b$
    - The returned spectrum has the slope removed but keeps the continuum level ($b$)
    - Useful for normalizing absorption features before stacking

    Examples
    --------
    >>> from astro_utils import fitting as aufit
    >>> import numpy as np
    >>> x = np.arange(100)
    >>> spec = 10.0 + 0.05 * x + np.random.normal(0, 0.1, 100)
    >>> flat_spec = aufit.flatten_spectrum(spec)
    >>> flat_spec, continuum = aufit.flatten_spectrum(spec, return_continuum=True)
    """
    
    def linear(x, a, b):
        """Simple linear function for continuum fitting."""
        return a * x + b
    
    # Create pixel array
    pixels = np.arange(len(spectrum))
    
    # Fit linear function to the spectrum
    popt, pcov = curve_fit(linear, pixels, spectrum, p0=[0, 0])
    
    # Calculate the continuum model
    continuum_model = linear(pixels, *popt)
    
    # Remove the slope but keep the continuum level
    # This is: spectrum - (a*x + b) + b = spectrum - a*x
    flattened = spectrum - continuum_model + popt[1]
    
    if return_continuum:
        return flattened, continuum_model
    else:
        return flattened
    
from astropy.modeling import models, fitting
from astropy.utils.exceptions import AstropyUserWarning

def fit_sersic(imgin, maskin, center, amp, reff, theta, ellip, n, bounds = {},
                fixed = {}, psf=None):
    """
    Fit a 2D Sersic profile to an image using astropy's modeling framework.

    Parameters
    ----------
    imgin : 2D array
        Input image to fit.
    maskin : 2D boolean array
        Mask indicating which pixels to include in the fit (True = include).
    center : tuple
        (x0, y0) coordinates of the Sersic profile center.
    amp : float
        Amplitude of the Sersic profile.
    reff : float
        Effective radius of the Sersic profile.
    theta : float
        Position angle of the Sersic profile in radians.
    ellip : float
        Ellipticity of the Sersic profile (0 = circular, 1 = line).
    n : float
        Sersic index (controls the concentration of the profile).
    bounds : dict, optional
        Dictionary of parameter bounds (keys: 'amplitude', 'x_0', 'y_0', 'r_eff', 'theta', 'ellip', 'n').
    fixed : dict, optional
        Dictionary of parameters to fix during fitting (keys same as bounds, values are booleans).
    psf : 2D array, optional
        Point spread function to convolve the model with before fitting. If None, no convolution is applied.

    Returns
    -------
    fit_mod : astropy.modeling.Model
        The fitted Sersic model.
    """


    sermod = models.Sersic2D(amplitude=amp, x_0=center[0], y_0=center[1],
                                    r_eff=reff, theta=theta, ellip=ellip, n=n,
                                    bounds=bounds, fixed=fixed)

    lmfitter = fitting.LevMarLSQFitter()
    y, x = np.mgrid[:np.shape(imgin)[0], :np.shape(imgin)[1]]
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.filterwarnings('ignore', message='Model is linear in parameters',
                                category=AstropyUserWarning)
        fit_mod = lmfitter(sermod, x[maskin], y[maskin], imgin[maskin], maxiter=10000)
    return fit_mod