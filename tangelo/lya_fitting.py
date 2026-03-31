"""
Lyman-alpha specific fitting functions.

This module contains high-level functions for fitting Lyman-alpha emission lines,
including functions that handle baseline selection, model comparison, and refitting.
"""

import numpy as np

from . import constants as const
from . import models as mdl
from . import spectroscopy as spectro
from . import plotting as plot
from .spectroscopy import generate_spec_mask
from .fitting import check_inputs, gen_bounds
from .lya_profile import LyaProfile
from . import io
from scipy.optimize import curve_fit
from astropy.table import Table, Column

w_lya = const.wavedict['LYALPHA']  # Lyman-alpha rest wavelength in Angstroms
c = const.c # Speed of light constant

def get_reduced_chisq(y, ymodel, yerr, nparams):
    """
    Calculate the reduced chi-squared statistic.
    
    Parameters
    ----------
    y : array-like
        Observed data.
    ymodel : array-like
        Model evaluated at the same points as y.
    yerr : array-like
        Errors associated with the observed data.
    nparams : int
        Number of parameters in the model.
    
    Returns
    -------
    float
        The reduced chi-squared statistic.
    """
    residuals = y - ymodel
    chisq = np.nansum((residuals / yerr) ** 2)
    dof = len(y) - nparams
    if dof <= 0:
        return np.inf
    return chisq / dof

def calculate_lya_rchsq(megatab, spec_source='R21', spec_type='weight_skysub', modify_inplace=True):
    """
    Calculate the reduced chi-squared for Lyman-alpha fits in the megatab.
    
    Parameters
    ----------
    megatab : Astropy Table
        Table containing the fitting results and spectra.
    spec_source : str, optional
        Source of the spectra to use for chi-squared calculation (default is 'R21').
    spec_type : str, optional
        Type of spectrum to use (default is 'weight_skysub').
    
    Returns
    -------
    astropy.table.Table
        The megatable with reduced chi-squared values added.
    """
    # Split table based on number of lyman alpha peaks and baseline types
    single_peaks = np.isnan(megatab['FLUXB'])
    double_peaks = ~single_peaks
    dla_baselines = ~np.isnan(megatab['TAU'])
    lin_baselines = ~np.isnan(megatab['SLOPE'])
    const_baselines = ~(dla_baselines | lin_baselines)

    sp_const = single_peaks & const_baselines
    sp_lin = single_peaks & lin_baselines
    sp_dla = single_peaks & dla_baselines
    dp_const = double_peaks & const_baselines
    dp_lin = double_peaks & lin_baselines
    dp_dla = double_peaks & dla_baselines

    # Dictionary to map model types to the appropriate model function
    model_mapping = {
        'sp_const': (mdl.lya_speak, sp_const),
        'sp_lin': (mdl.lya_speak_lin, sp_lin),
        'sp_dla': (mdl.lya_speak_damp, sp_dla),
        'dp_const': (mdl.lya_dpeak, dp_const),
        'dp_lin': (mdl.lya_dpeak_lin, dp_lin),
        'dp_dla': (mdl.lya_dpeak_damp, dp_dla)
    }

    # Initialise a column for reduced chi-squared values
    rchsq_col = np.full(len(megatab), np.nan)

    # Loop over each model type and calculate reduced chi-squared
    for model_type, (mdl_func, mask) in model_mapping.items():
        if not np.any(mask):
            print(f"No entries for model type {model_type}; skipping.")
            continue  # Skip if no entries for this model type

        for i in np.where(mask)[0]:
            row = megatab[i]
            spectab = io.load_spec(row['CLUSTER'], row['iden'], row['idfrom'], 
                                   spec_source=spec_source, spec_type=spec_type)
            if spectab is None:
                raise ValueError(f"Could not load spectrum for {row['CLUSTER']} {row['iden']} {row['idfrom']}. Cannot calculate reduced chi-squared.")

            # Extract wavelength, spectrum, and error arrays
            wave = spectab['wave']
            spec = spectab['spec']
            spec_err = spectab['spec_err']

            # Generate fit mask
            lya_peak = row['LPEAKR']
            fitmask = generate_spec_mask(wave, spec, spec_err, lya_peak, 30, 'LYALPHA')

            # Extract best-fit parameters
            if 'dp' in model_type:
                pnames = ['AMPB', 'LPEAKB', 'DISPB', 'ASYMB',
                          'AMPR', 'LPEAKR', 'DISPR', 'ASYMR',
                          'CONT']
                if 'lin' in model_type:
                    pnames.append('SLOPE')
                elif 'dla' in model_type:
                    pnames.extend(['TAU', 'FWHM_ABS', 'LPEAK_ABS'])
            else:
                pnames = ['AMPR', 'LPEAKR', 'DISPR', 'ASYMR',
                          'CONT']
                if 'lin' in model_type:
                    pnames.append('SLOPE')
                elif 'dla' in model_type:
                    pnames.extend(['TAU', 'FWHM_ABS', 'LPEAK_ABS'])

            popt = [row[p] for p in pnames]

            # Raise warning if any of the parameters are NaN, as this will affect the chi-squared calculation
            if np.any(np.isnan(popt)):
                print(f"Warning: NaN values found in parameters for {row['CLUSTER']} {row['iden']} {row['idfrom']} with model {model_type}. This will affect the reduced chi-squared calculation.")
                continue  # Skip this entry for chi-squared calculation

            # Calculate model values
            ymodel = mdl_func(wave[fitmask], *popt)

            # Calculate reduced chi-squared
            rchsq = get_reduced_chisq(spec[fitmask], ymodel,
                                      spec_err[fitmask], len(popt))
 
            # Store the result
            rchsq_col[i] = rchsq

    # Add the reduced chi-squared column to the table
    if modify_inplace:
        position = megatab.colnames.index('FLUXB_UB') + 1 if 'FLUXB_UB' in megatab.colnames else len(megatab.colnames)
        megatab.add_column(Column(rchsq_col, name='RCHSQ'), index=position)
    else:
        return rchsq_col


# Default Monte Carlo parameters for bootstrapping
default_bootstrap_params = {
    'niter': 1000,
    'autocorrelation': False,
    'max_nfev': 2000,
    'errfunc': '84-16'  # Use 68% confidence interval for errors
}

def fit_lya_line(wave, spec, spec_err, initial_guesses, iden, cluster, 
                 baseline='auto', width=50, bounds={}, plot_result=False, 
                 bootstrap_params=default_bootstrap_params, use_bootstrap=True, 
                 rchsq_thresh=2.0, save_plots=False, plot_dir='./', 
                 spec_type='APER', convolve_model=False):
    """
    Master function to fit a Lyman alpha profile to the provided spectrum. Can handle different baseline types,
    and automatically selects between single and double-peaked profiles unless specified otherwise, can use Monte
    Carlo resampling (bootstrapping) to estimate parameter uncertainties, and can produce diagnostic plots.

    Parameters
    ----------
    wave : array-like
        Wavelength array.
    spec : array-like
        Spectrum array.
    spec_err : array-like
        Spectrum error array.
    initial_guesses : dict or Astropy Table row
        Initial guesses for the fit parameters. If a Table row is provided, it should contain the necessary columns.
    baseline : str, optional
        Baseline type to use ('const', 'lin', 'damp', or 'auto' to try all).
    width : float, optional
        Width around the line center to consider for fitting (default is 50).
    bounds : dict, optional
        Dictionary of bounds for the fitting parameters.
    plot_result : bool, optional
        Whether to plot the fitting result.
    bootstrap_params : dict, optional
        Parameters for bootstrap error estimation.
    use_bootstrap : bool, optional
        Whether to use bootstrap resampling for error estimation.
    rchsq_thresh : float, optional
        Reduced chi-squared threshold for accepting a fit when baseline='auto' (default is 2.0).
    save_plots : bool, optional
        Whether to save the plots to disk.
    plot_dir : str, optional
        Directory to save plots if save_plots is True.
    spec_type : str, optional
        Type of spectrum being fitted (for labeling purposes, default: 'APER').
    convolve_model : bool, optional
        Whether to convolve the model with the instrumental resolution (default: False).
    
    Returns
    -------
    fit_result : dict
        Dictionary containing fit parameters, errors, model, reduced chi-squared, and baseline type.
        Keys include: 'param_dict', 'error_dict', 'model', 'reduced_chisq', 'baseline', 
        'fit_mask', 'wl_fit', 'spec_fit', 'err_fit'.
        Empty dict {} if fit failed.
    """
    if baseline == 'auto':
        # Try constant, then linear, then damped baselines
        fit_result = fit_lya_autobase(wave, spec, spec_err, initial_guesses, iden,
                                        cluster, bounds=bounds, 
                                        width=width, plot_result=plot_result,
                                        use_bootstrap=use_bootstrap, bootstrap_params=bootstrap_params,
                                        rchsq_thresh=rchsq_thresh,
                                        save_plots=save_plots, plot_dir=plot_dir,
                                        spec_type=spec_type, convolve_model=convolve_model)
        return fit_result
    else:
        # Fit with specified baseline type
        fit_result = fit_lya(wave, spec, spec_err, initial_guesses, iden, cluster,
                                    bounds=bounds, 
                                    width=width, baseline=baseline,
                                    plot_result=plot_result,
                                    use_bootstrap=use_bootstrap,
                                    bootstrap_params=bootstrap_params,
                                    save_plots=save_plots, plot_dir=plot_dir,
                                    spec_type=spec_type, convolve_model=convolve_model)
        return fit_result


def fit_lya_autobase(wave, spec, spec_err, initial_guesses, iden, cluster, 
                     width=50, bounds={}, plot_result=True, use_bootstrap=True,
                     bootstrap_params=default_bootstrap_params,
                     rchsq_thresh=2.0, save_plots = False, plot_dir = './',
                     spec_type='APER', convolve_model=False):
    """
    Fit the Lyman alpha line using multiple baseline types and select the best fit
    
    Parameters
    ----------
    wave : array-like
        Wavelength array.
    spec : array-like
        Spectrum array.
    spec_err : array-like
        Spectrum error array.
    initial_guesses : dict or Astropy Table row
        Initial guesses for the fitting parameters.
    iden : str
        Identifier for the object being fitted.
    cluster : str
        Cluster name for the object being fitted.
    width : float, optional
        The width (in Angstroms) around the Lya peak to use for fitting.
    bounds : dict, optional
        Dictionary of bounds for the fitting parameters.
    plot_result : bool, optional
        Whether to plot the fitting result.
    mc_niter : int, optional
        Number of Monte Carlo iterations for error estimation.
    rchsq_thresh : float, optional
        Reduced chi-squared threshold for accepting a fit.
    save_plots : bool, optional
        Whether to save the plots to disk.
    plot_dir : str, optional
        Directory to save plots if save_plots is True.
    spec_type : str, optional
        Type of spectrum being fitted (for labeling purposes, default: 'APER').
    convolve_model : bool, optional
        Whether to convolve the model with the instrumental resolution (default: False).

    Returns
    -------
    fit_result : dict
        Dictionary containing fit parameters, errors, model, reduced chi-squared, and baseline type.
        Keys include: 'param_dict', 'error_dict', 'model', 'reduced_chisq', 'baseline', 
        'fit_mask', 'wl_fit', 'spec_fit', 'err_fit'.
        Empty dict {} if all fits failed.
    """
    # First use a constant baseline with the refit_lya_line function
    # Suppress plotting for intermediate fits
    fit_const = fit_lya(wave, spec, spec_err, initial_guesses, iden, cluster, 
                        bounds=bounds, width=width, baseline='const', 
                        plot_result=False, use_bootstrap=use_bootstrap, 
                        bootstrap_params=bootstrap_params,
                        save_plots=False, plot_dir=plot_dir,
                        spec_type=spec_type, convolve_model=convolve_model)
    
    # Check the reduced chi-squared of the fit -- if it's good enough, return it
    if fit_const and fit_const.get('reduced_chisq') and fit_const['reduced_chisq'] < rchsq_thresh:
        print("Constant baseline fit is good enough; returning result.")
        if plot_result:
            plot.plot_lya_fit_result(fit_const, iden, cluster, save_plots=save_plots, 
                               plot_dir=plot_dir, spec_type=spec_type)
        return fit_const
    
    # If not, try a linear baseline
    print("Trying linear baseline fit...")
    initial_guesses_lin = initial_guesses.copy()
    initial_guesses_lin['SLOPE'] = 0.0  # Add a slope initial guess
    fit_lin = fit_lya(wave, spec, spec_err, initial_guesses_lin, iden, cluster, 
                      bounds=bounds, width=width, baseline='lin',
                      plot_result=False, use_bootstrap=use_bootstrap, 
                      bootstrap_params=bootstrap_params,
                      save_plots=False, plot_dir=plot_dir,
                      spec_type=spec_type, convolve_model=convolve_model)

    if fit_lin and fit_lin.get('reduced_chisq') and fit_lin['reduced_chisq'] < rchsq_thresh:
        print("Linear baseline fit is good enough; returning result.")
        if plot_result:
            plot.plot_lya_fit_result(fit_lin, iden, cluster, save_plots=save_plots, 
                                    plot_dir=plot_dir, spec_type=spec_type)
        return fit_lin
    
    # If still not good enough, try a damped Lyman alpha baseline
    print("Trying damped Lyman alpha baseline fit...")
    initial_guesses_damp = initial_guesses.copy()
    initial_guesses_damp['TAU'] = 20.0  # Add initial guess for tau
    initial_guesses_damp['FWHM_ABS'] = (150 / c) * 1215.67 * (initial_guesses['LPEAKR'] / w_lya)  # Initial guess for fwhm in observed frame
    initial_guesses_damp['LPEAK_ABS'] = initial_guesses['LPEAKR'] - 2.5  # Initial guess for absorption peak
    fit_damp = fit_lya(wave, spec, spec_err, initial_guesses_damp, iden, cluster, 
                       bounds=bounds, width=width, baseline='damp',
                       plot_result=False, use_bootstrap=use_bootstrap, 
                       bootstrap_params=bootstrap_params,
                       save_plots=False, plot_dir=plot_dir,
                       spec_type=spec_type, convolve_model=convolve_model)
    
    # Criteria for accepting the damped Lyman alpha fit -- if not met, reduced chi-squared
    # is set to infinity so that it won't be selected as the best fit below
    if fit_damp['param_dict']['CONT'] <= 0:
        print("Damped Lyman alpha fit has non-positive continuum; rejecting fit.")
        fit_damp['reduced_chisq'] = np.inf

    if fit_damp and fit_damp.get('reduced_chisq') and fit_damp['reduced_chisq'] < rchsq_thresh:
        print("Damped Lyman alpha baseline fit is good enough; returning result.")
        if plot_result:
            plot.plot_lya_fit_result(fit_damp, iden, cluster, save_plots=save_plots, 
                               plot_dir=plot_dir, spec_type=spec_type)
        return fit_damp
    
    # If none of the fits were good enough, return the one with the lowest reduced chi-squared
    fits = [fit_const, fit_lin, fit_damp]
    best_fit = min(fits, key=lambda x: x.get('reduced_chisq', np.inf))
    # Get the fit type for reporting
    best_fit_type = ['const', 'lin', 'damp'][fits.index(best_fit)] if best_fit in fits else 'unknown'
    rchsq_value = best_fit.get('reduced_chisq', np.nan)
    print(f"Returning best fit ({best_fit_type}) with reduced chi-squared = {rchsq_value:.2f}")
    
    # Plot the best fit if requested
    if plot_result and best_fit:
        plot.plot_lya_fit_result(best_fit, iden, cluster, save_plots=save_plots, 
                           plot_dir=plot_dir, spec_type=spec_type)
    
    return best_fit

def fit_lya(wave, spec, spec_err, initial_guesses, iden, cluster, 
            bounds={}, width=50, baseline='const', plot_result=True, use_bootstrap=True, 
            bootstrap_params=default_bootstrap_params,
            save_plots=False, plot_dir='./', spec_type='APER', convolve_model=False):
    """
    Fit the Lyman alpha line with specified baseline type using provided initial parameters.
    
    Parameters
    ----------
    wave : array-like
        Wavelength array.
    spec : array-like
        Spectrum array.
    spec_err : array-like
        Spectrum error array.
    initial_guesses : dict-like
        Dictionary or table row containing initial guesses for the fitting parameters.
    iden : str
        Identifier for the source being fitted.
    cluster : str
        Cluster name for the source being fitted.
    bounds : dict, optional
        Dictionary of bounds for the fitting parameters.
    width : float, optional
        The width (in Angstroms) around the Lya peak to use for fitting.
    baseline : str, optional
        Baseline type ('const', 'lin', 'damp').
    plot_result : bool, optional
        Whether to plot the fitting result.
    use_bootstrap : bool, optional
        Whether to use bootstrap resampling for error estimation.
    bootstrap_params : dict, optional
        Parameters for bootstrap error estimation.
    save_plots : bool, optional
        Whether to save the plots to disk.
    plot_dir : str, optional
        Directory to save plots if save_plots is True.
    spec_type : str, optional
        Type of spectrum being fitted (for labeling purposes, default: 'APER').
    convolve_model : bool, optional
        Whether to convolve the model with the instrumental resolution (default: False).
    

    Returns
    -------
    fit_result : dict
        Dictionary containing fit parameters, errors, model, reduced chi-squared, and baseline type.
        Keys include: 'param_dict', 'error_dict', 'model', 'reduced_chisq', 'baseline', 
        'fit_mask', 'wl_fit', 'spec_fit', 'err_fit', 'popt', 'pcov', 'method'.
        Empty dict {} if fit failed.
    """

    # Convert initial_guesses to a dict if it's an Astropy Table row
    if not isinstance(initial_guesses, dict):
        initial_guesses = {col: initial_guesses[col] for col in initial_guesses.colnames if not np.isnan(initial_guesses[col])}
    else: # remove any NaN entries
        initial_guesses = {k: v for k, v in initial_guesses.items() if not np.isnan(v)}


    # First try a double-peaked fit
    print(f"\nFitting Lyman-α line to {cluster} {iden}...")

    # Get the wavelength of the red Lya peak from the table
    lya_peak = initial_guesses['LPEAKR']
    if np.isnan(lya_peak):
        raise ValueError("\nInitial guess for LPEAKR is NaN; cannot proceed with fitting.")
    
    # Get initial redshift guess
    z_init = (lya_peak / w_lya) - 1
    
    # Initial guesses from the table, or semi-generic if not available
    try:
        amp_r_init = initial_guesses['AMPR'] #These will throw an error if not present, which is the desired behavior
        cen_r_init = initial_guesses['LPEAKR']
        wid_r_init = initial_guesses['DISPR']
        asy_r_init = initial_guesses['ASYMR']
    except KeyError as e:
        raise KeyError(f"\nMissing required initial guess parameter: {e}. Cannot proceed with fitting.")

    amp_b_init = initial_guesses.get('AMPB', 0.1 * initial_guesses['AMPR'])
    default_br_sep = -2 * (1 + z_init)  # Default separation of blue peak from red peak in Angstroms, scaled by redshift
    cen_b_init = initial_guesses.get('LPEAKB', initial_guesses['LPEAKR'] + default_br_sep)
    wid_b_init = initial_guesses.get('DISPB', initial_guesses['DISPR'])
    asy_b_init = -1 * initial_guesses['ASYMR']
    cont_init  = [initial_guesses.get('CONT', 0.0)] # This needs to be a list for appending later

    # Ensure the blue peak parameters are in the initial_guesses dict (required for bounds generation later)
    initial_guesses['AMPB'] = amp_b_init
    initial_guesses['LPEAKB'] = cen_b_init
    initial_guesses['DISPB'] = wid_b_init
    initial_guesses['ASYMB'] = asy_b_init

    # Make a list of parameter names for reference in the order used by the model
    param_names = ['AMPB', 'LPEAKB', 'DISPB', 'ASYMB',
                   'AMPR', 'LPEAKR', 'DISPR', 'ASYMR',
                   'CONT']
    
    # Append baseline parameters if needed
    if baseline == 'lin':
        slope_init = initial_guesses['SLOPE'] # If not present, this will throw an error as desired
        param_names.append('SLOPE')
        cont_init = [*cont_init, slope_init]
    elif baseline == 'damp':
        tau_init = initial_guesses['TAU'] # If not present, this will throw an error as desired
        fwhm_init = initial_guesses['FWHM_ABS'] # If not present, this will throw an error as desired
        lpeak_abs_init = initial_guesses['LPEAK_ABS'] # If not present, this will throw an error as desired
        param_names.extend(['TAU', 'FWHM_ABS', 'LPEAK_ABS'])
        cont_init = [*cont_init, tau_init, fwhm_init, lpeak_abs_init]

    # Define initial parameters for the double-peaked model
    p0 = [amp_b_init, cen_b_init, wid_b_init, asy_b_init,
            amp_r_init, cen_r_init, wid_r_init, asy_r_init,
            *cont_init]  # baseline
    
    # Remove any parameters from initial_guesses that aren't in param_names
    initial_guesses = {k: v for k, v in initial_guesses.items() if k in param_names}

    # Define bounds for the parameters
    dpeak_bounds = gen_bounds(initial_guesses, 'LYALPHA', input_bounds=bounds, force_sign='positive')
    
    dpeak_bounds = [[dpeak_bounds[k][0] for k in param_names],
                    [dpeak_bounds[k][1] for k in param_names]]

    # Figure out which function to use based on baseline type
    mdl_func = mdl.lya_dpeak_lin if baseline == 'lin' else \
               mdl.lya_dpeak_damp if baseline == 'damp' else \
               mdl.lya_dpeak
    
    if convolve_model:
        lsf_fwhm = spectro.muse_lsf_fwhm_poly(initial_guesses['LPEAKR'])
        print(f"Convolving model with Gaussian kernel of FWHM = {lsf_fwhm:.2f} Å"
              f" to match instrumental resolution.")
        # Get the name of the convolved model function based on mdl_func
        mdl_func_name = mdl_func.__name__
        conv_func_name = f"convolved_{mdl_func_name}"
        if hasattr(mdl, conv_func_name):
            mdl_func = getattr(mdl, conv_func_name)(lsf_fwhm)
        else:
            raise AttributeError(f"Model function {conv_func_name} not found in models module. Cannot convolve model.")

    # Get mask for sky lines and bad values
    fitmask = generate_spec_mask(wave, spec, spec_err,
                                 lya_peak, width, 'LYALPHA')
    
    # Make sure there are enough good points to fit
    if fitmask.sum() < 10:
        print(f"Not enough good points to fit Lya for {cluster} {iden}.")
        return {}

    best_popt, popt_double, popt_single = None, None, None
    best_perr, perr_double, perr_single = None, None, None
    best_pcov, pcov_double, pcov_single = None, None, None
    best_rchsq, rchsq_double, rchsq_single = np.inf, np.inf, np.inf
    best_param_names = None
    best_method = None

    # First fit the double-peaked model, trying multiple initial guesses for the blue peak
    shifts = np.array([0, -0.5, 0.5, -0.9, 0.9]) * (1 + z_init)  # Angstrom shifts for blue peak initial guess
    for shift in shifts: # Perform five initial fits, moving the blue peak initial guess each time
        p0[1] = cen_b_init + shift
        
        # Make sure the initial guesses are always within bounds
        p0, dpeak_bounds = check_inputs(p0, dpeak_bounds)
        try:
            popt, pcov = curve_fit(mdl_func, wave[fitmask], spec[fitmask],
                                   p0=p0, sigma=spec_err[fitmask],
                                   bounds=dpeak_bounds, absolute_sigma=True,
                                   max_nfev = 100000, method = 'trf')
            perr = np.sqrt(np.diag(pcov))
            if spectro.is_reasonable_dpeak(popt, perr):
                popt_double  = popt
                perr_double  = perr
                pcov_double  = pcov
                rchsq_double = get_reduced_chisq(spec[fitmask], 
                                               mdl_func(wave[fitmask], *popt), 
                                               spec_err[fitmask], 
                                               len(popt))
                break  # Exit the loop if a good fit is found
        except (RuntimeError, ValueError) as e:
            print(f"Fit attempt with blue peak shift {shift} failed: {e}")
            continue

    if popt_double is None or perr_double is None:
        print(f"No reasonable double-peaked fit found for {cluster} {iden}.")
        print("Moving to single-peaked fit...")

    # Now perform a single-peaked fit (with multiple attempts in case of failure)
    p0_single = [amp_r_init, cen_r_init, wid_r_init, asy_r_init, *cont_init]  # baseline
    # Make initial_guesses dict for single peak
    single_initial_guesses = {
        'AMPR': amp_r_init,
        'LPEAKR': cen_r_init,
        'DISPR': wid_r_init,
        'ASYMR': asy_r_init,
        'CONT': cont_init[0]
    }
    if baseline == 'lin':
        single_initial_guesses['SLOPE'] = slope_init
    elif baseline == 'damp':
        single_initial_guesses['TAU'] = tau_init
        single_initial_guesses['FWHM_ABS'] = fwhm_init
        single_initial_guesses['LPEAK_ABS'] = lpeak_abs_init
    
    # Generate bounds for single peak
    speak_bounds = gen_bounds(single_initial_guesses, 'LYALPHA', input_bounds=bounds, force_sign='positive')
    speak_bounds = [[speak_bounds[k][0] for k in param_names if k[-1] != 'B'],
                     [speak_bounds[k][1] for k in param_names if k[-1] != 'B']]
    
    # Define which model to use
    mdl_func_single = mdl.lya_speak_lin if baseline == 'lin' else \
                mdl.lya_speak_damp if baseline == 'damp' else \
                mdl.lya_speak
    
    # If convolving the double-peaked model, also convolve the single-peaked model
    if convolve_model:
        conv_func_name_single = f"convolved_{mdl_func_single.__name__}"
        if hasattr(mdl, conv_func_name_single):
            mdl_func_single = getattr(mdl, conv_func_name_single)(lsf_fwhm)
        else:
            raise AttributeError(f"Model function {conv_func_name_single} not found in models module. Cannot convolve model.")
    
    # Ensure initial guesses are within bounds
    p0_single, speak_bounds = check_inputs(p0_single, speak_bounds)

    for _ in range(3): # Try up to three times
        try:
            popt, pcov = curve_fit(mdl_func_single, wave[fitmask], spec[fitmask],
                                    p0=p0_single, sigma=spec_err[fitmask],
                                    bounds=speak_bounds, absolute_sigma=True,
                                    max_nfev = 100000, method = 'trf')
            popt_single  = popt
            perr_single = np.sqrt(np.diag(pcov))
            pcov_single = pcov
            rchsq_single = get_reduced_chisq(spec[fitmask], 
                                            mdl_func_single(wave[fitmask], *popt), 
                                            spec_err[fitmask], 
                                            len(popt))

            print(f"Single-peaked fit successful for {cluster} {iden}.")

            break  # Exit the loop if fit was successful

        except (RuntimeError, ValueError) as e:
            print(f"Single-peaked fit also failed for {cluster} {iden}: {e}")
        
    # Now compare the single and double peaked fits if both were successful
    if popt_double is not None and perr_double is not None and popt_single is not None and perr_single is not None:
        if rchsq_double < rchsq_single:
            print(f"Double-peaked fit is better (reduced chi-squared = {rchsq_double:.2f}) than single-peaked fit ({rchsq_single:.2f}).")
            best_popt = popt_double
            best_perr = perr_double
            best_pcov = pcov_double
            best_rchsq = rchsq_double
            best_param_names = param_names  # Use full parameter names
            best_bounds = dpeak_bounds
            best_method = 'double-peaked'
        else:
            print(f"Single-peaked fit is better (reduced chi-squared = {rchsq_single:.2f}) than double-peaked fit ({rchsq_double:.2f}).")
            best_popt = popt_single
            best_perr = perr_single
            best_pcov = pcov_single
            best_rchsq = rchsq_single
            # Exclude blue peak parameters from the names
            best_param_names = [n for n in param_names if n[-1] != 'B']
            best_bounds = speak_bounds
            best_method = 'single-peaked'
    elif popt_double is not None and perr_double is not None:
        best_popt = popt_double
        best_perr = perr_double
        best_pcov = pcov_double
        best_rchsq = rchsq_double
        best_param_names = param_names  # Use full parameter names
        best_bounds = dpeak_bounds
        best_method = 'double-peaked'
    elif popt_single is not None and perr_single is not None:
        best_popt = popt_single
        best_perr = perr_single
        best_pcov = pcov_single
        best_rchsq = rchsq_single
        best_param_names = [n for n in param_names if n[-1] != 'B']
        best_bounds = speak_bounds
        best_method = 'single-peaked'
    else:
        print(f"Both single and double-peaked fits failed for {cluster} {iden}.")
        return {}
    
    # If we got here, we have a best fit. Populate the parameter dictionary
    param_dict = dict(zip(best_param_names, best_popt))
    error_dict = dict(zip(best_param_names, best_perr))

    # Now get the final parameters and errors using bootstrapping
    # Using the LyaProfile class which is in this module
    initial_profile = LyaProfile(param_dict, error_dict, )
    # Now use the class method to generate uncertainties
    final_params, final_errors, \
        final_function, final_reduced_chisq \
            = initial_profile.fit_to(wave, spec, spec_err, mask=fitmask,
                                     bounds=best_bounds, use_bootstrap=use_bootstrap,
                                     bootstrap_params=bootstrap_params)
    
    # Catch cases where the fit failed
    if final_params is None or final_errors is None:
        print(f"Final fitting with bootstrapping failed for {cluster} {iden}.")
        return {}

    # # Add dummy values for the blue peak parameters if needed
    # if 'AMPB' not in final_params.keys():
    #     for pname in param_names[:4]:
    #         final_params[pname] = np.nan
    #         final_errors[pname] = np.nan
    
    # Build the fit_result dictionary to return
    fit_result = {
        'param_dict': final_params,
        'error_dict': final_errors,
        'model': final_function,
        'reduced_chisq': final_reduced_chisq,
        'baseline': baseline,
        'fit_mask': fitmask,
        'wl_fit': wave[fitmask],
        'spec_fit': spec[fitmask],
        'err_fit': spec_err[fitmask],
        'popt': best_popt,
        'pcov': best_pcov,
        'method': best_method
    }

    # Plot the fit result if requested
    if plot_result and final_function is not None:
        plot.plot_lya_fit_result(fit_result, iden, cluster, save_plots=save_plots, 
                           plot_dir=plot_dir, spec_type=spec_type)

    return fit_result

def refit_lya_line(wave, spec, spec_err, tabrow, baseline='auto', width=50, plot_result=True,
                   use_bootstrap=True, bootstrap_params=default_bootstrap_params, rchsq_thresh=2.0,
                   save_plots=True, plot_dir=None, spec_type='aper'):
    """
    Re-fit a Lyman alpha line using the parameters from a given table row as initial guesses, and optionally 
    trying multiple baseline types.

    Parameters
    ----------
    wave : array-like
        Wavelength array.
    spec : array-like
        Spectrum array.
    spec_err : array-like
        Spectrum error array.
    tabrow : Astropy Table row
        The row of the megatab containing the fitting results to use as priors.
    baseline : str, optional
        Baseline type to use for fitting ('const', 'lin', 'damp', or 'auto' to try all).
    width : float, optional
        Width around the line center to consider for fitting (default is 50).
    plot_result : bool, optional
        Whether to plot the fitting result (default is True).
    use_bootstrap : bool, optional
        Whether to use bootstrap resampling for error estimation (default is True).
    bootstrap_params : dict, optional
        Parameters for bootstrap error estimation (default is default_bootstrap_params).
    rchsq_thresh : float, optional
        Reduced chi-squared threshold for accepting a fit when baseline='auto' (default is 2.0).
    save_plots : bool, optional
        Whether to save the plots to disk (default is True).
    plot_dir : str, optional
        Directory to save plots if save_plots is True (default is None, which means current directory).
    spec_type : str, optional
        Type of spectrum being fitted (for labeling purposes, default is 'aper').

    Returns
    -------
    fit_result : dict
        Dictionary containing fit parameters, errors, model, reduced chi-squared, and baseline type.
        Keys include: 'param_dict', 'error_dict', 'model', 'reduced_chisq', 'baseline', 
        'fit_mask', 'wl_fit', 'spec_fit', 'err_fit', 'popt', 'pcov', 'method'.
        Empty dict {} if fit failed.
    """
    iden = tabrow['iden']
    cluster = tabrow['CLUSTER']

    # Extract initial guesses from the table row
    initial_guesses = {}
    for param in ['AMPB', 'LPEAKB', 'DISPB', 'ASYMB',
                  'AMPR', 'LPEAKR', 'DISPR', 'ASYMR',
                  'CONT', 'SLOPE', 'TAU', 'FWHM_ABS', 'LPEAK_ABS']:
        if param in tabrow.colnames and not np.isnan(tabrow[param]):
            initial_guesses[param] = tabrow[param]

    # Depending on baseline type, call the fitting function
    if baseline == 'auto':
        fit_result = fit_lya_autobase(wave, spec, spec_err, initial_guesses, iden,
                                        cluster, bounds={}, 
                                        width=width, plot_result=plot_result,
                                        use_bootstrap=use_bootstrap, bootstrap_params=bootstrap_params,
                                        rchsq_thresh=rchsq_thresh,
                                        save_plots=save_plots, plot_dir=plot_dir,
                                        spec_type=spec_type)
    else:
        fit_result = fit_lya(wave, spec, spec_err, initial_guesses, iden, cluster,
                                    bounds={}, 
                                    width=width, baseline=baseline,
                                    plot_result=plot_result,
                                    use_bootstrap=use_bootstrap,
                                    bootstrap_params=bootstrap_params,
                                    save_plots=save_plots, plot_dir=plot_dir,
                                    spec_type=spec_type)
        
    return fit_result