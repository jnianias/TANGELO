# Plotting functions for spectral lines

import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from . import spectroscopy as spectro
from . import models
from . import io
import copy
import os
from pathlib import Path

import matplotlib
from matplotlib.colors import PowerNorm

def safe_show():
    """
    Show matplotlib figure only if not running in a non-GUI environment.
    This function checks the current matplotlib backend and only calls plt.show()
    if the backend supports GUI operations. This prevents annoying warnings when running
    in headless environments (e.g., remote servers without display).

    Returns
    -------
    None
    """    
    if matplotlib.get_backend() not in ['agg', 'template']:
        plt.show()
    else:
        pass  # Do nothing in non-GUI backends


def plot_muse_spectrum(wave, spec, spec_err=None, ax=None, label=None, color='slateblue', alpha=0.8, step='mid',
                       y_label=r'f$_{\lambda}$ $[10^{-20}$ erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}]$', x_label=r'$\lambda$ [\AA]',
                       save_plot=False, save_dir='./', plot_name='spectrum_plot.png'):
    """
    Plot a MUSE spectrum with optional error bars

    Parameters
    ----------
    wave : array-like
        Wavelength array in Angstroms or km/s if velocity=True.
    spec : array-like
        Flux array corresponding to the wavelengths.
    spec_err : array-like, optional
        Error array for the flux values. If provided, shaded error region will be plotted.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axis to plot on. If None, a new figure and axis will be created.
    label : str, optional
        Label for the spectrum (for legend).
    color : str, optional
        Color for the spectrum line (default is 'slateblue').
    alpha : float, optional
        Transparency level for the spectrum line (default is 0.8).
    step : str, optional
        Step style for plotting the spectrum (default is 'mid').
    y_label : str, optional
        Label for the y-axis (default is r'f$_{\lambda}$ $[10^{-20}$ erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}]$').
    x_label : str, optional
        Label for the x-axis (default is r'$\lambda$ [\AA]').
    
    Returns
    -------
    None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5), facecolor='w')
    
    ax.plot(wave, spec, drawstyle=f'steps-{step}', color=color, alpha=alpha, label=label)
    
    if spec_err is not None:
        ax.fill_between(wave, spec - spec_err, spec + spec_err,
                        color=color, alpha=0.3, step=step, edgecolor='none')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    if label:
        ax.legend()

    if save_plot:
        if not Path(save_dir).exists():
            os.makedirs(save_dir)
        plt.savefig(Path(save_dir) / plot_name, dpi=300)
    
    if ax is None:
        safe_show()
        plt.close()

def plotline(iden, clus, idfrom, wln, ax_in, spec_source = '2fwhm', width=100, model=None, title = False,
             hline = None, hspan = None, vline = None, vspan = None,
             # plot_sky = 'red', plot_cluslines = 'magenta', plot_bkg = 'goldenrod', plot_clusspec = 'seagreen', 
             set_ylim = 'mpl', normalise=False, label = None, return_spectrum = False, 
             plotcolor='slateblue', modcolor='maroon'):
    """
    Plot a spectrum from the R21 catalog or 2FWHM catalog with various options.

    Parameters
    ----------
    iden : str
        Identifier for the source in the catalog.
    clus : str
        Cluster name (e.g., 'A2744').
    idfrom : str
        Source of the identifier (e.g., 'R21' or '2fwhm').
    wln : float
        Central wavelength to plot around.
    ax_in : matplotlib.axes.Axes
        Matplotlib axis to plot on.
    spec_source : str, optional
        'R21' for R21 catalog, '2fwhm' for 2FWHM catalog (default is '2fwhm').
    width : float, optional
        Wavelength range around wln to plot (total width = 2*width, default is 100).
    model : tuple, optional
        Tuple of (function, params...) to overplot a model fit.
    title : str or bool, optional
        Title for the plot (if False, no title).
    hline : float, optional
        Horizontal line value to plot (if None, no line).
    hspan : tuple, optional
        Horizontal span to plot (if None, no span).
    vline : list, optional
        Vertical line values to plot (if None, no line).
    vspan : tuple, optional
        Vertical span to plot (if None, no span).
    set_ylim : str, optional
        Y-axis limits (if 'mpl', use Matplotlib defaults, if 'manual', set from data).
    normalise : bool, optional
        Whether to normalise the spectrum (default is False).
    label : str, optional
        Label for the spectrum (for legend).
    return_spectrum : bool, optional
        Whether to return the plotted spectrum data (default is False).
    plotcolor : str, optional
        Color for the main spectrum plot (default is 'slateblue').
    modcolor : str, optional
        Color for the model fit plot (default is 'maroon').

    Returns
    -------
    None or astropy Table
    """

    ax = ax_in

    # Find spectrum file and load it as a table
    load_method = spectro.load_r21_spec if spec_source == 'R21' else spectro.load_aper_spec
    spectab = load_method(clus, iden, idfrom, 'weight_skysub' if spec_source == 'R21' else '2fwhm')

    if spectab is None:
        print(f'No spectrum found for {clus} ID{iden} in {spec_source} catalog.')
        return None

    if normalise: # normalise the spectrum if requested
        normfac = copy.deepcopy(np.abs(np.nanmedian(spectab['spec'].data)))
        spectab['spec'] /= normfac
        spectab['spec_err'] /= normfac

    # region to plot
    region = np.logical_and(wln - width < spectab['wave'], spectab['wave'] < wln + width) 
    if np.sum(region) < 2:
        print(f'No data in the specified wavelength range for {clus} ID{iden}.')
        return None

    # Plot the spectrum
    plot_muse_spectrum(spectab['wave'][region], spectab['spec'][region], spec_err=spectab['spec_err'][region],
                       ax=ax, label=label, color=plotcolor, alpha=0.8, step='mid')
    
    # Axis labels and title
    if isinstance(title, str):
        ax.set_title(f'{clus}  ID{iden}' + ' ' + title)

    # Add lines/spans if requested
    if hline:
        ax.axhline(hline, linestyle = '--', alpha=0.5, color='gray')
    if hspan:
        ax.axhspan(hspan[0], hspan[1], alpha=0.3, color='gray')
    if vline:
        for l in vline:
            ax.axvline(l, linestyle = '--', alpha=0.5, color='green')
    if vspan:
        ax.axvspan(vspan[0], vspan[1], alpha=0.3, color='orange')

    # Overplot model if provided
    if model is not None:
        func = model[0]
        highreswl = np.arange(np.nanmin(spectab['wave'][region]), np.nanmax(spectab['wave'][region]), 0.125)
        modplot = func(highreswl, *model[1:])
        if normalise:
            modplot /= normfac
        ax.plot(highreswl, modplot, linestyle = '--', color=modcolor, alpha=0.3, label='fit')
    
    if set_ylim == 'manual':
        ax.set_ylim(np.min([0., np.nanmin(spectab['spec'][region])]), np.max(spectab['spec'][region]) * 1.1)

    # Legend
    if label or model is not None:
        ax.legend()

    if return_spectrum:
        return spectab[region]


def lya_mod_plot(row, axin, eml=False):
    """
    Plot Lyman alpha models on a given axis in velocity space.
    
    This function generates a high-resolution model of the Lyman alpha line profile
    and plots it in velocity space on the provided axis. It can handle both single-peak
    and double-peak Lyman alpha models with different baseline types (constant, linear,
    or damped) depending on the fit parameters available in the row.
    
    Parameters
    ----------
    row : dict or table row
        A dictionary or table row containing the fit parameters for the Lyman alpha profile.
        Must contain the following keys:
        - 'LPEAKR': Rest-frame wavelength of the red peak (Angstroms)
        - 'AMPR': Amplitude of the red peak
        - 'DISPR': Dispersion of the red peak 
        - 'ASYMR': Asymmetry parameter of the red peak
        - 'CONT': Continuum level
        - 'SNRB': Signal-to-noise ratio of the blue peak
        If SNRB > 3.0, also requires:
        - 'AMPB': Amplitude of the blue peak
        - 'LPEAKB': Rest-frame wavelength of the blue peak (Angstroms)
        - 'DISPB': Dispersion of the blue peak
        - 'ASYMB': Asymmetry parameter of the blue peak
        For linear baseline (if 'SLOPE' is not NaN):
        - 'SLOPE': Linear slope parameter
        For damped baseline (if 'TAU' is not NaN):
        - 'TAU': Damping parameter
        - 'FWHM_ABS': Full width at half maximum for damped profile
        - 'LPEAK_ABS': Peak wavelength of absorption component
        If eml=True, also requires:
        - 'DELTAV_LYA': Velocity offset for emission lines (km/s)
    
    axin : matplotlib.axes.Axes
        The matplotlib axis object on which to plot the model.
    
    eml : bool, optional
        If True, applies a velocity offset (DELTAV_LYA) to the model.
        Default is False.
    
    Returns
    -------
    None
        The function plots directly on the provided axis and does not return anything.
    
    Notes
    -----
    - The function creates a high-resolution wavelength grid spanning ±40 Angstroms
      around the red peak wavelength with 1000 points.
    - Model selection hierarchy:
      1. If 'SLOPE' is not NaN: uses linear baseline models (lya_speak_lin/lya_dpeak_lin)
      2. Elif 'TAU' is not NaN: uses damped baseline models (lya_speak_damp/lya_dpeak_damp)  
      3. Else: uses constant baseline models (lya_speak/lya_dpeak)
    - Uses single-peak model if blue SNR <= 3.0, otherwise uses double-peak model.
    - Converts wavelength to velocity using the Lyman alpha rest wavelength (1215.67 Å).
    - The flux is converted from wavelength space to velocity space using dλ/dv.
    - The model is plotted as a dashed maroon line with 60% opacity.
    """
    # Create high-resolution wavelength grid around the red peak
    hireswl = np.linspace(row['LPEAKR'] - 40, row['LPEAKR'] + 40, 1000)
    
    # Determine baseline type (matching logic in lya_profile.py)
    # Check for SLOPE first (linear baseline), but TAU takes precedence if present
    baseline_type = 'const'  # default
    if not np.isnan(row['SLOPE']):
        baseline_type = 'lin'
    elif not np.isnan(row['TAU']):
        baseline_type = 'damp'  # damped overrides linear if both present
    
    if baseline_type == 'lin':
        # Use linear baseline models
        if row['SNRB'] > 3.0:
            # Double-peak with linear baseline
            hiresmod = models.lya_dpeak_lin(hireswl, row['AMPB'], row['LPEAKB'], row['DISPB'], row['ASYMB'],
                                           row['AMPR'], row['LPEAKR'], row['DISPR'], row['ASYMR'], 
                                           row['CONT'], row['SLOPE'])
        else:
            # Single-peak with linear baseline
            hiresmod = models.lya_speak_lin(hireswl, row['AMPR'], row['LPEAKR'], row['DISPR'], row['ASYMR'], 
                                           row['CONT'], row['SLOPE'])
    elif baseline_type == 'damp':
        # Use damped baseline models
        if row['SNRB'] > 3.0:
            # Double-peak with damped baseline
            hiresmod = models.lya_dpeak_damp(hireswl, row['AMPB'], row['LPEAKB'], row['DISPB'], row['ASYMB'],
                                            row['AMPR'], row['LPEAKR'], row['DISPR'], row['ASYMR'], 
                                            row['CONT'], row['TAU'], row['FWHM_ABS'], row['LPEAK_ABS'])
        else:
            # Single-peak with damped baseline
            hiresmod = models.lya_speak_damp(hireswl, row['AMPR'], row['LPEAKR'], row['DISPR'], row['ASYMR'], 
                                            row['CONT'], row['TAU'], row['FWHM_ABS'], row['LPEAK_ABS'])
    else:
        # Use constant baseline models (original behavior)
        if row['SNRB'] > 3.0:
            # Double-peak with constant baseline
            hiresmod = models.lya_dpeak(hireswl, row['AMPB'], row['LPEAKB'], row['DISPB'], row['ASYMB'],
                                       row['AMPR'], row['LPEAKR'], row['DISPR'], row['ASYMR'], row['CONT'])
        else:
            # Single-peak with constant baseline
            hiresmod = models.lya_speak(hireswl, row['AMPR'], row['LPEAKR'], row['DISPR'], row['ASYMR'], row['CONT'])
    
    # Convert wavelength to velocity
    hiresvel = spectro.wave2vel(hireswl, 1215.67, redshift=row['LPEAKR'] / 1215.67 - 1)
    
    # Apply velocity offset if requested
    if eml and 'DELTAV_LYA' in row:
        hiresvel += row['DELTAV_LYA']
    
    # Convert flux from wavelength space to velocity space (dλ/dv)
    dldv = np.ediff1d(hireswl, to_end=np.ediff1d(hireswl)[-1]) / np.ediff1d(hiresvel, to_end=np.ediff1d(hiresvel)[-1])
    
    # Plot the model
    axin.plot(hiresvel[1:-1], dldv[1:-1] * hiresmod[1:-1], 
              color='maroon', alpha=0.6, label=r"model", linestyle='--')
    

def plot_lya_fit_result(fit_result, iden, cluster, save_plots=False, plot_dir='./', spec_type='aper'):
    """
    Plot the Lyman-alpha fit result, decomposed into emission and baseline components.
    
    Parameters
    ----------
    fit_result : dict
        Dictionary containing fit results from fit_lya, including 'wl_fit', 'spec_fit',
        'err_fit', 'popt', 'model', 'baseline', 'method', and 'param_dict'.
    iden : str
        Identifier for the source being fitted.
    cluster : str
        Cluster name for the source being fitted.
    save_plots : bool, optional
        Whether to save the plots to disk.
    plot_dir : str, optional
        Directory to save plots if save_plots is True.
    spec_type : str, optional
        Type of spectrum being fitted (for labeling purposes, default: 'aper').
    """
    # Extract data from fit_result
    wave = fit_result['wl_fit']
    spec = fit_result['spec_fit']
    spec_err = fit_result['err_fit']
    popt = fit_result['popt']
    model_func = fit_result['model']
    baseline = fit_result['baseline']
    method = fit_result['method']
    
    # Get the Lya peak position for plotting range
    lya_peak = fit_result['param_dict']['LPEAKR']
    
    # Determine the number of continuum parameters based on baseline type
    if baseline == 'lin':
        n_cont_params = 2  # CONT and SLOPE
    elif baseline == 'damp':
        n_cont_params = 4  # CONT, TAU, FWHM_ABS, LPEAK_ABS
    else:  # 'const'
        n_cont_params = 1  # CONT only
    
    # Dictionary of full baseline names for plotting
    basenames = {
        'const': 'Constant',
        'lin': 'Linear',
        'damp': 'Absorption'
    }
    
    # Make a fine wavelength grid for plotting the model
    finegrid = np.linspace(wave.min(), wave.max(), 1000)
    
    # Total model
    final_model = model_func(finegrid, *popt)
    
    # Emission model only (extract the emission parameters)
    emission_popt = popt[:-n_cont_params]
    rpeak_popt = emission_popt[:4] if len(emission_popt) == 4 else emission_popt[4:8]
    bpeak_model = 0
    
    if len(emission_popt) == 8:  # double peaked
        rpeak_popt = emission_popt[4:8]
        bpeak_popt = emission_popt[:4]
        bpeak_model = models.lya_speak(finegrid, *bpeak_popt, 0.0)  # Append zero for continuum
    
    rpeak_model = models.lya_speak(finegrid, *rpeak_popt, 0.0)  # Append zero for continuum
    
    # Baseline only -- just subtract the emission model from the total
    baseline_model = final_model - rpeak_model - bpeak_model
    
    # Now plot
    plt.figure(figsize=(6, 3), facecolor='white')
    
    # Data with error bars
    plt.step(wave, spec, where='mid', color='black', alpha=0.75, label='Data')
    plt.fill_between(wave, spec - spec_err, spec + spec_err,
                     color='gray', step='mid', alpha=0.3, label='Error', edgecolor='none')
    
    # Model components - Total model
    plt.plot(finegrid, final_model, color='fuchsia', label='Best Fit')
    
    # Emission only
    plt.plot(finegrid, rpeak_model, color='red', linestyle='--', label='Red Peak')
    
    if len(emission_popt) == 8:  # double peaked
        plt.plot(finegrid, bpeak_model, color='royalblue', linestyle='--', label='Blue Peak')
    
    # Baseline only
    plt.plot(finegrid, baseline_model, color='tab:green', linestyle=':', 
             label=f'{basenames[baseline]} Baseline')
    
    plt.xlim(lya_peak - 50, lya_peak + 50)
    plt.xlabel('Wavelength [\AA]')
    plt.ylabel('Flux Density [$10^{-20}$\,erg\,s$^{-1}$\,cm$^{-2}$\,\AA$^{-1}$]')
    plt.title(f"{cluster} {iden} " + r"Lyman-$\alpha$ Fit")
    plt.legend()
    
    # if save_plots:
    #     plt.savefig(f"{plot_dir}/LYALPHA_fit_{spec_type}.png", dpi=300)
    
    # safe_show()

    # Save plot if requested
    if save_plots:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plot_path = os.path.join(plot_dir, f'LYALPHA_fit_{spec_type}.png')
        plt.savefig(plot_path, dpi=250)
        print(f'Lyman alpha fit plot saved to {plot_path}')

    safe_show()
    plt.close()

def plot_line_fit(wave, spec, spec_err, popt, func, line_name,
                  save_plots=False, plot_dir=None, ax_in=None,
                  cluster='', full_iden='', method='single', 
                  spec_type='aper', initial_guesses=None):
    """
    Plot the spectral line fit results along with the data.

    Parameters
    ----------
    wave : array-like
        Wavelength array of the spectrum.
    spec : array-like
        Flux array of the spectrum.
    spec_err : array-like
        Error array of the spectrum.
    popt : array-like
        Optimal fit parameters from the fitting procedure.
    func : callable
        The model function used for fitting.
    line_name : str
        Name of the spectral line being fitted.
    save_plots : bool, optional
        Whether to save the plot to disk. Default is False.
    plot_dir : str, optional
        Directory to save the plot if save_plots is True. Default is None, which uses a default directory.
    ax_in : matplotlib.axes.Axes, optional
        Matplotlib axis to plot on. If None, a new figure and axes are created.
    cluster : str, optional
        Cluster name (e.g., 'A2744', 'MACS0416', etc.). Used in title and filename if saving. Default is ''.
    full_iden : str, optional
        Full identifier string of the source (e.g., 'E1234', 'X5678', etc.). Used in title and filename if saving. Default is ''.
    method : str, optional
        Fitting method: 'single' for single line, 'doublet' for doublet line. Default is 'single'.
    spec_type : str, optional
        Type of spectrum being plotted (e.g., 'aper' for aperture). Used in filename if saving. Default is 'aper'.
    initial_guesses : list, optional
        Initial guesses used for fitting, for reference in the plot.  Default is None.

    Returns
    -------
    None
    """

    # Create figure and axes
    if ax_in is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), facecolor='w')
    else:
        ax = ax_in

    # If no cluster or iden or plot directory provided, raise warning when saving
    if save_plots and (cluster == '' or full_iden == '') and plot_dir is None:
        print("Warning: cluster name and full identifier not provided! Plot will be saved to working directory.")

    # Plot data with shaded region for errors
    ax.plot(wave, spec, drawstyle='steps-mid', label='Fitted Data', color='black', alpha=0.7)
    ax.fill_between(wave, spec - spec_err, spec + spec_err, color='grey', alpha=0.5, step='mid', edgecolor='none')

    # Generate a finely sampled wavelength array for plotting the model
    hires_wave = np.linspace(np.min(wave), np.max(wave), 1000)
    
    # Plot model based on method
    if method == 'doublet':
        # For doublets, we need to extract the wavelength ratio from the model function
        # The doublet model is created as gaussian_doublet(rest_ratio)
        # We can infer the ratio from the fitted parameters (popt[1] vs popt[1]*rest_ratio)
        # Or we can call the full model and plot components separately
        
        # Plot total model
        model_spec = func(hires_wave, *popt)
        ax.plot(hires_wave, model_spec, color='red', label='Doublet Fit', lw=2)
        # Plot initial guesses if provided
        if initial_guesses is not None:
            init_model_spec = func(hires_wave, *initial_guesses)
            ax.plot(hires_wave, init_model_spec, color='tab:blue', ls=':', label='Initial Guess', lw=1, alpha=0.5)
        
        # Plot individual components
        # Primary component: uses popt[0] (flux), popt[1] (center), popt[2] (fwhm)
        primary_comp = models.gaussian(hires_wave, popt[0], popt[1], popt[2], 0, 0)
        ax.plot(hires_wave, primary_comp, color='orange', ls='--', label='Primary Component')
        
        # Secondary component: uses popt[3] (flux2), and we need to get the wavelength ratio
        # The secondary wavelength is embedded in the doublet model
        # We can get it from the function's closure
        if hasattr(func, '__closure__') and func.__closure__:
            # Extract rest_ratio from the closure
            rest_ratio = func.__closure__[0].cell_contents
            secondary_comp = models.gaussian(hires_wave, popt[3], popt[1]*rest_ratio, popt[2], 0, 0)
            ax.plot(hires_wave, secondary_comp, color='green', ls='--', label='Secondary Component')
    else:
        # Single line fit
        model_spec = func(hires_wave, *popt)
        ax.plot(hires_wave, model_spec, color='red', label='Single Line Fit', lw=2)
        # Plot initial guesses if provided
        if initial_guesses is not None:
            init_model_spec = func(hires_wave, *initial_guesses)
            ax.plot(hires_wave, init_model_spec, color='tab:blue', ls=':', label='Initial Guess', lw=1, alpha=0.5)

    # Labels and legend
    ax.set_xlabel(r'Wavelength (\AA)')
    ax.set_ylabel('Flux Density')
    ax.set_title(f'{cluster} {full_iden} {line_name} Fit')
    ax.legend()

    plt.tight_layout()

    # Save plot if requested
    if save_plots:
        if plot_dir is None and cluster != '' and full_iden != '':
            plot_dir = io.get_plot_dir(cluster, full_iden)  # Get directory for saving plots for this source
        elif plot_dir is None:
            plot_dir = './'  # Default to current directory
        if not os.path.exists(plot_dir): # Create directory if it doesn't exist
            os.makedirs(plot_dir)
        plot_path = os.path.join(plot_dir, f'{line_name}_fit_{spec_type}.png')
        plt.savefig(plot_path, dpi=250)
        print(f'{line_name} fit plot saved to {plot_path}')

    # Show and close if no axis was provided
    if ax_in is None:
        safe_show()
        plt.close()

def plot_lya_peak_detection(lya_nb_img, ra, dec, ra_opt, dec_opt, cluster, full_iden, peak_locs_world, save_plot=False):
    """
    Plot the Lyman-alpha narrowband image with detected peaks and source positions.

    Parameters
    ----------
    lya_nb_img : MPDAF Image
    The Lyman-alpha narrowband image object with WCS information.
    ra : float
        Original right ascension of the source (degrees).
    dec : float
        Original declination of the source (degrees).
    ra_opt : float
        Optimised right ascension of the source (degrees).
    dec_opt : float
        Optimised declination of the source (degrees).
    cluster : str
        Cluster name (e.g., 'A2744', 'MACS0416', etc.).
    full_iden : str
        Full identifier string of the source (e.g., 'E1234', 'X5678', etc.).
    peak_locs_world : array-like, shape (N, 2)
        World coordinates (RA, Dec) of detected peaks to plot.
    save_plot : bool, optional
        Whether to save the plot to disk. Default is False.

    Returns
    -------
    None
    """
    # Get sensible vmin, vmax for display
    vmin, vmax = np.percentile(lya_nb_img.data.compressed(), [5, 99.9])
    plt.figure(figsize=(6, 6), facecolor='white')
    plt.imshow(lya_nb_img.data, origin='lower', cmap='viridis', norm=PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax * 1.05))
    
    # Plot detected peaks if any remain after filtering
    if len(peak_locs_world) > 0:
        peak_locs_for_plot = lya_nb_img.wcs.sky2pix(peak_locs_world)
        plt.scatter(peak_locs_for_plot[:, 1], peak_locs_for_plot[:, 0], c='red', marker='x', s=100, label='Detected Peaks')
    
    # Always mark the optimised position (which may be the original position)
    opt_y, opt_x = lya_nb_img.wcs.sky2pix([[dec_opt, ra_opt]])[0]
    plt.scatter(opt_x, opt_y, c='cyan', marker='o', s=100, facecolors='none', label='Source Position')
    
    # Mark original position if different from optimised position
    if (ra_opt, dec_opt) != (ra, dec):
        orig_y, orig_x = lya_nb_img.wcs.sky2pix([[dec, ra]])[0]
        plt.scatter(orig_x, orig_y, c='black', marker='+', s=100, label='Original Position')
    
    plt.legend(loc='upper right')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    plt.title(f"{full_iden} " r"Lyman-$\alpha$ " f"NB image with Detected Peaks")
    plt.colorbar(label='Flux')
    if save_plot:
        plot_dir = io.get_plot_dir(cluster, full_iden) # Get directory for saving plots for this source
        plt.savefig(str(plot_dir / f"lya_peak_{full_iden}.png"), bbox_inches='tight', dpi=150)
    safe_show()
    plt.close()

def get_centmax(arr, n):
    """
    Get the maximum value in the central n x n region of a 2D array.
    If the array is smaller than n x n, return the maximum of the whole array.

    Parameters
    ----------
    arr : 2D array-like
        The input array from which to find the central maximum.
    n : int
        The size of the central region to consider (n x n).

    Returns
    -------
    float
        The maximum value in the central n x n region, or the maximum of the whole array 
        if the array is smaller than n x n.
    """
    # Check if the array is large enough to have a central n x n region
    if np.shape(arr)[0] >= n and np.shape(arr)[1] >= n:
        arrsh = np.shape(arr)
        arrcent_x, arrcent_y = int(np.floor(arrsh[1]/2)), int(np.floor(arrsh[0]/2))
        miniarr = arr[arrcent_y - int(np.floor(n/2)):arrcent_y + int(np.floor(n/2)) + 1,
                        arrcent_x - int(np.floor(n/2)):arrcent_x + int(np.floor(n/2)) + 1]
        return np.abs(np.nanmax(miniarr))
    else:
        print("Array is smaller than the specified central region. Returning the maximum of the whole array.")
        return np.abs(np.nanmax(arr))
    
def sensible_colorbar(fig, ax_in, img_in, label = None):
    """
    Add a colorbar to a Matplotlib figure next to a given image axis.
    This function calculates the position of the provided axis and places a colorbar
    next to it without overlapping. It also allows for an optional label for the colorbar.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The Matplotlib figure object to which the colorbar will be added.
    ax_in : matplotlib.axes.Axes
        The Matplotlib axis object that contains the image for which the colorbar is being added.
    img_in : matplotlib.image.AxesImage
        The image object for which the colorbar is being created. This is typically the result of an imshow() call.
    label : str, optional
        The label for the colorbar. If None, no label will be set. Default is None.

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The colorbar object that was created and added to the figure.
    """
    # Get the position of the image Axes and its size
    pos = ax_in.get_position()
    [ax_x, ax_y, ax_width, ax_height] = pos.bounds

    # Define the position for the colorbar
    cax = fig.add_axes([ax_x + ax_width + 0.01, ax_y, 0.02, ax_height])

    # Add a colorbar next to the image
    cbar = fig.colorbar(img_in, cax=cax)
    cax.yaxis.set_ticks_position('right')

    # Set a label for the colorbar
    cbar.set_label(label)

    return cbar

def gen_mpdaf_img_ticks(cutout, pixscale, tickspace = 1.0):
    """
    Generate tick positions and labels for an MPDAF image cutout based on its size and pixel scale.
    The ticks are generated such that they are spaced by a specified physical distance (tickspace)
    in arcseconds, and the labels are in arcseconds relative to the center of the image.

    Parameters
    ----------
    cutout : MPDAF Image
        The image cutout for which to generate ticks. The size of the cutout will determine the range of the ticks.
    pixscale : float
        The pixel scale of the image in arcseconds per pixel. This is used to convert the desired tick spacing from arcseconds to pixels.
    tickspace : float, optional
        The desired spacing between major ticks in arcseconds. Default is 1.0 arcsecond.
    """
    halfsize = (np.shape(cutout)[0]) / 2.
    tickspace_pix = tickspace / pixscale
    #how many times the major tick space goes into the half size?
    nticks = halfsize // tickspace_pix
    upbound = nticks * tickspace_pix
    xticks = yticks = np.arange(halfsize - upbound, halfsize + upbound + tickspace_pix, tickspace_pix)
    xlabels = ylabels = np.round(2.*(xticks - halfsize) * pixscale)/2.
    
    return xticks, yticks, xlabels, ylabels

import matplotlib.patches as patches

def plot_2d_model(cutout, model, markers=[], iden=None, cluster=None, save_plot=True,
                  aperture=None, marker_type='x', title='contaminant source model'):
    """
    Plot a 2D model of a source (data, model, data-model) with optional markers and aperture.

    Parameters
    ----------
    cutout : array-like
        The image cutout of the source.
    model : array-like
        The model image to plot.
    markers : list of tuples, optional
        List of (x, y) coordinates for markers to overlay on the images.
    iden : str, optional
        Identifier for the source.
    cluster : str, optional
        Cluster name.
    save_plot : bool, optional
        Whether to save the plot to a file (default is True).
    aperture : tuple, optional
        Tuple of (center, radius) for an aperture to overlay on the images.
    marker_type : str, optional
        Marker style for the overlay markers (default is 'x').
    title : str, optional
        Title for the plot (default is 'contaminant source model').

    Returns
    -------
    None
    """

    # Initialise figure and axis for plotting (horizontal layout)

    fig, axs = plt.subplots(1, 3, figsize=(24, 6), facecolor='white')
    # Ensure axs is a flat array for consistent indexing
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    vmin = -0.05 * get_centmax(cutout.data, 20.)
    vmax = get_centmax(cutout.data, 20.)

    bbimg = axs[0].imshow(cutout.data, norm=PowerNorm(0.5, vmin=vmin, vmax=vmax))
    modimg = axs[1].imshow(model, norm=PowerNorm(0.5, vmin=vmin, vmax=vmax))
    subimg = axs[2].imshow(cutout.data.data - model, norm=PowerNorm(0.5, vmin=vmin, vmax=vmax))

    # Adjust colorbar placement for horizontal layout
    cbars = []
    for a in axs:
        pos = a.get_position()
        # Place colorbar to the right of each subplot
        cax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.015, pos.height])
        cbar = fig.colorbar(a.get_images()[0], cax=cax)
        cax.yaxis.set_ticks_position('right')
        cbar.set_label(r"Flux density ($10^{-20}$\,erg\,s$^{-1}$\,cm$^{-2}$\,\AA$^{-1}$)")
        cbars.append(cbar)

    ticks = gen_mpdaf_img_ticks(cutout, 0.2, tickspace=3.0)
    for n, a in enumerate(axs):
        # Plot the provided markers
        if markers:
            a.scatter(*markers, s=250, color='red', marker=marker_type)

        a.set_yticks(ticks[1])
        a.set_xticks(ticks[0])
        a.set_yticklabels(ticks[3])
        a.set_xticklabels(ticks[2])
        a.set_xlabel(r"$ \upDelta $R.A. ($''$)")
        a.set_ylabel(r"$ \upDelta $Dec ($''$)")

        # If an aperture was provided, plot it
        if isinstance(aperture, tuple):
            apcenter = aperture[0]
            ap_radius = aperture[1]
            circ = patches.Circle(apcenter, radius=ap_radius, facecolor='none', edgecolor='coral', linestyle='--')
            a.add_artist(circ)
    axs[1].set_title(f"{cluster} source {iden} {title}")

    if save_plot:
        plot_dir = io.get_plot_dir(cluster, iden)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        print(f'Saving 2D model plot to {plot_dir}/{cluster}_{iden}_sersic_sub.pdf')
        fig.savefig(f"{plot_dir}/{cluster}_{iden}_sersic_sub.pdf")
    
    safe_show()
    plt.close()
