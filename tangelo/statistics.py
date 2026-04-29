"""
Tools for performing statistical analysis on spectra and tables.
"""

from .catalogue_operations import generate_source_mask
from . import constants as const
from . import spectroscopy as spectro
from . import models
from . import fitting
from . import plotting as plot

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
from scipy.stats import ks_2samp

from typing import Optional

def stack_and_plot_lines(
    megatab,
    emission_lines,
    selection_criteria,
    absorption=False,
    velbounds=(-5000, 5000),
    velstep=75.0,
    fit_guesses=None,
    fit_bounds=None,
    br_cutoff=0.09,
    weight_types=None,
    colors=None,
    spec_source='APER',
    spec_type='2fwhm_opt',
    save_plots=True,
    plot_name='line_stacks',
):
    """
    Stack, fit, and plot emission lines (and EW histograms) for single- and
    double-peaked Lyman-alpha sources.

    Parameters
    ----------
    megatab : astropy.table.Table
        Source catalogue (pre-filtered for significant Lya detections).
    emission_lines : list of str or list of list of str
        Line names to stack (must match keys in const.wavedict).
    selection_criteria : list of (str, float, str)
        Criteria passed to generate_source_mask as
        (column_name, threshold, bound_type).
    absorption : bool, optional
        Whether to interpret the lines as absorption features (in which case EWs will be negative).
    velbounds : tuple, optional
        (v_min, v_max) in km/s for the stacking velocity grid.
    velstep : float, optional
        Velocity bin width in km/s.
    fit_guesses : list, optional
        Initial parameter guesses for curve fitting. If None, defaults will be used.
    fit_bounds : list of (list, list), optional
        Bounds for curve fitting parameters. If None, defaults will be used.
    br_cutoff : float, optional
        Blue-to-red amplitude ratio threshold that separates single- from
        double-peaked sources.
    weight_types : dict, optional
        Mapping of {weight_exponent: label} controlling the weighting scheme.
        Defaults to {-2: 'Inverse Variance Weighting'}.
    colors : dict, optional
        Mapping of {'single': color, 'double': color} for plotting.
        Defaults to {'single': 'firebrick', 'double': 'slateblue'}.
    spec_source : str, optional
        Spectral source identifier (e.g. 'APER', 'R21').
    spec_type : str, optional
        Spectral type identifier (e.g. '2fwhm_opt').
    save_plots : bool, optional
        Whether to save stacking plots as PDF files under plots/.
    plot_name : str, optional
        Base name for saved plot files (e.g. 'line_stacks').

    Returns
    -------
    stacks : dict
        Nested dict stacks[npeak][line] = (vel, flux, err, n).
    mc_samples : dict
        Nested dict mc_samples[npeak][line] with Monte Carlo fit parameter arrays.
    peaked_masks : dict
        Boolean masks {'single': ..., 'double': ...} used for the stacking.
    """
    if weight_types is None:
        weight_types = {-2: 'Inverse Variance Weighting'}
    if colors is None:
        colors = {'single': 'firebrick', 'double': 'slateblue'}

    # If line list contains strings, convert to list of lists for uniform processing
    if all(isinstance(line, str) for line in emission_lines):
        emission_lines = [[line] for line in emission_lines]

    # --- Build source mask from selection criteria ---
    source_mask = generate_source_mask(megatab, selection_criteria)

    # Split into single- and double-peaked based on blue-to-red amplitude ratio
    brratio  = megatab['AMPB'] / megatab['AMPR']
    br_upper = 3.0 / (megatab['AMPR'] / megatab['AMPR_ERR'])

    peaked_masks = {
        "single": source_mask & ~(brratio >= br_cutoff) & (br_upper < br_cutoff) & (megatab['z'] < 4.1),
        "double": source_mask & (brratio >= br_cutoff),
    }

    print("Selection criteria applied:")
    for colname, threshold, bound_type in selection_criteria:
        op = '>' if bound_type == 'lower' else '<='
        print(f"  {colname} {op} {threshold}")
    print(f"\nSingle-peaked sources: {peaked_masks['single'].sum()}")
    print(f"Double-peaked sources: {peaked_masks['double'].sum()}")

    # --- Stacking, fitting, and plotting ---
    stacks      = {}
    mc_samples  = {}

    initial_guesses = fit_guesses if fit_guesses is not None else [10, -200, 100, 0.2, 0]
    parameter_bounds = fit_bounds if fit_bounds is not None else [[-1000, -300, velstep, 0., -np.inf],
                                                                   [1000, 300, 400, 5, np.inf]]

    for weight_exp, weight_label in weight_types.items():
        print(f"\nStacking with {weight_label} (weight exponent = {weight_exp}):")

        ncols_plot = len(emission_lines)
        ncols_plot = min(ncols_plot, 3)
        nrows_plot = int(np.ceil(len(emission_lines) / ncols_plot))
        fig, axs = plt.subplots(nrows_plot, ncols_plot,
                                figsize=(4 * ncols_plot, 4 * nrows_plot),
                                facecolor='w', sharey=True, sharex=True)
        ax = np.atleast_1d(axs).flatten()
        fig.subplots_adjust(wspace=0, hspace=0)

        for npeak, mask in peaked_masks.items():
            stacktab = megatab[mask]
            stacks.setdefault(npeak, {})
            mc_samples.setdefault(npeak, {})

            for j, line in enumerate(emission_lines):
                line_label = ' + '.join(line)
                _vel, _flux, _err, _n = spectro.stack_spectra_across_sources(
                    stacktab, line, velocity_frame='lyalpha',
                    velbounds=list(velbounds), velstep=velstep,
                    weighting=weight_exp, sigclip_weights=3,
                    spec_source=spec_source, spec_type=spec_type,
                )
                stacks[npeak][line_label] = (_vel, _flux, _err, _n)

                print(f"\nStacking of {line_label} complete:")
                print(f"  {npeak.capitalize()}-peaked: {_n} sources")

                ax[j].plot(_vel, _flux, drawstyle='steps-mid',
                           label=f"{npeak.capitalize()}-peaked (N={_n})",
                           alpha=0.75, color=colors[npeak])
                ax[j].fill_between(_vel, _flux - _err, y2=_flux + _err,
                                   alpha=0.15, step='mid', edgecolor='none',
                                   color=colors[npeak], linewidth=0)
                ax[j].legend(loc='upper left', bbox_to_anchor=(0, 0.90))

                fit_func  = models.gaussian
                fit_range = (-3000, 3000)

                if len(line) == 1 and line[0] in const.doublets:
                    print("  Using doublet Gaussian fit function.")
                    wave1     = const.wavedict[const.doublets[line[0]][0]]
                    wave2     = const.wavedict[const.doublets[line[0]][1]]
                    fit_func  = models.gaussian_doublet_vel((wave1, wave2))
                    parameter_bounds = [[-1000, -300, velstep, -1000, 0., -np.inf],
                                 [ 1000,  300,    400,  1000,  5,  np.inf]]
                    initial_guesses  = [10, -200, 100, 10, 0, 0]

                _fitreg = (_vel >= fit_range[0]) & (_vel <= fit_range[1])

                try:
                    _init = curve_fit(fit_func, _vel[_fitreg], _flux[_fitreg],
                                      p0=initial_guesses, bounds=parameter_bounds,
                                      max_nfev=100000, method='trf')[0]
                except RuntimeError:
                    print("  Fit failed; skipping this stack.")
                    continue

                _fit, _mcarr = fitting.fit_mc(
                    fit_func, _vel[_fitreg], _flux[_fitreg], _err[_fitreg],
                    _init, bounds=parameter_bounds, return_sample=True, niter=500,
                    autocorrelation=False, chisq_thresh=np.inf,
                )

                hires_vel     = np.linspace(_vel[0], _vel[-1], 1000)
                ax[j].plot(hires_vel, fit_func(hires_vel, *_fit[0]),
                           linestyle='-', color='fuchsia', alpha=0.2)
                _model_curves = np.array([fit_func(hires_vel, *p) for p in _mcarr])
                ax[j].fill_between(hires_vel,
                                   np.percentile(_model_curves, 16, axis=0),
                                   np.percentile(_model_curves, 84, axis=0),
                                   color='fuchsia', alpha=0.15, edgecolor='none')

                _mcarr = np.array(_mcarr)
                mc_samples[npeak][line_label] = {
                    'fluxes': _mcarr[:, 0],
                    'v_cs'  : _mcarr[:, 1],
                    'widths': _mcarr[:, 2],
                    'ews'   : np.abs(_mcarr[:, 0] + (_mcarr[:, 3] if len(_mcarr[0]) > 5 else 0) / _mcarr[:, -2]),
                }

        # Statistical comparison of emission-line EWs
        for line_list in emission_lines:
            line_label = ' + '.join(line_list)
            if 'single' not in mc_samples or line_label not in mc_samples.get('single', {}):
                continue
            if 'double' not in mc_samples or line_label not in mc_samples.get('double', {}):
                continue
            ew_single = mc_samples['single'][line_label]['ews']
            ew_double = mc_samples['double'][line_label]['ews']
            ew_ratio  = ew_single / ew_double
            p_ew      = np.sum(ew_single >= ew_double) / len(ew_single)
            print(f"\nStatistical comparison for {line_label}:")
            print(f"  P(single EW >= double EW) = {p_ew:.4f}")
            median = np.median(ew_ratio)
            p16, p84 = np.percentile(ew_ratio, [16, 84])
            p2,  p97 = np.percentile(ew_ratio, [2.5, 97.5])
            print(f"  EW Ratio: median={median:.3f}, 68% CI=({p16:.3f},{p84:.3f}), 95% CI=({p2:.3f},{p97:.3f})")

        # Plot formatting
        ax[0].set_ylabel(r"$f_{v}$ [$10^{-20}$\,erg\,s$^{-1}$\,cm$^{-2}$\,(km\,s$^{-1}$)$^{-1}$]")
        for a in ax:
            a.axvline(0., linestyle='--', alpha=0.25, color='k')
            a.set_xlabel(r"Velocity [km\,s$^{-1}$]")
            a.set_xlim(-3500, 3500)
            a.set_xticks(np.arange(-3000, 3000, 1000))
        if save_plots:
            fig.savefig(
                f"plots/{plot_name}_{spec_type}_{weight_label.replace(' ', '').lower()}.pdf",
                bbox_inches='tight',
            )
        plt.show()
        plt.close(fig)

    # --- EW histograms (cell 8 logic) ---
    ew_bins = np.geomspace(0.05, 20, 15)
    for line_list in emission_lines:
        for line in line_list:
            if f"FLUX_{line}" not in megatab.colnames or f"CONT_{line}" not in megatab.colnames:
                print(f"  Missing FLUX_{line} or CONT_{line} columns; skipping EW histogram for {line}.")
                continue
            ew_arrays = {}
            plt.figure(figsize=(6, 4))
            for npeak in ['single', 'double']:
                ew_data = (megatab[peaked_masks[npeak]][f"FLUX_{line}"]
                        / megatab[peaked_masks[npeak]][f"CONT_{line}"])
                ew_data /= megatab[peaked_masks[npeak]]['z'] + 1
                ew_data  = ew_data[np.where(~np.isnan(ew_data))]
                if absorption:
                    ew_data = -ew_data
                ew_arrays[npeak] = ew_data
                plt.hist(ew_data, bins=ew_bins, alpha=0.5, density=False,
                        label=f"{npeak.capitalize()}-peaked", color=colors[npeak])
            plt.xlabel(f"Equivalent Width of {line} [Å]")
            plt.ylabel("Number of sources")
            plt.legend()
            plt.title(f"Equivalent Width Distribution of {line}")
            plt.xscale('log')
            plt.show()
            plt.close()

            ks_stat, ks_pval = ks_2samp(ew_arrays['single'], ew_arrays['double'],
                                        alternative='two-sided')
            print(f"KS test for {line} EW distributions (single vs double peaked):")
            print(f"  KS Statistic: {ks_stat:.4f}, P-value: {ks_pval:.4f}")

    return stacks, mc_samples, peaked_masks


from matplotlib import colormaps as cm
blured = cm.get_cmap('seismic')
black = cm.get_cmap('bone_r')

def mask_bad_scatter_points(colx: list, coly: list, upper_bounds: Optional[np.ndarray] = None, c: Optional[np.ndarray] = None, 
                            mask_in: Optional[np.ndarray] = None) -> tuple[list, list, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Masks out points in colx and coly that have NaN or inf values in either the values or error bars.

    Parameters
    ----------
    colx : list of arrays
        List of arrays for x values and optional error bars. First element is x values, second is symmetric error bars, third is asymmetric error bars.
    coly : list of arrays
        List of arrays for y values and optional error bars. First element is y values, second is symmetric error bars, third is asymmetric error bars.
    upper_bounds : np.ndarray, optional
        Mask for points that are upper limits. Should have the same length as colx[0] and coly[0]. Default is 
        None (no upper limits).
    c : array-like, optional
        Values for coloring the points. Should have the same length as colx[0] and coly[0]. Default is None (no coloring).
    mask_in : array-like, optional
        Optional boolean array to further mask the points. Should have the same length as colx[0] and coly[0]. 
        Default is None (no additional masking).
        
    Returns
    -------
    tuple of lists
        Masked versions of colx and coly with NaN and inf points removed.
    """
    mask = np.ones(np.size(colx[0])).astype(bool)
    if mask_in is not None:
        mask &= mask_in
    mask &= ~np.isnan(colx[0])
    mask &= ~np.isnan(coly[0])
    mask &= ~np.isinf(colx[0])
    mask &= ~np.isinf(coly[0])
    if len(colx) > 1:
        mask &= ~np.isnan(colx[1])
        mask &= ~np.isinf(colx[1])
    if len(colx) > 2:
        mask &= ~np.isnan(colx[2])
        mask &= ~np.isinf(colx[2])
    if len(coly) > 1:
        mask &= ~np.isnan(coly[1])
        mask &= ~np.isinf(coly[1])
    if len(coly) > 2:
        mask &= ~np.isnan(coly[2])
        mask &= ~np.isinf(coly[2])

    masked_colx = [arr[mask] for arr in colx]
    masked_coly = [arr[mask] for arr in coly]
    masked_upper_bounds = upper_bounds[mask] if upper_bounds is not None else None
    masked_c = c[mask] if c is not None else None

    return masked_colx, masked_coly, masked_upper_bounds, masked_c

def make_scatter(colx: list, coly: list, ax: plt.Axes, mask: Optional[np.ndarray] = None, 
                 c = None, upper_bounds: Optional[np.ndarray] = None, edgecolor = 'black',
                 show_colorbar = False, alpha = 0.6, msize = 25, cmap = blured, alpha_e = 0.3,
                 label = None, marker='o', alpha_ubs = 0.1, vmin = None, vmax = None,
                 clabel = None, cnorm = 'linear'):
    """
    Makes a scatter plot of colx vs coly, with optional coloring by a third parameter.

    Parameters
    ----------
    colx : list of arrays
        List of arrays for x values and optional error bars. First element is x values, second is symmetric error bars, third is asymmetric error bars.
    coly : list of arrays
        List of arrays for y values and optional error bars. First element is y values, second is symmetric error bars, third is asymmetric error bars.
    ax : matplotlib.axes.Axes
        The axes on which to plot.
    mask : array-like, optional
        Optional boolean array to mask the points. Should have the same length as colx[0] and coly[0]. Default 
        is None (no masking).
    c : array-like, optional
        Values used to color the points via the colormap. Can be any numeric quantity (e.g. redshift, 
        luminosity, S/N). Points with NaN or inf values in ``c`` are plotted in grey. Default is None 
        (uniform color).
    upper_bounds : np.ndarray, optional
        Mask for points that are upper limits. Should have the same length as colx[0] and coly[0]. Default is 
        None (no upper limits).
    edgecolor : str, optional
        Color for the edges of the points. Default is 'black'.
    show_colorbar : bool, optional
        Whether to add a colorbar to the axes. Default is False.
    alpha : float, optional
        Alpha value for the points. Default is 0.6.
    alpha_ubs : float, optional
        Alpha value for the upper bound arrows. Default is 0.1.
    msize : float, optional
        Marker size for the points. Default is 25.
    cmap : matplotlib.colors.Colormap, optional
        Colormap to use for coloring points by c. Default is blured (seismic colormap).
    alpha_e : float, optional
        Alpha value for the error bars. Default is 0.3.
    label : str, optional
        Label for the points to be used in the legend. Default is None.
    marker : str, optional
        Marker style for the points. Default is 'o' (circle).
    vmin : float, optional
        Lower limit of the colormap range. If None, set automatically from the data. Default is None.
    vmax : float, optional
        Upper limit of the colormap range. If None, set automatically from the data. Default is None.
    clabel : str, optional
        Label for the colorbar. Default is None (no label).
    cnorm : str, optional
        Scaling to apply to the colormap. One of ``'linear'`` (default), ``'log'``, or
        ``'sqrt'``. Log and square-root scales are useful when ``c`` has extreme outliers.

    Returns
    -------
    None
    """
    if upper_bounds is None:
        upper_bounds = np.zeros(np.size(colx[0])).astype(bool)
    # If there are any NaN or inf values or error bars, remove all the corresponding points from the plot
    colx, coly, upper_bounds, c = mask_bad_scatter_points(colx, coly, upper_bounds = upper_bounds, c = c,
                                                           mask_in=mask)
    
    # Identify points whose c value is NaN/inf so they can be given a special colour
    c_nan_mask = np.zeros(np.size(colx[0]), dtype=bool)
    if c is not None:
        c_nan_mask = ~np.isfinite(c)

    if c is None:
        # If c is not provided, use a default color for all points
        c = np.zeros(np.size(colx[0]))

    # Extract x and y values and error bars from colx and coly
    xvals = colx[0] # Values for x-axis
    xerrsm = xerrsp = np.zeros(len(xvals)) # Initialize error bars to zero
    if len(colx) == 2:
        xerrsm = xerrsp = colx[1] # Symmetric error bars if only one error array is provided
    elif len(colx) == 3:
        xerrsm = colx[1] # Lower error bars
        xerrsp = colx[2] # Upper error bars
    yvals = coly[0] # Values for y-axis
    yerrsm = yerrsp = np.zeros(len(yvals)) # Initialize error bars to zero
    if len(coly) == 2:
        yerrsm = yerrsp = coly[1] # Symmetric error bars if only one error array is provided
    elif len(coly) == 3:
        yerrsm = coly[1] # Lower error bars
        yerrsp = coly[2] # Upper error bars
    
    # Determine colormap limits: use data range if not explicitly supplied
    _vmin = vmin if vmin is not None else np.nanmin(c)
    _vmax = vmax if vmax is not None else np.nanmax(c)

    # Build colormap normalisation
    _cnorm = cnorm.lower() if cnorm is not None else 'linear'
    if _cnorm == 'log':
        norm = mcolors.LogNorm(vmin=_vmin, vmax=_vmax)
    elif _cnorm == 'sqrt':
        norm = mcolors.PowerNorm(gamma=0.5, vmin=_vmin, vmax=_vmax)
    else:  # 'linear' or anything else
        norm = mcolors.Normalize(vmin=_vmin, vmax=_vmax)

    # Split detections into those with valid c values and those with NaN c values
    valid_c_det = ~upper_bounds & ~c_nan_mask
    nan_c_det   = ~upper_bounds & c_nan_mask

    # Create scatter plots with error bars for the detections and upper limits
    # Detections with valid c: use colormap
    sc = ax.scatter(xvals[valid_c_det], yvals[valid_c_det], c=c[valid_c_det], cmap=cmap, 
                    edgecolor=edgecolor, s=msize, norm=norm, 
                    alpha = alpha, label=label, marker=marker) # points
    # Detections with NaN c: plot in grey
    if np.any(nan_c_det):
        ax.scatter(xvals[nan_c_det], yvals[nan_c_det], color='grey',
                   edgecolor=edgecolor, s=msize, alpha=alpha, marker=marker)
    ax.errorbar(xvals[~upper_bounds], yvals[~upper_bounds], xerr=[xerrsm[~upper_bounds], xerrsp[~upper_bounds]], 
                yerr=[yerrsm[~upper_bounds], yerrsp[~upper_bounds]], marker='', color=edgecolor, 
                zorder=0, alpha = alpha_e, linestyle='') # error bars
    # Then upper limits
    ax.errorbar(xvals[upper_bounds], yvals[upper_bounds], xerr=[xerrsm[upper_bounds], xerrsp[upper_bounds]], 
                yerr=[3*yerrsm[upper_bounds], yerrsp[upper_bounds]], marker='v', color='red', zorder=0, 
                alpha = alpha_ubs, linestyle='', uplims=upper_bounds[upper_bounds]) # upper limit arrows

    if show_colorbar and np.any(valid_c_det):
        cbar = plt.colorbar(sc, ax=ax)
        if clabel is not None:
            cbar.set_label(clabel)
    
def make_histo(cola, colb, ax, filtsa = [], filtsb = [], erra = None, errb = None,
                 binsa = None, binsb = None, density=False, plotavg=True,
                laba = None, labb = None, show_stat = None, kwargs = {}):
    """
    Makes two histograms on the same axes for cola and colb, with optional filtering by
    filtsa and filtsb, and optional error bars erra and errb.
    
    Parameters
    ----------
    cola : array-like
        Data for the first histogram.
    colb : array-like
        Data for the second histogram.
    ax : matplotlib.axes.Axes
        The axes on which to plot the histograms.
    filtsa : list of arrays, optional
        List of boolean arrays for filtering cola. Each array should have the same length as cola, and the final mask will be the logical AND of all the filters. Default is an empty list (no filtering).
    filtsb : list of arrays, optional
        List of boolean arrays for filtering colb. Each array should have the same length as colb, and the final mask will be the logical AND of all the filters. Default is an empty list (no filtering).
    erra : array-like, optional
        Symmetric error bars for cola. Should have the same length as cola. Default is None (no error bars).
    errb : array-like, optional
        Symmetric error bars for colb. Should have the same length as colb. Default is None (no error bars).
    binsa : array-like, optional
        Bin edges for the histogram of cola. If None, bins will be automatically determined. Default is None.
    binsb : array-like, optional
        Bin edges for the histogram of colb. If None, bins will be automatically determined. Default is None.
    density : bool, optional
        If True, the histograms will be normalized to form a probability density. Default is False.
    plotavg : bool, optional
        Whether to plot vertical lines for the medians of cola and colb. Default is True.
    laba : str, optional
        Label for the first histogram (used in the legend). Default is None.
    labb : str, optional
        Label for the second histogram (used in the legend). Default is None.
    show_stat : str, optional
        Column name in megatab to print the mean of for the filtered data. Default is None (no statistic printed).
    kwargs : dict, optional
        Additional keyword arguments to pass to ax.hist for both histograms. Default is an empty dictionary (no additional arguments).

    Returns
    -------
    None
    """
    maska = np.ones(np.size(cola)).astype(bool)
    maskb = np.ones(np.size(colb)).astype(bool)
    for cond in filtsa:
        maska *= cond
    for cond in filtsb:
        maskb *= cond
        
    if binsa is None:
        binsa = np.linspace(np.nanmin(cola), np.nanmax(cola), 50)
    if binsb is None:
        binsb = np.linspace(np.nanmin(colb), np.nanmax(colb), 50)
        
    if erra is None:
        erra = np.zeros(np.size(cola))
    if errb is None:
        errb = np.zeros(np.size(colb))
    
    # Calculate medians - check if x-axis is log-scaled
    if ax.get_xscale() == 'log':
        # For log scale, use log of values (add small offset to avoid log(0))
        offset = 1e-10  # Small offset to handle zeros/negative values
        log_cola = np.log10(cola[maska] + offset)
        log_colb = np.log10(colb[maskb] + offset)
        avga = 10**np.nanmedian(log_cola) - offset
        avgb = 10**np.nanmedian(log_colb) - offset
    else:
        # For linear scale, use normal median
        avga = np.nanmedian(cola[maska])
        avgb = np.nanmedian(colb[maskb])
    
    avga_err = np.sqrt(np.nanmean(np.square(erra[maska])))
    avgb_err = np.sqrt(np.nanmean(np.square(errb[maskb])))
    
    ax.hist(cola[maska], bins=binsa, density=density, label=laba, color='mediumslateblue', **kwargs)
    ax.hist(colb[maskb], bins=binsb, density=density, label=labb, color='coral', **kwargs)

    megatab=None  # Placeholder for megatab, which should be defined in the context where this function is used
    
    if show_stat is not None:
        print(show_stat, np.nanmean(megatab[show_stat].data[maska]), np.nanmean(megatab[show_stat].data[maskb]))
    
    if plotavg:
        ax.axvline(avga, alpha=0.5, linestyle='--', color='darkslateblue', linewidth=2, 
                  label=f"{laba} median")
        ax.axvline(avgb, alpha=0.5, linestyle='--', color='darkred', linewidth=2,
                  label=f"{labb} median")
        


from .spectroscopy import muse_lsf_fwhm_poly
from .constants import wavedict
from astropy.table import Table
from astropy.cosmology import Planck18
import astropy.units as u


# Helper functions to get line any lyman alpha properties that aren't directly in the megatable, and to apply rest-frame correction if needed.

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


def get_line_property(megatab: Table, line: str, prop: str, 
                      rest_frame: bool = True, correct_inst: bool = True,
                      abs: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper function to get the line property column from the megatable, handling parameters that aren't
    directly in the megatable (e.g. LYA_EW, BRRATIO, DISPR, FWHMR) and applying rest-frame correction if needed.
    Note that absorption lines have their equivalent widths flipped to maintain the convention of positive values 
    indicating stronger absorption, which is important for interpreting correlations with Lyman alpha properties.

    Parameters
    ----------
    megatab : astropy.table.Table
        The megatable containing the data.
    line : str
        The name of the line (e.g. 'CIV1548', 'SiII1260').
    prop : str
        The property to retrieve (e.g. 'EW', 'FWHM').
    rest_frame : bool, optional
        Whether to apply rest-frame correction for wavelength-based properties, by default True.
    correct_inst : bool, optional
        Whether to apply instrumental resolution correction for FWHM using the MUSE LSF polynomial,
        by default True.
    abs : bool, optional
        Whether the line is an absorption line, which requires flipping the sign of the equivalent width
        to be consistent with the convention that positive values indicate stronger absorption

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The requested line property column and its associated error column.
    """
    col_name = f"{prop}_{line}"
    err_name = f"{prop}_ERR_{line}"
    if line in ["LI_ABS", "HI_ABS", "TOT_ABS"]:
        col_name = f"{prop}_{line}"
        err_name = f"{prop}_{line}_ERR"
        return megatab[col_name].copy(), megatab[err_name].copy()
    if prop == "EW" and col_name not in megatab.colnames:
        # Calculate EW from FLUX, FLUX_ERR, CONT and CONT_ERR if not directly available
        flux_col = f"FLUX_{line}"
        flux_err_col = f"FLUX_ERR_{line}"
        cont_col = f"CONT_{line}"
        cont_err_col = f"CONT_ERR_{line}"
        if flux_col in megatab.colnames and flux_err_col in megatab.colnames and cont_col in megatab.colnames and cont_err_col in megatab.colnames:
            flux = megatab[flux_col].copy()
            flux_err = megatab[flux_err_col].copy()
            cont = megatab[cont_col].copy()
            cont_err = megatab[cont_err_col].copy()
            ew = flux / cont
            ew_err = np.abs(ew * np.sqrt((flux_err / flux)**2 + (cont_err / cont)**2))
            if rest_frame:
                ew /= (1 + megatab['z'])
                ew_err /= (1 + megatab['z'])
            if abs:
                ew *= -1  # Flip sign for absorption lines to maintain convention of positive values 
                            # indicating stronger absorption
            return ew, ew_err
        else:
            raise ValueError(f"Required columns for calculating EW of {line} not found in megatable.")
    elif prop == "FWHM":
        # Calculate FWHM from FWHM_OPT and redshift if not directly available
        fwhm_col = f"FWHM_{line}"
        if fwhm_col in megatab.colnames and err_name in megatab.colnames:
            fwhm = megatab[fwhm_col].copy()
            fwhm_err = megatab[err_name].copy()
            # Apply instrumental resolution correction using MUSE LSF polynomial if not already applied
            if correct_inst:
                fwhm, fwhm_err = correct_inst_res(fwhm, fwhm_err, megatab[f'LPEAK_{line}'], "FWHM")
            if rest_frame:
                fwhm /= (1 + megatab['z'])
                fwhm_err /= (1 + megatab['z'])
            return fwhm, fwhm_err
        else:
            raise ValueError(f"Required column for calculating FWHM of {line} not found in megatable.")
    elif prop == "CVEL":
        # Calculate velocity centroid of the line relative to the systemic redshift, which can
        # be derived from the Lyman alpha peak redshift LPEAKR and the offset from systemic
        # DELTAV_LYA
        lya_z = megatab['LPEAKR'] / 1215.67 - 1
        sys_z = lya_z - megatab['DELTAV_LYA'] / 299792.458 * (1 + lya_z)  # Convert velocity offset to redshift
        line_peak_rest = megatab[f'LPEAK_{line}'] / (1 + sys_z)  # Convert observed line peak to rest-frame wavelength
        line_peak_rest_err = megatab[f'LPEAK_ERR_{line}'].copy() / (1 + sys_z)  # Propagate error through rest-frame conversion
        line_rest_wave = wavedict[line]  # Get rest-frame wavelength of the line from wavedict
        cvel = (line_peak_rest - line_rest_wave) / line_rest_wave * 299792.458  # Convert to velocity offset from rest wavelength
        cvel_err = line_peak_rest_err / line_rest_wave * 299792.458  # Propagate error through velocity conversion
        return cvel, cvel_err
    elif prop == "LUM":
        # Emission line luminosity: L = 4π D_L² × (FLUX_{line} / MU) × 1e-20
        flux_col = f"FLUX_{line}"
        flux_err_col = f"FLUX_ERR_{line}"
        if flux_col not in megatab.colnames:
            raise ValueError(f"Column {flux_col} not found in megatable.")
        return flux_to_luminosity(
            megatab[flux_col].copy(), megatab[flux_err_col].copy(),
            np.asarray(megatab['z']), np.asarray(megatab['MU']),
            is_continuum=False,
        )
    elif prop == "CONT_LUM":
        # Emission line continuum luminosity (rest-frame, per Å)
        cont_col = f"CONT_{line}"
        cont_err_col = f"CONT_ERR_{line}"
        if cont_col not in megatab.colnames:
            raise ValueError(f"Column {cont_col} not found in megatable.")
        return flux_to_luminosity(
            megatab[cont_col].copy(), megatab[cont_err_col].copy(),
            np.asarray(megatab['z']), np.asarray(megatab['MU']),
            is_continuum=True,
        )
    elif col_name in megatab.colnames and err_name in megatab.colnames:
        return megatab[col_name].copy(), megatab[err_name].copy()
    else:
        raise ValueError(f"Column {col_name} not found in megatable.")
    
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
        
def get_lya_property(megatab: Table, prop: str, rest_frame: bool = True,
                     correct_inst: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper function to get the Lyman alpha property column from the megatable, handling parameters that aren't
    directly in the megatable (e.g. LYA_EW, BRRATIO, DISPR, FWHMR).

    Parameters
    ----------
    megatab : astropy.table.Table
        The megatable containing the data.
    prop : str
        The Lyman alpha property to retrieve (e.g. 'EW', 'BRRATIO', 'DISPR', 'FWHMR').
    rest_frame : bool, optional
        Whether to apply rest-frame correction for wavelength-based properties, by default True.
    correct_inst : bool, optional
        Whether to apply instrumental resolution correction for FWHMR and DISPR using the MUSE
        LSF polynomial, by default True.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The requested Lyman alpha property column and its associated error column.
    """
    if prop in ["LYA_EW", "EW"]:
        # Calculate Lya EW from FLUXR, FLUXR_ERR, CONT and CONT_ERR if not directly available
        fluxr_col = "FLUXR"
        fluxr_err_col = "FLUXR_ERR"
        fluxb_col = "FLUXB"
        fluxb_err_col = "FLUXB_ERR"
        cont_col = "CONT"
        cont_err_col = "CONT_ERR"
        if fluxr_col in megatab.colnames and fluxr_err_col in megatab.colnames and fluxb_col in megatab.colnames and fluxb_err_col in megatab.colnames and cont_col in megatab.colnames and cont_err_col in megatab.colnames:
            fluxr = megatab[fluxr_col].copy()
            fluxr_err = megatab[fluxr_err_col].copy()
            fluxb = megatab[fluxb_col].copy()
            fluxb_err = megatab[fluxb_err_col].copy()
            # replace any NaN fluxb and fluxb_err values with 0 to avoid issues in EW calculation for sources without significant blue peak detection
            fluxb = np.nan_to_num(fluxb, nan=0.0)
            fluxb_err = np.nan_to_num(fluxb_err, nan=0.0)
            flux_total = fluxr + fluxb
            flux_total_err = np.sqrt(fluxr_err**2 + fluxb_err**2)
            cont = megatab[cont_col].copy()
            cont_err = megatab[cont_err_col].copy()
            ew = flux_total / cont
            ew_err = np.abs(ew * np.sqrt((flux_total_err / flux_total)**2 + (cont_err / cont)**2))
            if rest_frame:
                ew /= (1 + megatab['z'])  # Rest-frame correction
                ew_err /= (1 + megatab['z'])
            return ew, ew_err
        else:
            raise ValueError("Required columns for calculating Lya EW not found in megatable.")
    elif prop == "BRRATIO":
        # Calculate blue-to-red flux ratio from FLUXB and FLUXR if not directly available
        blue_flux_col = "FLUXB"
        red_flux_col = "FLUXR"
        if blue_flux_col in megatab.colnames and red_flux_col in megatab.colnames:
            blue_flux = megatab[blue_flux_col].copy()
            red_flux = megatab[red_flux_col].copy()
            with np.errstate(divide='ignore', invalid='ignore'):
                br_ratio = blue_flux / red_flux
                br_ratio[red_flux == 0] = np.nan  # Avoid division by zero
            br_ratio_err = br_ratio * np.sqrt((megatab["FLUXB_ERR"].copy() / blue_flux)**2 
                                              + (megatab["FLUXR_ERR"].copy() / red_flux)**2)
            return br_ratio, br_ratio_err
        else:
            raise ValueError("Required columns for calculating Lya blue-to-red flux ratio not found in megatable.")
    elif prop == "BRSEP":
        # Calculate blue-red peak separation from LPEAKR and LPEAKB if not directly available
        red_peak_col = "LPEAKR"
        blue_peak_col = "LPEAKB"
        if red_peak_col in megatab.colnames and blue_peak_col in megatab.colnames:
            red_peak = megatab[red_peak_col].copy()
            blue_peak = megatab[blue_peak_col].copy()
            if rest_frame:
                br_sep = (red_peak - blue_peak) / (1 + megatab['z'])  # Rest-frame correction
                br_sep_err = np.sqrt(megatab["LPEAKR_ERR"].copy()**2 + megatab["LPEAKB_ERR"].copy()**2) / (1 + megatab['z'])
            else:
                br_sep = red_peak - blue_peak
                br_sep_err = np.sqrt(megatab["LPEAKR_ERR"].copy()**2 + megatab["LPEAKB_ERR"].copy()**2)
            br_sep[(red_peak == 0) | (blue_peak == 0)] = np.nan  # Avoid invalid values
            return br_sep, br_sep_err
        else:
            raise ValueError("Required columns for calculating Lya blue-red peak separation not found in megatable.")
    elif prop[:-1] in ["DISP", "FWHM", "FWHM_AB"] and prop in megatab.colnames:
        err_name = f"{prop}_ERR"
        # For these parameters, just apply rest-frame correction to the existing column
        col = megatab[prop].copy()
        col_err = megatab[err_name].copy()
        if correct_inst:
            lpeakr_col = "LPEAKR"
            if lpeakr_col in megatab.colnames:
                lpeakr = megatab[lpeakr_col].copy()
                lsf_fwhm = muse_lsf_fwhm_poly(lpeakr)
                if prop[:-1] == "DISP":
                    lsf_disp = lsf_fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to dispersion
                    col, col_err = correct_inst_res(col, col_err, lpeakr, "DISP")
                else:  # FWHM or FWHM_AB
                    col, col_err = correct_inst_res(col, col_err, lpeakr, "FWHM")
            else:
                raise ValueError("Required column for instrumental resolution correction not found in megatable.")
        if rest_frame:
            col /= (1 + megatab['z'])  # Rest-frame correction
            col_err /= (1 + megatab['z'])
        return col, col_err
    elif "ZELDA" in prop and prop in megatab.colnames:
        # for ZELDA parameters, no need to apply corrections; however, the error bars are
        # stored in two separate columns (ERRM and ERRP) for negative and positive errors, 
        # so we take the average of these for simplicity in plotting and correlation analysis
        prop_base = prop.rsplit('_', 1)[0]  # Get the base property name without the ZELDA suffix
        errm_name = f"{prop_base}_ERRM_ZELDA"
        errp_name = f"{prop_base}_ERRP_ZELDA"
        if errm_name in megatab.colnames and errp_name in megatab.colnames:
            col_err = (megatab[errm_name].copy() + megatab[errp_name].copy()) / 2
            return megatab[prop].copy(), col_err
        else:
            return megatab[prop].copy(), None
    elif prop == "LUM_LYA":
        # Total Lya line luminosity (red + blue peak)
        fluxr = megatab['FLUXR'].copy()
        fluxr_err = megatab['FLUXR_ERR'].copy()
        fluxb = np.nan_to_num(megatab['FLUXB'].copy(), nan=0.0)
        fluxb_err = np.nan_to_num(megatab['FLUXB_ERR'].copy(), nan=0.0)
        flux_total = fluxr + fluxb
        flux_total_err = np.sqrt(fluxr_err**2 + fluxb_err**2)
        return flux_to_luminosity(
            flux_total, flux_total_err,
            np.asarray(megatab['z']), np.asarray(megatab['MU']),
            is_continuum=False,
        )
    elif prop == "LUM_CONT_LYA":
        # Lya continuum luminosity (rest-frame, per Å)
        return flux_to_luminosity(
            megatab['CONT'].copy(), megatab['CONT_ERR'].copy(),
            np.asarray(megatab['z']), np.asarray(megatab['MU']),
            is_continuum=True,
        )
    elif prop in megatab.colnames:
        err_name = f"{prop}_ERR"
        if err_name in megatab.colnames:
            return megatab[prop].copy(), megatab[err_name].copy()
        else:
            return megatab[prop].copy(), None
    else:
        raise ValueError(f"Column {prop} not found in megatable.")
    
def prepare_scatter_mask(megatab: Table, line: str, line_col: np.ndarray, line_prop: str,
                         lya_col: np.ndarray, lya_prop: str, abs_lines: list[str],
                         include_upper_limits: bool = False, sig_thresh: float = 3.0) -> np.ndarray:
    """
    Prepare a mask for scatter plot analysis between a given line and Lyman alpha property.

    Parameters
    ----------
    megatab : astropy.table.Table
        The megatable containing the data.
    line : str
        The line to analyze (e.g., "SiII1260").
    line_col : np.ndarray
        The column for the line property (e.g., equivalent width or FWHM).
    line_prop : str
        The line property being analyzed (e.g., "EW", "FWHM").
    lya_col : np.ndarray
        The Lyman alpha property column.
    lya_prop : str
        The Lyman alpha property to analyze (e.g., "LYA_EW").
    abs_lines : list[str]
        List of lines that should be treated as absorption.
    include_upper_limits : bool, optional
        Whether to include upper limits for for non-detections. If True, sources with non-significant detections
        are not masked.
    sig_thresh : float, optional
        The significance threshold (in sigma) for including sources based on their SNR

    Returns
    -------
    np.ndarray
        A boolean mask indicating valid data points for scatter plot analysis.
    """
    mask = np.isfinite(line_col) & np.isfinite(lya_col)

    # Special condition for stacked absorption lines
    is_stacked_abs = line in ["TOT_ABS", "HI_ABS", "LI_ABS"]

    if is_stacked_abs:
        # Calculate SNR based on EW and mask
        mask &= -megatab[f"EW_{line}"] / megatab[f"EW_{line}_ERR"] > sig_thresh
        return mask

    # Only include sources with significant line and continuum detection
    if not include_upper_limits and not is_stacked_abs:
        # For emission, require SNR > sig_thresh, for absorption, require < -sig_thresh
        mask &= (megatab[f"SNR_{line}"] > sig_thresh) if line not in abs_lines else (megatab[f"SNR_{line}"] < -sig_thresh)
    else:
        # If including upper limits, only require that the line is not significantly detected in the opposite direction
        mask &= (megatab[f"SNR_{line}"] > -sig_thresh) if line not in abs_lines else (megatab[f"SNR_{line}"] < sig_thresh)
    
    if line_prop in ["EW", "CONT_LUM"] and line_prop not in megatab.colnames:
        # Always require significant continuum for EW and continuum luminosity measurements
        mask &= (megatab[f"CONT_{line}"] / megatab[f"CONT_ERR_{line}"] > sig_thresh)
    elif line_prop in ["FWHM", "CVEL"]:
        mask &= line_col > 0  # Only consider positive FWHM and CVEL values to avoid unphysical results from poor fits
    
    # Only include unflagged lines
    mask &= (megatab[f"FLAG_{line}"] == '')
    # If fitting Lya EW, CONT, or continuum luminosity, require significant continuum detection to ensure reliable measurement
    if lya_prop in ["LYA_EW", "CONT", "EW", "LUM_CONT_LYA"]:
        mask &= (megatab['CONT'] / megatab['CONT_ERR'] > sig_thresh)
    
    # Only take positive Lya ASYMR values to focus on sources with stronger red peaks, which are more likely to have reliable Lya EW measurements and be less affected by IGM absorption.
    if lya_prop in ["ASYMR", "DELTAV_LYA", "FWHMR", "DISPR", "VEXP_ZELDA"]:
        mask &= lya_col > 0
    if lya_prop == 'ASYMR':
        # Mask outlier values of asymmetry (above 0.3)
        mask &= lya_col < 0.3

    if lya_prop in ["BRRATIO", "FLUXB", "ASYMB", "FWHMB", "DISPB"]:
        # Mask insignificant blue peaks
        mask &= (megatab["FLUXB"] / megatab["FLUXB_ERR"] > sig_thresh)

    return mask


from typing import Optional
from linmix import LinMix
from scipy.odr import ODR, Model, RealData
from scipy import stats
from scipy.stats import linregress

def get_mcmc_p_value(chain: np.ndarray) -> float:
    """
    Calculate a p-value from the MCMC posterior distribution of the slope parameter (beta) in LinMix.
    """
    # Calculate probability of positive/negative slope
    prob_positive = np.mean(chain['beta'] > 0)
    prob_negative = 1 - prob_positive

    # For two-tailed test, this is the probability of the opposite sign
    p_value = 2 * min(prob_positive, prob_negative)

    # Add a warning if p_value hits resolution limit
    min_possible_p = 2.0 / len(chain)
    if p_value <= min_possible_p * 1.1:  # Within 10% of minimum
        print(f"Warning: p-value ({p_value:.2e}) is at resolution limit. "
            f"Consider longer MCMC run for more precision. "
            f"All {len(chain)} posterior samples have the same sign.")

    return p_value

from matplotlib.axes import Axes

def do_linregress(x: np.ndarray, y: np.ndarray, x_err: np.ndarray, y_err: np.ndarray,
                  mcmc: bool = True, ax_in: Optional[Axes] = None, 
                  delta: Optional[np.ndarray] = None, log_transformed: bool = False) -> tuple[float, float, float, float, float]:
    """
    Perform linear regression using either the LinMix MCMC method, which accounts for measurement errors in both x and y, 
    or a simple ODR regression if MCMC is disabled.

    Parameters
    ----------
    x : np.ndarray
        The x-values of the data points.
    y : np.ndarray
        The y-values of the data points.
    x_err : np.ndarray
        The uncertainties in the x-values.
    y_err : np.ndarray
        The uncertainties in the y-values.
    mcmc : bool, optional
        Whether to perform MCMC regression using LinMix, by default True. If False, will
        perform a simple ODR regression.
    ax_in : matplotlib.axes.Axes, optional
        An optional matplotlib Axes object to plot the regression line and confidence interval on, by default None.
    delta : np.ndarray, optional
        An optional array indicating upper limits (1 for DETECTION, 0 for upper limit) to
        be passed to LinMix for proper handling of censored data, by default None.
    log_transformed : bool, optional
        Whether the data has been log-transformed, which affects how upper limits should be handled in
            LinMix, by default False.

    Returns
    -------
    tuple[float, float, float, float, float]
        The slope, slope uncertainty, intercept, intercept uncertainty, and p-value of the fit.
    """
    if mcmc:
        # If fitting with upper bounds in log-space, we need to translate all the y values up
        # by an arbitrary amount (10, corresponding to 10 dex) to ensure that upper limits
        # are properly handled in log-space without resulting in -inf values that can cause issues for LinMix
        shift_up = log_transformed and delta is not None and np.any(delta == 0)
        if shift_up:
            print("Data is log-transformed with upper limits, shifting y values up by 10 dex for LinMix...")
            y = y + 10  # Shift log-transformed values up by 10 dex to avoid -inf for upper limits
        
        lm = LinMix(x, y, xsig=x_err, ysig=y_err, delta=delta, K=2)

        try:
            lm.run_mcmc(silent=True)

            # Shift the intercept chain back down by 10 dex if log-transformed with upper limits to get the correct intercept values in log-space
            if shift_up:
                print("Shifting intercept chain back down by 10 dex for correct log-space values...")
                lm.chain['alpha'] -= 10
            
            slope = np.mean(lm.chain['beta'])
            sloperr = np.std(lm.chain['beta'])
            inter = np.mean(lm.chain['alpha'])
            intererr = np.std(lm.chain['alpha'])

            # Calculate p-value using the slope posterior distribution
            p_value = get_mcmc_p_value(lm.chain)
        except ValueError as e:
            print(f"LinMix MCMC failed to converge: {e}")
            return None, None, None, None, None
        # Plot the result by taking a sample of the posterior distribution of slopes and intercepts to show the confidence interval
        if ax_in is not None:
            x_fit = np.linspace(np.min(x), np.max(x), 10, endpoint=True)
            y_fit_samples = np.array([lm.chain[i]['alpha'] + lm.chain[i]['beta'] * x_fit for i in range(0, len(lm.chain), 25)])
            ax_in.plot(x_fit, y_fit_samples.T, color='red', alpha=0.01)

        # Quick check - create a new figure just for this (won't interfere with your main plot)
        fig_hist, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.hist(lm.chain['beta'], bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_title('Slope posterior')
        ax1.set_xlabel(r'$\beta$')
        ax1.axvline(0, color='r', ls='--', label='Zero')
        ax1.legend()
        ax2.hist(lm.chain['alpha'], bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        ax2.set_title('Intercept posterior')
        ax2.set_xlabel(r'$\alpha$')
        
        return slope, inter, sloperr, intererr, p_value  # Return order: slope, intercept, slope_err, intercept_err, p_value

    else:
        # Perform ODR regression as a fallback if MCMC is disabled
        def linear_model(B, x):
            return B[0] * x + B[1]
        data = RealData(x, y, sx=x_err, sy=y_err)
        model = Model(linear_model)
        odr = ODR(data, model, beta0=[0., 1.])
        output = odr.run()
        slope, intercept = output.beta
        slope_err, intercept_err = output.sd_beta
        # Calculate p-value using the t-statistic for the slope
        t_stat = slope / slope_err if slope_err > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=len(x) - 2))  # Two-tailed test
        return slope, slope_err, intercept, intercept_err, p_value


        
_log_quantities = ["LYA_EW", "FWHMR", "DISPR", 
                   "CONT", "EW", "FWHM", "DISP", 
                   "VEXP_ZELDA", "BRSEP", "BRRATIO",
                   "LUM", "CONT_LUM", "LUM_LYA", "LUM_CONT_LYA"]

def check_line_correlations(line_property: str, lya_properties: list[str], lines: list[str],
                       abs_lines: list[str], megatab: Table, min_points: int = 10,
                       significance_thresh: float = 0.01, mcmc: bool = True, 
                       fit_upper_limits: bool = False, upper_limit: float = 3.0,
                       logify: bool = False, plot_upper_limits: bool = False,
                       save_fig: bool = False, point_sig_thresh: float = 3.0,
                       c: Optional[str] = 'z', clip_extreme_errors: Optional[float] = None,
                       **scatter_kwargs) -> dict:
    """
    Check for correlations between a given line property (e.g. EW, FWHM) and a list of Lyman alpha properties.

    Parameters
    ----------
    line_property : str
        The line property to check (e.g. "EW", "FWHM").
    lya_properties : list[str]
        List of Lyman alpha properties to check against (e.g. ["LYA_EW", "DISPR", "CONT", "ASYMR", 
        "FWHMR", "BRRATIO", "BRSEP"]).
    lines : list[str]
        List of lines to check (e.g. ["SiII1260", "CII1334", "SiIV1394", "SiIV1403", "CIV1548", 
        "HeII1640", "OIII1660", "CIII1907"]).
    abs_lines : list[str]
        List of lines that should be treated as absorption (e.g. ["SiII1260", "CII1334", 
        "SiIV1394", "SiIV1403"]).
    megatab : astropy.table.Table
        The megatable containing the data.
    min_points : int, optional
        Minimum number of points required to attempt fitting a correlation, by default 10.
    significance_thresh : float, optional
        P-value (or Bayesian equivalent) threshold for determining significant correlations, by default 0.01.
    mcmc : bool, optional
        Whether to use MCMC regression for fitting the correlation, by default True.
    fit_upper_limits : bool, optional
        Whether to attempt to fit upper limits for non-detections, by default False.
    upper_limit : float, optional
        The sigma level to use for upper limits, by default 3.0.
    logify : bool, optional
        Whether to log-transform the line property for fitting, by default False.
    plot_upper_limits : bool, optional
        Whether to plot upper limits for non-detections, by default False.
    save_fig : bool, optional
        Whether to save the figure, by default False.
    point_sig_thresh : float, optional
        The significance threshold (in sigma) for plotting individual points, by default 3.0
    c : str, optional
        Name of the ``megatab`` column used to colour scatter-plot points. Default is ``'z'``
        (redshift). Pass ``None`` to disable point colouring.
    clip_extreme_errors : Optional[float], optional
        The threshold for clipping extreme error values in terms of the standard deviation of
        the corresponding data, by default None (no clipping).
    **scatter_kwargs
        Additional keyword arguments forwarded to :func:`make_scatter` (e.g. ``cnorm='log'``,
        ``cmap``, ``vmin``, ``vmax``, ``show_colorbar``, ``clabel``).

    Returns
    -------
    dict
        A dictionary containing the correlation summaries for each line and Lyman alpha property.
    """
    summaries = {}
    for line in lines:
        summaries[line] = {}
        for lya_prop in lya_properties:
            lya_col, lya_col_err = get_lya_property(megatab, lya_prop)

            # Prepare line EW column
            line_col, line_err = get_line_property(megatab, line, line_property, abs=line in abs_lines)

            # Prepare mask
            mask = prepare_scatter_mask(megatab, line, line_col, line_property, lya_col, lya_prop,
                                               abs_lines, include_upper_limits=fit_upper_limits,
                                               sig_thresh=point_sig_thresh)
            
            # x-axis is Lya property, y-axis is line property
            x_vals = lya_col[mask]
            y_vals = line_col[mask]
            x_errs = lya_col_err[mask]
            y_errs = line_err[mask]

            # Optionally clip points with extreme error bars before log-transform
            if clip_extreme_errors is not None:
                x_scatter = np.std(x_vals)
                y_scatter = np.std(y_vals)
                error_mask = (x_errs < clip_extreme_errors * x_scatter) & (y_errs < clip_extreme_errors * y_scatter)
                if np.sum(error_mask) < min_points:
                    print(f"Warning: Clipping extreme errors with threshold {clip_extreme_errors}\u03c3 leaves fewer than "
                          f"{min_points} points for {line} {line_property} vs {lya_prop}. Skipping correlation analysis.")
                    continue
                x_vals = x_vals[error_mask]
                y_vals = y_vals[error_mask]
                x_errs = x_errs[error_mask]
                y_errs = y_errs[error_mask]
                print(f"Clipped extreme errors with threshold {clip_extreme_errors}\u03c3, leaving {len(x_vals)} points "
                      f"for {line} {line_property} vs {lya_prop}.")
                mask[mask] = mask[mask] & error_mask

            # Re-initialise detection mask after any clipping so its length matches y_vals
            det_mask = np.ones_like(y_vals, dtype=bool)
            
            # Generate a boolean array for the upper limits and replace any such values with 
            # the specified sigma upper limit value
            nondet_mask = np.zeros_like(y_vals, dtype=bool)
            if fit_upper_limits:
                 # Identify non-detections based on the specified sigma threshold
                nondet_mask = y_vals < upper_limit * y_errs
                # Identify detections based on the specified sigma threshold
                det_mask = y_vals >= upper_limit * y_errs
                # Ensure that n_det + n_nondet = total number of points after masking
                assert np.sum(det_mask) + np.sum(nondet_mask) == len(y_vals), \
                        "Error in non-detection masking: number of detections + non-detections does not" \
                        " equal total number of points after masking."
                y_vals[~det_mask] = upper_limit * y_errs[~det_mask]  # Replace non-detections with specified sigma upper limit
 
            # Transform EWs and FWHMs/DISP into log space
            plot_log_x = False
            plot_log_y = False
            if lya_prop in _log_quantities and logify:
                x_vals_orig = x_vals.copy()  # Keep a copy of the original values for error transformation
                x_vals = np.log10(x_vals)
                x_errs = x_errs / (x_vals_orig * np.log(10))
                plot_log_x = True
            if line_property in _log_quantities and logify:
                y_vals_orig = y_vals.copy()  # Keep a copy of the original values for error transformation
                y_vals = np.log10(y_vals)
                y_errs = y_errs / (y_vals_orig * np.log(10))
                plot_log_y = True

            # Raise error if there are negative error bars in log space, which can happen if there are negative 
            # values or values consistent with zero.
            if np.any(x_errs < 0) or np.any(y_errs < 0):
                print(f"Warning: Negative error bars detected for {line} {line_property} vs {lya_prop}. Skipping"
                    " correlation analysis.")
                print(x_errs[x_errs < 0], y_errs[y_errs < 0])
                continue

            # Make scatter plot
            fig, ax = plt.subplots(figsize=(6, 4))

            make_scatter(
                [x_vals, x_errs],
                [y_vals, y_errs],
                ax=ax,
                c=megatab[c][mask] if c is not None else None,
                upper_bounds = nondet_mask,
                **scatter_kwargs
            )

            # Only attempt to fit a line if there are enough valid points after masking to avoid unreliable
            #  fits and overinterpreting small number statistics
            if np.sum(mask) >= min_points:

                # Final check on data quality: look for NaNs, negative error bars, or infinite values
                if any(np.any(np.isnan(arr)) for arr in [x_vals, y_vals, x_errs, y_errs]):
                    print(f"Warning: NaN values detected in data for {line} {line_property} vs {lya_prop}. Skipping"
                        " correlation analysis.")
                    plt.close(fig)
                    continue
                if any(np.any(np.isinf(arr)) for arr in [x_vals, y_vals, x_errs, y_errs]):
                    print(f"Warning: Infinite values detected in data for {line} {line_property} vs {lya_prop}. Skipping"
                        " correlation analysis.")
                    plt.close(fig)
                    continue
                if np.any(x_errs < 0) or np.any(y_errs < 0):
                    print(f"Warning: Negative error bars detected for {line} {line_property} vs {lya_prop}. Skipping"
                        " correlation analysis.")
                    plt.close(fig)
                    continue

                # Do a preliminary fit to pre-screen for significant correlations before attempting the more computationally expensive MCMC fit
                _, _, _, prelim_p_value, _ = linregress(x_vals[det_mask], y_vals[det_mask])
                if prelim_p_value >= significance_thresh:
                    print(f"No significant correlation found for {line} {line_property} vs {lya_prop} (prelim p={prelim_p_value:.3e}). Skipping MCMC fit.")
                    plt.close(fig)  # Close the plot if not significant to avoid clutter
                    continue
                elif prelim_p_value < significance_thresh and mcmc:
                    print(f"Preliminary fit suggests significant correlation for {line} {line_property} vs "
                        f"{lya_prop} (prelim p={prelim_p_value:.3e}). Proceeding with MCMC fit.")
                elif prelim_p_value < significance_thresh and not mcmc:
                    print(f"Preliminary fit suggests significant correlation for {line} {line_property} vs "
                        f"{lya_prop} (prelim p={prelim_p_value:.3e}). Proceeding with ODR fit.")

                # Fit line to original data with MCMC: line_property (y) vs lya_property (x)
                slope, intercept, slope_err, intercept_err, p_value = do_linregress(x_vals, y_vals, x_errs, y_errs, 
                                                                                    delta=det_mask, mcmc=mcmc,
                                                                                    ax_in=ax, log_transformed=plot_log_y)
            
                # Plot best-fit line
                print(f"Correlation for {line} {line_property} vs {lya_prop}: slope={slope:.2f}±{slope_err:.2f}, "
                    f"intercept={intercept:.2f}±{intercept_err:.2f}, p={p_value:.3e}")
                x_fit = np.linspace(np.min(x_vals), np.max(x_vals), 100)
                y_fit = slope * x_fit + intercept
                ax.plot(x_fit, y_fit, color='red', 
                        label=f"Slope={slope:.2f}±{slope_err:.2f}\n"
                        f"Intercept={intercept:.2f}±{intercept_err:.2f}\n"
                        f"(p={p_value:.3e})")
                ax.legend()
            else:
                print(f"Not enough points to fit line for {line} {line_property} vs {lya_prop} (n={np.sum(mask)})")
                plt.close(fig)  # Close the plot if not enough points to avoid clutter
                continue


            ax.set_xlabel(f"{'log ' if plot_log_x else ''}{plot.get_plot_name(lya_prop)}")
            ax.set_ylabel(f"{'log ' if plot_log_y else ''}{plot.get_plot_name(line)} {plot.get_plot_name(line_property)}")
            ax.set_title(f"{plot.get_plot_name(line, unit=False)} {plot.get_plot_name(line_property, unit=False)} "
                         f"vs {plot.get_plot_name(lya_prop, unit=False)}")

            # ax.set_xscale('log') if plot_log_x else ax.set_xscale('linear')
            # if np.size(line_col[mask]) > 0:  # Only set yscale if there are valid points to avoid warnings
            #     ax.set_yscale('log') if plot_log_y else ax.set_yscale('linear')

            if save_fig:
                fig.savefig(f"plots/{line}_{line_property}_vs_{lya_prop}.png", dpi=300, bbox_inches='tight')

            plt.show()

            summaries[line][lya_prop] = {
                'slope': slope,
                'slope_err': slope_err,
                'intercept': intercept,
                'intercept_err': intercept_err,
                'p_value': p_value,
                'n_points': np.sum(mask)
            }
    return summaries


def check_lya_correlations(lya_properties_y: list[str], lya_properties_x: list[str],
                            megatab: Table, min_points: int = 10,
                            significance_thresh: float = 0.01, mcmc: bool = True,
                            logify: bool = False, save_fig: bool = False,
                            point_sig_thresh: float = 3.0, clip_extreme_errors: Optional[float] = None,
                            c: Optional[str] = 'z', **scatter_kwargs) -> dict:
    """
    Check for correlations between pairs of Lyman alpha properties.

    Both axes use ``get_lya_property`` for data retrieval and Lya-appropriate masking is applied
    to each axis independently.

    Parameters
    ----------
    lya_properties_y : list[str]
        Lyman alpha properties to place on the y-axis (e.g. ["FWHMR", "DISPR", "ASYMR"]).
    lya_properties_x : list[str]
        Lyman alpha properties to place on the x-axis (e.g. ["LYA_EW", "CONT", "BRRATIO"]).
    megatab : astropy.table.Table
        The megatable containing the data.
    min_points : int, optional
        Minimum number of points required to attempt fitting a correlation, by default 10.
    significance_thresh : float, optional
        P-value (or Bayesian equivalent) threshold for determining significant correlations, by default 0.01.
    mcmc : bool, optional
        Whether to use MCMC regression for fitting the correlation, by default True.
    logify : bool, optional
        Whether to log-transform quantities that appear in ``_log_quantities``, by default False.
    save_fig : bool, optional
        Whether to save the figure, by default False.
    point_sig_thresh : float, optional
        The significance threshold (in sigma) for source-level quality cuts, by default 3.0.
    clip_extreme_errors : Optional[float], optional
        The threshold for clipping extreme error values in terms of the standard deviation of
        the corresponding data, by default None (no clipping).
    c : str, optional
        Name of the ``megatab`` column used to colour scatter-plot points. Default is ``'z'``
        (redshift). Pass ``None`` to disable point colouring.
    **scatter_kwargs
        Additional keyword arguments forwarded to :func:`make_scatter` (e.g. ``cnorm='log'``,
        ``cmap``, ``vmin``, ``vmax``, ``show_colorbar``, ``clabel``).

    Returns
    -------
    dict
        A nested dictionary ``summaries[lya_prop_y][lya_prop_x]`` containing the correlation
        summaries (slope, intercept, errors, p-value, n_points) for each pair.
    """
    def _lya_mask(megatab, prop, col, sig_thresh):
        """Build a quality mask for a single Lya property column."""
        mask = np.isfinite(col)
        if prop in ["LYA_EW", "CONT", "EW", "LUM_CONT_LYA"]:
            mask &= (megatab['CONT'] / megatab['CONT_ERR'] > sig_thresh)
        if prop in ["ASYMR", "DELTAV_LYA", "FWHMR", "DISPR", "VEXP_ZELDA"]:
            mask &= col > 0
        if prop == 'ASYMR':
            mask &= col < 0.3
        if prop in ["BRRATIO", "FLUXB", "ASYMB", "FWHMB", "DISPB", "BRSEP"]:
            mask &= (megatab["FLUXB"] / megatab["FLUXB_ERR"] > sig_thresh)
        if prop in ["VEXP_ZELDA"]:
            mask &= (megatab[prop] - 3 * megatab['VEXP_ERRM_ZELDA'] > 0)
        return mask

    summaries = {}
    seen_pairs = set()
    for prop_y in lya_properties_y:
        summaries[prop_y] = {}
        for prop_x in lya_properties_x:
            # Skip trivial self-correlations
            if prop_x == prop_y:
                print(f"Skipping trivial self-correlation: {prop_y} vs {prop_x}.")
                continue
            # Skip duplicate pairs (e.g. (A, B) already computed as (B, A))
            pair = frozenset((prop_x, prop_y))
            if pair in seen_pairs:
                print(f"Skipping duplicate pair: {prop_y} vs {prop_x}.")
                continue
            seen_pairs.add(pair)

            x_col, x_col_err = get_lya_property(megatab, prop_x)
            y_col, y_col_err = get_lya_property(megatab, prop_y)

            mask = _lya_mask(megatab, prop_x, x_col, point_sig_thresh) \
                 & _lya_mask(megatab, prop_y, y_col, point_sig_thresh)
            
            mask &= np.isfinite(x_col) & np.isfinite(y_col)  # Ensure both x and y are finite for valid points
            mask &= np.isfinite(x_col_err) & np.isfinite(y_col_err)  # Ensure error bars are finite for valid points

            x_vals = x_col[mask]
            y_vals = y_col[mask]
            x_errs = x_col_err[mask]
            y_errs = y_col_err[mask]

            # Optionally clip points which have error bars so large that it exceeds the scatter of the actual
            # data itself, in which case they are not very informative for the correlation analysis and can cause issues for MCMC convergence
            if clip_extreme_errors is not None:
                x_scatter = np.std(x_vals)
                y_scatter = np.std(y_vals)
                error_mask = (x_errs < clip_extreme_errors * x_scatter) & (y_errs < clip_extreme_errors * y_scatter)
                if np.sum(error_mask) < min_points:
                    print(f"Warning: Clipping extreme errors with threshold {clip_extreme_errors}σ leaves fewer than {min_points} points for {prop_y} vs {prop_x}. Skipping correlation analysis.")
                    continue
                x_vals = x_vals[error_mask]
                y_vals = y_vals[error_mask]
                x_errs = x_errs[error_mask]
                y_errs = y_errs[error_mask]
                print(f"Clipped extreme errors with threshold {clip_extreme_errors}σ, leaving {len(x_vals)} points for {prop_y} vs {prop_x}.")
                mask[mask] = mask[mask] & error_mask  # Update the original mask to reflect the clipping for accurate n_points count in summaries

            # Transform into log space if requested
            plot_log_x = False
            plot_log_y = False
            if prop_x in _log_quantities and logify:
                x_vals_orig = x_vals.copy()
                x_vals = np.log10(x_vals)
                x_errs = x_errs / (x_vals_orig * np.log(10))
                plot_log_x = True
            if prop_y in _log_quantities and logify:
                y_vals_orig = y_vals.copy()
                y_vals = np.log10(y_vals)
                y_errs = y_errs / (y_vals_orig * np.log(10))
                plot_log_y = True

            if np.any(x_errs < 0) or np.any(y_errs < 0):
                print(f"Warning: Negative error bars detected for {prop_y} vs {prop_x}. Skipping"
                      " correlation analysis.")
                print(x_errs[x_errs < 0], y_errs[y_errs < 0])
                continue

            # Make scatter plot
            fig, ax = plt.subplots(figsize=(6, 4))

            make_scatter(
                [x_vals, x_errs],
                [y_vals, y_errs],
                ax=ax,
                c=megatab[c][mask] if c is not None else None,
                **scatter_kwargs
            )

            if np.sum(mask) >= min_points:

                # Final check on data quality
                if any(np.any(np.isnan(arr)) for arr in [x_vals, y_vals, x_errs, y_errs]):
                    print(f"Warning: NaN values detected in data for {prop_y} vs {prop_x}. Skipping"
                          " correlation analysis.")
                    plt.close(fig)
                    continue
                if any(np.any(np.isinf(arr)) for arr in [x_vals, y_vals, x_errs, y_errs]):
                    print(f"Warning: Infinite values detected in data for {prop_y} vs {prop_x}. Skipping"
                          " correlation analysis.")
                    plt.close(fig)
                    continue
                if np.any(x_errs < 0) or np.any(y_errs < 0):
                    print(f"Warning: Negative error bars detected for {prop_y} vs {prop_x}. Skipping"
                          " correlation analysis.")
                    plt.close(fig)
                    continue

                # Preliminary OLS screen
                _, _, _, prelim_p_value, _ = linregress(x_vals, y_vals)
                if prelim_p_value >= significance_thresh:
                    print(f"No significant correlation found for {prop_y} vs {prop_x} "
                          f"(prelim p={prelim_p_value:.3e}). Skipping MCMC fit.")
                    plt.close(fig)
                    continue
                elif prelim_p_value < significance_thresh and mcmc:
                    print(f"Preliminary fit suggests significant correlation for {prop_y} vs "
                          f"{prop_x} (prelim p={prelim_p_value:.3e}). Proceeding with MCMC fit.")
                else:
                    print(f"Preliminary fit suggests significant correlation for {prop_y} vs "
                          f"{prop_x} (prelim p={prelim_p_value:.3e}). Proceeding with ODR fit.")

                slope, intercept, slope_err, intercept_err, p_value = do_linregress(
                    x_vals, y_vals, x_errs, y_errs, mcmc=mcmc,
                    ax_in=ax, log_transformed=plot_log_y)

                print(f"Correlation for {prop_y} vs {prop_x}: slope={slope:.2f}±{slope_err:.2f}, "
                      f"intercept={intercept:.2f}±{intercept_err:.2f}, p={p_value:.3e}")
                x_fit = np.linspace(np.min(x_vals), np.max(x_vals), 100)
                y_fit = slope * x_fit + intercept
                ax.plot(x_fit, y_fit, color='red',
                        label=f"Slope={slope:.2f}±{slope_err:.2f}\n"
                              f"Intercept={intercept:.2f}±{intercept_err:.2f}\n"
                              f"(p={p_value:.3e})")
                ax.legend()
            else:
                print(f"Not enough points to fit line for {prop_y} vs {prop_x} (n={np.sum(mask)})")
                plt.close(fig)
                continue

            ax.set_xlabel(f"{'log ' if plot_log_x else ''}{plot.get_plot_name(prop_x)}")
            ax.set_ylabel(f"{'log ' if plot_log_y else ''}{plot.get_plot_name(prop_y)}")
            ax.set_title(f"{plot.get_plot_name(prop_y, unit=False)} vs {plot.get_plot_name(prop_x, unit=False)}")

            if save_fig:
                fig.savefig(f"plots/{prop_y}_vs_{prop_x}.png", dpi=300, bbox_inches='tight')

            plt.show()

            summaries[prop_y][prop_x] = {
                'slope': slope,
                'slope_err': slope_err,
                'intercept': intercept,
                'intercept_err': intercept_err,
                'p_value': p_value,
                'n_points': np.sum(mask)
            }
    return summaries