"""
Tools for performing statistical analysis on spectra and tables.
"""

import matplotlib

from .catalogue_operations import generate_source_mask
from . import constants as const
from . import spectroscopy as spectro
from . import models
from . import fitting
from . import plotting as plot

import warnings
import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from scipy.stats import ks_2samp

from typing import Optional, Union

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
        Boolean mask identifying which points are upper limits. Those points are
        plotted as downward-pointing triangles at their face value (which should
        already be the upper-limit value, e.g. from ``_insert_upper_limits``).
        No y error bar manipulation is applied. Default is None (no upper limits).
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
    # Upper limits: plot as downward-pointing triangles at the upper-bound value.
    # The y value is already the upper limit; no y error bar manipulation is applied.
    if np.any(upper_bounds):
        ax.errorbar(xvals[upper_bounds], yvals[upper_bounds],
                    xerr=[xerrsm[upper_bounds], xerrsp[upper_bounds]],
                    yerr=np.zeros((2, np.sum(upper_bounds))),
                    marker='v', color='red', zorder=0,
                    alpha=alpha_ubs, linestyle='',
                    uplims=np.ones(np.sum(upper_bounds), dtype=bool))

    if show_colorbar and np.any(valid_c_det):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(sc, cax=cax)
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
        

from astropy.table import Table
from .source_properties import (
    get_line_property, get_lya_property, normalise_prop,
    _log_quantities, _known_line_tokens, flux_to_luminosity,
)


def _effective_snr(megatab: Table, line: str, abs_lines: list[str],
                   line_prop: str = "EW", combine_doublets: bool = True) -> np.ndarray:
    """
    Return the signed effective SNR for each source for a given line.

    For doublet lines when ``combine_doublets`` is True and ``line_prop`` is an
    additive property (EW, FLUX, LUM), the quadrature-combined SNR of both
    components is returned, signed according to the titular line's SNR sign.
    For all other cases the titular line's SNR column is returned directly.

    The sign convention is: positive for emission detections, negative for
    absorption detections.  A threshold test against ±sig_thresh on the
    returned array is therefore always of the form::

        detected = snr > +sig_thresh   # emission
        detected = snr < -sig_thresh   # absorption

    Parameters
    ----------
    megatab : astropy.table.Table
        The megatable.
    line : str
        Titular line name (e.g. ``'CIV1548'``, ``'SiII1260'``).
    abs_lines : list[str]
        Lines treated as absorption.
    line_prop : str, optional
        The property being analysed.  Only matters for deciding whether the
        doublet quadrature path applies (additive properties). Default ``'EW'``.
    combine_doublets : bool, optional
        Whether doublet components are being combined. Default True.

    Returns
    -------
    np.ndarray
        Signed effective SNR, one value per source.
    """
    snr_col = f"SNR_{line}"
    snr = np.asarray(megatab[snr_col], dtype=float)

    _use_doublet_snr = (combine_doublets and line in const.doublets
                        and line_prop in ("EW", "FLUX", "LUM", "FWHM", "CVEL"))
    if _use_doublet_snr:
        _line2 = const.doublets[line][1]
        snr2_col = f"SNR_{_line2}"
        if snr2_col in megatab.colnames:
            snr2 = np.asarray(megatab[snr2_col], dtype=float)
            snr = np.sign(snr) * np.sqrt(snr**2 + snr2**2)

    return snr


def _prepare_scatter_mask(megatab: Table, line: str, line_col: np.ndarray, line_prop: str,
                         lya_col: np.ndarray, lya_prop: str, abs_lines: list[str],
                         include_upper_limits: bool = False, sig_thresh: float = 3.0,
                         combine_doublets: bool = True,
                         delta: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Prepare a mask for scatter plot analysis between a given line and Lyman alpha property.

    Parameters
    ----------
    megatab : astropy.table.Table
        The megatable containing the data.
    line : str
        The line to analyze (e.g., "SiII1260").
    line_col : np.ndarray
        The column for the line property (e.g., equivalent width or FWHM).  When
        ``delta`` is supplied this should already be the upper-limit-substituted
        array returned by ``_insert_upper_limits``.
    line_prop : str
        The line property being analyzed (e.g., "EW", "FWHM").
    lya_col : np.ndarray
        The Lyman alpha property column.
    lya_prop : str
        The Lyman alpha property to analyze (e.g., "EW_LYA").
    abs_lines : list[str]
        List of lines that should be treated as absorption.
    include_upper_limits : bool, optional
        Whether to include upper limits for non-detections. Ignored when
        ``delta`` is provided (the presence of ``delta`` implies upper limits
        are already handled). Default False.
    sig_thresh : float, optional
        The significance threshold (in sigma) for including sources based on
        their SNR. Default 3.0.
    combine_doublets : bool, optional
        Whether doublet lines are being combined. When True and ``line`` is the
        titular line of a doublet, the SNR threshold uses the
        quadrature-combined SNR of both components and flags on *both*
        components must be clear for additive properties (EW/FLUX/LUM).
        Default True.
    delta : np.ndarray of int, optional
        Detection indicator array from ``_insert_upper_limits`` (1 = detected,
        0 = upper limit).  When provided, the SNR cut is replaced by
        ``delta == 1``, and sources with ``delta == 0`` (upper limits) are kept
        in the mask provided they pass all other quality cuts (continuum SNR,
        flags, Lya quality).  Default None.

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

    is_abs = line in abs_lines

    # SNR threshold — skip for sources already classified as upper limits by delta
    if delta is not None:
        # Upper-limit sources (delta==0) are admitted regardless of SNR; detections
        # must still pass the SNR cut so we don't admit noisy non-upper-limit sources.
        snr = _effective_snr(megatab, line, abs_lines,
                             line_prop=line_prop, combine_doublets=combine_doublets)
        snr_ok = (snr < -sig_thresh) if is_abs else (snr > sig_thresh)
        mask &= snr_ok | (delta == 0)
    elif not include_upper_limits:
        snr = _effective_snr(megatab, line, abs_lines,
                             line_prop=line_prop, combine_doublets=combine_doublets)
        mask &= (snr < -sig_thresh) if is_abs else (snr > sig_thresh)

    if line_prop in ["EW", "CONT_LUM", "CONT"] and line_prop not in megatab.colnames:
        # Always require significant continuum for EW and continuum luminosity measurements
        mask &= (megatab[f"CONT_{line}"] / megatab[f"CONT_ERR_{line}"] > 3)
    elif line_prop in ["FWHM"]:
        mask &= line_col > 0  # Only consider positive FWHM values to avoid unphysical results from poor fits

    # Flag quality: for doublets with combine_doublets, both component flags must be clear.
    # Upper-limit sources (delta==0) bypass flag cuts — the fit flag reflects a poor
    # detection fit, which is irrelevant when we are only using the source as a census point.
    is_upper_limit = (delta == 0) if delta is not None else np.zeros(len(mask), dtype=bool)
    flag_col = f"FLAG_{line}"
    if flag_col in megatab.colnames:
        mask &= (megatab[flag_col] == '') | is_upper_limit
    if combine_doublets and line in const.doublets:
        _line2_flag = f"FLAG_{const.doublets[line][1]}"
        if _line2_flag in megatab.colnames:
            mask &= (megatab[_line2_flag] == '') | is_upper_limit

    # The remaining cuts are Lya-specific. Guard them so they don't fire when
    # _prepare_scatter_mask is called from check_line_line_correlations with a
    # non-Lya property name in the lya_prop slot.
    _lya_props = {"EW_LYA", "CONT_LUM_LYA", "ASYMR", "DELTAV_LYA",
                  "FWHMR", "DISPR", "VEXP_ZELDA", "BRRATIO", "FLUXB", "ASYMB",
                  "FWHMB", "DISPB", "BRSEP"}
    if lya_prop not in _lya_props:
        return mask

    # If fitting Lya EW, CONT, or continuum luminosity, require significant continuum detection to ensure reliable measurement
    if lya_prop in ["EW_LYA", "CONT", "EW", "CONT_LUM_LYA"]:
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
        mask &= megatab['z'] < 4 # At z>4, the blue peak is often completely absorbed by the IGM, so we exclude those sources when analyzing blue peak properties to avoid biasing the results with unreliable measurements.

    return mask


from typing import Optional, Union
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
                  niter: int = 5000, delta: Optional[np.ndarray] = None) -> tuple[float, float, float, float, float, float, float]:
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
    niter : int, optional
        The minimum number of MCMC iterations per chain to run in LinMix, by default 5000. The posterior
        chain stored after convergence contains approximately ``nchains * niter / 2`` samples (4 chains
        by default, so ~10,000 samples at the default). Increase this value to obtain a denser posterior
        for reliable detection of weak signals (e.g. use 25000 for ~50,000 samples).
    delta : np.ndarray, optional
        Optional array of 0/1 values indicating which data points are upper limits (1 for detections, 0 for upper limits)
        Should have the same length as x and y. Default is None (no upper limits).

    Returns
    -------
    tuple[float, float, float, float, float, float, float]
        The slope, slope uncertainty, intercept, intercept uncertainty, LinMix/ODR p-value,
        Spearman rho, and Spearman rho p-value.
    """
    if mcmc:
        rho, rho_p = stats.spearmanr(x, y)

        lm = LinMix(x, y, xsig=x_err, ysig=y_err, K=2, delta=delta)

        _mcmc_exc: list[Optional[Exception]] = [None]
        _done = threading.Event()

        def _run_mcmc():
            try:
                lm.run_mcmc(miniter=niter, maxiter=max(niter, 100000), silent=True)
            except Exception as e:
                _mcmc_exc[0] = e
            finally:
                _done.set()

        _thread = threading.Thread(target=_run_mcmc, daemon=True)
        _thread.start()
        _completed = _done.wait(timeout=600)

        if not _completed:
            warnings.warn(
                "LinMix MCMC hung (likely due to numerical instability — overflow in "
                "covariance matrix or NaN probabilities in multivariate sampling). "
                "Returning None.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None, None, None, None, None, None, None

        if _mcmc_exc[0] is not None:
            warnings.warn(
                f"LinMix MCMC failed: {_mcmc_exc[0]}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None, None, None, None, None, None, None

        slope = np.mean(lm.chain['beta'])
        sloperr = np.std(lm.chain['beta'])
        slope_median = np.median(lm.chain['beta'])
        slope_16th = np.percentile(lm.chain['beta'], 16)
        slope_84th = np.percentile(lm.chain['beta'], 84)
        inter = np.mean(lm.chain['alpha'])
        intererr = np.std(lm.chain['alpha'])
        inter_median = np.median(lm.chain['alpha'])
        inter_16th = np.percentile(lm.chain['alpha'], 16)
        inter_84th = np.percentile(lm.chain['alpha'], 84)

        # Calculate posterior probability of positive/negative slope and convert to a two-tailed p-value
        p_value = get_mcmc_p_value(lm.chain)
        posterior_probability = 1 - p_value
        # Plot the posterior predictive band: evaluate y = alpha + beta*x for every
        # chain sample, then show the 16th–84th percentile envelope with fill_between.
        if ax_in is not None:
            x_fit = np.linspace(np.min(x - x_err), np.max(x + x_err), 200)
            # Shape: (n_chain, n_x)
            y_samples = lm.chain['alpha'][:, None] + lm.chain['beta'][:, None] * x_fit[None, :]
            y_lo  = np.percentile(y_samples, 16, axis=0)
            y_hi  = np.percentile(y_samples, 84, axis=0)
            y_med = np.median(y_samples, axis=0)
            ineq = '>' if slope > 0 else '<'
            _fit_label = (f"slope $={slope_median:.4g}^{{+{slope_84th - slope_median:.4g}}}_{{-{slope_median - slope_16th:.4g}}}$\n"
                        #   f"$\\alpha={inter_median:.2f}^{{+{inter_84th - inter_median:.2f}}}_{{-{inter_median - inter_16th:.2f}}}$\n"
                          r"$ P(\mathrm{slope}"+f" {ineq} 0)={posterior_probability:.3f}$\n"
                          f"Spearman $\\rho={rho:.3f}$")
            ax_in.plot(x_fit, y_med, color='red', lw=1.5, alpha=0.75)
            ax_in.fill_between(x_fit, y_lo, y_hi, color='red', alpha=0.25,
                               label=_fit_label)
            ax_in.legend(framealpha=0.6, fancybox=True, loc='upper right')

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

        return slope, inter, sloperr, intererr, p_value, rho, rho_p # Return order: slope, intercept, slope_err, intercept_err, p_value, rho, rho_p

    else:
        # Perform ODR regression as a fallback if MCMC is disabled
        # Raise a warning and mask non-detections if delta is provided, since ODR does not natively handle upper limits
        if delta is not None:
            warnings.warn(
                "ODR regression does not natively handle upper limits. Masking non-detections based on provided delta array.",
                RuntimeWarning,
                stacklevel=2,
            )
            mask = delta == 1
            x = x[mask]
            y = y[mask]
            x_err = x_err[mask]
            y_err = y_err[mask]
        def linear_model(B, x):
            return B[0] * x + B[1]
        data = RealData(x, y, sx=x_err, sy=y_err)
        model = Model(linear_model)
        odr = ODR(data, model, beta0=[0., 1.])
        output = odr.run()
        rho, rho_p = stats.spearmanr(x, y)
        slope, intercept = output.beta
        slope_err, intercept_err = output.sd_beta
        # Calculate p-value using the t-statistic for the slope
        t_stat = slope / slope_err if slope_err > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=len(x) - 2))  # Two-tailed test
        if ax_in is not None:
            x_fit = np.linspace(np.min(x), np.max(x), 200)
            _fit_label = (f"Slope={slope:.2f}\u00b1{slope_err:.2f}\n"
                          f"Intercept={intercept:.2f}\u00b1{intercept_err:.2f}\n"
                          f"(p={p_value:.3e})\n"
                          f"$\\rho_s={rho:.3f}$, $p_s={rho_p:.3e}$")
            ax_in.plot(x_fit, slope * x_fit + intercept, color='red', lw=1.5,
                       label=_fit_label)
            ax_in.legend(framealpha=0.6, fancybox=True)
        return slope, slope_err, intercept, intercept_err, p_value, rho, rho_p




def ks_test_distributions(groups: list[np.ndarray],
                          labels: Optional[list[str]] = None) -> dict:
    """
    Perform pairwise two-sample Kolmogorov-Smirnov tests on a list of value arrays.

    Parameters
    ----------
    groups : list of np.ndarray
        Value arrays, one per group (NaN/inf already removed).
    labels : list of str, optional
        Names for each group. Defaults to ``'Group 0'``, ``'Group 1'``, …

    Returns
    -------
    dict
        Mapping ``(label_i, label_j)`` → ``{'statistic': float, 'p_value': float}``
        for every unique pair ``i < j``.
    """
    from scipy.stats import ks_2samp

    n = len(groups)
    if labels is None:
        labels = [f"Group {i}" for i in range(n)]
    results = {}
    for i in range(n):
        for j in range(i + 1, n):
            if len(groups[i]) == 0 or len(groups[j]) == 0:
                results[(labels[i], labels[j])] = {'statistic': np.nan, 'p_value': np.nan}
                continue
            stat, p = ks_2samp(groups[i], groups[j])
            results[(labels[i], labels[j])] = {'statistic': stat, 'p_value': p}
    return results


def ks_test_lya_mc(line: str, lya_prop: str, megatab: Table, line_property: str = "EW",
                   abs_lines: Optional[list] = None, threshold: Union[float, str] = 'auto',
                   significance_thresh: float = 3.0) -> tuple[float, float]:
    """
    Perform a KS test comparing the distribution of a Lyman alpha property
    (e.g. DISPR, FWHMR, ASYMR) for sources with high vs low values of a given line property
    (e.g. EW). Non-detections are included in the low-value group when their upper limit
    falls below the threshold.

    Parameters
    ----------
    line : str
        The line to analyse (e.g. ``"SiII1260"``).
    lya_prop : str
        The Lyman alpha property to split on (e.g. ``"DISPR"``).
    megatab : astropy.table.Table
        The megatable containing the data.
    line_property : str, optional
        The line property used for splitting the sample (e.g. ``"EW"``), by default ``"EW"``.
    abs_lines : list, optional
        Lines that should be treated as absorption (SNR sign flipped). Default is ``[]``.
    threshold : float or str, optional
        Value used to split the sample into high/low groups. If ``'auto'``, the median
        detection value is used, by default ``'auto'``.
    significance_thresh : float, optional
        SNR threshold (in sigma) for classifying a source as a detection, by default 3.0.

    Returns
    -------
    tuple[float, float]
        The KS statistic and p-value for the test.
    """
    from scipy.stats import ks_2samp

    if abs_lines is None:
        abs_lines = []

    # Retrieve columns
    line_col, line_err = get_line_property(megatab, line, line_property, abs=line in abs_lines)
    lya_col, lya_col_err = get_lya_property(megatab, lya_prop)

    # Valid-data mask (include upper limits so non-detections are retained)
    mask = _prepare_scatter_mask(megatab, line, line_col, line_property, lya_col, lya_prop,
                                 abs_lines, include_upper_limits=True,
                                 sig_thresh=significance_thresh)

    snr = np.asarray(megatab[f"SNR_{line}"], dtype=float)
    if line in abs_lines:
        significance_mask = snr < -significance_thresh
    else:
        significance_mask = snr > significance_thresh

    detections   = mask & significance_mask
    upper_limits = mask & ~significance_mask

    # Replace non-detections with 3-sigma upper bounds
    line_col = np.array(line_col, dtype=float)
    line_col[upper_limits] = 3.0 * np.asarray(line_err, dtype=float)[upper_limits]

    # Determine splitting threshold
    if threshold == 'auto':
        median_line = np.nanmedian(line_col[detections])
        print(f"Using median {line} {line_property} value of {median_line:.2f} for splitting groups.")
    else:
        median_line = threshold

    group_high = lya_col[detections & (line_col >= median_line)]
    group_low  = lya_col[(detections & (line_col < median_line)) | (upper_limits & (line_col < median_line))]

    # Log-transform if appropriate
    if lya_prop in _log_quantities:
        group_high = np.log10(group_high)
        group_low  = np.log10(group_low)

    if len(group_high) == 0 or len(group_low) == 0:
        print(f"Warning: one of the groups for {line} {line_property} vs {lya_prop} is empty. "
              "Cannot perform KS test.")
        return np.nan, np.nan

    ks_statistic, p_value = ks_2samp(group_high, group_low)

    if p_value < 0.05:
        print(f"Significant difference in {lya_prop} distribution for high vs low "
              f"{line} {line_property} (KS p={p_value:.3e}).")
    else:
        print(f"No significant difference in {lya_prop} distribution for high vs low "
              f"{line} {line_property} (KS p={p_value:.3e}).")
        return ks_statistic, p_value

    print(f"KS test for {line} {line_property} vs {lya_prop}: "
          f"KS statistic={ks_statistic:.3f}, p-value={p_value:.3e}")

    n_bins = 20
    bins = np.linspace(min(np.min(group_high), np.min(group_low)),
                       max(np.max(group_high), np.max(group_low)), n_bins)

    plt.figure(figsize=(6, 4))
    plt.hist(group_high, bins=bins, alpha=0.7,
             label=f"{line} {line_property} $\\geq {median_line:.2f}$",
             color='steelblue', edgecolor='black')
    plt.hist(group_low, bins=bins, alpha=0.7,
             label=f"{line} {line_property} $< {median_line:.2f}$",
             color='salmon', edgecolor='black')
    plt.xlabel(plot.get_plot_name(lya_prop))
    plt.ylabel("Number of sources")
    plt.title(f"{line} {line_property} vs {lya_prop}\nKS p-value={p_value:.3e}")
    plt.legend()
    plt.savefig(f"plots/{line}_{line_property}_vs_{lya_prop}_KS_hist.png",
                dpi=300, bbox_inches='tight')
    plt.show()

    return ks_statistic, p_value


def generate_histogram(prop: str, megatab: Table, masks: Optional[list] = None,
                       labels: Optional[list] = None, colors: Optional[list] = None,
                       line: Optional[str] = None, ax: Optional[plt.Axes] = None,
                       logify: bool = True, bins: Union[int, str, np.ndarray] = 'auto',
                       density: bool = False, plot_median: bool = True,
                       sig_thresh: float = 3.0, alpha: float = 0.6,
                       combine_doublets: bool = True,
                       abs_lines: Optional[list] = None,
                       ks_test: bool = False,
                       hist_kwargs: Union[dict, list, None] = None) -> plt.Axes:
    """
    Plot one or more overlaid histograms of a named property, handling derived
    quantities (Lya properties, line properties, luminosities) automatically.

    Parameters
    ----------
    prop : str
        Property to histogram. Interpreted as follows:

        - A Lya property recognised by :func:`get_lya_property` (e.g. ``'EW_LYA'``,
          ``'DISPR'``, ``'LUM_LYA'``, ``'CONT_LUM_LYA'``).
        - A line property recognised by :func:`get_line_property` when ``line`` is
          also supplied (e.g. ``prop='EW'``, ``line='CIV1548'``).
        - A combined ``"{param}_{line}"`` string (e.g. ``'EW_CIV1548'``,
          ``'CONT_LUM_CIV1548'``). The line name is inferred automatically by trying
          right-to-left splits on ``_``.
        - Any column name present directly in ``megatab``.

    megatab : astropy.table.Table
        The source catalogue.
    masks : list of array-like, optional
        List of boolean masks, one per group to plot. Each mask selects a subset of
        ``megatab``. If ``None``, a single group containing all finite, valid rows is used.
    labels : list of str, optional
        Legend labels, one per group. Defaults to ``None`` (no labels).
    colors : list of str, optional
        Colours, one per group. Defaults to the matplotlib colour cycle.
    line : str, optional
        Line name required when ``prop`` is a line property (e.g. ``'CIV1548'``).
        Ignored for Lya properties and plain column names.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw. A new figure and axes are created if ``None``.
    logify : bool, optional
        If ``True`` (default) and ``prop`` is in ``_log_quantities``, apply a
        log10 transform before binning. The x-axis label is prefixed with ``log ``.
    bins : int, str, or array-like, optional
        Number of bins, a numpy bin-selection string (e.g. ``'auto'``, ``'fd'``,
        ``'scott'``), or explicit bin edges shared across all groups.
        Default is ``'auto'`` (numpy Freedman-Diaconis / Sturges selector).
    density : bool, optional
        Normalise histograms to form a probability density. Default is ``False``.
    plot_median : bool, optional
        Draw a vertical dashed line at the median for each group. Default is ``True``.
    sig_thresh : float, optional
        SNR threshold for the base quality mask. For emission lines this requires
        ``SNR > sig_thresh``; for absorption lines ``SNR < -sig_thresh``.
        Continuum-dependent quantities additionally require ``CONT SNR > sig_thresh``.
        Default is 3.0.
    alpha : float, optional
        Transparency for the histogram bars. Default is 0.6.
    combine_doublets : bool, optional
        When ``True`` (default) and the resolved line is a doublet key, the two lines
        are summed for ``FLUX``, ``EW``, and ``LUM`` properties.
    abs_lines : list of str, optional
        Lines to treat as absorption (SNR sign flipped). Defaults to ``[]``.
    ks_test : bool, optional
        When ``True`` and more than one mask is provided, run pairwise two-sample
        KS tests via :func:`ks_test_distributions` and annotate the plot with the
        resulting p-values. Default is ``False``.
    hist_kwargs : dict or list of dict, optional
        Extra keyword arguments forwarded to ``ax.hist``. Can be a single dictionary
        (applied to all groups) or a list of dictionaries (one per group). Default is ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes on which the histograms were drawn.
    """
    if hist_kwargs is None:
        hist_kwargs = {}
    # Convert single dict to list of dicts for uniform processing
    if isinstance(hist_kwargs, dict):
        hist_kwargs = [hist_kwargs]  # Will be replicated to match n_groups below
    if abs_lines is None:
        abs_lines = []

    prop = normalise_prop(prop)

    # --- Step 1: Resolve line and param from prop name or explicit line= ---
    # This is done BEFORE data retrieval so that SNR masking is always applied
    # correctly regardless of which path retrieves the data.
    _is_lya_prop = False
    _is_line_prop = False
    _resolved_param = prop   # updated below if a line suffix is found
    _resolved_line = line    # from explicit parameter; may be updated by auto-parse

    if _resolved_line is None and '_' in prop:
        # Try right-to-left splits to find a known line token as the suffix
        parts = prop.split('_')
        for split_idx in range(len(parts) - 1, 0, -1):
            candidate_line = '_'.join(parts[split_idx:])
            if candidate_line in _known_line_tokens:
                _resolved_line = candidate_line
                _resolved_param = '_'.join(parts[:split_idx])
                break
    elif _resolved_line is not None:
        # Explicit line= given — strip the line suffix from _resolved_param if present
        if prop.endswith(f'_{_resolved_line}'):
            _resolved_param = prop[:-(len(_resolved_line) + 1)]

    # --- Step 2: Retrieve the raw column ---
    # Resolution order:
    #   1. Explicit line= or auto-parsed line → get_line_property
    #   2. get_lya_property (Lya-specific props: EW_LYA, BRRATIO, FWHMR, etc.)
    #      NOTE: get_lya_property has a catch-all for any megatab column, so it
    #      must come AFTER the line-specific path to avoid bypassing SNR masking.
    #   3. Direct column lookup (final fallback)
    col = None
    _is_abs = _resolved_line in abs_lines

    if _resolved_line is not None:
        try:
            col, _ = get_line_property(megatab, _resolved_line, _resolved_param,
                                       combine_doublets=combine_doublets,
                                       abs=_is_abs)
            _is_line_prop = True
        except (ValueError, KeyError):
            pass
        # If get_line_property failed (e.g. column is pre-computed in megatab rather than
        # being derivable from FLUX/CONT), fall back to direct column lookup.
        if col is None and prop in megatab.colnames:
            col = np.asarray(megatab[prop], dtype=float)

    if col is None:
        try:
            col, _ = get_lya_property(megatab, prop)
            _is_lya_prop = True
        except (ValueError, KeyError):
            pass

    if col is None:
        if prop in megatab.colnames:
            col = np.asarray(megatab[prop], dtype=float)
        else:
            raise ValueError(
                f"Property '{prop}' not recognised as a Lya property, line property, or "
                f"column in megatab. If it is a line property, supply `line=` or use the "
                f"'{{param}}_{{line}}' form (e.g. 'EW_CIV1548')."
            )

    col = np.asarray(col, dtype=float)

    # --- Build a base quality mask (mirrors _prepare_scatter_mask logic) ---
    base_mask = np.isfinite(col)

    # Lya continuum quality: only for Lya-derived properties (not line-specific ones)
    if _is_lya_prop and prop in ["EW_LYA", "EW", "CONT", "CONT_LUM_LYA"] \
            and 'CONT' in megatab.colnames and 'CONT_ERR' in megatab.colnames:
        base_mask &= (megatab['CONT'] / megatab['CONT_ERR'] > sig_thresh)

    # Blue Lya peak quality: properties that depend on the blue peak require a
    # significant blue detection (mirrors _prepare_scatter_mask / check_lya_correlations)
    _blue_lya_props = {"BRRATIO", "BRSEP", "FLUXB", "ASYMB", "FWHMB", "DISPB"}
    if _is_lya_prop and prop in _blue_lya_props \
            and 'FLUXB' in megatab.colnames and 'FLUXB_ERR' in megatab.colnames:
        base_mask &= (megatab['FLUXB'] / megatab['FLUXB_ERR'] > sig_thresh)

    # Line SNR quality masking — applied for ANY property once a line is resolved,
    # regardless of whether data came from get_line_property or a direct column.
    if _resolved_line is not None:
        # SNR cut — emission: > +thresh; absorption: < -thresh
        # When combining doublets, use the quadrature-combined SNR of both components
        # (sqrt(SNR1^2 + SNR2^2)), since we are testing the significance of the combined flux.
        snr_col = f"SNR_{_resolved_line}"
        if snr_col in megatab.colnames:
            _use_doublet_snr = (combine_doublets
                                and _resolved_line in const.doublets
                                and _resolved_param in ("EW", "FLUX", "LUM"))
            if _use_doublet_snr:
                _line1, _line2 = const.doublets[_resolved_line]
                _snr2_col = f"SNR_{_line2}"
                if _snr2_col in megatab.colnames:
                    _snr1 = np.asarray(megatab[snr_col], dtype=float)
                    _snr2 = np.asarray(megatab[_snr2_col], dtype=float)
                    _combined_snr = np.sqrt(_snr1**2 + _snr2**2)
                    # Sign: negative combined SNR only makes sense for absorption — use titular-line sign
                    _sign = np.sign(_snr1)
                    _signed_combined_snr = _sign * _combined_snr
                    if _is_abs:
                        base_mask &= _signed_combined_snr < -sig_thresh
                    else:
                        base_mask &= _signed_combined_snr > sig_thresh
                else:
                    # Second component not in table — fall back to titular line
                    if _is_abs:
                        base_mask &= megatab[snr_col] < -sig_thresh
                    else:
                        base_mask &= megatab[snr_col] > sig_thresh
            else:
                if _is_abs:
                    base_mask &= megatab[snr_col] < -sig_thresh
                else:
                    base_mask &= megatab[snr_col] > sig_thresh

        # Line-continuum SNR for EW / CONT_LUM
        if _resolved_param in ["EW", "CONT_LUM"] and f"CONT_{_resolved_line}" in megatab.colnames:
            base_mask &= (megatab[f"CONT_{_resolved_line}"] / megatab[f"CONT_ERR_{_resolved_line}"] > sig_thresh)

        # Quality flag — for doublets, both component flags must be clear
        flag_col = f"FLAG_{_resolved_line}"
        if flag_col in megatab.colnames:
            base_mask &= (megatab[flag_col] == '')
        if combine_doublets and _resolved_line in const.doublets:
            _line2_flag = f"FLAG_{const.doublets[_resolved_line][1]}"
            if _line2_flag in megatab.colnames:
                base_mask &= (megatab[_line2_flag] == '')

        # FWHM must be positive after instrumental correction
        if _resolved_param in ["FWHM"]:
            base_mask &= col > 0

    # --- Optional log transform ---
    do_log = logify and _resolved_param in _log_quantities
    plot_col = np.log10(col) if do_log else col.copy()

    # --- Default: single group with no extra masking ---
    if masks is None:
        masks = [base_mask]
    else:
        masks = [base_mask & np.asarray(m, dtype=bool) for m in masks]

    n_groups = len(masks)
    if labels is None:
        labels = [None] * n_groups
    if colors is None:
        colors = [f"C{i}" for i in range(n_groups)]
    # Replicate hist_kwargs if only a single dict was provided
    if len(hist_kwargs) == 1 and n_groups > 1:
        hist_kwargs = hist_kwargs * n_groups
    elif len(hist_kwargs) != n_groups:
        raise ValueError(
            f"hist_kwargs must be either a single dict or a list of {n_groups} "
            f"dict(s), got {len(hist_kwargs)}"
        )

    # --- Shared bin edges across all groups (auto-selected if bins is a string) ---
    all_vals = np.concatenate([plot_col[m] for m in masks if np.any(m)])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        raise ValueError(f"No finite values found for property '{prop}' after masking.")
    if isinstance(bins, str):
        # numpy auto-selector ('auto', 'fd', 'scott', etc.) — compute shared edges
        bin_edges = np.histogram_bin_edges(all_vals, bins=bins)
    elif isinstance(bins, int):
        bin_edges = np.linspace(np.nanmin(all_vals), np.nanmax(all_vals), bins + 1)
    else:
        bin_edges = np.asarray(bins)

    # --- Create axes if needed ---
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    # --- Plot each group ---
    for m, label, color, kws in zip(masks, labels, colors, hist_kwargs):
        vals = plot_col[m]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=bin_edges, density=density, label=label,
                color=color, alpha=alpha, **kws)
        if plot_median:
            ax.axvline(np.nanmedian(vals), linestyle='--', color=color,
                       alpha=0.8, linewidth=1.5,
                       label=f"{label} median" if label else None)

    # --- Axis labels — same pattern as check_line_correlations ---
    if _resolved_line is not None:
        # Try the full compound key first (e.g. 'EW_LYA' → nice Lya-specific label);
        # fall back to composing from line + param parts for generic line properties.
        compound_label = plot.get_plot_name(prop)
        if compound_label == prop:  # not in plot_names — compose from parts
            xlabel = f"{plot.get_plot_name(_resolved_line)} {plot.get_plot_name(_resolved_param)}"
        else:
            xlabel = compound_label
    else:
        xlabel = plot.get_plot_name(prop)
    if do_log:
        xlabel = f"log {xlabel}"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density" if density else "N")

    if any(l is not None for l in labels):
        ax.legend()

    # --- Optional KS tests ---
    if ks_test and n_groups > 1:
        group_vals = []
        for m in masks:
            v = plot_col[m]
            group_vals.append(v[np.isfinite(v)])
        ks_results = ks_test_distributions(group_vals, labels=labels)
        ks_lines = []
        for (l1, l2), res in ks_results.items():
            pair = f"{l1} vs {l2}" if (l1 is not None and l2 is not None) else "KS"
            ks_lines.append(f"{pair}: D={res['statistic']:.3f}, p={res['p_value']:.3e}")
        ax.text(0.98, 0.97, '\n'.join(ks_lines),
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8, family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    return ax


def _lya_quality_mask(megatab: Table, prop: str, col: np.ndarray, sig_thresh: float) -> np.ndarray:
    """Build a quality mask for a single Lya property column."""
    mask = np.isfinite(col)
    if prop in ["EW_LYA", "CONT", "EW", "CONT_LUM_LYA"]:
        mask &= (megatab['CONT'] / megatab['CONT_ERR'] > sig_thresh)
    if prop in ["ASYMR", "DELTAV_LYA", "FWHMR", "DISPR", "VEXP_ZELDA"]:
        mask &= col > 0
    if prop == 'ASYMR':
        mask &= col < 0.3
    if prop in ["BRRATIO", "FLUXB", "ASYMB", "FWHMB", "DISPB", "BRSEP"]:
        mask &= (megatab["FLUXB"] / megatab["FLUXB_ERR"] > sig_thresh)
    if prop == "VEXP_ZELDA":
        mask &= (megatab[prop] - 3 * megatab['VEXP_ERRM_ZELDA'] > 0)
    return mask


def _scatter_qc(
    x_vals: np.ndarray, x_errs: np.ndarray,
    y_vals: np.ndarray, y_errs: np.ndarray,
    c_col: Optional[np.ndarray] = None,
    delta: Optional[np.ndarray] = None,
    sigma_clip: Optional[float] = None,
    nn_clip: Optional[float] = None,
    min_points: int = 10,
    pair_label: str = '',
) -> Optional[tuple]:
    """
    Apply quality-control filtering to a scatter dataset.

    Applies sigma clipping and/or nearest-neighbour outlier rejection in
    sequence, returning filtered copies of the input arrays.  Returns
    ``None`` if fewer than ``min_points`` survive filtering.

    Parameters
    ----------
    x_vals, x_errs : np.ndarray
        X-axis values and uncertainties.
    y_vals, y_errs : np.ndarray
        Y-axis values and uncertainties.
    c_col : np.ndarray, optional
        Colour column to filter alongside the data arrays.
    delta : np.ndarray, optional
        Upper-limit flags (1 = detection, 0 = upper limit).
    sigma_clip : float, optional
        Remove points whose deviation from the sample median exceeds this
        many sigma, computed in quadrature with the point's own uncertainty.
        This avoids unfairly clipping points whose apparent deviation is
        explained by a large error bar.
    nn_clip : float, optional
        Remove points whose mean distance to all other points in the normalised
        (x / std_x, y / std_y) plane exceeds ``median_mean_dist + nn_clip * MAD_mean_dist``.
        Because every point is compared against the whole sample, small clusters of
        outliers cannot protect each other.  Applied after sigma clipping.
        By default None.
    min_points : int, optional
        Minimum number of points required to survive filtering.
    pair_label : str, optional
        Identifier used in printed warning messages.

    Returns
    -------
    tuple or None
        ``(x_vals, x_errs, y_vals, y_errs, c_col, delta)`` — filtered
        copies of the inputs — or ``None`` if too few points survive.
    """
    # --- Sigma clipping ---
    if sigma_clip is not None:
        # Clip in quadrature: a point is kept if its deviation from the sample
        # median is within sigma_clip times the quadrature sum of the sample
        # scatter and the point's own uncertainty.  This avoids unfairly
        # clipping points whose apparent deviation is explained by a large
        # error bar.
        x_thresh = np.sqrt(np.std(x_vals) ** 2 + x_errs ** 2)
        y_thresh = np.sqrt(np.std(y_vals) ** 2 + y_errs ** 2)
        sc_mask = (
            (np.abs(x_vals - np.median(x_vals)) <= sigma_clip * x_thresh) &
            (np.abs(y_vals - np.median(y_vals)) <= sigma_clip * y_thresh)
        )
        if np.sum(sc_mask) < min_points:
            print(f"Warning: Sigma clipping ({sigma_clip}\u03c3) leaves fewer than "
                  f"{min_points} points for {pair_label}. Skipping.")
            return None
        n_clipped = len(x_vals) - np.sum(sc_mask)
        x_vals, x_errs = x_vals[sc_mask], x_errs[sc_mask]
        y_vals, y_errs = y_vals[sc_mask], y_errs[sc_mask]
        c_col = c_col[sc_mask] if c_col is not None else None
        delta = delta[sc_mask] if delta is not None else None
        if n_clipped:
            print(f"Sigma clipped {n_clipped} point(s) ({sigma_clip}\u03c3), leaving "
                  f"{len(x_vals)} points for {pair_label}.")

    # --- Mean-distance outlier rejection ---
    if nn_clip is not None and len(x_vals) >= 2:
        std_x = np.std(x_vals)
        std_y = np.std(y_vals)
        if std_x < 1e-10 or std_y < 1e-10:
            print(f"Warning: Near-zero axis scatter prevents mean-distance clipping "
                  f"for {pair_label}. Skipping.")
        else:
            xn = x_vals / std_x
            yn = y_vals / std_y
            coords = np.column_stack([xn, yn])
            # Pairwise distances; exclude self via the diagonal
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=-1)
            np.fill_diagonal(sq_dist, 0.0)
            mean_dist = np.sum(np.sqrt(sq_dist), axis=1) / (len(x_vals) - 1)
            med_md = np.median(mean_dist)
            mad_md = np.median(np.abs(mean_dist - med_md))
            if mad_md < 1e-10:
                mad_md = np.std(mean_dist) if np.std(mean_dist) > 1e-10 else 1.0
            nn_mask = mean_dist <= med_md + nn_clip * mad_md
            if np.sum(nn_mask) < min_points:
                print(f"Warning: Mean-distance clipping ({nn_clip}\u03c3) leaves fewer than "
                      f"{min_points} points for {pair_label}. Skipping.")
                return None
            n_nn_clipped = len(x_vals) - np.sum(nn_mask)
            x_vals, x_errs = x_vals[nn_mask], x_errs[nn_mask]
            y_vals, y_errs = y_vals[nn_mask], y_errs[nn_mask]
            c_col = c_col[nn_mask] if c_col is not None else None
            delta = delta[nn_mask] if delta is not None else None
            if n_nn_clipped:
                print(f"Mean-distance clipped {n_nn_clipped} point(s) ({nn_clip}\u03c3), "
                      f"leaving {len(x_vals)} points for {pair_label}.")

    return x_vals, x_errs, y_vals, y_errs, c_col, delta


def _correlate_pair(
    x_vals: np.ndarray, x_errs: np.ndarray,
    y_vals: np.ndarray, y_errs: np.ndarray,
    x_prop: str, y_prop: str,
    x_label: str, y_label: str, title: str,
    pair_label: str,
    ax_in: Optional[plt.Axes] = None,
    c_col: Optional[np.ndarray] = None,
    c_label: Optional[str] = None,
    logify: bool = False,
    min_points: int = 10,
    significance_thresh: float = 0.01,
    delta: Optional[np.ndarray] = None,
    clip_extreme_errors: Optional[float] = None,
    sigma_clip: Optional[float] = None,
    nn_clip: Optional[float] = None,
    mcmc: bool = True,
    niter: int = 5000,
    save_fig: bool = False,
    fig_path: Optional[str] = None,
    plot_all: bool = False,
    **scatter_kwargs,
) -> Optional[dict]:
    """
    Shared inner loop for all correlation-analysis functions.

    Handles log-transform, error-bar clipping, scatter plot, OLS pre-screen,
    MCMC/ODR fit, axis labelling, and returns a result summary dict.
    Returns ``None`` when the pair is skipped.

    Parameters
    ----------
    x_vals, x_errs : np.ndarray
        X-axis values and 1-sigma uncertainties (already quality-masked).
    y_vals, y_errs : np.ndarray
        Y-axis values and 1-sigma uncertainties (already quality-masked).
    x_prop, y_prop : str
        Property token used to determine whether log-transform applies
        (checked against ``_log_quantities``).
    x_label, y_label : str
        Base axis label strings (``"log "`` prefix added automatically when
        logified).
    title : str
        Plot title.
    pair_label : str
        Human-readable description for print messages
        (e.g. ``"SiII1260 EW vs DISPR"``).
    ax_in : matplotlib.axes.Axes, optional
        Axes on which to draw the scatter plot. A new figure and axes are created
        if ``None``. Default is ``None``.
    c_col : np.ndarray, optional
        Colour values for scatter plot points (already masked), by default None.
    logify : bool, optional
        Log-transform axes whose property is listed in ``_log_quantities``,
        by default False.
    min_points : int, optional
        Minimum points required to attempt a fit, by default 10.
    significance_thresh : float, optional
        OLS pre-screening p-value threshold; pairs above this are skipped,
        by default 0.01.
    delta : np.ndarray, optional
        Optional boolean array indicating upper limits (1 detection, 0 for upper limit) to be used
        by LinMix MCMC fitting
    clip_extreme_errors : float, optional
        Clip points whose error exceeds this multiple of the data scatter,
        by default None.
    sigma_clip : float, optional
        If given, remove points whose deviation from the sample median exceeds
        ``sigma_clip`` times the quadrature sum of the sample standard deviation
        and the point's own uncertainty (applied independently on each axis).
        This avoids clipping points whose apparent deviation is explained by a
        large error bar. By default None.
    nn_clip : float, optional
        If given, remove points whose mean distance to all other points in the
        normalised (x / std_x, y / std_y) plane exceeds
        ``median_mean_dist + nn_clip * MAD_mean_dist``.  Applied after sigma
        clipping.  By default None.
    mcmc : bool, optional
        Use LinMix MCMC; falls back to ODR if False, by default True.
    niter : int, optional
        Minimum MCMC iterations per chain, by default 5000.
    save_fig : bool, optional
        Save the figure to ``fig_path``, by default False.
    fig_path : str, optional
        File path for saving; required when ``save_fig=True``.
    plot_all : bool, optional
        Whether to plot regardless of whether the pair passes the significance threshold. 
        Default is False (only plot significant pairs).
    **scatter_kwargs
        Forwarded to :func:`make_scatter`.

    Returns
    -------
    dict or None
        Keys: ``slope``, ``slope_err``, ``intercept``, ``intercept_err``,
        ``p_value``, ``n_points``. ``None`` if the pair was skipped.
    """
    from scipy.stats import linregress

    x_vals = np.array(x_vals, dtype=float)
    y_vals = np.array(y_vals, dtype=float)
    x_errs = np.array(x_errs, dtype=float)
    y_errs = np.array(y_errs, dtype=float)

    # --- Log transform ---
    plot_log_x = logify and x_prop in _log_quantities
    plot_log_y = logify and y_prop in _log_quantities
    if plot_log_x:
        x_orig = x_vals.copy()
        x_vals = np.log10(x_orig)
        x_errs = x_errs / (x_orig * np.log(10))
    if plot_log_y:
        y_orig = y_vals.copy()
        y_vals = np.log10(y_orig)
        y_errs = y_errs / (y_orig * np.log(10))

    # --- Clip extreme errors ---
    if clip_extreme_errors is not None:
        err_mask = ((x_errs < clip_extreme_errors * np.std(x_vals)) &
                    (y_errs < clip_extreme_errors * np.std(y_vals)))
        x_vals, x_errs = x_vals[err_mask], x_errs[err_mask]
        y_vals, y_errs = y_vals[err_mask], y_errs[err_mask]
        c_col = c_col[err_mask] if c_col is not None else None
        delta = delta[err_mask] if delta is not None else None
        print(f"Clipped extreme errors ({clip_extreme_errors}\u03c3), leaving {len(x_vals)} points "
              f"for {pair_label}.")

    # --- Sigma clipping and nearest-neighbour outlier rejection ---
    qc_result = _scatter_qc(
        x_vals, x_errs, y_vals, y_errs,
        c_col=c_col, delta=delta,
        sigma_clip=sigma_clip, nn_clip=nn_clip,
        min_points=min_points, pair_label=pair_label,
    )
    if qc_result is None:
        return None
    x_vals, x_errs, y_vals, y_errs, c_col, delta = qc_result

    # --- Sanity checks ---
    if np.any(x_errs < 0) or np.any(y_errs < 0):
        print(f"Warning: Negative error bars for {pair_label}. Skipping.")
        return None

    # --- Scatter plot ---
    if ax_in is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        ax = ax_in
    _scatter_kw: dict = {}
    if c_col is not None:
        _scatter_kw['show_colorbar'] = True
        if c_label is not None:
            _scatter_kw['clabel'] = c_label
    _scatter_kw.update(scatter_kwargs)
    upper_bounds = (delta == 0) if delta is not None else None
    make_scatter([x_vals, x_errs], [y_vals, y_errs], ax=ax, c=c_col,
                 upper_bounds=upper_bounds, **_scatter_kw)

    n_pts = len(x_vals)
    if n_pts < min_points:
        print(f"Not enough points to fit for {pair_label} (n={n_pts}).")
        if ax_in is None:
            plt.close(fig)
        return None

    for arr_name, arr in [("x_vals", x_vals), ("y_vals", y_vals),
                          ("x_errs", x_errs), ("y_errs", y_errs)]:
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            print(f"Warning: Non-finite values in {arr_name} for {pair_label}. Skipping.")
            if ax_in is None:
                plt.close(fig)
            return None

    # --- OLS pre-screen ---
    _, _, _, prelim_p, _ = linregress(x_vals, y_vals)
    if prelim_p >= significance_thresh:
        print(f"No significant correlation for {pair_label} (prelim p={prelim_p:.3e}). Skipping.")
        if plot_all:
            if save_fig and fig_path:
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.show()
        if ax_in is None:
            plt.close(fig)
        return None
    print(f"Preliminary fit suggests significant correlation for {pair_label} "
          f"(prelim p={prelim_p:.3e}). Proceeding with {'MCMC' if mcmc else 'ODR'} fit.")

    # --- MCMC / ODR fit ---
    slope, intercept, slope_err, intercept_err, p_value, rho, rho_p = do_linregress(
        x_vals, y_vals, x_errs, y_errs, mcmc=mcmc, ax_in=ax, niter=niter, delta=delta
    )
    print(f"Correlation for {pair_label}: slope={slope:.2f}±{slope_err:.2f}, "
          f"intercept={intercept:.2f}±{intercept_err:.2f}, p={p_value:.3e}, "
          f"Spearman ρ={rho:.3f} (p={rho_p:.3e})")

    ax.set_xlabel(f"{'log ' if plot_log_x else ''}{x_label}")
    ax.set_ylabel(f"{'log ' if plot_log_y else ''}{y_label}")
    ax.set_title(title)

    if save_fig and fig_path and ax_in is None:
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.show()

    return {
        'slope': slope,
        'slope_err': slope_err,
        'intercept': intercept,
        'intercept_err': intercept_err,
        'p_value': p_value,
        'spearman_rho': rho,
        'spearman_rho_p': rho_p,
        'n_points': n_pts,
    }


def _insert_upper_limits(
    megatab, line: str, line_col: np.ndarray, line_err: np.ndarray,
    abs_lines: list[str], line_prop: str = 'EW',
    sig_thresh: float = 3.0, combine_doublets: bool = True,
    rest_frame: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Replace non-detected sources' line property values with upper limits, and
    return a LinMix-compatible delta array marking detections vs. censored points.

    For EW, bootstrapped 3-sigma flux upper bounds stored in ``FLUX_UB_{line}``
    columns are used to derive the EW upper limit via ``FLUX_UB / CONT / (1+z)``.
    This is preferred over ``3 * FLUX_ERR`` because the raw curve_fit errors
    underestimate the true uncertainties. Falls back to ``3 * line_err`` if the
    ``FLUX_UB`` column is absent or if ``line_prop != 'EW'``.

    For doublets with ``combine_doublets=True``, both components' ``FLUX_UB``
    columns are summed (if both are present) and divided by the first component's
    continuum, matching the convention used by ``get_line_property``.

    Parameters
    ----------
    megatab : astropy.table.Table
        The megatable.
    line : str
        Titular line name (e.g. ``'CIV1548'``).
    line_col : np.ndarray
        Property values for all sources (output of ``get_line_property``).
    line_err : np.ndarray
        Property errors for all sources (output of ``get_line_property``).
    abs_lines : list[str]
        Lines treated as absorption.
    line_prop : str, optional
        The property being analysed. Controls whether the bootstrapped EW upper
        limit path is taken. Default ``'EW'``.
    sig_thresh : float, optional
        Detection significance threshold in sigma. Default 3.0.
    combine_doublets : bool, optional
        Whether doublet components are combined. Default True.
    rest_frame : bool, optional
        Apply rest-frame ``1/(1+z)`` correction when computing EW upper limits.
        Should match the setting used in ``get_line_property``. Default True.

    Returns
    -------
    modified_col : np.ndarray
        Copy of ``line_col`` with non-detection values replaced by upper limits.
    delta : np.ndarray of int
        1 for detected sources, 0 for censored (upper-limit) sources. Suitable
        for passing directly to ``LinMix(..., delta=delta)``.
    """
    snr = _effective_snr(megatab, line, abs_lines,
                         line_prop=line_prop, combine_doublets=combine_doublets)
    is_abs = line in abs_lines
    detected = (snr < -sig_thresh) if is_abs else (snr > sig_thresh)
    delta = detected.astype(int)

    modified_col = np.array(line_col, dtype=float)
    ul_mask = ~detected

    if not np.any(ul_mask):
        return modified_col, delta

    if line_prop == 'EW':
        # --- Determine FLUX_UB and CONT columns ---
        flux_ub: Optional[np.ndarray] = None
        cont: Optional[np.ndarray] = None

        if combine_doublets and line in const.doublets:
            line1, line2 = const.doublets[line]
            ub1_col  = f"FLUX_UB_{line1}"
            ub2_col  = f"FLUX_UB_{line2}"
            cont_col = f"CONT_{line1}"
            if ub1_col in megatab.colnames and cont_col in megatab.colnames:
                flux_ub = np.asarray(megatab[ub1_col], dtype=float)
                if ub2_col in megatab.colnames:
                    flux_ub = flux_ub + np.asarray(megatab[ub2_col], dtype=float)
                else:
                    print(f"Warning: Second component upper limit column '{ub2_col}' not found for doublet '{line}'. "
                          f"Using only '{ub1_col}' for upper limits.")
                cont = np.asarray(megatab[cont_col], dtype=float)
        else:
            ub_col   = f"FLUX_UB_{line}"
            cont_col = f"CONT_{line}"
            if ub_col in megatab.colnames and cont_col in megatab.colnames:
                flux_ub = np.asarray(megatab[ub_col], dtype=float)
                cont    = np.asarray(megatab[cont_col], dtype=float)

        if flux_ub is not None and cont is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                ew_ub = flux_ub / cont
            if rest_frame:
                ew_ub /= (1.0 + np.asarray(megatab['z'], dtype=float))
            modified_col[ul_mask] = ew_ub[ul_mask]
            return modified_col, delta
        # else fall through to the generic fallback below

    # Fallback for non-EW properties or missing FLUX_UB columns
    print(f"Warning: Using fallback upper limit for {line_prop} of '{line}' due to missing FLUX_UB or non-EW property.")
    modified_col[ul_mask] = 3.0 * np.asarray(line_err, dtype=float)[ul_mask]
    return modified_col, delta


def check_line_correlations(
        line_property: str, lya_properties: list[str], 
        lines: list[str], abs_lines: list[str], 
        megatab: Table,
        ax_in: Optional[plt.Axes] = None,
        min_points: int = 10, 
        significance_thresh: float = 0.01, 
        mcmc: bool = True,
        logify: bool = False, 
        save_fig: bool = False,
        point_sig_thresh: float = 3.0, 
        fit_upper_limits: bool = False,
        c: Optional[str] = 'z', 
        clip_extreme_errors: Optional[float] = None,
        sigma_clip: Optional[float] = None,
        nn_clip: Optional[float] = None,
        combine_doublets: bool = True, 
        niter: int = 5000,
        plot_all: bool = False, 
        **scatter_kwargs
        ) -> dict:
    """
    Check for correlations between a given line property (e.g. EW, FWHM) and a list of Lyman alpha properties.

    Parameters
    ----------
    line_property : str
        The line property to check (e.g. "EW", "FWHM").
    lya_properties : list[str]
        List of Lyman alpha properties to check against (e.g. ["EW_LYA", "DISPR", "CONT", "ASYMR", 
        "FWHMR", "BRRATIO", "BRSEP"]).
    lines : list[str]
        List of lines to check (e.g. ["SiII1260", "CII1334", "SiIV1394", "SiIV1403", "CIV1548", 
        "HeII1640", "OIII1660", "CIII1907"]).
    abs_lines : list[str]
        List of lines that should be treated as absorption (e.g. ["SiII1260", "CII1334", 
        "SiIV1394", "SiIV1403"]).
    megatab : astropy.table.Table
        The megatable containing the data.
    ax_in : matplotlib.axes.Axes, optional
        An existing Axes object to plot on. If None, a new figure and axes will be created.
    min_points : int, optional
        Minimum number of points required to attempt fitting a correlation, by default 10.
    significance_thresh : float, optional
        P-value (or Bayesian equivalent) threshold for determining significant correlations, by default 0.01.
    mcmc : bool, optional
        Whether to use MCMC regression for fitting the correlation, by default True.
    logify : bool, optional
        Whether to log-transform the line property for fitting, by default False.
    save_fig : bool, optional
        Whether to save the figure, by default False.
    point_sig_thresh : float, optional
        The significance threshold (in sigma) for plotting individual points, by default 3.0
    fit_upper_limits : bool, optional
        Whether to include upper limits (in y values) in the fit. Default is False (only use detections).
    c : str, optional
        Name of the ``megatab`` column used to colour scatter-plot points. Default is ``'z'``
        (redshift). Pass ``None`` to disable point colouring.
    clip_extreme_errors : Optional[float], optional
        The threshold for clipping extreme error values in terms of the standard deviation of
        the corresponding data, by default None (no clipping).
    sigma_clip : float, optional
        If given, remove points more than this many standard deviations from the median along
        either axis before fitting, by default None.
    nn_clip : float, optional
        If given, remove points whose mean distance to all other points in the
        normalised (x / std_x, y / std_y) plane exceeds
        ``median_mean_dist + nn_clip * MAD_mean_dist``.  Applied after sigma
        clipping. By default None.
    niter : int, optional
        Minimum number of MCMC iterations per chain passed to :func:`do_linregress`, by default 5000
        (~10,000 posterior samples). Increase for more reliable detection of weak signals.
    plot_all : bool, optional
        Whether to plot all pairs regardless of significance, or only those that pass the
        threshold. Default is False (only plot significant pairs).
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
            lya_prop = normalise_prop(lya_prop)
            lya_col, lya_col_err = get_lya_property(megatab, lya_prop)
            line_col_raw, line_err_raw = get_line_property(megatab, line, line_property,
                                                           abs=line in abs_lines,
                                                           combine_doublets=combine_doublets)
            
            line_col = np.array(line_col_raw, dtype=float)
            line_err_arr = np.array(line_err_raw, dtype=float)
            delta = None

            mask = _prepare_scatter_mask(megatab, line, line_col_raw, line_property,
                                        lya_col, lya_prop,
                                        abs_lines, sig_thresh=point_sig_thresh,
                                        combine_doublets=combine_doublets)

            if fit_upper_limits:
                line_col, delta = _insert_upper_limits(megatab, line, line_col_raw, line_err_raw,
                                                       abs_lines, line_prop=line_property,
                                                       sig_thresh=point_sig_thresh,
                                                       combine_doublets=combine_doublets)
                mask = _prepare_scatter_mask(megatab, line, line_col, line_property,
                                            lya_col, lya_prop,
                                            abs_lines, delta=delta,
                                            sig_thresh=point_sig_thresh,
                                            combine_doublets=combine_doublets)

            result = _correlate_pair(
                lya_col[mask], lya_col_err[mask],
                line_col[mask], line_err_arr[mask],
                x_prop=lya_prop, y_prop=line_property,
                x_label=plot.get_plot_name(lya_prop),
                y_label=f"{plot.get_plot_name(line)} {plot.get_plot_name(line_property)}",
                title=(f"{plot.get_plot_name(line, unit=False)} "
                       f"{plot.get_plot_name(line_property, unit=False)} "
                       f"vs {plot.get_plot_name(lya_prop, unit=False)}"),
                pair_label=f"{line} {line_property} vs {lya_prop}",
                ax_in=ax_in,
                c_col=megatab[c][mask] if c is not None else None,
                c_label=plot.get_plot_name(c) if c is not None else None,
                delta=delta[mask] if delta is not None else None,
                logify=logify,
                min_points=min_points,
                significance_thresh=significance_thresh,
                clip_extreme_errors=clip_extreme_errors,
                sigma_clip=sigma_clip,
                nn_clip=nn_clip,
                mcmc=mcmc,
                niter=niter,
                save_fig=save_fig,
                fig_path=f"plots/{line}_{line_property}_vs_{lya_prop}.png",
                plot_all=plot_all,
                **scatter_kwargs,
            )
            if result is not None:
                summaries[line][lya_prop] = result
    return summaries


def check_lya_correlations(
        lya_properties_y: list[str], lya_properties_x: list[str],
        megatab: Table, 
        min_points: int = 10,
        significance_thresh: float = 0.01, 
        mcmc: bool = True,
        logify: bool = False, 
        save_fig: bool = False,
        point_sig_thresh: float = 3.0, 
        clip_extreme_errors: Optional[float] = None,
        sigma_clip: Optional[float] = None,
        nn_clip: Optional[float] = None,
        c: Optional[str] = 'z', 
        niter: int = 5000, 
        plot_all: bool = False, 
        ax_in: Optional[matplotlib.axes.Axes] = None,
        **scatter_kwargs
    ) -> dict:
    """
    Check for correlations between pairs of Lyman alpha properties.

    Both axes use ``get_lya_property`` for data retrieval and Lya-appropriate masking is applied
    to each axis independently.

    Parameters
    ----------
    lya_properties_y : list[str]
        Lyman alpha properties to place on the y-axis (e.g. ["FWHMR", "DISPR", "ASYMR"]).
    lya_properties_x : list[str]
        Lyman alpha properties to place on the x-axis (e.g. ["EW_LYA", "CONT", "BRRATIO"]).
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
    sigma_clip : float, optional
        If given, remove points more than this many standard deviations from the median along
        either axis before fitting, by default None.
    nn_clip : float, optional
        If given, remove points whose mean distance to all other points in the
        normalised (x / std_x, y / std_y) plane exceeds
        ``median_mean_dist + nn_clip * MAD_mean_dist``.  Applied after sigma
        clipping. By default None.
    c : str, optional
        Name of the ``megatab`` column used to colour scatter-plot points. Default is ``'z'``
        (redshift). Pass ``None`` to disable point colouring.
    niter : int, optional
        Minimum number of MCMC iterations per chain passed to :func:`do_linregress`, by default 5000
        (~10,000 posterior samples). Increase for more reliable detection of weak signals.
    plot_all : bool, optional
        Whether to plot all pairs regardless of significance, or only those that pass the 
        threshold. Default is False (only plot significant pairs).
    ax_in : matplotlib.axes.Axes, optional
        An existing Axes object to plot on. If None, a new figure and axes will be created. Default is None.
    **scatter_kwargs
        Additional keyword arguments forwarded to :func:`make_scatter` (e.g. ``cnorm='log'``,
        ``cmap``, ``vmin``, ``vmax``, ``show_colorbar``, ``clabel``).

    Returns
    -------
    dict
        A nested dictionary ``summaries[lya_prop_y][lya_prop_x]`` containing the correlation
        summaries (slope, intercept, errors, p-value, n_points) for each pair.
    """
    summaries = {}
    seen_pairs = set()
    for prop_y in lya_properties_y:
        prop_y = normalise_prop(prop_y)
        summaries[prop_y] = {}
        for prop_x in lya_properties_x:
            prop_x = normalise_prop(prop_x)
            if prop_x == prop_y:
                print(f"Skipping trivial self-correlation: {prop_y} vs {prop_x}.")
                continue
            pair = frozenset((prop_x, prop_y))
            if pair in seen_pairs:
                print(f"Skipping duplicate pair: {prop_y} vs {prop_x}.")
                continue
            seen_pairs.add(pair)

            x_col, x_col_err = get_lya_property(megatab, prop_x)
            y_col, y_col_err = get_lya_property(megatab, prop_y)
            mask = (_lya_quality_mask(megatab, prop_x, x_col, point_sig_thresh)
                  & _lya_quality_mask(megatab, prop_y, y_col, point_sig_thresh)
                  & np.isfinite(x_col) & np.isfinite(y_col)
                  & np.isfinite(x_col_err) & np.isfinite(y_col_err))

            result = _correlate_pair(
                x_col[mask], x_col_err[mask],
                y_col[mask], y_col_err[mask],
                x_prop=prop_x, y_prop=prop_y,
                x_label=plot.get_plot_name(prop_x),
                y_label=plot.get_plot_name(prop_y),
                title=(f"{plot.get_plot_name(prop_y, unit=False)} "
                       f"vs {plot.get_plot_name(prop_x, unit=False)}"),
                pair_label=f"{prop_y} vs {prop_x}",
                c_col=megatab[c][mask] if c is not None else None,
                c_label=plot.get_plot_name(c) if c is not None else None,
                logify=logify,
                min_points=min_points,
                significance_thresh=significance_thresh,
                clip_extreme_errors=clip_extreme_errors,
                sigma_clip=sigma_clip,
                nn_clip=nn_clip,
                mcmc=mcmc,
                niter=niter,
                save_fig=save_fig,
                fig_path=f"plots/{prop_y}_vs_{prop_x}.png",
                plot_all=plot_all,
                ax_in=ax_in,
                **scatter_kwargs,
            )
            if result is not None:
                summaries[prop_y][prop_x] = result
    return summaries


def check_line_line_correlations(
        property_x: str, lines_x: list[str],
        property_y: str, lines_y: list[str],
        abs_lines: list[str], megatab: Table,
        min_points: int = 10,
        significance_thresh: float = 0.01,
        mcmc: bool = True,
        logify: bool = False,
        save_fig: bool = False,
        point_sig_thresh: float = 3.0,
        c: Optional[str] = 'z',
        clip_extreme_errors: Optional[float] = None,
        combine_doublets: bool = True,
        sigma_clip: Optional[float] = None,
        nn_clip: Optional[float] = None,
        niter: int = 5000,
        plot_all: bool = False,
        fit_upper_limits: bool = False,
        ax_in: Optional[matplotlib.axes.Axes] = None,
        **scatter_kwargs,
) -> dict:
    """
    Check for correlations between a property of one set of lines and a property
    of another set of lines.

    Parameters
    ----------
    property_x : str
        Line property for the x-axis (e.g. ``"EW"``).
    lines_x : list[str]
        Lines to iterate over for the x-axis.
    property_y : str
        Line property for the y-axis (e.g. ``"FWHM"``).
    lines_y : list[str]
        Lines to iterate over for the y-axis.
    abs_lines : list[str]
        Lines treated as absorption (SNR sign-flipped for masking).
    megatab : astropy.table.Table
        The megatable.
    min_points : int, optional
        Minimum number of points required to attempt a fit, by default 10.
    significance_thresh : float, optional
        OLS pre-screening p-value threshold, by default 0.01.
    mcmc : bool, optional
        Use LinMix MCMC; falls back to ODR if False, by default True.
    logify : bool, optional
        Log-transform axes whose property appears in ``_log_quantities``,
        by default False.
    save_fig : bool, optional
        Save each figure, by default False.
    point_sig_thresh : float, optional
        SNR threshold for quality masking, by default 3.0.
    c : str, optional
        Column name for scatter point colouring. Default ``'z'``; pass
        ``None`` to disable.
    clip_extreme_errors : float, optional
        Clip points whose error exceeds this multiple of the scatter,
        by default None.
    combine_doublets : bool, optional
        Combine doublet components for additive properties, by default True.
    sigma_clip : float, optional
        If given, remove points more than this many standard deviations from the median along
        either axis before fitting, by default None.
    nn_clip : float, optional
        If given, remove points whose mean distance to all other points in the
        normalised (x / std_x, y / std_y) plane exceeds
        ``median_mean_dist + nn_clip * MAD_mean_dist``.  Applied after sigma
        clipping. By default None.
    niter : int, optional
        Minimum MCMC iterations per chain, by default 5000
        (~10,000 posterior samples).
    plot_all : bool, optional
        Whether to plot all pairs regardless of significance, or only those that pass the 
        threshold. Default is False (only plot significant pairs).
    fit_upper_limits : bool, optional
        Whether to include upper limits for non-detections on the y-axis, using bootstrapped
        ``FLUX_UB`` columns to derive EW upper limits. Only applies when ``property_y == 'EW'``
        (or whichever property has ``FLUX_UB`` columns). The x-axis always requires detections.
        Default False.
    ax_in : matplotlib.axes.Axes, optional
        An existing Axes object to plot on. If None, a new figure and axes will be created. Default is None.
    **scatter_kwargs
        Forwarded to :func:`make_scatter`.

    Returns
    -------
    dict
        Nested ``summaries[line_y][line_x]`` containing slope, intercept,
        errors, p-value, and n_points for each pair that passed the
        pre-screen.
    """
    summaries = {}
    for line_y in lines_y:
        summaries[line_y] = {}
        for line_x in lines_x:
            if line_x == line_y and property_x == property_y:
                print(f"Skipping trivial self-correlation: {line_y} {property_y}.")
                continue

            x_col, x_err = get_line_property(megatab, line_x, property_x,
                                              abs=line_x in abs_lines,
                                              combine_doublets=combine_doublets)
            y_col_raw, y_err_raw = get_line_property(megatab, line_y, property_y,
                                                     abs=line_y in abs_lines,
                                                     combine_doublets=combine_doublets)

            y_col = np.array(y_col_raw, dtype=float)
            y_err = np.array(y_err_raw, dtype=float)
            delta = None

            # Build independent quality masks for each line then combine.
            # Pass lya_prop='' so Lya-specific quality cuts in _prepare_scatter_mask
            # (Lya continuum SNR, positivity, blue-peak) never fire here.
            # x-axis always requires detections; y-axis optionally admits upper limits.
            mask_x = _prepare_scatter_mask(megatab, line_x, x_col, property_x,
                                           x_col, '',
                                           abs_lines, include_upper_limits=False,
                                           sig_thresh=point_sig_thresh,
                                           combine_doublets=combine_doublets)
            if fit_upper_limits:
                y_col, delta = _insert_upper_limits(megatab, line_y, y_col_raw, y_err_raw,
                                                    abs_lines, line_prop=property_y,
                                                    sig_thresh=point_sig_thresh,
                                                    combine_doublets=combine_doublets)
                mask_y = _prepare_scatter_mask(megatab, line_y, y_col, property_y,
                                               y_col, '',
                                               abs_lines, delta=delta,
                                               sig_thresh=point_sig_thresh,
                                               combine_doublets=combine_doublets)
            else:
                mask_y = _prepare_scatter_mask(megatab, line_y, y_col, property_y,
                                               y_col, '',
                                               abs_lines, include_upper_limits=False,
                                               sig_thresh=point_sig_thresh,
                                               combine_doublets=combine_doublets)

            mask = mask_x & mask_y

            result = _correlate_pair(
                x_col[mask], x_err[mask],
                y_col[mask], y_err[mask],
                x_prop=property_x, y_prop=property_y,
                x_label=f"{plot.get_plot_name(line_x)} {plot.get_plot_name(property_x)}",
                y_label=f"{plot.get_plot_name(line_y)} {plot.get_plot_name(property_y)}",
                title=(f"{plot.get_plot_name(line_y, unit=False)} "
                       f"{plot.get_plot_name(property_y, unit=False)} "
                       f"vs {plot.get_plot_name(line_x, unit=False)} "
                       f"{plot.get_plot_name(property_x, unit=False)}"),
                pair_label=f"{line_y} {property_y} vs {line_x} {property_x}",
                c_col=megatab[c][mask] if c is not None else None,
                c_label=plot.get_plot_name(c) if c is not None else None,
                delta=delta[mask] if delta is not None else None,
                logify=logify,
                min_points=min_points,
                significance_thresh=significance_thresh,
                clip_extreme_errors=clip_extreme_errors,
                sigma_clip=sigma_clip,
                nn_clip=nn_clip,
                mcmc=mcmc,
                niter=niter,
                save_fig=save_fig,
                fig_path=f"plots/{line_y}_{property_y}_vs_{line_x}_{property_x}.png",
                plot_all=plot_all,
                ax_in=ax_in,
                **scatter_kwargs,
            )
            if result is not None:
                summaries[line_y][line_x] = result
    return summaries