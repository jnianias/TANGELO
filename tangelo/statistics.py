"""
Tools for performing statistical analysis on spectra and tables.
"""

from .catalogue_operations import generate_source_mask
from . import constants as const
from . import spectroscopy as spectro
from . import models
from . import fitting

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import ks_2samp

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

def make_scatter(colx, coly, ax, z = None, plotxubs=False, plotyubs=False, color = None, 
                   skymask = 'total', edgecolor = 'black', filts = [],
                   color_redshift = False, highlight = False, alpha = 0.6, 
                   alpha_ubs = 0.3, msize = 25, z_cmap = blured, alpha_e = 0.3,
                  label = None, marker='o'):
    """
    Makes a scatter plot of colx vs coly, with optional coloring by z and filtering by filts.

    Parameters
    ----------
    colx : list of arrays
        List of arrays for x values and optional error bars. First element is x values, second is symmetric error bars, third is asymmetric error bars.
    coly : list of arrays
        List of arrays for y values and optional error bars. First element is y values, second is symmetric error bars, third is asymmetric error bars.
    ax : matplotlib.axes.Axes
        The axes on which to plot.
    z : array-like, optional
        Values for coloring the points. If None, points will not be colored by z.
    plotxubs : bool, optional
        Whether to plot upper bounds in x as arrows.
    plotyubs : bool, optional
        Whether to plot upper bounds in y as arrows.
    color : str or array-like, optional
        Color for the points. If None, color will be determined by z if color_redshift is True, otherwise default color will be used.
    skymask : str, optional
        Sky mask to apply for filtering the data. Default is 'total'.
    edgecolor : str, optional
        Color for the edges of the points. Default is 'black'.
    filts : list of arrays, optional
        List of boolean arrays for filtering the data. Each array should have the same length as colx[0] and coly[0], and the final mask will be the logical AND of all the filters.
    color_redshift : bool, optional
        Whether to color the points by redshift using the z_cmap. If True, z values will be used for coloring the points. If False, color will be determined by the color parameter or default color.
    highlight : bool, optional
        Whether to highlight points that meet certain criteria (e.g. high redshift). If True, points that meet the criteria will be colored differently or plotted with a different marker. The specific criteria for highlighting should be defined within the function based on the data and requirements.
    alpha : float, optional
        Alpha value for the points. Default is 0.6.
    alpha_ubs : float, optional
        Alpha value for the upper bound arrows. Default is 0.3.
    msize : float, optional
        Marker size for the points. Default is 25.
    z_cmap : matplotlib.colors.Colormap, optional
        Colormap to use for coloring points by z if color_redshift is True. Default is blured (seismic colormap).
    alpha_e : float, optional
        Alpha value for the error bars. Default is 0.3.
    label : str, optional
        Label for the points to be used in the legend. Default is None.
    marker : str, optional
        Marker style for the points. Default is 'o' (circle).

    Returns
    -------
    None
    """
    mask = np.ones(np.size(colx[0])).astype(bool) # Initialize mask to include all points
    mask *= ~(np.isnan(colx[0]) + np.isnan(coly[0])) # Exclude points where x or y is NaN

    for cond in filts:
        # Apply additional filters to the mask
        mask *= cond
    
    if z is None:
        # If z is not provided, use a default color for all points
        z = np.zeros(np.size(colx[0]))
    
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
    
    # Create scatter plot with error bars and optional upper bound arrows
    sc = ax.scatter(xvals[mask], yvals[mask], c=z[mask], 
               cmap=z_cmap, edgecolor=edgecolor, s=msize, vmin=3.0, 
               vmax=6.7, alpha = alpha, label=label, marker=marker)
    ax.errorbar(xvals[mask], yvals[mask], xerr=[xerrsm[mask], xerrsp[mask]], 
                yerr=[yerrsm[mask], yerrsp[mask]], marker='', color=edgecolor, 
                zorder=0, alpha = alpha_e, linestyle='')
    if plotxubs:
        x_ubs_mask = mask & (xerrsp > 0) & ~np.isnan(xvals)
        ax.scatter(xvals[x_ubs_mask] + xerrsp[x_ubs_mask], yvals[x_ubs_mask],
                   marker='^', color=edgecolor, s=msize * 0.5, alpha=alpha_ubs, zorder=1)
    if plotyubs:
        y_ubs_mask = mask & (yerrsp > 0) & ~np.isnan(yvals)
        ax.scatter(xvals[y_ubs_mask], yvals[y_ubs_mask] + yerrsp[y_ubs_mask],
                   marker='^', color=edgecolor, s=msize * 0.5, alpha=alpha_ubs, zorder=1)
    if highlight:
        highlight_mask = mask & (z > 5.0)  # Example criterion for highlighting high-redshift sources
        ax.scatter(xvals[highlight_mask], yvals[highlight_mask],
                   marker='*', color='gold', s=msize * 1.5, alpha=1.0, zorder=2, label='High-z Highlight')
    if color_redshift:
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Redshift (z)')
    
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