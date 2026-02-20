"""
Lyman-alpha profile fitting class.

This module contains the LyaProfile class for fitting Lyman-alpha emission line profiles 
using asymmetric Gaussian functions. Supports both single-peak and double-peak profiles, 
as well as constant, linear, or damped Lyman alpha baselines.
"""

import numpy as np
from scipy.optimize import curve_fit
from error_propagation import Complex
from typing import Optional

from . import models as mdl
from .fitting import fit_mc, avgfunc, BootstrapParams


class LyaProfile:
    """Class to fit Lyman-alpha emission line profiles using asymmetric Gaussian functions.
    Supports both single-peak and double-peak profiles, as well as constant, linear or 
    damped Lyman alpha baselines.
    """

    # Define parameter order and bounds for single and double peak models (class-level defaults)
    lya_param_order_default = {
        1: ['AMPR', 'LPEAKR', 'DISPR', 'ASYMR', 'CONT'],
        2: ['AMPB', 'LPEAKB', 'DISPB', 'ASYMB', 'AMPR', 'LPEAKR', 'DISPR', 'ASYMR', 'CONT']
    }
    lya_bounds_default = {
        1: ([0,      1215, 0.1,   -0.1, -np.inf],
            [np.inf, 1225, 10,     0.1,  np.inf]),
        2: ([0,      4600, 1.25,  -0.5,  0,      4600, 1.25,  -0.5, -50],
            [1e7,    9350, np.inf, 0.5,  1e7,    9350, np.inf, 0.5,  1e4])
    } # These are hard coded and suitable for MUSE spectra -- modify as needed


    def __init__(self, paramdict, uncertainties):
        """Initialize the lya_profile class using dictionaries of parameters and their uncertainties.
        paramdict: Dictionary of initial parameter guesses.
        uncertainties: Dictionary of uncertainties for the parameters.
        """
        # Determine if single or double peak based on presence of LPEAKB
        self.ncomp = 2 if 'LPEAKB' in paramdict.keys() else 1
        # Determine baseline type based on presence of SLOPE or TAU
        self.baseline = 'lin' if 'SLOPE' in paramdict.keys() else 'const'
        self.baseline = 'damp' if 'TAU' in paramdict.keys() else self.baseline

        # Make per-instance copies of param order and bounds to avoid shared mutation
        self.lya_param_order = {
            1: list(self.__class__.lya_param_order_default[1]),
            2: list(self.__class__.lya_param_order_default[2])
        }
        self.lya_bounds = {
            1: [list(self.__class__.lya_bounds_default[1][0]), list(self.__class__.lya_bounds_default[1][1])],
            2: [list(self.__class__.lya_bounds_default[2][0]), list(self.__class__.lya_bounds_default[2][1])]
        }

        # Extend the lya_param_order and lya_bounds if needed
        if self.baseline == 'lin':
            self.lya_param_order[self.ncomp].extend(['SLOPE'])
            self.lya_bounds[self.ncomp][0].append(-np.inf)
            self.lya_bounds[self.ncomp][1].append(np.inf)
        elif self.baseline == 'damp':
            self.lya_param_order[self.ncomp].extend(['TAU', 'FWHM_ABS', 'LPEAK_ABS'])
            self.lya_bounds[self.ncomp][0].extend([0   , -10, 1215.67 - 2.5])
            self.lya_bounds[self.ncomp][1].extend([1000,  20, 1215.67 + 2.5])

        # Populate parameter and uncertainty dictionaries with only relevant parameters
        self.param_dict = {k: paramdict[k] for k in self.lya_param_order[self.ncomp]}
        self.err_dict   = {k: uncertainties[k] for k in self.lya_param_order[self.ncomp]}
        # Select appropriate model function
        if self.baseline == 'lin':
            self.func = mdl.lya_speak_lin if self.ncomp == 1 else mdl.lya_dpeak_lin
        elif self.baseline == 'damp':
            self.func = mdl.lya_speak_damp if self.ncomp == 1 else mdl.lya_dpeak_damp
        elif self.baseline == 'const':
            self.func = mdl.lya_speak if self.ncomp == 1 else mdl.lya_dpeak
        else:
            raise ValueError(f"Unknown baseline type: {self.baseline}")

        # Compute advanced parameters and errors immediately
        # Use Complex for error propagation if uncertainties are provided
        try:
            popt = [self.param_dict[k] for k in self.param_dict.keys()]
            perr = [self.err_dict[k] for k in self.err_dict.keys()]
            complex_pars = [Complex(val, err) for val, err in zip(popt, perr)]
            complex_dict = {k: v for k, v in zip(self.param_dict.keys(), complex_pars)}
            advpars = self.get_adv_params(complex_dict)
            self.adv_params = {k: v.value for k, v in advpars.items()}
            self.adv_errors = {k: v.error for k, v in advpars.items()}
        except Exception:
            self.adv_params = {}
            self.adv_errors = {}
    
    def get_adv_params(self, param_dict):
        """Calculate advanced parameters such as FWHM and integrated fluxes from the fit parameters.
        Returns a dictionary of advanced parameters.
        """
        advdict = {} # Dictionary to hold advanced parameters

        # Calculate FWHM
        if self.ncomp == 2:
            advdict['FWHMB'] = 2 * np.sqrt(2 * np.log(2)) * param_dict['DISPB'] / (1. - 2 * np.log(2) * param_dict['ASYMB'] ** 2.)
        advdict['FWHMR'] = 2 * np.sqrt(2 * np.log(2)) * param_dict['DISPR'] / (1. - 2 * np.log(2) * param_dict['ASYMR'] ** 2.)
        
        # Calculate integrated fluxes over a minimum intrinsic range (4 Angstroms rest-frame)
        intrange_rest = 4.0
        intrange = intrange_rest * (1. + self.param_dict['LPEAKR'] / 1215.67) # Observed frame
        if self.ncomp == 2: # Integrate blue component if present
            asymfrac = np.abs(mdl.lya_swhm(param_dict['DISPB'], param_dict['ASYMB'], -1) \
                              / mdl.lya_swhm(param_dict['DISPB'], param_dict['ASYMB'], +1))
            intstart = param_dict['LPEAKB'] - asymfrac * intrange / (1. + asymfrac)
            intend = param_dict['LPEAKB'] + intrange / (1. + asymfrac)
            xaxis = np.arange(intstart, intend, 0.125)
            modfuncb = mdl.lya_speak(
                xaxis, 
                param_dict['AMPB'], 
                param_dict['LPEAKB'], 
                param_dict['DISPB'], 
                param_dict['ASYMB'], 
                0.
            )
            posmask = modfuncb > 0.
            advdict['FLUXB'] = np.trapz(modfuncb[posmask], x = xaxis[posmask])
        asymfrac = np.abs(mdl.lya_swhm(param_dict['DISPR'], param_dict['ASYMR'], -1) \
                          / mdl.lya_swhm(param_dict['DISPR'], param_dict['ASYMR'], +1))
        intstart = param_dict['LPEAKR'] - asymfrac * intrange / (1. + asymfrac)
        intend = param_dict['LPEAKR'] + intrange / (1. + asymfrac)
        xaxis = np.arange(intstart, intend, 0.125)
        modfuncr = mdl.lya_speak(
            xaxis, 
            param_dict['AMPR'], 
            param_dict['LPEAKR'], 
            param_dict['DISPR'], 
            param_dict['ASYMR'], 
            0.
        )
        posmask = modfuncr > 0.
        advdict['FLUXR'] = np.trapz(modfuncr[posmask], x = xaxis[posmask])
        return advdict
    
    def fit_to(self, x, y, yerr, mask = None, bounds = None, use_bootstrap = False,
               bootstrap_params: Optional[BootstrapParams] = None):
        """Fit the Lyman-alpha profile to the provided data. Returns dictionaries of fit results and uncertainties."""
        if mask is None:
            mask = np.ones(np.size(y))
        if bounds is None:
            bounds = self.lya_bounds[self.ncomp]

        popt, perrs = None, None
        advpars = None
        reduced_chisq = None

        # Use fit_mc for bootstrapping, passing only user-provided params
        if use_bootstrap:
            print("Fitting with bootstrapping...")
            fit_mc_kwargs = {
                'f': self.func,
                'x': x[mask],
                'y': y[mask],
                'yerr': yerr[mask],
                'p0': list(self.param_dict.values()),
                'bounds': bounds,
                'return_sample': True
            }
            if bootstrap_params:
                fit_mc_kwargs.update(bootstrap_params)

            fitres = fit_mc(**fit_mc_kwargs) # Pass to fit_mc

            if fitres is None:
                print("Fitting failed.")
                return None, None, None, None
            fitted_params, distributions = fitres
            popt = np.array(fitted_params[0]) # Best-fit parameters from fit_mc
            perrs = np.array(fitted_params[1]) # Errors from fit_mc
            
            # Make a dictionary to pass to get_adv_params (does not need to be Complex here)
            param_dicts = []
            for pset in distributions:
                param_dicts.append({k: v for k, v in zip(self.param_dict.keys(), pset)})
            
            # Calculate advanced parameters for each distribution and then get median and errors
            advparams_distr = [self.get_adv_params(p) for p in param_dicts]
            adv_params = {}
            adv_errors = {}
            for key in advparams_distr[0].keys():
                vals = np.array([d[key] for d in advparams_distr])
                med, err = avgfunc(vals, bootstrap_params.get('errfunc', 'stddev') if bootstrap_params else 'stddev')
                adv_params[key] = med
                adv_errors[key] = err

            # Calculate reduced chi-squared of the best fit
            residuals = y[mask] - self.func(x[mask], *popt)
            reduced_chisq = np.nansum((residuals / yerr[mask])**2) / (np.sum(mask) - len(popt))
        else:
            print("Fitting without bootstrapping. Errors may be underestimated.")
            max_retries = 3
            success = False
            popt, pcov = None, None
            for attempt in range(max_retries):
                try:
                    popt, pcov = curve_fit(
                        self.func, x[mask], y[mask], sigma=yerr[mask], absolute_sigma=True,
                        p0=list(self.param_dict.values()), method='trf', max_nfev=100000, bounds=bounds
                    )
                    success = True
                    break
                except (RuntimeError, ValueError) as e:
                    print(f"curve_fit attempt {attempt+1} failed: {e}")
            if not success:
                print("All curve_fit attempts failed.")
                return None, None, None, None
            perrs = np.sqrt(np.diag(pcov))
            complex_pars = [Complex(val, err) for val, err in zip(popt, perrs)]
            advpars = self.get_adv_params(complex_pars)
            adv_params = {k: v.value for k, v in advpars.items()}
            adv_errors = {k: v.error for k, v in advpars.items()}
            residuals = y[mask] - self.func(x[mask], *popt)
            reduced_chisq = np.nansum((residuals / yerr[mask])**2) / (np.sum(mask) - len(popt))

        # Prepare output dictionaries
        fit_params = {k: v for k, v in zip(self.param_dict.keys(), popt)} if popt is not None else {}
        fit_errors = {k: v for k, v in zip(self.param_dict.keys(), perrs)} if perrs is not None else {}
        # Merge advanced params
        fit_params.update(adv_params)
        fit_errors.update(adv_errors)

        # Update instance attributes to reflect latest fit
        self.adv_params = adv_params
        self.adv_errors = adv_errors

        return fit_params, fit_errors, self.func, reduced_chisq
