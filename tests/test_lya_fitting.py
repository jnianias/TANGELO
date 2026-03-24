"""
Tests for the Lyman alpha fitting functions in tangelo/lya_fitting.py.
To be run with pytest.
"""

import pytest
import numpy as np
from tangelo import lya_fitting as lyafit
from tangelo import models as mdl

def test_fit_lya():
    """
    Test the fit_lya function using a model spectrum with known parameters to ensure it can 
    recover those parameters accurately.
    """
    wave = np.arange(5500, 5600, 1.25) # Wavelength array in Angstroms using same sampling as MUSE
    true_params = {
        'AMPB'   :  100.0,
        'LPEAKB' :  5540.0,
        'DISPB'  :  3.0 / 2.355,
        'ASYMB'  :  -0.2,
        'AMPR'   :  500.0,
        'LPEAKR' :  5545.0,
        'DISPR'  :  3.0 / 2.355,
        'ASYMR'  :  0.2,
        'CONT'   :  10.0
    }
    # Generate a model spectrum using the true parameters
    spec = mdl.lya_dpeak(wave, **true_params)
    # Add some noise to the spectrum
    snr = 50.0
    noise = np.random.normal(0, np.max(spec) / snr, size=spec.shape)
    spec_noisy = spec + noise
    spec_err = np.full_like(spec, np.max(spec) / snr)

    # Initial guesses for fitting (intentionally offset from true parameters)
    initial_guesses = {
        'AMPB'   :  80.0,
        'LPEAKB' :  5535.0,
        'DISPB'  :  4.0 / 2.355,
        'ASYMB'  :  -0.1,
        'AMPR'   :  400.0,
        'LPEAKR' :  5550.0,
        'DISPR'  :  4.0 / 2.355,
        'ASYMR'  :  0.1,
        'CONT'   :  8.0
    }

    # Fit the spectrum using the fit_lya function
    fit_result = lyafit.fit_lya(wave, spec_noisy, spec_err, initial_guesses,
                                iden='TEST', cluster='TEST_CLUSTER', 
                                bounds={}, width=50, baseline='const', 
                                plot_result=False, use_bootstrap=False)
    
    # Check that the fitted parameters are close to the true parameters
    for key in true_params:
        assert fit_result[key] == pytest.approx(true_params[key], rel=0.2) # Allowing 20% relative tolerance due to noise and fitting uncertainties

def test_fit_lya_convolved():
    """
    Test the fit_lya function with convolution enabled to ensure it can recover parameters 
    when the model is convolved with the instrumental resolution.
    """
    wave = np.arange(5500, 5600, 1.25) # Wavelength array in Angstroms using same sampling as MUSE
    true_params = {
        'AMPB'   :  100.0,
        'LPEAKB' :  5540.0,
        'DISPB'  :  1.0 / 2.355,
        'ASYMB'  :  -0.2,
        'AMPR'   :  500.0,
        'LPEAKR' :  5545.0,
        'DISPR'  :  1.0 / 2.355,
        'ASYMR'  :  0.2,
        'CONT'   :  10.0
    }
    lsf_fwhm = 2.5 # Example instrumental resolution in Angstroms

    # Generate a model spectrum using the true parameters and convolve it with the LSF
    spec = mdl.convolved_lya_dpeak(wave, lsf_fwhm, **true_params)
    
    # Add some noise to the spectrum
    snr = 50.0
    noise = np.random.normal(0, np.max(spec) / snr, size=spec.shape)
    spec_noisy = spec + noise
    spec_err = np.full_like(spec, np.max(spec) / snr)

    # Initial guesses for fitting (intentionally offset from true parameters)
    initial_guesses = {
        'AMPB'   :  80.0,
        'LPEAKB' :  5535.0,
        'DISPB'  :  2.0 / 2.355,
        'ASYMB'  :  -0.1,
        'AMPR'   :  400.0,
        'LPEAKR' :  5550.0,
        'DISPR'  :  2.0 / 2.355,
        'ASYMR'  :  0.1,
        'CONT'   :  8.0
    }

    # Fit the spectrum using the fit_lya function with convolution enabled
    fit_result = lyafit.fit_lya(wave, spec_noisy, spec_err, initial_guesses,
                                iden='TEST', cluster='TEST_CLUSTER', 
                                bounds={}, width=50, baseline='const', 
                                plot_result=False, use_bootstrap=False, lsf_fwhm=lsf_fwhm)
    
    # Check that the fitted parameters are close to the true parameters
    for key in true_params:
        assert fit_result[key] == pytest.approx(true_params[key], rel=0.2) # Allowing 20% relative tolerance due to noise and fitting uncertainties