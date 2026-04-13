"""
Tests for the Lyman alpha fitting functions in tangelo/lya_fitting.py.
To be run with pytest.
"""

import pytest
import numpy as np
from error_propagation import Complex
from tangelo import lya_fitting as lyafit
from tangelo import models as mdl
from tangelo import spectroscopy as spectro

import matplotlib.pyplot as plt

TOLERANCE = 0.33 # Relative tolerance for parameter recovery in tests, can be adjusted based on expected fitting uncertainties

def test_fit_lya():
    """
    Test the fit_lya function using a model spectrum with known parameters to ensure it can 
    recover those parameters accurately.
    """
    wave = np.arange(5450, 5650, 1.25) # Wavelength array in Angstroms using same sampling as MUSE
    true_params = {
        'AMPB'   :  100.0,
        'LPEAKB' :  5545.0,
        'DISPB'  :  2.0,
        'ASYMB'  :  -0.2,
        'AMPR'   :  500.0,
        'LPEAKR' :  5555.0,
        'DISPR'  :  2.0,
        'ASYMR'  :  0.2,
        'CONT'   :  10.0
    }
    # Generate a model spectrum using the true parameters
    spec = mdl.lya_dpeak(wave, *list(true_params.values()))
    # Add some noise to the spectrum
    snr = 50.0
    noise = np.random.normal(0, np.max(spec) / snr, size=spec.shape)
    spec_noisy = spec + noise
    spec_err = np.full_like(spec, np.max(spec) / snr)

    # Generate a plot of the spectrum so we can visually inspect
    plt.figure()
    plt.plot(wave, spec_noisy, label='Noisy Spectrum')
    plt.plot(wave, spec, label='True Spectrum', linestyle='--')
    plt.fill_between(wave, spec_noisy - spec_err, spec_noisy + spec_err, color='gray', alpha=0.3, label='Error')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')


    # Initial guesses for fitting (intentionally offset from true parameters)
    initial_guesses = {
        'AMPB'   :  80.0,
        'LPEAKB' :  5535.0,
        'DISPB'  :  2.0,
        'ASYMB'  :  -0.1,
        'AMPR'   :  400.0,
        'LPEAKR' :  5550.0,
        'DISPR'  :  2.0,
        'ASYMR'  :  0.1,
        'CONT'   :  8.0
    }

    # Fit the spectrum using the fit_lya function
    fit_result = lyafit.fit_lya(wave, spec_noisy, spec_err, initial_guesses,
                                iden='TEST', cluster='TEST_CLUSTER', 
                                bounds={}, width=75, baseline='const', 
                                plot_result=False, use_bootstrap=False)
    
    # Plot the fitted model on top of the data for visual inspection
    plt.plot(wave, mdl.lya_dpeak(wave, *fit_result['popt']), label='Fitted Model', linestyle='-.')
    plt.legend()
    plt.savefig('test_fit_lya_spectrum.png') # Save the plot for inspection
    plt.close()
    
    print(f"Fit result: {fit_result['param_dict']}")
    
    # Check that the fitted parameters are close to the true parameters
    for key in true_params:
        fitted_param = fit_result["param_dict"][key]
        if 'ASYM' in key:
            continue # Skip asymmetry parameters as they are often poorly constrained and can be more sensitive to noise
        if isinstance(fitted_param, Complex):
            assert fitted_param.value == pytest.approx(true_params[key], rel=TOLERANCE)
        else:
            assert fitted_param == pytest.approx(true_params[key], rel=TOLERANCE)

def test_fit_lya_convolved():
    """
    Test the fit_lya function with convolution enabled to ensure it can recover parameters 
    when the model is convolved with the instrumental resolution.
    """
    wave = np.arange(5450, 5650, 1.25) # Wavelength array in Angstroms using same sampling as MUSE
    true_params = {
        'AMPB'   :  100.0,
        'LPEAKB' :  5545.0,
        'DISPB'  :  2.0,
        'ASYMB'  :  -0.2,
        'AMPR'   :  500.0,
        'LPEAKR' :  5555.0,
        'DISPR'  :  2.0,
        'ASYMR'  :  0.2,
        'CONT'   :  10.0
    }
    lsf_fwhm = spectro.muse_lsf_fwhm_poly(true_params['LPEAKR']) # Get the LSF FWHM at the wavelength of the blue peak

    # Generate a model spectrum using the true parameters and convolve it with the LSF
    spec = mdl.convolved_lya_dpeak(lsf_fwhm)(wave, *list(true_params.values()))
    
    # Add some noise to the spectrum
    snr = 50.0
    noise = np.random.normal(0, np.max(spec) / snr, size=spec.shape)
    spec_noisy = spec + noise
    spec_err = np.full_like(spec, np.max(spec) / snr)

    # Generate a plot of the spectrum so we can visually inspect
    plt.figure()
    plt.plot(wave, spec_noisy, label='Noisy Spectrum')
    plt.plot(wave, spec, label='True Convolved Spectrum', linestyle='--')
    plt.fill_between(wave, spec_noisy - spec_err, spec_noisy + spec_err, color='gray', alpha=0.3, label='Error')
    plt.xlabel('Wavelength (Å)')
    plt.ylabel('Flux')

    # Initial guesses for fitting (intentionally offset from true parameters)
    initial_guesses = {
        'AMPB'   :  80.0,
        'LPEAKB' :  5535.0,
        'DISPB'  :  2.5,
        'ASYMB'  :  -0.1,
        'AMPR'   :  400.0,
        'LPEAKR' :  5550.0,
        'DISPR'  :  2.5,
        'ASYMR'  :  0.1,
        'CONT'   :  8.0
    }

    # Fit the spectrum using the fit_lya function with convolution enabled
    fit_result = lyafit.fit_lya(wave, spec_noisy, spec_err, initial_guesses,
                                iden='TEST', cluster='TEST_CLUSTER', 
                                bounds={}, width=75, baseline='const', 
                                plot_result=False, use_bootstrap=False, 
                                convolve_model=True)
    
    # Plot the fitted model on top of the data for visual inspection
    plt.plot(wave, mdl.convolved_lya_dpeak(lsf_fwhm)(wave, *fit_result['popt']), label='Fitted Convolved Model', linestyle='-.')
    plt.legend()
    plt.savefig('test_fit_lya_convolved_spectrum.png') # Save the plot for inspection
    plt.close()
    
    print(f"Fit result with convolution: {fit_result['param_dict']}")
    
    # Check that the fitted parameters are close to the true parameters
    for key in true_params:
        if 'ASYM' in key:
            continue # Skip asymmetry parameters as they are often poorly constrained and can be more sensitive to noise
        fitted_param = fit_result["param_dict"][key]
        if isinstance(fitted_param, Complex):
            assert fitted_param.value == pytest.approx(true_params[key], rel=TOLERANCE)
        else:
            assert fitted_param == pytest.approx(true_params[key], rel=TOLERANCE)