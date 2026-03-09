import numpy as np
from scipy.optimize import curve_fit
from astropy.stats import median_absolute_deviation as mad
from scipy.special import wofz
from . import constants as const


def gaussian(x, flux, center, fwhm, continuum, slope):
    """
    Single Gaussian function with a linear continuum.
    x: Wavelength array
    flux:       Integrated flux of the Gaussian
    center:     Center wavelength of the Gaussian
    fwhm:       Full Width at Half Maximum (FWHM) of the Gaussian
    continuum:  Continuum level at the center wavelength
    slope:      Slope of the continuum
    """
    # Convert FWHM to standard deviation
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    
    # Calculate amplitude from integrated flux
    amp = flux / (sigma * np.sqrt(2 * np.pi))
    
    # Gaussian function
    gauss_func = amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    
    # Continuum (linear)
    continuum_term = continuum + slope * (x - center)
    
    return gauss_func + continuum_term


def gaussian_doublet(wavelength_ratio):
    """
    Double Gaussian function for fitting doublet lines with a fixed or variable wavelength ratio.
    wavelength_ratio: The ratio of the wavelengths of the two lines in the doublet.
    Returns a function that can be used for curve fitting.
    """
    def doublet_func(x, flux1, center, fwhm, flux2, continuum, slope):
        """
        x: Wavelength array
        flux1:      Integrated flux of the first Gaussian
        center:     Center wavelength of the first Gaussian
        fwhm:       Full Width at Half Maximum (FWHM) of both Gaussians
        flux2:      Integrated flux of the second Gaussian
        continuum:  Continuum level at the center wavelength
        slope:      Slope of the continuum
        """
        # Convert FWHM to standard deviation
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        
        # Calculate amplitudes from integrated fluxes
        amp1 = flux1 / (sigma * np.sqrt(2 * np.pi))
        amp2 = flux2 / (sigma * np.sqrt(2 * np.pi))
        
        # Second Gaussian center
        center2 = center * wavelength_ratio

        # Midpoint between the two centers for continuum calculation
        mid_center = (center + center2) / 2
        
        # Gaussian functions
        gauss1 = amp1 * np.exp(-0.5 * ((x - center) / sigma) ** 2)
        gauss2 = amp2 * np.exp(-0.5 * ((x - center2) / sigma) ** 2)
        
        # Continuum (linear)
        continuum_term = continuum + slope * (x - mid_center)
        
        return gauss1 + gauss2 + continuum_term
    
    return doublet_func

def gaussian_doublet_vel(wavelengths, z=0):
    """
    Double Gaussian function for fitting doublet lines in velocity space with a fixed wavelength ratio.

    Parameters
    ----------
    wavelengths: tuple
        The wavelengths of the two lines in the doublet.
    z: float, optional
        Redshift to apply to the wavelengths (default is 0).
    
    Returns
    -------
    doublet_func : function
        A double-gaussian function with a fixed velocity separation that can be used for curve fitting.
    """
    # Calculate observed wavelengths
    wave1 = wavelengths[0] * (1 + z)
    wave2 = wavelengths[1] * (1 + z)

    # Calculate the separation in velocity space using the relativistic Doppler formula
    delta_v = const.c * ( (wave2**2 - wave1**2) / (wave2**2 + wave1**2) )


    def doublet_func(x, flux1, center, fwhm, flux2, continuum, slope):
        """
        x: Wavelength array
        flux1:      Integrated flux of the first Gaussian
        center:     Center velocity of the first Gaussian
        fwhm:       Full Width at Half Maximum (FWHM) of both Gaussians
        flux2:      Integrated flux of the second Gaussian
        continuum:  Continuum level at the center wavelength
        slope:      Slope of the continuum
        """
        # Convert FWHM to standard deviation
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        
        # Calculate amplitudes from integrated fluxes
        amp1 = flux1 / (sigma * np.sqrt(2 * np.pi))
        amp2 = flux2 / (sigma * np.sqrt(2 * np.pi))
        
        # Second Gaussian center
        center2 = center + delta_v

        # Midpoint between the two centers for continuum calculation
        mid_center = (center + center2) / 2
        
        # Gaussian functions
        gauss1 = amp1 * np.exp(-0.5 * ((x - center) / sigma) ** 2)
        gauss2 = amp2 * np.exp(-0.5 * ((x - center2) / sigma) ** 2)
        
        # Continuum (linear)
        continuum_term = continuum + slope * (x - mid_center)
        
        return gauss1 + gauss2 + continuum_term
    
    return doublet_func


def asym_gauss(x, amp, lpeak, disp, asym):
        """Single asymmetric Gaussian function for Lyman-alpha emission line.
        amp: Amplitude of the Gaussian
        lpeak: Peak wavelength of the Gaussian
        disp: Dispersion parameter of the Gaussian
        asym: Asymmetry parameter of the Gaussian
        """
        # Calculate modified dispersion based on asymmetry
        mod_disp = disp + asym * (x - lpeak)
        gauss = amp * np.exp(-0.5 * ((x - lpeak) / mod_disp) ** 2)
        
        return gauss


def lya_speak(x, amp, lpeak, disp, asym, const):
    """Single asymmetric Gaussian function with constant baseline for Lyman-alpha emission line."""
    # Continuum (constant)
    continuum_term = const

    return asym_gauss(x, amp, lpeak, disp, asym) + continuum_term


def lya_speak_lin(x, amp, lpeak, disp, asym, const, slope):
    """Single asymmetric Gaussian function with linear baseline for Lyman-alpha emission line."""
    continuum_term = const + slope * (x - lpeak)

    return asym_gauss(x, amp, lpeak, disp, asym) + continuum_term


def lya_dpeak(x, ampb, lpeakb, dispb, asymb, ampr, lpeakr, dispr, asymr, const):
    """Double asymmetric Gaussian function with constant baseline for Lyman-alpha emission line."""
    # Continuum (constant)
    continuum_term = const

    return (asym_gauss(x, ampb, lpeakb, dispb, asymb) + 
            asym_gauss(x, ampr, lpeakr, dispr, asymr) + 
            continuum_term)


def lya_dpeak_lin(x, ampb, lpeakb, dispb, asymb, ampr, lpeakr, dispr, asymr, const, slope):
    """Double asymmetric Gaussian function with linear baseline for Lyman-alpha emission line."""
    continuum_term = const + slope * (x - (lpeakb + lpeakr) / 2.)
    
    return (asym_gauss(x, ampb, lpeakb, dispb, asymb) + 
            asym_gauss(x, ampr, lpeakr, dispr, asymr) + 
            continuum_term)

def fast_voigt_profile(x, x0, sigma, gamma, amplitude=1.0):
    """Fast Voigt profile normalized to peak = amplitude"""
    z = ((x - x0) + 1j * gamma) / (sigma * np.sqrt(2))
    voigt = wofz(z).real / (sigma * np.sqrt(2 * np.pi))
    # Normalize to specified amplitude
    if amplitude != 1.0:
        peak = wofz(1j * gamma / (sigma * np.sqrt(2))).real / (sigma * np.sqrt(2 * np.pi))
        voigt = amplitude * voigt / peak
    return voigt

def lya_speak_damp(x, amp, lpeak, disp, asym, const, tau, fwhm, lpeak_abs):
    """Single asymmetric Gaussian function with DLA baseline for Lyman-alpha emission line.
    x: Wavelength array
    amp: Amplitude of the asymmetric Gaussian
    lpeak: Peak wavelength of the asymmetric Gaussian
    disp: Dispersion parameter of the asymmetric Gaussian
    asym: Asymmetry parameter of the asymmetric Gaussian
    const: Continuum level at the center wavelength
    tau: Optical depth at line center for the DLA
    fwhm: FWHM for the Voigt profile of the DLA
    lpeak_abs: Absolute peak wavelength for the DLA
    """
    # Get the emission component
    emission = asym_gauss(x, amp, lpeak, disp, asym)
   
    # Convert FWHM to native Voigt parameters
    sigma  = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gamma  = fwhm / 2.0

    # Use fast Voigt
    voigt = fast_voigt_profile(x, lpeak_abs, sigma, gamma)
    voigt_norm = voigt / np.max(voigt)
    dla_baseline = np.exp(-tau * voigt_norm)
    
    return const * dla_baseline + emission

def lya_dpeak_damp(x, ampb, lpeakb, dispb, asymb, ampr, lpeakr, dispr, asymr, const, tau, fwhm, lpeak_abs):
    """Double asymmetric Gaussian function with DLA baseline for Lyman-alpha emission line.
    x: Wavelength array
    amp: Amplitude of the asymmetric Gaussian
    lpeak: Peak wavelength of the asymmetric Gaussian
    disp: Dispersion parameter of the asymmetric Gaussian
    asym: Asymmetry parameter of the asymmetric Gaussian
    const: Continuum level at the center wavelength
    tau: Optical depth at line center for the DLA
    fwhm: FWHM for the Voigt profile of the DLA
    lpeak_abs: Absolute peak wavelength for the DLA
    """
    # Get the emission component
    emission = asym_gauss(x, ampb, lpeakb, dispb, asymb) \
                + asym_gauss(x, ampr, lpeakr, dispr, asymr)
   
   # Convert FWHM to native Voigt parameters
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gamma = fwhm / 2.0

    # Use fast Voigt
    voigt = fast_voigt_profile(x, lpeak_abs, sigma, gamma)
    voigt_norm = voigt / np.max(voigt)
    dla_baseline = np.exp(-tau * voigt_norm)
    
    return const * dla_baseline + emission

def lya_swhm(disp, asym, side):
    """Calculate the half-width at half-maximum (HWHM) of an asymmetric Gaussian.
    disp: Dispersion parameter of the asymmetric Gaussian
    asym: Asymmetry parameter of the asymmetric Gaussian
    side: +1 for red side, -1 for blue side
    """
    if side in [+1, -1]:
        numerator = np.sqrt(np.log(4)) * disp
        denominator = 1. - side * np.sqrt(np.log(4)) * asym

        return side * numerator / denominator
    else:
        raise ValueError("Side must be +1 or -1.")
