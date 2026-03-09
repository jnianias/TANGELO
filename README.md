
# TANGELO

**Tools for Analysing leNsed Galaxies with intEgral fieLd spectrOscopy**

A Python package for astronomical spectroscopy analysis, originally designed for the MUSE catalogues presented in Richard et al. 2021 (R21): https://arxiv.org/abs/2009.09784

Please note that much of the documentation is, at present, machine-generated. Contributions and corrections are welcome.

## Features

- Spectral line fitting with support for single and double-peaked profiles
- Asymmetric Gaussian models for Lyman-alpha emission
- Support for damped Lyman-alpha absorption (DLA) profiles
- Automated masking of sky lines and contaminating features
- Monte Carlo bootstrap error estimation with correlated noise support
- Integration with R21 MUSE catalogs
- Velocity/wavelength conversion utilities
- Stacking of multiply-lensed sources
- Stacking of multiple spectra and lines
- Extraction of MUSE spectra using apertures

## Installation

```bash
# Clone the repository
git clone https://github.com/jnianias/TANGELO.git
cd TANGELO

# Install in development mode
pip install -e .
```

### Dependencies

- numpy
- scipy
- astropy
- matplotlib
- mpdaf
- error_propagation
- beautifulsoup4
- requests


## Configuration

### Quick Start

By default, the package expects your MUSE catalogs and spectra to be organized under a single base data directory. You can specify this directory when using the package, or configure it in your scripts as needed.

A few environment variables are required for the package to know where to look for spectra and catalogues both locally and online; see [CONFIG.md](CONFIG.md) for detailed configuration documentation.

## Usage

### Loading Spectra

```python
from tangelo import spectroscopy as spectro

# Load R21 spectrum
spec = spectro.load_r21_spec('A2744', '1234', 'PRIOR', 'weight_skysub')

# Load aperture spectrum
spec = spectro.load_aper_spec('A2744', '1234', 'PRIOR', '2fwhm')
```

### Fitting Emission Lines

```python
from tangelo import fitting

# Fit a single emission line
result = fitting.fit_line(
    wavelength, spectrum, errors, 
    linename='CIV1548',
    initial_guesses={'LPEAK': 6850, 'FLUX': 100, 'FWHM': 3.5},
    bounds={'LPEAK': (6845, 6855)}
)

# Fit Lyman-alpha (automatic single/double peak detection)
lya_result = fitting.fit_lya_complete(
    wavelength, spectrum, errors, row, 
    width=50, mc_niter=500
)
```

### Velocity Conversions

```python
from tangelo import spectroscopy as spectro

# Convert wavelength to velocity
vel = spectro.wave2vel(observed_wavelength=6850, 
                       rest_wavelength=1215.67, 
                       redshift=4.64)

# Convert velocity to wavelength
wave = spectro.vel2wave(vel=200, restLambda=1215.67, z=4.64)
```

### Plotting

```python
from tangelo import plotting
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plotting.plotline(
    iden='1234', clus='A2744', idfrom='E',
    wln=6850, ax_in=ax, width=50,
    spec_source='R21'
)
plt.show()
```

## Module Overview

### `spectroscopy.py`
- Spectrum loading and preprocessing
- Wavelength/velocity conversions
- MUSE LSF utilities

### `fitting.py`
- Line fitting routines
- Monte Carlo bootstrap error estimation
- Masking utilities
- Lyman-alpha profile fitting

### `models.py`
- Gaussian and asymmetric Gaussian models
- Voigt profiles
- DLA absorption models

### `catalogue_operations.py`
- R21 catalog loading and downloading
- Counterpart matching
- Result insertion

### `constants.py`
- Physical constants
- Line wavelength dictionary
- Doublet definitions

### `plotting.py`
- Spectrum visualization
- Model overlay plotting


## Data Structure


The package expects the following directory structure (see CONFIG.md for details):

```
<base_data_dir>/
├── <CLUSTER>/
│   ├── catalogs/
│   │   ├── fit_results/         # results of user fitting
│   │   │   ├── <CLUSTER>lya_lines_?fwhm.fits
│   │   │   └── <CLUSTER>lines_?fwhm.fits
│   │   └── R21/                # R21 catalogues
│   │       ├── <CLUSTER>_v?.?.fits
│   │       └── <CLUSTER>_v?.?_lines.fits
│   ├── spectra/
│   │   ├── aper/                # User-extracted aperture spectra
│   │   └── R21/                 # R21 spectra
│   ├── cubes/                   # MUSE data cubes
│   └── misc/                    # Miscellaneous auxiliary files (e.g. weight maps, segmentation maps)
├── <CLUSTER2>/
│   ├── catalogs/
│   ├── spectra/
│   ├── cubes/
│   └── misc/
├── ... (other clusters)
```

## Examples

See the `notebooks/` directory for Jupyter notebook examples [to be added]

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Citation

If you use this package in your research, please cite:
- Richard et al. (2021) for the MUSE catalogs

## Contact

nianias@hku.hk

## Acknowledgments

This package uses data from the Richard et al. (2021) MUSE lensing cluster survey (https://arxiv.org/abs/2009.09784)
