# Configuration Guide for TANGIERS

This document describes the environment variables used by TANGIERS for configuring data paths and URLs.

## Quick Start

TANGIERS needs to know where to look for the data that is to be used, namely MUSE data cubes, catalogues,
sourc spectra etc. The quickest way to get started is to set the MUSE_DATA_DIR environment variable

```bash
export MUSE_DATA_DIR="/path/to/your/data"
```

All other paths will be automatically derived from this base directory using the following structure:

```
$MUSE_DATA_DIR/
├── A2744/
│    ├── catalogs/          # catalog files
│    │    ├── fit_results/  # results of user fitting
│    │    └── R21/          # R21 catalogues
│    ├── spectra/           # R21 spectra files
│    │    ├── aper/         # User-extracted aperture spectra
│    │    └── R21/          # R21 spectra
│    ├── cubes/             # MUSE data cubes
│    └── misc/              # Miscellaneous auxiliary files (e.g. weight maps, segmentation maps)
├── MACS0416/
│   ├── catalogs/
│   ├── spectra/
│   ├── cubes/
│   └── misc/
├── BULLET/
│   ├── catalogs/
│   ├── spectra/
│   ├── cubes/
│   └── misc/
└── ... (other clusters)
```

## Environment Variables

### Required Variables

The MUSE_DATA_DIR environment variable must be set by the user -- an error is raised when it is not.

### Optional Variables

#### `MUSE_DATA_DIR`
- **Description**: Base directory for all MUSE data
- **Example**: `export ASTRO_DATA_DIR="~/muse_data"` (in your home directory)

#### `R21_CAT_URL`
- **Description**: Base URL for downloading Richard+21 (R21) catalogs from the official MUSE data release.
- **Default**: Will prompt interactively when needed
- **Example**: `export R21_CAT_URL="<your_R21_data_url>"`
- **Note**: The package will automatically append the appropriate path to the individual clusters

#### `R21_SPEC_URL`
- **Description**: Base URL for downloading R21 spectra. As these are not included in the data release, permission must be sought from the authors.
- **Default**: Will prompt interactively when needed
- **Example**: `export R21_SPEC_URL="<your_R21_data_url>"`
- **Note**: The package will automatically append the appropriate path to the clusters


## Setting Environment Variables

### Temporary (Current Session Only)

```bash
export MUSE_DATA_DIR="/path/to/your/data"
export R21_CAT_URL="<your_R21_catalog_url>"
```

### Permanent (Add to Shell Configuration)

Add these lines to your shell configuration file:
- **Bash**: `~/.bashrc` or `~/.bash_profile`
- **Zsh**: `~/.zshrc`
- **Fish**: `~/.config/fish/config.fish`

Example for Bash/Zsh:
```bash
# Add to ~/.bashrc or ~/.zshrc
export MUSE_DATA_DIR="/path/to/your/data"
export R21_CAT_URL="<your_R21_catalog_url>"
```

After editing, reload your configuration:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

### Using a .env File (Python)

You can also use a `.env` file with the `python-dotenv` package:

1. Create a `.env` file in your project directory:
```bash
ASTRO_DATA_DIR=/path/to/your/data
R21_URL=<your_R21_data_url>
```

2. Load it in your Python script:
```python
from dotenv import load_dotenv
load_dotenv()

# Now import tangiers
from tangiers import spectroscopy, fitting
```

## Data Directory Structure

The package expects the following directory structure (created automatically as needed):

```
$MUSE_DATA_DIR/
├── A2744/
│    ├── catalogs/          # catalog files
│    │    ├── fit_results/  # results of user fitting
│    │         ├── A2744lya_lines_?fwhm.fits
│    │         └── A2744lines_?fwhm.fits
│    │    └── R21/          # R21 catalogues
│    │         ├── A2744_v?.?.fits
│    │         └── A2744_v?.?_lines.fits
│    ├── spectra/           # R21 spectra files
│    │    ├── aper/         # User-extracted aperture spectra
│    │    └── R21/          # R21 spectra
│    ├── cubes/             # MUSE data cubes
│    └── misc/              # Miscellaneous auxiliary files (e.g. weight maps, segmentation maps)
├── MACS0416/
│   ├── catalogs/
│   ├── spectra/
│   ├── cubes/
│   └── misc/
├── BULLET/
│   ├── catalogs/
│   ├── spectra/
│   ├── cubes/
│   └── misc/
└── ... (other clusters)
```

## Troubleshooting

### Package can't find my data files
1. Check that your environment variable is set: `echo $MUSE_DATA_DIR`
2. Verify the directory structure matches the expected layout
3. Ensure the package has read/write permissions for the directory

### Downloads are failing
1. Check that `R21_CAT_URL` and `R21_SPEC_URL` is set correctly
2. Verify you have internet connectivity
3. Check that the destination directory has write permissions

## Example Setup Script

Save this as `setup_astro_env.sh`:

```bash
#!/bin/bash
# Configuration for TANGIERS

# Base data directory (customize this)
export MUSE_DATA_DIR="/path/to/your/data"

# R21 catalog URL (optional, will prompt if not set)
export R21_CAT_URL="<your_R21_data_url>"

# R21 spectra URL (optional, will prompt if not set)
export R21_SPEC_URL="<your_R21_spec_url>"

# Optional: Override specific directories if needed
# export R21_SPECTRA_DIR="/custom/path/to/spectra"
# export R21_CATALOG_DIR="/custom/path/to/catalogs"
# export SOURCE_SPECTRA_DIR="/custom/path/to/source_spectra"

echo "TANGIERS environment configured:"
echo "  MUSE_DATA_DIR = $MUSE_DATA_DIR"
echo "  R21_CAT_URL = $R21_CAT_URL"
echo "  R21_SPEC_URL = $R21_SPEC_URL"
```

Source it before running your scripts:
```bash
source setup_astro_env.sh
python your_analysis_script.py
```
