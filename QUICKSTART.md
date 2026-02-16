# Quick Reference: Environment Setup

## Quick Start (30 seconds)


### Option 1: Use Default Location
```bash
# No setup needed! Data will be stored in ~/.astro_data
pip install -e .
```

### Option 2: Custom Location
```bash
# One-time setup
echo 'export MUSE_DATA_DIR="/your/data/path"' >> ~/.bashrc
echo 'export R21_CAT_URL="<your_R21_catalog_url>"' >> ~/.bashrc
echo 'export R21_SPEC_URL="<your_R21_spec_url>"' >> ~/.bashrc
source ~/.bashrc

# Install
pip install -e .
```

### Option 3: Per-Session Setup
```bash
# Run before each session
export MUSE_DATA_DIR="/your/data/path"
export R21_CAT_URL="<your_R21_catalog_url>"
export R21_SPEC_URL="<your_R21_spec_url>"
```

## Environment Variables

| Variable | What It Does | Example |
|----------|--------------|---------|
| `MUSE_DATA_DIR` | Base directory for all MUSE data | `/home/user/muse_data` |
| `R21_CAT_URL` | Base URL for downloading R21 catalogs | (see above) |
| `R21_SPEC_URL` | Base URL for downloading R21 spectra | (see above) |

## Common Scenarios

### Scenario 1: First Time User
```bash
# Let it use defaults
pip install -e .
python your_script.py
# Data will be in ~/.astro_data
```

### Scenario 2: Migrating from Old Code
```bash
# Keep using your existing data location
export MUSE_DATA_DIR="/media/james/63C4C5633F1EAE9F/phd/lya_outflows"
```

### Scenario 3: Shared Server
```bash
# Each user gets their own space
export MUSE_DATA_DIR="$HOME/muse_data"
```

### Scenario 4: Using .env File
```bash
# One-time setup
cp .env.example .env
nano .env  # Edit with your paths

# In your Python code
from dotenv import load_dotenv
load_dotenv()
```

## Check Your Setup

```python
from tangiers import spectroscopy as spectro

# Where will data be stored?
print(spectro.get_data_dir())

# Check specific directories
print(spectro.get_spectra_dir())
print(spectro.get_source_spectra_dir())
```

## Expected Directory Structure

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

## Troubleshooting

### "Can't find data files"
```bash
# Check your environment
echo $MUSE_DATA_DIR
# Make sure directory exists
ls -la $MUSE_DATA_DIR
```

### "Permission denied"
```bash
# Check directory permissions
ls -ld $MUSE_DATA_DIR
# Fix if needed
chmod 755 $MUSE_DATA_DIR
```

### "Download failing"
```bash
# Check URLs are set
echo $R21_CAT_URL
echo $R21_SPEC_URL
# Test connectivity
curl -I $R21_CAT_URL
curl -I $R21_SPEC_URL
```

## More Information

- **Detailed config**: See `CONFIG.md`
- **Full documentation**: See `README.md`

## Example Session

```bash
# 1. Set environment
export MUSE_DATA_DIR="$HOME/my_muse_data"

# 2. Run Python
python3
>>> from tangiers import spectroscopy as spectro
>>> spec = spectro.load_r21_spec('A2744', '1234', 'E', 'weight_skysub')
>>> # First time: downloads file to $HOME/my_muse_data/A2744/spectra/R21/
>>> # Next time: uses cached file
```