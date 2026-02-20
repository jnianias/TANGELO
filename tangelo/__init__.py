"""
TANGELO - Tools for ANalysing lensed Galaxies with IntEgRal field Spectroscopy
"""

# Import all modules to make them easily accessible
from . import catalogue_operations
from . import constants
from . import cosmology
from . import fitting
from . import image_processing
from . import io
from . import models
from . import plotting
from . import sky_tools
from . import spectroscopy
from . import ifs
from . import lya_profile
from . import lya_fitting

# Define what gets imported with "from tangelo import *"
__all__ = [
    'catalogue_operations',
    'constants',
    'cosmology',
    'fitting',
    'image_processing',
    'io',
    'models',
    'plotting',
    'sky_tools',
    'spectroscopy',
    'lya_profile',      
    'lya_fitting',      
    'ifs',
]

__version__ = '1.0.0'
