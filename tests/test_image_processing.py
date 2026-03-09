
"""
Tests for the image processing utilities in tangelo.image_processing module.
To be run with pytest.
"""

# Make imports robust to running from project root or tests directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from astropy.io import fits
from tangelo import image_processing as imp
from tangelo import io
import pytest

def test_get_segmap_peak_composite():
    """
    Test the get_segmap_peak function to ensure it correctly identifies the peak position of a composite source.

    The test uses source P848 from the Bullet cluster. This source is a composite consisting of several regions along an extended 
    arc, with the brightest continuum peak being in the far south end of the arc. The source position in the catalogue 
    is off the arc. 

    There are several other composite sources in the segmentation map for this cluster, so this test checks that the function
    correctly identifies the closest of these as well as the correct peak within that source.

    The expected output coordinates are taken from visual inspection of the segmentation map and the HST image.
    """

    expected_ra = 104.6299134
    expected_dec = -55.9437245

    # Load segmentation and weight map for the Bullet cluster
    seg_map = fits.open('tests/test_data/BULLET/misc/test_segmap_bullet_p848.fits') # returns HDUList
    weight_map = fits.open('tests/test_data/BULLET/misc/test_weightmap_bullet_p848.fits') # returns HDUList

    # Get optimised peak position for source P848
    ra_opt, dec_opt = imp.get_segmap_peak('P848', 'BULLET', seg_map=seg_map, weight_map=weight_map)
    # Assert that the returned coordinates are within 0.1 arcsec of the expected values
    precision = 0.1 / 3600  # degrees
    assert ra_opt == pytest.approx(expected_ra, abs=precision)
    assert dec_opt == pytest.approx(expected_dec, abs=precision)