"""
Tests for the fitting utilities in tangelo.fitting module.
To be run with pytest.
"""

import pytest
from tangelo import fitting as aufit
from tangelo import spectroscopy as auspec
from tangelo import catalogue_operations as aucat

wavedict = auspec.wavedict

def test_get_initial_guesses_from_catalog():
    """
    Test the get_initial_guesses_from_catalog function to ensure it retrieves correct initial guesses from the catalog.
    """

    # Test case 1: Lyman alpha line from P594 in MACS0416NE
    full_iden = 'P594'
    cluster   = 'MACS0416NE'
    line_name = 'LYALPHA'

    line_table = aucat.get_muse_cand(full_iden, cluster)

    initial_guesses = aufit.get_initial_guesses_from_catalog(line_table, line_name, type='em')

    expected_guesses = {
        'AMPB'   :  380.14799202510267,
        'LPEAKB' :  5152.057737999559 - 2 * (1 + (5152.057737999559 / wavedict['LYALPHA'] - 1)),
        'DISPB'  :  3.9649905781616326 / 2.355,
        'ASYMB'  :  -0.1,
        'AMPR'   :  380.14799202510267,
        'LPEAKR' :  5152.057737999559,
        'DISPR'  :  3.9649905781616326 / 2.355,
        'ASYMR'  :  0.1,
        'CONT'   :  6.827872333173079
    }

    assert initial_guesses == pytest.approx(expected_guesses, rel=1e-5)

def test_gen_bounds():
    """
    Test the gen_bounds function to ensure it generates correct parameter bounds.
    """

    full_iden = 'P594'
    cluster   = 'MACS0416NE'
    line_name = 'LYALPHA'

    # Retrieve line table from catalog
    line_table = aucat.get_muse_cand(full_iden, cluster)

    # Test case 1: automatic bounds for Lyman alpha
    initial_guesses = aufit.get_initial_guesses_from_catalog(line_table, line_name, type='em')

    z = ( initial_guesses['LPEAKR'] / wavedict['LYALPHA'] ) - 1

    lpeakb_tol = [-3 * (1 + z), -1 * (1 + z)]
    lpeakr_tol = [-1 * (1 + z), 3 * (1 + z)]

    expected_bounds = {
        'AMPB'   : (0, 10000),
        'LPEAKB' : (initial_guesses['LPEAKR'] + lpeakb_tol[0], 
                    initial_guesses['LPEAKR'] + lpeakb_tol[1]),
        'DISPB'  : (0.2 * (1 + z), 1 * (1 + z)),
        'ASYMB'  : (-0.5, 0.2),
        'AMPR'   : (0, 10000),
        'LPEAKR' : (initial_guesses['LPEAKR'] + lpeakr_tol[0], 
                    initial_guesses['LPEAKR'] + lpeakr_tol[1]),
        'DISPR'  : (0.2 * (1 + z), 1 * (1 + z)),
        'ASYMR'  : (-0.5, 0.5),
        'CONT'   : (-50, 2000)
    }
    
    bounds = aufit.gen_bounds(initial_guesses, line_name, force_sign='positive')

    # Check keys are the same
    assert set(bounds.keys()) == set(expected_bounds.keys())
    # Check values are approximately equal
    for key in expected_bounds:
        assert bounds[key] == pytest.approx(expected_bounds[key], rel=1e-5)