"""
Test catalogue operations functions
to be run with pytest.
"""

from astropy.io import fits
from astropy.table import Table, Column
from tangelo import catalogue_operations as catops

import numpy as np
import pytest

lya_param_names = ['AMPB', 'FLUXB', 'DISPB', 'FWHMB', 'ASYMB', 'LPEAKB',
                   'AMPR', 'FLUXR', 'DISPR', 'FWHMR', 'ASYMR', 'LPEAKR',
                   'CONT', 'SLOPE', 'TAU', 'FWHM_ABS', 'LPEAK_ABS']

lya_add_colnames = ['SNRB', 'SNRR', 'Z_LYA', 'FLUXB_UB', 'RCHSQ'] # additional derived columns to be added to 
                                                                    #the catalogue for Lyman alpha fits

line_param_names = ['FLUX', 'LPEAK', 'FWHM', 'CONT', 'SLOPE']
other_line_info  = ['RCHSQ', 'FLAG', 'SNR'] # additional derived columns to be added to the catalogue 
                                            # for other lines, in addition to the fit parameters

def generate_dummy_catalogue():
    """Helper: Generate a dummy megatable in which to insert fit results. Contains
    columns for Lyman alpha fit results, one absorption singlet, and one emission doublet."""
    
    other_lines = ['SiII1260', 'CIII1907', 'CIII1909']

    line_colnames = {
        line: [f"{param}_{line}" for param in line_param_names] + \
                [f"{param}_ERR_{line}" for param in line_param_names] + \
                [f"{info}_{line}" for info in other_line_info]
        for line in other_lines
    }

    # Generate a dummy catalogue
    source_colnames = ['iden', 'CLUSTER', 'RA', 'DEC']

    lya_err_colnames = [f"{param}_ERR" for param in lya_param_names]
    lya_colnames = lya_param_names + lya_err_colnames + lya_add_colnames

    all_colnames = source_colnames + lya_colnames
    for line in other_lines:
        all_colnames += line_colnames[line]

    dummy_table = Table()

    dummy_info = {
        'iden': ['T0001', 'T0002', 'T0003'],
        'CLUSTER': ['TEST_CLUSTER'] * 3,
        'RA': [150.0, 150.1, 150.2],
        'DEC': [2.0, 2.1, 2.2],
    }

    dummy_param_val = 1.0
    dummy_err_val = 0.1
    dummy_flag = 'test'

    for colname in source_colnames:
        dummy_table.add_column(Column(data=dummy_info[colname], name=colname))

    for colname in lya_param_names:
        dummy_table.add_column(Column(data=[dummy_param_val] * 3, name=colname))
    
    for colname in lya_err_colnames:
        dummy_table.add_column(Column(data=[dummy_err_val] * 3, name=colname))

    for colname in lya_add_colnames:
        if colname in ['SNRB', 'SNRR', 'Z_LYA', 'FLUXB_UB', 'RCHSQ']:
            dummy_table.add_column(Column(data=[dummy_param_val] * 3, name=colname))
        elif colname == 'FLAG':
            dummy_table.add_column(Column(data=[dummy_flag] * 3, name=colname))

    for line in other_lines:
        for param in line_param_names:
            dummy_table.add_column(Column(data=[dummy_param_val] * 3, name=f"{param}_{line}"))
            dummy_table.add_column(Column(data=[dummy_err_val] * 3, name=f"{param}_ERR_{line}"))
        for info in other_line_info:
            if info == 'FLAG':
                dummy_table.add_column(Column(data=[dummy_flag] * 3, name=f"{info}_{line}"))
            else:
                dummy_table.add_column(Column(data=[dummy_param_val] * 3, name=f"{info}_{line}"))

    return dummy_table


def test_insert_fit_results():
    """Test that insert_fit_results correctly inserts Lya and other line results
    into the targeted row, and that other rows are not modified."""
    dummy_catalogue = generate_dummy_catalogue()

    dummy_lya_results = {
        'param_dict': {param: 2.0 for param in lya_param_names},
        'error_dict': {param: 0.2 for param in lya_param_names},
        'reduced_chisq': 2.0,
    }

    dummy_line_results = {
        'SiII1260': {
            'param_dict': {'FLUX': 2.0, 'LPEAK': 2.0, 'FWHM': 2.0, 'CONT': 2.0, 'SLOPE': 2.0},
            'error_dict': {'FLUX': 0.2, 'LPEAK': 0.2, 'FWHM': 0.2, 'CONT': 0.2, 'SLOPE': 0.2},
            'reduced_chisq': 2.0,
        },
        'CIII1907': {
            'param_dict': {'FLUX': 2.0, 'LPEAK': 2.0, 'FWHM': 2.0, 'CONT': 2.0, 'SLOPE': 2.0},
            'error_dict': {'FLUX': 0.2, 'LPEAK': 0.2, 'FWHM': 0.2, 'CONT': 0.2, 'SLOPE': 0.2},
            'reduced_chisq': 2.0,
        },
    }

    # insert_fit_results modifies in place; args are (megatab, clus, iden, ...)
    catops.insert_fit_results(dummy_catalogue, 'TEST_CLUSTER', 'T0001',
                              dummy_lya_results, dummy_line_results)

    row = dummy_catalogue[dummy_catalogue['iden'] == 'T0001'][0]

    # Lya parameters and errors
    for param in lya_param_names:
        assert row[param] == pytest.approx(2.0), f"{param}: expected 2.0, got {row[param]}"
        assert row[f'{param}_ERR'] == pytest.approx(0.2), f"{param}_ERR: expected 0.2, got {row[f'{param}_ERR']}"

    # Derived Lya quantities
    assert row['SNRR'] == pytest.approx(2.0 / 0.2)
    assert row['SNRB'] == pytest.approx(2.0 / 0.2)
    assert row['RCHSQ'] == pytest.approx(2.0)

    # Other line parameters, errors, and derived quantities
    for line in ['SiII1260', 'CIII1907']:
        for param in line_param_names:
            assert row[f'{param}_{line}'] == pytest.approx(2.0), \
                f"{param}_{line}: expected 2.0, got {row[f'{param}_{line}']}"
            assert row[f'{param}_ERR_{line}'] == pytest.approx(0.2), \
                f"{param}_ERR_{line}: expected 0.2, got {row[f'{param}_ERR_{line}']}"
        assert row[f'RCHSQ_{line}'] == pytest.approx(2.0)
        assert row[f'SNR_{line}'] == pytest.approx(2.0 / 0.2)

    # CIII1909 should have been cleared (it was in the catalogue but not in other_results)
    assert np.isnan(row['FLUX_CIII1909']), "FLUX_CIII1909 should be NaN (not in other_results)"

    # Other rows must not be modified
    for other_iden in ['T0002', 'T0003']:
        other_row = dummy_catalogue[dummy_catalogue['iden'] == other_iden][0]
        for param in lya_param_names:
            assert other_row[param] == pytest.approx(1.0), \
                f"Row {other_iden}, {param}: expected 1.0, got {other_row[param]}"


def test_blue_peak_loss_clears_b_columns():
    """Test that when a previously double-peaked Lya fit loses its blue peak,
    all blue component columns and SNRB become NaN and do not retain stale values."""
    dummy_catalogue = generate_dummy_catalogue()
    bluecols = ['AMPB', 'FLUXB', 'DISPB', 'FWHMB', 'ASYMB', 'LPEAKB']

    # Stage 1: insert a double-peaked fit (all params valid)
    double_peaked_results = {
        'param_dict': {param: 2.0 for param in lya_param_names},
        'error_dict': {param: 0.2 for param in lya_param_names},
        'reduced_chisq': 2.0,
    }
    catops.insert_fit_results(dummy_catalogue, 'TEST_CLUSTER', 'T0001',
                              double_peaked_results, None)

    row = dummy_catalogue[dummy_catalogue['iden'] == 'T0001'][0]
    assert not np.isnan(row['SNRB']), "Prerequisite: SNRB should be valid after double-peaked fit"
    assert not np.isnan(row['FLUXB']), "Prerequisite: FLUXB should be valid after double-peaked fit"

    # Stage 2: insert results where the blue peak is absent (FLUXB and other blue cols are NaN)
    single_peaked_params  = {param: (np.nan if param in bluecols else 3.0) for param in lya_param_names}
    single_peaked_errors  = {param: (np.nan if param in bluecols else 0.3) for param in lya_param_names}
    single_peaked_results = {
        'param_dict': single_peaked_params,
        'error_dict': single_peaked_errors,
        'reduced_chisq': 1.5,
    }
    catops.insert_fit_results(dummy_catalogue, 'TEST_CLUSTER', 'T0001',
                              single_peaked_results, None)

    row = dummy_catalogue[dummy_catalogue['iden'] == 'T0001'][0]

    # All blue component columns and their errors must be NaN
    for col in bluecols:
        assert np.isnan(row[col]),         f"{col} should be NaN after losing blue peak, got {row[col]}"
        assert np.isnan(row[f'{col}_ERR']), f"{col}_ERR should be NaN after losing blue peak, got {row[f'{col}_ERR']}"
    assert np.isnan(row['SNRB']), f"SNRB should be NaN after losing blue peak, got {row['SNRB']}"

    # Red component should reflect the new fit
    assert row['FLUXR']  == pytest.approx(3.0)
    assert row['SNRR']   == pytest.approx(3.0 / 0.3)
    assert row['RCHSQ']  == pytest.approx(1.5)