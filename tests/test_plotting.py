"""
Tests for plotting utilities in tangelo.plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

from tangelo import plotting


def _make_lya_row(deltav):
    """Build a minimal row that lya_mod_plot can evaluate."""
    tab = Table(
        {
            'LPEAKR': [1217.0],
            'AMPR': [10.0],
            'DISPR': [1.2],
            'ASYMR': [0.1],
            'CONT': [0.5],
            'SNRB': [0.0],
            'SLOPE': [np.nan],
            'TAU': [np.nan],
            'DELTAV_LYA': [deltav],
        }
    )
    return tab[0]


def test_lya_mod_plot_eml_applies_velocity_offset():
    """Setting eml=True should shift the model x-axis by DELTAV_LYA."""
    row = _make_lya_row(220.0)

    fig, ax = plt.subplots()
    plotting.lya_mod_plot(row, ax, eml=False, velocity=True)
    x_no_offset = np.array(ax.lines[-1].get_xdata())
    plt.close(fig)

    fig, ax = plt.subplots()
    plotting.lya_mod_plot(row, ax, eml=True, velocity=True)
    x_with_offset = np.array(ax.lines[-1].get_xdata())
    plt.close(fig)

    assert x_no_offset.shape == x_with_offset.shape
    assert np.allclose(x_with_offset - x_no_offset, row['DELTAV_LYA'], atol=1e-8, rtol=0)
