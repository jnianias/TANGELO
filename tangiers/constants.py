import numpy as np
from . import io
import astropy.table as aptb

# Fundamental constants used extensively
c = 299792.458  # speed of light in km/s
rwlya = 1215.67 # The rest wavelength of Lyman alpha

# Some common absorption lines
abslines = [
    'SiII1260',
    'CII1334',
    'SiIV1394',
    'SiIV1403',
]

# Some common emission lines
emlines = [
    'CIV1548',
    'CIV1551',
    'HeII1640',
    'OIII1660',
    'OIII1666',
    'CIII1907',
    'CIII1909'
]

# Doublets (old)
# doublets = {
#     'OVI1032': ('OVI1032', 'OVI1038'),
#     'NV1238': ('NV1238','NV1243'),
#     'SiIV1394': ('SiIV1394','SiIV1403'),
#     'NIV1483': ('NIV1483','NIV1487'),
#     'CIV1548': ('CIV1548','CIV1551'),
#     'FeII1608': ('FeII1608','FeII1611'),
#     'OIII1660': ('OIII1660','OIII1666'),
#     'AlIII1854': ('AlIII1854','AlIII1862'),
#     'SiIII1883': ('SiIII1883','SiIII1892'),
#     'CIII1907': ('CIII1907','CIII1909')
# }

# Master wavelength dictionary made using the lines from the MACS0416NE catalogue
cat = io.load_r21_catalogue('MACS0416NE', type='lines')
# Get unique lines and their rest wavelengths
cat_unique = aptb.unique(cat, keys='LINE')
wavedict = {row['LINE']: row['LBDA_REST'] for row in cat_unique}

# Some special lines not in the catalogue
wavedict['LYALPHA']  = 1215.67
wavedict['SiII1527'] = 1526.71
wavedict['DUST']     = 2175.0

# identify doublets by looking for lines from the same species within 10 angstroms
doublets = {}
for line1 in wavedict:
    for line2 in wavedict:
        if line1 != line2 and wavedict[line1] < wavedict[line2]:
            if abs(wavedict[line1] - wavedict[line2]) < 10.0:
                species1 = ''.join([i for i in line1 if not i.isdigit()])
                species2 = ''.join([i for i in line2 if not i.isdigit()])
                if species1 == species2:
                    doublets[line1] = (line1, line2)

# Compute first and second lines of each doublet
flines = np.array([x[0] for x in list(doublets.values())])
slines = np.array([x[1] for x in list(doublets.values())])

skylines = {
    'Raman OVI': 6825.4,
    'OI6300': 6300.304,
    'OI5577': 5577.339,
}