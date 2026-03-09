"""
Quality control utilities for processing of MUSE spectra
"""

from . import io
from . import image_processing as improc
from . import plotting as plot
from . import fitting
from . import spectroscopy

import astropy.table as aptb
import astropy.units as u
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from astropy.table import Column, MaskedColumn
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib import patches
from astropy.wcs import WCS

def replace_nan_lya_errors(megatab):
    """
    Replace NaN error values in the megatable with reasonable estimates based on the SNR of the Lyman alpha line.

    Parameters
    ----------
    megatab : astropy.table.Table
        The megatable containing the features and their errors.

    Returns
    -------
    astropy.table.Table
        The megatable with NaN errors replaced.
    """
    # list of parameter names to check
    param_names = [
        'FLUXR', 'FWHMR', 'ASYMR', 'DISPR', 'AMPR',
        'FLUXB', 'FWHMB', 'ASYMB', 'DISPB', 'AMPB',
        'CONT', 'SLOPE', 'TAU', 'FWHM_ABS', 'LPEAK_ABS',
        'Z_LYA', 'MU']
    
    # Look for any cases where the parameter is finite but the error is NaN, and replace the error with an estimate based on the SNR
    for param in param_names:
        param_err = f"{param}_ERR"
        if param in megatab.colnames and param_err in megatab.colnames:
            has_nan_error = np.isnan(megatab[param_err]) & np.isfinite(megatab[param])
            if np.any(has_nan_error):
                # Estimate error as parameter value divided by SNR, with a minimum error threshold to avoid zero errors
                scale_fac = np.maximum(1 / megatab['SNRR'][has_nan_error], 5e-2) # Set a minimum error of 5% of the parameter value to avoid zero errors
                estimated_error = np.abs(megatab[param][has_nan_error]) * scale_fac
                megatab[param_err][has_nan_error] = estimated_error
 
    return megatab

def prep_feature_matrix(megatab, features, nan_policy='raise_error', snr_cap=100, rchsq_cap=10):
    """
    Prepare the feature matrix for LOF calculation, handling NaN values according to the specified policy.

    Parameters
    ----------
    megatab : astropy.table.Table
        The megatable containing the features.
    features : list
        List of feature names to include in the matrix.
    nan_policy : str, optional
        Policy for handling NaN values ('raise_error', 'impute', or 'remove'; default is 'raise_error').
    snr_cap : float, optional
        SNR threshold above which to consider imputation (default is 100).
    rchsq_cap : float, optional
        Reduced chi-squared threshold above which to consider imputation (default is 10).

    Returns
    -------
    np.ndarray
        The prepared feature matrix ready for LOF calculation.
    np.ndarray
        Boolean mask indicating which rows have NaN values (if nan_policy is 'remove').
    int
        The number of rows with NaN values.
    """
    # Extract features for LOF
    feature_matrix = np.array([megatab[name] for name in features]).T
    
    # Check for and handle NaNs
    has_nan = np.isnan(feature_matrix).any(axis=1)
    n_nan = np.sum(has_nan)
    
    # Extremely high SNR sources sometimes have high reduced chi-squared values simply because
    # any small deviation from the model is statistically significant at such high SNR even though
    # the fit is qualitatively reasonable. We impute the SNR and reduced chi-squared values of 
    # these sources to the column medians for outlier detection purposes.
    
    # Create a copy of the feature matrix for imputation
    feature_matrix_imputed = feature_matrix.copy()
    
    # Find indices of SNRR and RCHSQ in the feature list
    snr_idx = features.index('SNRR') if 'SNRR' in features else None
    rchsq_idx = features.index('RCHSQ') if 'RCHSQ' in features else None
    
    # Identify sources to impute: both high SNR AND high reduced chi-squared
    impute_mask = np.zeros(len(megatab), dtype=bool)
    if snr_idx is not None and rchsq_idx is not None:
        high_snr = feature_matrix[:, snr_idx] > snr_cap
        high_rchsq = feature_matrix[:, rchsq_idx] > rchsq_cap
        impute_mask = high_snr & high_rchsq
        
        # Impute to column medians
        if np.any(impute_mask):
            print(f"Imputing {np.sum(impute_mask)} high-SNR, high-χ² sources to median values for outlier detection.")
            snr_median = np.median(feature_matrix[:, snr_idx][~impute_mask])
            rchsq_median = np.median(feature_matrix[:, rchsq_idx][~impute_mask])
            feature_matrix_imputed[impute_mask, snr_idx] = snr_median
            feature_matrix_imputed[impute_mask, rchsq_idx] = rchsq_median
    
    # Handle NaN values according to the specified policy
    if n_nan > 0:
        if nan_policy == 'raise_error':
            raise ValueError("NaN values found in features. Set nan_policy to 'impute' or 'remove' to handle NaNs.")
        elif nan_policy == 'impute':
            # Impute remaining NaNs (not already imputed) with column medians
            col_medians = np.nanmedian(feature_matrix_imputed, axis=0)
            inds = np.where(np.isnan(feature_matrix_imputed))
            feature_matrix_imputed[inds] = np.take(col_medians, inds[1])
        elif nan_policy == 'remove':
            # NaN entries will be flagged as outliers later due to NaN LOF scores.
            pass
        else:
            raise ValueError(f"Invalid nan_policy: {nan_policy}. Choose from 'raise_error', 'impute', or 'remove'.")
        
    return feature_matrix_imputed

def get_lof_score(megatab, features, nan_policy='remove', n_neighbors=20, snr_cap=100, rchsq_cap=10):
    """
    Calculate Local Outlier Factor (LOF) scores for the megatable based on specified features.
    LOF identifies points that are isolated in multi-dimensional parameter space
    by comparing local density around each point to the density around its neighbors.
    Sources far from all neighbors (in multiple dimensions) get high LOF scores.

    Parameters
    ----------
    megatab : astropy.table.Table
        The megatable to analyze.
    features : list
        List of feature names to use for LOF calculation.
    nan_policy : str, optional
        Policy for handling NaN values ('raise_error', 'impute', or 'remove'; default is 'remove').
         - 'raise_error': Raise an error if NaNs are found.
         - 'impute': Replace NaNs with column medians.
         - 'remove': Flag entries with NaNs as outliers (they will get NaN LOF scores and can be filtered out later).
    n_neighbors : int, optional
        Number of neighbors to use for LOF calculation (default is 20).
    snr_cap : float, optional
        SNR threshold above which values are imputed to the median for outlier detection (default is 100).
    rchsq_cap : float, optional
        Reduced chi-squared threshold above which values are imputed to the median (default is 10).

    Returns
    -------
    np.ndarray
        Array of LOF scores for each source (higher scores indicate more anomalous sources).
    dict
        A report containing details about the LOF calculation and any imputed sources.
    """    
    # Initialize StandardScaler for feature normalization
    scaler = StandardScaler()
    
    # Initialize LOF in fit_predict mode (novelty=False means detecting outliers in training set)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination='auto', novelty=False)

    # Generate feature matrix and handle NaNs according to the specified policy
    feature_matrix = prep_feature_matrix(megatab, features, nan_policy=nan_policy,
                                         snr_cap=snr_cap, rchsq_cap=rchsq_cap)
    
    # Scale features to have mean=0 and std=1 for LOF
    feature_matrix = scaler.fit_transform(feature_matrix)
    
    # Fit LOF model and get negative outlier factor scores (higher = more anomalous)
    lof.fit(feature_matrix)
    lof_scores = -lof.negative_outlier_factor_ # Invert to get positive scores

    return lof_scores
    

def megatable_qc(megatab, features='default', return_report=False, 
                 lof_threshold=None, nan_policy='raise_error',
                 snr_cap=100, rchsq_cap=10, n_neighbors=20):
    """
    Perform quality control on the megatable using Local Outlier Factor (LOF) to identify 
    sources that are isolated in parameter space (i.e., unusual in multiple dimensions simultaneously).
    
    Implementation strategy:
    1. Impute extreme high-SNR, high-χ² sources to median values for outlier detection purposes,
       since these are physically-driven extremes rather than problematic fits.
    2. Compute LOF scores for all sources to quantify isolation in multi-dimensional parameter space.
    3. Optionally flag sources above a user-specified LOF threshold (for manual inspection workflow).

    Parameters
    ----------
    megatab : astropy.table.Table
        The megatable to perform quality control on.
    features : list, optional
        Features to use for quality control (default is ['FWHMR', 'ASYMR', 'CONT', 'RCHSQ', 'SNRR']).
    return_report : bool, optional
        If True, return a detailed report of the quality control process (default is False).
    lof_threshold : float or None, optional
        LOF score threshold above which sources are flagged as outliers (default is None).
        If None, no sources are auto-flagged; LOF scores are returned for manual inspection.
        Typical LOF scores: ~1.0 for normal sources, >2.0 for clear outliers.
        Recommended workflow: run with None first, inspect score distribution, then set threshold.
    nan_policy : str, optional
        Policy for handling NaN values ('raise_error', 'impute', or 'remove'; default is 'raise_error').
        - 'raise_error': Raise an error if NaNs are found.
        - 'impute': Replace NaNs with column medians.
        - 'remove': Flag entries with NaNs as outliers.
    snr_cap : float, optional
        SNR threshold above which values are imputed to the median for outlier detection (default is 100).
        High-SNR sources naturally have poor reduced chi-squared, so we impute them to avoid false positives.
    rchsq_cap : float, optional
        Reduced chi-squared threshold above which values are imputed to the median (default is 10).
        Combined with high SNR, this identifies physically-driven extremes to exclude from outlier detection.
    n_neighbors : int, optional
        Number of neighbors to use for LOF calculation (default is 20).
        Higher values make the algorithm more robust but less sensitive to small local anomalies.
        
    Returns
    -------
    astropy.table.Table
        The quality-controlled megatable. If lof_threshold is None, returns the full table.
        If lof_threshold is set, returns only sources with LOF scores below the threshold.
    dict (optional)
        The quality control report (if return_report is True), containing:
        - 'total_entries': Total number of sources in input table
        - 'entries_after_qc': Number of sources passing quality control
        - 'removed_entries': Number of sources flagged as outliers
        - 'removed_with_nan': Number of sources removed due to NaN values
        - 'removed_by_lof': Number of sources removed by LOF threshold
        - 'imputed_mask': Boolean mask of sources with imputed values
        - 'removed_table': Astropy table of removed sources
        - 'removed_mask': Boolean mask of removed sources
        - 'lof_scores': LOF scores for all sources (higher = more anomalous)
    """
    # Convert all MaskedColumns to regular Columns to avoid mask persistence issues
    for colname in megatab.colnames:
        if isinstance(megatab[colname], MaskedColumn) and 'f' in megatab[colname].dtype.name:
            # Fill masked values with NaN and convert to regular Column
            filled_data = megatab[colname].filled(np.nan)
            megatab[colname] = Column(filled_data, name=colname)
    
    if features == 'default':
        feature_names = [
            'FWHMR',
            'ASYMR',
            'CONT',
            'RCHSQ',
            'SNRR'
        ] # Lyman-alpha features that can be used to identify outliers based on lyman alpha alone
    else:
        feature_names = features

    # Exclude any features that are not present in the megatable and warn about missing features
    missing_features = [name for name in feature_names if name not in megatab.colnames]
    if missing_features:
        print(f"Warning: The following features are missing from the megatable and will be excluded from quality control: {missing_features}")
    feature_names = [name for name in feature_names if name in megatab.colnames]

    # In some cases, errors are NaN due to parameters reaching fitting limits.
    # We replace these with reasonable errors based on the SNR of the line.
    megatab = replace_nan_lya_errors(megatab)
    
    # Get LOF scores based on specified features
    lof_scores = get_lof_score(megatab, feature_names, nan_policy=nan_policy, 
                               n_neighbors=n_neighbors, snr_cap=snr_cap, rchsq_cap=rchsq_cap)

    if lof_threshold is not None:
        # Create mask for sources with LOF scores above the threshold or NaN if nan_policy is 'remove'
        outlier_mask = lof_scores > lof_threshold
        nan_mask = np.isnan(lof_scores) if nan_policy == 'remove' else np.zeros(len(megatab), dtype=bool)
    else:
        # If no threshold is set, determine it automatically based on the 97th percentile of the 
        # LOF scores (this is a common heuristic, but can be adjusted based on the score distribution)
        lof_threshold = np.percentile(lof_scores, 97)
        outlier_mask = lof_scores > lof_threshold
        nan_mask = np.isnan(lof_scores) if nan_policy == 'remove' else np.zeros(len(megatab), dtype=bool)
    
    # Keep only inliers (outlier_mask == False)
    qc_megatab = megatab[~outlier_mask & ~nan_mask]

    # Create detailed quality control report
    qc_report = {
        'total_entries': len(megatab),
        'entries_after_qc': len(qc_megatab),
        'removed_entries': len(megatab) - len(qc_megatab),
        'removed_with_nan': np.sum(nan_mask),
        'removed_by_lof': np.sum(outlier_mask),
        'removed_table': megatab[outlier_mask | nan_mask],  # Table of removed sources for inspection
        'removed_mask': outlier_mask | nan_mask,  # Boolean mask of removed sources
        'lof_scores': lof_scores  # Higher scores = more anomalous
    }

    if return_report:
        return qc_megatab, qc_report
    else:
        return qc_megatab
    

def contamination_score(distance_arcsec, apsum, iso_area_px, psf_fwhm_arcsec,
                        hst_pixel_scale=0.03):
    """
    Estimate the contamination score for a candidate foreground source: a proxy for how much
    flux the source deposits at the target's position. Sources are ranked in descending order
    of this score; the highest score is the most likely contaminant.

    The model assumes an exponential (Sérsic n=1) surface brightness profile convolved with the
    MUSE PSF. The effective scale radius is derived from the isophotal area measured by
    SExtractor, which is available for all R21 catalogue sources regardless of whether elliptical
    morphology parameters could be measured. Using the same method for all sources avoids biasing
    the ranking towards particular source types.

    Parameters
    ----------
    distance_arcsec : float
        Angular separation between the candidate source and the target, in arcseconds.
    apsum : float
        Total (absolute) flux measured within an aperture around the candidate source in the
        MUSE white-light image. Used as a proxy for source brightness.
    iso_area_px : float
        Isophotal area of the source in HST pixels (``ISOAREAF_IMAGE`` from the R21 catalogue).
        Floored at 1 pixel before use to avoid zero effective radii.
    psf_fwhm_arcsec : float
        FWHM of the MUSE PSF in arcseconds, used to set a minimum effective scale radius so
        that unresolved (stellar) sources are correctly handled.
    hst_pixel_scale : float, optional
        HST pixel scale in arcseconds per pixel (default is 0.03"/px, matching the drizzled
        pixel scale used in the R21 catalogues).

    Returns
    -------
    float
        Contamination score. Higher values indicate a more likely contaminant.
    """
    psf_sigma = psf_fwhm_arcsec / 2.355
    # Effective radius from isophotal area, floored at 1 px to avoid zero
    r_e = np.sqrt(max(iso_area_px, 1) / np.pi) * hst_pixel_scale  # arcsec
    sigma_eff = np.sqrt(r_e**2 + psf_sigma**2)
    score = apsum * np.exp(-distance_arcsec / sigma_eff) / sigma_eff
    return score

def find_nearby_sources(row, maxdist=5.0 * u.arcsec, flux_filter='HST_F814W'):
    """
    Find nearby sources from the R21 catalogue that could potentially contaminate the target spectrum based on
    their proximity and brightness.
    
    Parameters
    ----------
    row : astropy.table.Row
        The row from the megatable corresponding to the target source.
    maxdist : astropy.units.Quantity, optional
        The maximum distance to consider for nearby sources (default is 5 arcseconds).
    flux_filter : str, optional
        The filter to use for flux measurements (default is 'HST_F814W').

    Returns
    -------
    astropy.table.Table
        Table of nearby sources with their coordinates, magnitudes, and distances from the target.

    """

    # First perform a preliminary check to see whether any sources in the R21 catalogue
    # are within maxdist of the target coordinates. This prevents us from having to load
    # the data cube to create an image if there are no nearby sources.
    clus = row['CLUSTER'] # Get cluster name from the input row

    # If maxdist is just an integer or float, convert it to arcseconds
    if not isinstance(maxdist, u.Quantity):
        maxdist = maxdist * u.arcsec

    # Load the R21 cluster catalogue which contains foreground sources
    ortab = io.load_r21_catalogue(clus)
    ortab.add_column(aptb.Column([np.inf for p in ortab], name='DIST')) # Initialize NORMDIST column

    # Get target coordinates
    target_ra = row['RA']
    target_dec = row['DEC']
    target_coord = SkyCoord(target_ra, target_dec, unit='deg')

    ctab = aptb.Table([[] for _ in ortab.colnames],
                      names=ortab.colnames,
                      dtype=[ortab[col].dtype for col in ortab.colnames]) # Initialize empty table to hold close sources

    for crowidx, crow in enumerate(ortab):
        # Get galaxy coordinates and redshift
        galaxy_ra = crow['RA']
        galaxy_dec = crow['DEC']
        galaxy_coord = SkyCoord(galaxy_ra, galaxy_dec, unit='deg')
        galaxy_z = crow['z']

        # Calculate distance
        distance = target_coord.separation(galaxy_coord)
        if distance.to(u.arcsec) > maxdist or galaxy_z > 2.9 or crow['zconf'] < 2:
            continue # Skip if too far, high redshift, or low confidence (indicating poor SNR)
        ortab['DIST'][crowidx] = distance.to(u.arcsec).value
        ctab.add_row(crow)

    ctab.sort('DIST') # Sort by distance to find the most likely contaminants

    return ctab

def find_strongest_contaminant(row, nearby_sources, image_size=5*u.arcsec, mask_radius=2, bbcenter=6000, bbrange=1000):
    """
    Find the likely strongest contaminant of the target source among a list of nearby sources
    by creating a MUSE white-light image around the target and ranking candidates by their
    estimated flux contribution at the target's position.

    Each candidate is scored using an exponential (Sérsic n=1) surface brightness profile
    convolved with the MUSE PSF, with an effective scale radius derived from the source's
    isophotal area (``ISOAREAF_IMAGE``) measured in the R21 HST catalogue. This quantity is
    available for all sources, avoiding the bias that would arise from using elliptical
    morphology parameters (``A_WORLD``/``B_WORLD``) which are missing for many bright galaxies.
    See :func:`contamination_score` for the full scoring formula.

    Parameters
    ----------
    row : astropy.table.Row
        The row from the megatable corresponding to the target source.
    nearby_sources : astropy.table.Table
        Table of nearby sources from the R21 catalogue, as returned by :func:`find_nearby_sources`.
        Must contain ``RA``, ``DEC``, and ``ISOAREAF_IMAGE`` columns.
    image_size : astropy.units.Quantity, optional
        The size of the MUSE white light image cutout to create around the target (default is 5 arcseconds).
    mask_radius : float, optional
        The radius (in units of the MUSE PSF FWHM) to use for the aperture when measuring source
        flux in the image (default is 2).
    bbcenter : float, optional
        The central wavelength for making the image for assessing the nearby sources (default is 6000 Å).
    bbrange : float, optional
        The half-width of the wavelength range used to make the image (default is 1000 Å).

    Returns
    -------
    astropy.table.Table
        The input nearby_sources table with additional columns ``CONTAMSCORE`` (contamination score,
        higher is more likely to contaminate), ``BRIGHTSNR`` (aperture SNR in the MUSE image),
        ``IMPOS_X``, and ``IMPOS_Y`` (pixel positions in the cutout image), sorted by descending
        contamination score.
    MPDAF.obj.Image
        The MUSE white light image cutout used for assessing the nearby sources.
    """
    clus = row['CLUSTER']
    iden = row['iden']
    idfrom = row['idfrom']

    # Check to see whether a broadband MUSE image of the cluster already exists
    try:
        bb_image = io.load_bb_image(clus, bbcenter, bbrange)
        img = bb_image.subimage(center=(row['DEC'], row['RA']), size=image_size.value * 2)
    except FileNotFoundError:
        print(f"No existing broadband image found for {clus} at {bbcenter}±{bbrange} Å. "
              f"Creating new image cutout for contamination assessment.")
        # Load the MUSE cube and create a white light image cutout of the given size
        img = improc.make_muse_img(row, image_size.value * 2, lcenter=bbcenter, width=bbrange, verbose=False)
    
    img_fwhm = improc.get_muse_psf(clus) # Get the MUSE PSF FWHM for the cluster to use in scoring and aperture size
    maskrad = mask_radius * img_fwhm # Set the aperture radius for measuring source brightness in the MUSE image based on the PSF size

    # Use the white light image to assess the brightness and proximity of nearby sources.
    contamscores = []
    brightsnrs = []
    image_positions = []
    for crow in nearby_sources:
        # Get source position in sky and image coordinates
        galaxy_ra = crow['RA']
        galaxy_dec = crow['DEC']
        cpos = SkyCoord(galaxy_ra, galaxy_dec, unit='deg')
        cimpos = cpos.to_pixel(WCS(img.wcs.to_header()))
        image_positions.append(cimpos)

        # Measure aperture flux and SNR in the MUSE image
        cimcirc = improc.create_circular_mask(np.shape(img.data.data), cimpos[::-1], maskrad / 0.2)
        distance = np.sqrt((row['RA'] - galaxy_ra)**2 + (row['DEC'] - galaxy_dec)**2) * 3600.  # arcsec
        apsum = np.abs(np.nansum(img.data.data[cimcirc]))
        aperr = np.sqrt(np.nansum(img.var.data[cimcirc]))
        apsnr = apsum / aperr
        brightsnrs.append(apsnr)  # Store the brightness SNR for later use in masking and flagging

        # Score the candidate using an exponential surface brightness profile convolved with the PSF,
        # with effective radius derived from the isophotal area in the HST catalogue.
        score = contamination_score(distance, apsum, crow['ISOAREAF_IMAGE'], img_fwhm)
        contamscores.append(score)

    # Add columns for contamination score, brightness SNR, and image positions to the nearby sources table
    nearby_sources.add_column(Column(contamscores, name='CONTAMSCORE'))
    nearby_sources.add_column(Column(brightsnrs, name='BRIGHTSNR'))
    nearby_sources.add_column(Column([pos[0] for pos in image_positions], name='IMPOS_X'))
    nearby_sources.add_column(Column([pos[1] for pos in image_positions], name='IMPOS_Y'))
    # Sort descending: highest contamination score = most likely contaminant
    nearby_sources.sort(keys='CONTAMSCORE', reverse=True)

    return nearby_sources, img
    
def get_initial_sersic_params(contaminant, img, ccmem_impos, rough_rad):
    # Get initial parameters + bounds for fitting Sersic to contaminant

    # sersic_fixed = { # Dictionary that specifies which parameters to fix
    #     # 'n': False,
    #     # 'x_0': True, # Always fix the center based on R21 coords as they are accurate
    #     # 'y_0': True,
    #     # 'amplitude': True,
    # }

    # Get initial guess for position and amplitude based on the position and value of the maximum 
    # within a small circle around the R21 position
    tight_mask = improc.create_circular_mask(np.shape(img.data.data), ccmem_impos, 3)
    local_max = np.nanmax(img.data.data[tight_mask])
    amp_guess = np.abs(local_max) / 2  # Start with a guess of half the maximum pixel value in the region as the amplitude
    local_peak = np.where((img.data.data == local_max) & tight_mask)
    if len(local_peak[0]) == 0:
        x0_guess, y0_guess = ccmem_impos[0], ccmem_impos[1] # If no valid local maximum is found, fall back to the R21 position
    else:
        x0_guess  = np.where((img.data.data == local_max) & tight_mask)[1][0] # Get x position of the maximum pixel value
        y0_guess = np.where((img.data.data == local_max) & tight_mask)[0][0] # Get y position of the maximum pixel value

    # If ellipticity information is available, use it to inform the initial guesses for ellip and theta
    if contaminant['A_WORLD'] > 0 and contaminant['B_WORLD'] > 0:
        ellip_guess = np.nanmax([1 - (contaminant['B_WORLD'] / contaminant['A_WORLD']), 0.1]) # Floor ellipticity at 0.1 to avoid perfectly circular profiles which can cause fitting issues
        ellip_guess = np.nanmin([ellip_guess, 0.5]) # Cap ellipticity at 0.5 to avoid extremely elongated profiles which can also cause fitting issues
        theta_guess = -np.radians(contaminant['THETA_J2000'])
    else:
        ellip_guess = 0.1 # Default to a slightly elliptical profile if no information is available
        theta_guess = 0.0

    initial_guesses = {
        'r_eff': rough_rad / 10,
        'amplitude': amp_guess,
        'x_0': x0_guess,
        'y_0': y0_guess,
        'theta': theta_guess,
        'ellip': ellip_guess,
        'n': 1
    }

    pos_bounds = ((initial_guesses['x_0'] - 2.5, initial_guesses['x_0'] + 2.5), 
                  (initial_guesses['y_0'] - 2.5, initial_guesses['y_0'] + 2.5)) # Allow position to vary within a 5x5 pixel
        # box around the initial position
    sersic_bounds = {
        'amplitude': (0, 10 * np.nanmax(img.data.data)), # Amplitude must be positive, upper bound is arbitrary but should be well above the brightest sources in the image
        'x_0': pos_bounds[0],
        'y_0': pos_bounds[1],
        'n': (0.5, 4.0),
        'ellip': (0, 0.7), # Allow for some ellipticity but not extreme values
        'theta': (-np.pi / 2, np.pi / 2),
        'r_eff': (0.1, rough_rad * 2)
    }

    if contaminant['z'] < 0.01:
        # Local sources are often stars, so adjust initial guess and bounds for n=0.5
        initial_guesses['n'] = 0.5
        sersic_bounds['n'] = (0.3, 1.0)

    return initial_guesses, sersic_bounds

def estimate_contaminating_spectrum(nearby_sources, img, target_clus, target_iden, mask_radius=2, 
                                    target_coord=None, plot_result=False, aper_radius=1, outlier_removal=True,
                                    outlier_kwargs={'niter': 3, 'sigma_upper': 3.0}, bbcenter=6000, bbwidth=1000):
    """
    Estimate the spectrum of the most likely contaminating source by fitting a Sersic profile
    to the MUSE white light image and scaling a spectrum of the contaminant (taken from the 
    R21 catalogue) to match the flux of the fitted profile within an aperture around the target.
    
    Parameters
    ----------
    nearby_sources : astropy.table.Table
        Table of nearby sources sorted by normalised distance, with columns for brightness SNR and image positions.
    img : MPDAF.obj.Image
        The MUSE white light image cutout containing the target and nearby sources.
    target_clus : str
        Cluster identifier for the target source. Used for PSF estimation and other cluster-specific parameters.
    mask_radius : float, optional
        The radius (in units of the MUSE PSF FWHM) to use for masking nearby sources when fitting the contaminant profile (default is 2).
    target_coord : astropy.coordinates.SkyCoord, optional
        The sky coordinates of the target source. If provided, this will be used to define the aperture for scaling the contaminant spectrum. If None, the target position in the image will be used
    target_iden : str, optional
        Identifier for the target source. Used for labeling plots and outputs.
    plot_result : bool, optional
        Whether to plot the fitted Sersic profile and contaminant spectrum (default is False).
    aper_radius : float, optional
        The radius (in units of the MUSE PSF FWHM) of the aperture around the target position to use for scaling 
        the contaminant spectrum (default is 1).
    outlier_removal : bool, optional
        Whether to perform outlier removal on the image data before fitting the Sersic profile (default is True).
    outlier_kwargs : dict, optional
        Dictionary of keyword arguments to pass to the outlier removal function (default: {'niter': 3, 'sigma_upper': 3.0}).
    bbcenter : float, optional
        The central wavelength for the broadband normalization (default is 6000). This should be the same as the bbcenter 
        used for creating the MUSE white light image to ensure consistent flux scaling.
    bbwidth : float, optional
        The width of the broadband normalization (default is 1000). This should be the same as the bbrange used for 
        creating the MUSE white light image to ensure consistent flux scaling.

    Returns
    -------
    contaminant_spec : astropy.table.Table
        The estimated spectrum of the contaminant, scaled to match the flux of the fitted Sersic profile within
        the aperture around the target.

    """
    contaminant = nearby_sources[0] # The most likely contaminant is the closest source in normalised distance
    other_sources = nearby_sources[1:] # The other nearby sources that are not the closest contaminant

    img_fwhm = improc.get_muse_psf(target_clus)
    maskrad = mask_radius * img_fwhm
    maskrad_pix = maskrad / 0.2 # Convert mask radius to pixels (assuming MUSE pixel scale of 0.2 arcsec/pixel)
    aperrad = aper_radius * img_fwhm
    aperrad_pix = aperrad / 0.2 # Convert aperture radius to pixels

    # Mask all other nearby sources
    for mem in other_sources:
        distance = np.sqrt((contaminant['RA'] - mem['RA'])**2 + (contaminant['DEC'] - mem['DEC'])**2) * 3600.
        if distance > 1.1 * maskrad and mem['BRIGHTSNR'] > 10.0:
            # Only mask if the source isn't too far from the selected contaminant -- otherwise may mask the
            # contaminant itself.
            img.mask_region((mem['DEC'], mem['RA']), maskrad)

    # Warn if the top-ranked contaminant is unresolved from the target (separation < 1 PSF FWHM)
    if contaminant['DIST'] < img_fwhm and contaminant['BRIGHTSNR'] > 10.:
        print(f"Warning: The closest source is within the FWHM of the PSF and has a high SNR.")
        print(f"Spectra of contaminant and target may not be reliably separated.")

    # Get the position of the contaminant in pixel coordinates and an estimate of its effective 
    # radius based on the isophotal area. 
    ccmem_pos = SkyCoord(contaminant['RA'], contaminant['DEC'], unit='deg')
    ccmem_impos = ccmem_pos.to_pixel(WCS(img.wcs.to_header()))
    rough_rad = np.nanmax([np.sqrt(contaminant['ISOAREAF_IMAGE'] / np.pi), 1.0]) # minimum radius of 1 pixel

    initial_guesses, sersic_bounds = get_initial_sersic_params(contaminant, img, ccmem_impos, rough_rad)

    print(f"Fitting Sersic profile to the most likely contaminant...")
    sersic = fitting.fit_sersic(img.data, ~img.mask, (initial_guesses['x_0'], initial_guesses['y_0']),
                                initial_guesses['amplitude'], 
                                initial_guesses['r_eff'], initial_guesses['theta'], initial_guesses['ellip'], 
                                initial_guesses['n'], bounds=sersic_bounds, outlier_removal=outlier_removal, 
                                outlier_kwargs=outlier_kwargs)
    
    # If we used outlier removal, sersic is a tuple, so we need to take the first element
    if isinstance(sersic, tuple):
        sersic = sersic[0]
    
    print(f"Fitted the following profile:\n")
    for k, par in enumerate(sersic.param_names):
        print(f"{par} \t {sersic.parameters[k]}")

    y, x = np.mgrid[:np.shape(img.data)[0], :np.shape(img.data)[1]]
    sersic_img = sersic(x, y)

    # Get the coordinate of the target in pixels
    if target_coord is None:
        # Set target coord to center of the image if not provided
        target_img_coord = [img.data.shape[1] / 2, img.data.shape[0] / 2]
    else:
        target_img_coord = target_coord.to_pixel(WCS(img.wcs.to_header()))

    # Show figure with data, model, and residual of Sersic model if desired
    if plot_result:
        print(f"Generating figure of fitted Sersic profile and contaminant spectrum...")
        plot.plot_2d_model(img, sersic_img, markers=ccmem_impos, iden=target_iden, cluster=target_clus,
                           aperture=(target_img_coord, aperrad_pix), save_plot=True)
    
    # We now need to sum the Sersic profile over the aperture containing the target
    aper_mask = improc.create_circular_mask(np.shape(img.data), target_img_coord, aperrad_pix)
    mod_apersum = np.nansum(sersic_img[aper_mask])

    # Get the spectrum of the contaminant made by R21 with weighting and sky subtraction
    contaminant_spec = io.load_r21_spec(target_clus, contaminant['iden'], contaminant['idfrom'],
                                        spec_type='weight_skysub')
    
    # Normalise the model by flux over a specified wavelength range
    normrange = (bbcenter - bbwidth < contaminant_spec['wave']) & (contaminant_spec['wave'] < bbcenter + bbwidth)
    contaminant_spec['spec'] /= np.nanmean(contaminant_spec['spec'][normrange])
    contaminant_spec['spec'] *= mod_apersum

    plot.plot_muse_spectrum(contaminant_spec['wave'], contaminant_spec['spec'], contaminant_spec['spec_err'],
                            save_plot=True, save_dir=io.get_plot_dir(cluster=target_clus, iden=target_iden),
                            plot_name='contaminant_spectrum.png')
    
    # Save the contaminant spectrum for later use in checking for contaminant lines in the target spectrum
    contaminant_spec.write(io.get_misc_dir(cluster=target_clus) / f"{target_iden}_contaminant_spec.fits", overwrite=True)

    return contaminant_spec

from . import constants, models
from scipy.optimize import curve_fit

wavedict = constants.wavedict

def bound(sign):
    """
    Get the bounds for fitting a gaussian to the contaminant spectrum based on the sign of the line in the target spectrum.

    Parameters
    ----------
    sign : int
        The sign of the line in the target spectrum (positive for emission, negative for absorption).

    Returns    
    -------
    tuple
        The lower and upper bounds for the amplitude of the gaussian fit to the contaminant spectrum.
    """
    if sign == +1:
        return 0, np.inf
    if sign == -1:
        return -np.inf, 0

def check_contamination(row, contaminating_spec, target_spec, save_plot=True):
    """
    Check whether any lines in the target spectrum that are significant (SNR > 3) could potentially be caused by 
    contamination from the contaminant spectrum by fitting a gaussian to the contaminant spectrum at the same 
    wavelength and checking whether there is a significant feature with the same sign in the contaminant spectrum.

    Parameters
    ----------
    row : astropy.table.Row
        The row from the megatable corresponding to the target source.
    contaminating_spec : astropy.table.Table
        The estimated spectrum of the contaminant source, with columns 'wave', 'spec', and 'spec_err'.
    target_spec : astropy.table.Table
        The spectrum of the target source, with columns 'wave', 'spec', and 'spec_err'.
    save_plot : bool, optional
        Whether to save the plots of the line fits (default is True).

    Returns
    -------
    list
        A list of contaminated lines.
    """
    iden = row['iden']
    cluster = row['CLUSTER']

    # Initialise list to store contaminated lines
    contaminated_lines = []
    
    print(f"Attempting to match lines in target spectrum with contaminant...")
    fig = plt.figure(figsize=(13,30), facecolor='w')
    counter = 0
    for col in row.colnames:
        if 'SNR_' not in col:
            continue
        if not np.abs(row[col]) > 3.0:
            continue # Only check lines that are significant in the target spectrum

        lname = col.split('_')[1] # Get line name from column name
        print(f"Checking the {lname} line...")

        # Initialize fit properties for the target line from the megatable
        fitprops = {
            'LPEAK': 0.,
            'FWHM': 0.,
            'FLUX': 0.,
            'SNR': 0.,
            'CONT': 0.,
            'SLOPE': 0.
        }
        for p in fitprops.keys():
            fitprops[p] = row[f"{p}_{lname}"]
        sign = np.sign(fitprops['SNR']) # Get the sign of the line based on the SNR (positive for emission, negative for absorption)
        # Define a wavelength region around the line peak to check for contamination in the contaminant spectrum
        linereg = np.logical_and(fitprops['LPEAK'] - 100 < contaminating_spec['wave'], 
                                    contaminating_spec['wave'] < fitprops['LPEAK'] + 100)
        
        axnew = fig.add_subplot(10,5,counter+1)
        counter += 1 # Counter to keep track of subplot index

        targ_wl = target_spec['wave'][linereg]
        targ_flux = target_spec['spec'][linereg]
        targ_specerr = target_spec['spec_err'][linereg]

        # Make a plot of the line from the target spectrum
        plot.plot_muse_spectrum(targ_wl, targ_flux, targ_specerr, ax=axnew, label=lname, 
                                color='navy', alpha=0.5)

        # Extract the contaminant spectrum in the defined wavelength region around the line peak
        ccmem_wl = contaminating_spec['wave'][linereg]
        ccmem_flux = contaminating_spec['spec'][linereg]
        ccmem_specerr = contaminating_spec['spec_err'][linereg]

        # The relevant noise for estimating the significance of features in the contaminant spectrum is the error 
        # on the target spectrum itself, as we are trying to determine whether any features in the contaminant 
        # spectrum are significant enough to be mistaken for the line in the target spectrum.
        ccmem_specerr = targ_specerr

        # Make a plot of the contaminant spectrum in the same wavelength region
        plot.plot_muse_spectrum(ccmem_wl, ccmem_flux, ccmem_specerr, ax=axnew, label='Contaminant',
                                color='teal', alpha=0.5)
        if np.nanmax(ccmem_flux / ccmem_specerr) < 3.0:
            # This checks whether there are any spectral channels in which the contaminant spectrum contributes
            # a statistically significant flux. If not, we skip this line.
            print(f"Contaminant spectrum is insignificant, skipping fit.")
            continue

        # Now fit a similar model (same lpeak) to the contaminant spectrum to see if there is a significant feature at 
        # the same wavelength.
        local_median = np.nanmedian(ccmem_flux)
        targz = row['z']
        velaxis = spectroscopy.wave2vel(ccmem_wl, wavedict[lname], redshift=targz)
        fit_reg = np.logical_and(-2000. < velaxis, velaxis < 2000.) * (~np.isnan(ccmem_flux)) * (~np.isnan(ccmem_specerr))
        fit, fitcov = curve_fit(lambda x,f,w,c,s: models.gaussian(x,f,fitprops['LPEAK'],w,c,s),
                                ccmem_wl[fit_reg], ccmem_flux[fit_reg], sigma = ccmem_specerr[fit_reg], 
                                absolute_sigma=True, p0 = [fitprops['FLUX'], fitprops['FWHM'], local_median, 0.],
                                bounds=[[bound(sign)[0], 2.0, -np.inf, -np.inf], 
                                        [bound(sign)[1], 5 * fitprops['FWHM'], np.inf, np.inf]],
                                max_nfev=10000)
        
        # Checks to make sure the fit is (i) significant and (ii) has the right sign to potentially be the 
        # cause of the putative line in the target spec.
        checks = {
            'significant': np.abs(fit[0] / np.sqrt(np.abs(fitcov[0,0]))) > 3.0,
            'right sign': np.sign(fit[0]) == sign
        }
        if all(list(checks.values())):
            print(f"Found a significant fit at the same wavelength in the contaminant spec.")
            contaminated_lines.append(lname)
        
        # Plot the fit to the contaminant spectrum
        highreswl = np.arange(np.nanmin(ccmem_wl), np.nanmax(ccmem_wl), 0.05)
        modplot = models.gaussian(highreswl, fit[0], fitprops['LPEAK'], fit[1], fit[2], fit[3])
        axnew.plot(highreswl, modplot, linestyle = '--', color='maroon', alpha=0.4)
        
        axnew.legend()
    
    if save_plot:
        plot_dir = io.get_plot_dir(cluster=cluster, iden=iden)
        fig.savefig(f"{plot_dir}/{row['iden']}_contaminants.pdf", bbox_inches='tight')
    plot.safe_show()
    plt.close()

    return contaminated_lines

from astropy.io import fits
from pathlib import Path

def flag_contamination(megatab, maxdist=5.0 * u.arcsec, imgbuff=1.0 * u.arcsec, flux_filter='HST_F814W',
                        spec_source='APER', spec_type='1fwhm_opt', bbcenter=6000, bbwidth=1000,
                        save_contamination_plots=True, use_existing_contaminant_spectra=False, plot_model=True,
                        save_bb_images=False, outlier_removal=True, outlier_kwargs={'niter': 3, 'sigma_upper': 3.0}):
    """
    Flag likely cases of contaminated emission/absorption lines using the spectrum of the most likely
    contaminating source based on proximity and brightness and comparing its spectrum with target
    fit results. Enters flags into the appropriate column of the megatable, modifying in place.

    Parameters
    ----------    
    megatab : astropy.table.Table
        The megatable containing the target sources and their fit results.
    maxdist : astropy.units.Quantity or float, optional
        The maximum distance to consider for nearby sources that could be contaminants (default is 5 arcseconds).
        If a float is provided, it will be assumed to be in arcseconds.
    imgbuff : astropy.units.Quantity or float, optional
        The buffer distance around the target source to consider when making the image for assessing nearby sources 
        (default is 1 arcsecond). If a float is provided, it will be assumed to be in arcseconds.
    flux_filter : str, optional
        The filter to use for flux measurements (default: 'HST_F814W').
    spec_source : str, optional
        The source of the spectrum (default: 'APER').
    spec_type : str, optional
        The type of the spectrum (default: '1fwhm_opt').
    bbcenter : float, optional
        The central wavelength used to make the image for assessing the nearby sources (default is 6000).
    bbwidth : float, optional
        The half-width of the wavelength range used to make the image for assessing the nearby sources (default is 1000).
    use_existing_contaminant_spectra : bool, optional
        Whether to use existing contaminant spectra if they have already been estimated and saved (default: False). 
        If True, the function will look for saved contaminant spectra and use them if available, rather than re-estimating 
        the contaminant spectrum from the MUSE image.
    plot_model : bool, optional
        Whether to plot the fitted Sersic profile for the contaminant and the contaminant spectrum (default: True).
    save_contamination_plots : bool, optional
        Whether to save plots of the contamination checks for each source (default: True)
    save_bb_images : bool, optional
        Whether to save broad-band images used for contamination assessment (default: False).
    outlier_removal : bool, optional
        Whether to perform outlier removal when fitting the Sersic profile to the contaminant in the MUSE image 
        (default: True).
    outlier_kwargs : dict, optional
        Dictionary of keyword arguments to pass to the outlier removal function (default: {'niter': 3, 'sigma_upper': 3.0}).

    Returns
    -------
        None

    """
    # Convert maxdist and imgbuff to astropy quantities if they are provided as floats
    if isinstance(maxdist, (int, float)):
        maxdist = maxdist * u.arcsec
    if isinstance(imgbuff, (int, float)):
        imgbuff = imgbuff * u.arcsec

    # Make sure that FLAG columns are of datatype string
    for col in megatab.colnames:
        if 'FLAG_' in col and megatab[col].dtype != 'str':
            megatab[col] = megatab[col].astype(str)

    # Rearrange the table by cluster to minimise number of times we need to load MUSE cubes
    # (the cubes persist in memory as we loop through sources in the same cluster, so we only need to load each cube once)
    megatab.sort('CLUSTER')

    for rowidx, row in enumerate(megatab):
        clus = row['CLUSTER']
        iden = row['iden']
        idfrom = row['idfrom']

        print("\n" + "="*80)
        print(f"Checking for contamination in {clus} object {iden}...")

        # If requested, generate a broadband image for the cluster and save to disc. This removes the need to 
        # constantly re-load the MUSE cube to generate images
        if save_bb_images:
            bb_image_path = io.get_misc_dir(cluster=clus) / f"{clus}_bb_image_{bbcenter}A_{bbwidth}A.fits"
            if not bb_image_path.exists():
                print(f"Generating broadband image for {clus} at {bbcenter}±{bbwidth} Å and saving to {bb_image_path}...")
                _ = improc.make_bb_image(clus, bbcenter, bbwidth, save=True)

        # Check to see if a contaminant spectrum has already been estimated and saved for this target, and load it if desired
        contaminant_spec_path = io.get_misc_dir(cluster=clus) / f"{iden}_contaminant_spec.fits"
        if use_existing_contaminant_spectra and contaminant_spec_path.exists():
            print(f"Loading existing contaminant spectrum for {clus} object {iden} from {contaminant_spec_path}")
            contaminant_spec_hdul = fits.open(contaminant_spec_path)
            contaminating_spec = aptb.Table(contaminant_spec_hdul[1].data)
        else:
            # Find the likely strongest contaminant based on a white light image made from the MUSE cube
            # Get nearby sources from the catalogues
            print(f"Finding nearby sources within {maxdist}...")
            nearby_sources = find_nearby_sources(row, maxdist=maxdist, flux_filter=flux_filter)

            if len(nearby_sources) == 0:
                print(f"No nearby sources found for {clus} object {iden}. Skipping contamination check.")
                continue

            print(f"Found {len(nearby_sources)} nearby sources. Assessing the most likely contaminant...")
            nearby_sources_sorted, img = find_strongest_contaminant(row, nearby_sources, image_size=maxdist, 
                                                                    bbcenter=bbcenter, bbrange=bbwidth)
            
            print(f"Estimating the contaminant spectrum for {clus} object {iden}...")
            try:
                # Estimate contaminant spectrum over an aperture of radius 1fwhm
                contaminating_spec = estimate_contaminating_spectrum(nearby_sources_sorted, img, clus, iden, 
                                                                     bbcenter=bbcenter, bbwidth=bbwidth, 
                                                                     plot_result=plot_model,
                                                                     outlier_removal=outlier_removal, 
                                                                     outlier_kwargs=outlier_kwargs)
            except fitting.BadDataError as e:
                # If the Sersic fitting fails due to bad data (e.g. all pixels masked), we catch the error and 
                # skip the contamination check for this source, rather than crashing the entire script.
                print(f"Error estimating contaminant spectrum: {e}")
                continue

        # If N_fwhm > 1, scale the contaminant up by appropriate factor
        N_fwhm = float(spec_type.split('fwhm')[0])
        if N_fwhm > 1:
            print(f"Scaling contaminant spectrum by factor of {N_fwhm} to match the aperture size of the target spectrum...")
            contaminating_spec['spec'] *= N_fwhm
            contaminating_spec['spec_err'] *= N_fwhm

        # Load the target spectrum and to obtain error bars for the contaminant spectrum. This
        # step is vital, as uncertainties in the actual target spectrum are needed to determine
        # whether the contaminant spectrum contributes statistically significant features.
        target_spec = io.load_spec(clus, iden, idfrom, spec_source=spec_source, spec_type=spec_type)

        # If the contaminating spectrum doesn't contribute statistically significant flux
        # to the target, skip the contamination check to save time.
        if np.nanmax(contaminating_spec['spec'] / target_spec['spec_err']) < 3.0:
            print(f"Contaminant spectrum does not contribute statistically significant flux to the target spectrum. "
                  f"Skipping contamination check.")
            continue

        print(f"Checking for contaminated lines...")
        contaminated_lines = check_contamination(row, contaminating_spec, target_spec, save_plot=save_contamination_plots)

        if len(contaminated_lines) == 0:
            print(f"No contaminated lines found for {clus} object {iden}.")
            continue

        print(f"Found {len(contaminated_lines)} contaminated lines. Entering flags into megatable...")

        # Enter contaminated flags into the megatable
        for line in contaminated_lines:
            flag_col = f"FLAG_{line}"
            if flag_col in megatab.colnames:
                megatab[flag_col][rowidx] = 'c' # 'c' for contaminated
            else:
                print(f"Warning: {flag_col} not found in megatable. Cannot enter contamination flag for {line} line.")