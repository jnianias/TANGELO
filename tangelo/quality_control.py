"""
Quality control utilities for processing of MUSE spectra
"""

from . import io
from . import image_processing as improc
from . import plotting as plot
from . import fitting
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
    

def normalised_distance(distance, brightness, exponent=0.5, magnitude=True):
    """
    Calculate the normalised distance to an object, taking into account its brightness.

    Parameters
    ----------
    distance : astropy.units.Quantity
        The angular distance to the object
    brightness : float
        The brightness of the object.
    exponent : float, optional
        The exponent to use for flux scaling (default is 0.5, which scales approximately with radius).
    magnitude : bool, optional
        Whether the brightness is given in magnitudes (default is True). If False, brightness is assumed 
        to be in linear flux units.

    Returns
    -------
    float
        The normalised distance.
    """
    dist_arcsec = distance.to(u.arcsec)
    if magnitude:
        flux = 10**(-0.4 * brightness) # Convert magnitude to linear flux
    else:
        flux = brightness # Assume linear flux units

    norm_dist = dist_arcsec.value / flux**(exponent)

    return norm_dist

def find_nearby_sources(row, maxdist=5.0 * u.arcsec, flux_filter='HST_F814W'):

    # First perform a preliminary check to see whether any sources in the R21 catalogue
    # are within maxdist of the target coordinates. This prevents us from having to load
    # the data cube to create an image if there are no nearby sources.
    clus = row['CLUSTER'] # Get cluster name from the input row

    # Load the R21 cluster catalogue which contains foreground sources
    ortab = io.load_r21_catalogue(clus)
    ortab.add_column(aptb.Column([np.inf for p in ortab], name='DIST')) # Initialize NORMDIST column

    # Get target coordinates
    target_ra = row['RA']
    target_dec = row['DEC']
    target_coord = SkyCoord(target_ra, target_dec, unit='deg')

    ctab = aptb.Table() # Initialize empty table to hold close sources

    for crowidx, crow in enumerate(ortab):
        # Get galaxy coordinates and redshift
        galaxy_ra = crow['RA']
        galaxy_dec = crow['DEC']
        galaxy_coord = SkyCoord(galaxy_ra, galaxy_dec, unit='deg')
        galaxy_z = crow['z']

        # Calculate distance and normalised distance
        distance = target_coord.separation(galaxy_coord)
        normdist = normalised_distance(distance, crow[f"MAG_ISO_{flux_filter}"])
        if distance.to(u.arcsec) > maxdist or galaxy_z > 2.9 or crow['zconf'] < 2:
            continue # Skip if too far, high redshift, or low confidence (indicating poor SNR)
        ortab['DIST'][crowidx] = distance.to(u.arcsec).value
        ctab.add_row(crow)

    ctab.sort('DIST') # Sort by distance to find the most likely contaminants

    return ctab

def find_strongest_contaminant(row, nearby_sources, image_size=5*u.arcsec, mask_radius=2):
    """
    Find the likely strongest contaminant of the target source among a list of nearby sources
    by creating a white light image from the MUSE cube surrrounding the target and identifying
    the source with the smallest flux-normalised distance to the target in the image.

    Parameters
    ----------
    row : astropy.table.Row
        The row from the megatable corresponding to the target source.
    nearby_sources : astropy.table.Table
        Table of nearby sources with their coordinates, magnitudes, and distances from the target.
    image_size : astropy.units.Quantity, optional
        The size of the MUSE white light image cutout to create around the target (default is 5 arcseconds).
    mask_radius : float, optional
        The radius (in units of the MUSE PSF FWHM) to use for masking nearby sources when fitting the contaminant 
        profile (default is 2).
    
    Returns
    -------
    astropy.table.Table
        The input nearby_sources table with additional columns for normalised distance, brightness SNR, and image 
        positions, sorted by normalised distance.
    MPDAF.obj.Image
        The MUSE white light image cutout used for assessing the nearby sources.
    """
    clus = row['CLUSTER']
    iden = row['iden']
    idfrom = row['idfrom']

    # Load the MUSE cube and create a white light image cutout of the given size
    img = improc.make_muse_img(row, image_size * 2, verbose=False)
    img_fwhm = improc.get_muse_psf(clus)
    maskrad = mask_radius * img_fwhm

    # Use the white light image to assess the brightness and proximity of nearby sources.
    normdists = []
    brightsnrs = []
    image_positions = []
    for crow in nearby_sources:
        # Get source position
        galaxy_ra = crow['RA']
        galaxy_dec = crow['DEC']
        cpos = SkyCoord(galaxy_ra, galaxy_dec, unit='deg')
        cimpos = cpos.to_pixel(WCS(img.wcs.to_header()))
        image_positions.append(cimpos)

        # Get source flux and calculate brightness SNR in the image
        cimcirc = improc.create_circular_mask(np.shape(img.data), cimpos[::-1], maskrad / 0.2)
        distance = np.sqrt((row['RA'] - galaxy_ra)**2 + (row['DEC'] - galaxy_dec)**2) * 3600.
        apsum = np.abs(np.nansum(img.data.data[cimcirc]))
        aperr = np.sqrt(np.nansum(img.var.data[cimcirc]))
        apsnr = apsum / aperr
        brightsnrs.append(apsnr) # Store the brightness SNR for later use in masking and flagging
        
        if distance <= img_fwhm and apsnr > 10.:
            # For very close and bright sources, assign a negative normalised distance as they are
            # highly likely to contaminate the target spectrum.
            normdists.append(-1. / np.sqrt(apsum))
        else:
            normdists.append(distance / np.sqrt(apsum))

    # Add columns for normalised distance, brightness SNR, and image positions to the nearby sources table
    nearby_sources.add_column(Column(normdists, name='BRIGHTDIST'))
    nearby_sources.add_column(Column(brightsnrs, name='BRIGHTSNR'))
    nearby_sources.add_column(Column([pos[0] for pos in image_positions], name='IMPOS_X'))
    nearby_sources.add_column(Column([pos[1] for pos in image_positions], name='IMPOS_Y'))
    # Sort the nearby sources by normalised distance to find the most likely contaminant (smallest normalised distance)
    nearby_sources.sort(keys='BRIGHTDIST', reverse=False)

    return nearby_sources, img

def estimate_n(z):
    """
    Estimate the Sersic index 'n' based on the redshift 'z'. This helps to fit gaussian-like profiles
    for nearby sources which are likely to be foreground stars.

    Parameters
    ----------
    z : float
        Redshift of the source.

    Returns
    -------
    n : float
        Estimated Sersic index.
    """
    if z < 0.01:
        return 0.5
    else:
        return 2.0

def estimate_contaminating_spectrum(nearby_sources, img, mask_radius=2, target_coord=None, target_iden=None,
                                    target_clus=None, plot_result=False, aper_radius=1):
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
    mask_radius : float, optional
        The radius (in units of the MUSE PSF FWHM) to use for masking nearby sources when fitting the contaminant profile (default is 2).
    target_coord : astropy.coordinates.SkyCoord, optional
        The sky coordinates of the target source. If provided, this will be used to define the aperture for scaling the contaminant spectrum. If None, the target position in the image will be used
    target_iden : str, optional
        Identifier for the target source. Used for labeling plots and outputs.
    target_clus : str, optional
        Cluster identifier for the target source. Used for PSF estimation and other cluster-specific parameters.
    plot_result : bool, optional
        Whether to plot the fitted Sersic profile and contaminant spectrum (default is False).
    aper_radius : float, optional
        The radius (in units of the MUSE PSF FWHM) of the aperture around the target position to use for scaling 
        the contaminant spectrum (default is 1).

    """
    contaminant = nearby_sources[0] # The most likely contaminant is the closest source in normalised distance
    other_sources = nearby_sources[1:] # The other nearby sources that are not the closest contaminant

    maskrad = mask_radius * improc.get_muse_psf(contaminant['CLUSTER'])
    maskrad_pix = maskrad / 0.2 # Convert mask radius to pixels (assuming MUSE pixel scale of 0.2 arcsec/pixel)
    aperrad = aper_radius * improc.get_muse_psf(contaminant['CLUSTER'])
    aperrad_pix = aperrad / 0.2 # Convert aperture radius to pixels

    # Mask all other nearby sources
    for mem in other_sources:
        distance = np.sqrt((contaminant['RA'] - mem['RA'])**2 + (contaminant['DEC'] - mem['DEC'])**2) * 3600.
        if distance > 1.1 * maskrad and mem['BRIGHTSNR'] > 10.0:
            # Only mask if the source isn't too far from the selected contaminant -- otherwise may mask the
            # contaminant itself.
            img.mask_region((mem['DEC'], mem['RA']), maskrad)
    if contaminant['BRIGHTDIST'] < 0.:
        print(f"Warning: The closest source is within the FWHM of the PSF and has a high SNR.")
        print(f"Spectra of contaminant and target may not be reliably separated.")

    # Get initial parameters + bounds for fitting Sersic to contaminant
    ccmem_pos = SkyCoord(contaminant['RA'], contaminant['DEC'], unit='deg')
    ccmem_impos = ccmem_pos.to_pixel(WCS(img.wcs.to_header()))
    rough_rad = np.nanmax([np.sqrt(contaminant['ISOAREAF_IMAGE'] / np.pi), 1.0]) # minimum radius of 1 pixel
    sersic_bounds = {
        'n': (1.0, 4.0),
        'ellip': (0, 0.2),
        'theta': (-np.pi, np.pi),
        'r_eff': (0.1, 2 * rough_rad)
    }
    sersic_fixed = { # Dictionary that specifies which parameters to fix
        'n': False,
        'x_0': True, # Always fix the center based on R21 coords as they are accurate
        'y_0': True,
        'amplitude': True,
    }
    if contaminant['z'] < 0.01:
        # For local sources, which are likely stars, fix sersic index to 0.5 to ensure gaussian profile.
        sersic_fixed['n'] = True
        sersic_fixed['amplitude'] = False
    initial_guesses = {
        'r_eff': rough_rad,
        'amplitude': 1.0,
        'theta': 0.0,
        'ellip': 0.0,
        'n': estimate_n(contaminant['z'])
    }

    print(f"Fitting Sersic profile to the most likely contaminant...")
    sersic = fitting.fit_sersic(img.data, ~img.mask, ccmem_impos, initial_guesses['amplitude'], initial_guesses['r_eff'], 
                            initial_guesses['theta'], initial_guesses['ellip'], initial_guesses['n'], bounds=sersic_bounds)
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


    if plot_result:
        plot.plot_2d_model(img, sersic_img, markers=ccmem_impos, iden=target_iden, cluster=target_clus,
                           aperture=(target_img_coord, aperrad_pix))
    
    # We now need to sum the Sersic profile over the aperture containing the target
    mod_apersum = np.ma.sum(
        np.ma.masked_array(
            data=sersic_img,
            mask=~improc.create_circular_mask(
                sersic_img, (target_img_coord[1], target_img_coord[0]), aperrad_pix
            )
        )
    )

    contaminant_spec = io.load_r21_spec(target_clus, contaminant['iden'], contaminant['idfrom'],
                                        spec_type='noweight_skysub')
    
    # Normalise the model by flux
    normrange = np.logical_and(bbcenter - bbrange < contaminant_spec['wavelength'], contaminant_spec['wavelength'] < bbcenter + bbrange)
    contaminant_spec['spec'] /= np.nanmean(contaminant_spec['spec'][normrange])
    contaminant_spec['spec'] *= mod_apersum

    fig, ax = plt.subplots(figsize = (24,4), facecolor='w')
    ax.step(contaminant_spec['wavelength'], contaminant_spec['spec'], where='mid')
    ax.errorbar(contaminant_spec['wavelength'], contaminant_spec['spec'], 
                yerr = 0., linestyle='', color='gray',
                capsize=0.)
    ax.set_xlabel(r"Wavelength ($\AA$)")
    ax.set_ylabel(r"Flux density ($10^{-20}$\,erg\,s$^{-1}$\,cm$^{-2}$\,\AA$^{-1}$)")
    ax.set_title(f"Possible contaminant spectrum (ID: {contaminant['idfrom'][0]}{contaminant['iden']})")
    plt.show()
    plt.close()



def flag_contamination_overhauled(megatab, maxdist=5.0 * u.arcsec, flux_filter='HST_F814W',
                                  spec_source='APER', spec_type='1fwhm_opt', save_checkpoint=True):
    """
    Flag likely cases of contaminated emission/absorption lines using the spectrum of the most likely
    contaminating source based on proximity and brightness and comparing its spectrum with target
    fit results. Enters flags into the appropriate column of the megatable, modifying in place.

    Parameters
    ----------    
    megatab : astropy.table.Table
        The megatable containing the target sources and their fit results.
    flux_filter : str, optional
        The filter to use for flux measurements (default: 'HST_F814W').
    spec_source : str, optional
        The source of the spectrum (default: 'APER').
    spec_type : str, optional
        The type of the spectrum (default: '1fwhm_opt').
    save_checkpoint : bool, optional
        Whether to save a checkpoint of the megatable after flagging each source (default: True).

    Returns
    -------
        None

    """
    for rowidx, row in enumerate(megatab):
        clus = row['CLUSTER']
        iden = row['iden']
        idfrom = row['idfrom']

        # Get nearby sources from the catalogues
        nearby_sources = find_nearby_sources(row, maxdist=maxdist, flux_filter=flux_filter)

        if len(nearby_sources) == 0:
            print(f"No nearby sources found for {clus} object {iden}. Skipping contamination check.")
            continue
        
        # Find the likely strongest contaminant based on a white light image made from the MUSE cube
        nearby_sources, img = find_strongest_contaminant(row, nearby_sources, image_size=maxdist)
        contaminating_spec = estimate_contaminating_spectrum(nearby_sources, img)

        # Load the target spectrum and to obtain error bars for the contaminant spectrum. This
        # step is vital, as uncertainties in the actual target spectrum are needed to determine
        # whether the contaminant spectrum contributes statistically significant features.
        target_spec = io.load_spec(clus, iden, idfrom, spec_source=spec_source, spec_type=spec_type)

        check_contamination(megatab, row, rowidx, contaminating_spec, target_spec)






def check_contamination(megatab, maxdist=5.0, imgbuff=2.0, bbrange=100, maskrad=1.0, 
                       skiplist=[], save_checkpoint=True):
    """
    
    """

    for rowidx, row in enumerate(megatab):
        iden = row['iden']
        cluster = row['CLUSTER']

        if rowidx in skiplist:
            print(f"{iden} from cluster {cluster} in skiplist. Skipping.")
            continue

        print(f"Searching for possible contaminants for {cluster} object {iden}.")

        ccmems = find_closest_cluster_member(row, maxdist = maxdist * u.arcsec, return_all = True)
        print(f"Found {len(ccmems)} candidates.")
        if not ccmems:
            print(f"No potential contaminants within {maxdist} arcseconds! Moving to next object.")
            continue
        ccmem = ccmems[0]
        print(f"Found closest source at z = {ccmem['z']:.3f}")

        fig, axs = plt.subplots(1,2,figsize = (11,4), facecolor='w')
        ax = axs[0]
        whtimg = mf.make_muse_img(row, 2 * (maxdist + imgbuff), bbcenter, bbrange)
        #add brightnesses based on our pseudo-broadband image:
        
        print(f"Attempting to match lines in target spectrum with contaminant...")
        fig = plt.figure(figsize=(13,30), facecolor='w')
        counter = 0
        for col in row.colnames:
            if 'SNR_' not in col:
                continue
            if not np.abs(row[col]) > 3.0:
                continue
            lname = col.split('_')[1]
            print(f"Checking the {lname} line...")
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
            sign = np.sign(fitprops['SNR'])
            linereg = np.logical_and(fitprops['LPEAK'] - 100 < ccmem_spec['wavelength'], 
                                     ccmem_spec['wavelength'] < fitprops['LPEAK'] + 100)
            axnew = fig.add_subplot(10,5,counter+1)
            counter += 1
            ccmem_wl = ccmem_spec['wavelength'][linereg]
            ccmem_flux = ccmem_spec['spec'][linereg]
            ccmem_specerr = ccmem_spec['error'][linereg]
            targspec = mf.plotline(row['iden'], row['CLUSTER'], fitprops['LPEAK'], axnew, spec_source = 'R21', plot_cluslines=False,
                           plot_bkg=False, plot_sky=False, plot_clusspec=False, normalise=False, label='putative '+lname,
                           model = [mf.gauss, fitprops['FLUX'], fitprops['LPEAK'], fitprops['FWHM'], fitprops['CONT'], fitprops['SLOPE']],
                           return_spectrum = True)
            ccmem_specerr = targspec['spec_err']
            axnew.step(ccmem_wl, ccmem_flux, alpha=0.7, where='mid', color='teal', label='contaminant spectrum')
            axnew.errorbar(ccmem_wl, ccmem_flux, alpha=0.5, yerr = ccmem_specerr, color='teal')
            if np.nanmax(ccmem_flux / ccmem_specerr) < 3.0:
                print(f"Contaminant spectrum is insignificant, skipping fit.")
                continue
            #now fit a similar model (same lpeak) to the contaminant spectrum:
            local_median = np.nanmedian(ccmem_flux)
            targz = row['z']
            velaxis = mf.wave2vel(ccmem_wl, fitprops['LPEAK'] / (1 + targz), z=targz)
            fit_reg = np.logical_and(-2000. < velaxis, velaxis < 2000.) * (~np.isnan(ccmem_flux)) * (~np.isnan(ccmem_specerr))
            fit, fitcov = curve_fit(lambda x,f,w,c,s: mf.gauss(x,f,fitprops['LPEAK'],w,c,s),
                                   ccmem_wl[fit_reg], ccmem_flux[fit_reg], sigma = ccmem_specerr[fit_reg], 
                                   absolute_sigma=True, p0 = [fitprops['FLUX'], fitprops['FWHM'], local_median, 0.],
                                   bounds=[[bound(sign)[0], 2.4, -inf, -inf], [bound(sign)[1], 5 * fitprops['FWHM'], inf, inf]],
                                   max_nfev=10000)
            checks = {
                'significant': np.abs(fit[0] / np.sqrt(np.abs(fitcov[0,0]))) > 3.0,
                'right sign': np.sign(fit[0]) == sign
            }
            highreswl = np.arange(np.nanmin(ccmem_wl), np.nanmax(ccmem_wl), 0.05)
            modplot = mf.gauss(highreswl, fit[0], fitprops['LPEAK'], fit[1], fit[2], fit[3])
            axnew.plot(highreswl, modplot, linestyle = '--', color='maroon', alpha=0.4)
            if all(list(checks.values())):
                print(f"Found a significant fit at the same wavelength in the contaminant spec.")
                megatab[f"FLAG_{lname}"][i] = 'c'
            axnew.legend()
#         fig.savefig(f"../plots/{row['CLUSTER']}/{row['CLUSTER']}/{row['idfrom'][0]}{row['iden']}_contaminants.pdf", bbox_inches='tight')
        plt.show()
        plt.close()

        if save_checkpoint:
            megatab.write(f"contaminant_flag_checkpoint.fits", overwrite=True)
            # File path to save the float value
            file_path = "./.contaminant_checkpoint.txt"
            # Save the float value to a text file
            with open(file_path, 'w') as file:
                file.write(str(rowidx))