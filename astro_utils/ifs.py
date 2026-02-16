"""
Module for extracting spectra from MUSE data cubes including aperture optimisation.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from . import io
from . import image_processing as imp
from . import constants as const
from . import catalogue_operations as aucat
from . import plotting as plot

class BadMUSEDataError(Exception):
    """Custom exception for when MUSE data is bad or unusable."""
    pass

wavedict = const.wavedict # Dictionary of important wavelengths

# Dictionary mapping detection method codes to descriptions
det_types = {
    'M': 'MUSELET',
    'X': 'EXTERNAL',
    'P': 'PRIOR'
}

def check_data_quality(quality_map, ra, dec, cube_wcs):
    """
    Checks if the given RA and DEC coordinates are within the bounds of the MUSE cube. and not in masked regions.

    Parameters
    ----------
    quality_map : np.ndarray
        2D quality map where good data pixels are marked as True and bad data as False.
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.

    Returns
    -------
    bool
        True if the coordinates are within the cube bounds, False otherwise.
    """
    yx = cube_wcs.sky2pix([[dec, ra]])[0] # returns array of shape (N, 2)
    y, x = yx[0], yx[1] # MPDAF uses (DEC, RA) order
    print(f"getting mask value at x={x}, y={y}")
    maskval = quality_map[int(np.round(y)), int(np.round(x))]
    print(f"mask value: {maskval}")
    if (0 <= x < quality_map.shape[1]) and (0 <= y < quality_map.shape[0]) and (maskval == 0):
        return True
    else:
        return False

def optimise_aperture(cube, ra, dec, full_iden, zlya, cluster, 
                      method='auto', kwargs=None, plot_image=False, 
                      save_plot=False, quality_map=None):
    """
    Optimises the aperture position for a source based on the specified method.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        The MUSE data cube.
    ra : float
        Right Ascension of the source.
    dec : float
        Declination of the source.
    full_iden : str
        Full identifier of the source including detection method prefix (e.g., 'M1234').
    zlya : float
        Redshift of the Lyman-alpha line.
    cluster : str
        Name of the cluster.
    method : str, optional
        Method for aperture optimisation. Default is 'auto'. Options are:
        - 'lya': optimise based on Lyman-alpha peak detection.
        - 'segmap': optimise based on brightest continuum peak in segmentation map.
        - 'auto': choose method based on detection type.
    kwargs : dict
        Additional keyword arguments for the peak_detection method: nstruct, niter
    plot_image : bool, optional
        Whether to plot a diagnostic image of the aperture placement. Default is False.
    save_plot : bool, optional
        Whether to save the diagnostic plot. Default is False.
    quality_map : np.ndarray, optional
        2D quality map to check for bad data regions. If None, quality check is not performed

    Returns
    -------
    tuple
        Optimised (RA, DEC) coordinates.
    """
    # Initial check for bad data (i.e. masked) at the provided coordinates
    if quality_map is not None:
        print(f"\nChecking data quality for source ID {full_iden} in {cluster}...")
        if not check_data_quality(quality_map, ra, dec, cube.wcs):
            raise BadMUSEDataError(f"Coordinates RA={ra}, DEC={dec} are out of cube bounds or in masked region.")
        print(f"Data quality check passed.")
    
    # Determine detection method for this source
    dm = full_iden[0]

    lya_rest = wavedict['LYALPHA']  # Angstroms
    lya_obs = lya_rest * (1 + zlya)

    ra_opt, dec_opt = ra, dec

    if method == 'lya':
        print(f"\nOptimising aperture for source ID {full_iden} in {cluster} using Lyman-α peak detection.")
        kwargs = kwargs or {}
        ra_opt, dec_opt = find_lya_peak(cube, ra, dec, cluster, full_iden, plot_image=plot_image, 
                                        save_plot=save_plot, **kwargs)
    elif method == 'segmap':
        print(f"\nOptimising aperture for source ID {full_iden} using brightest continuum peak in segmentation map.")
        # Load segmentation map for the cluster
        ra_opt, dec_opt = imp.get_segmap_peak(full_iden, cluster)
    elif method == 'auto':
        if dm == 'P':
            print(f"\nOptimising aperture for source ID {full_iden} using brightest continuum peak in segmentation map (auto mode).")
            ra_opt, dec_opt = imp.get_segmap_peak(full_iden, cluster)
        else:
            print(f"\nOptimising aperture for source ID {full_iden} using Lyman-α peak detection (auto mode).")
            kwargs = kwargs or {}
            ra_opt, dec_opt = find_lya_peak(cube, ra, dec, cluster, full_iden, plot_image=plot_image, 
                                            save_plot=save_plot, **kwargs)
    else:
        print(f"Warning: Unknown optimisation method '{method}'. Using original coordinates.")
    
    print(f"Optimised aperture position for source ID {full_iden}: RA={ra_opt}, DEC={dec_opt}")
    return ra_opt, dec_opt

def find_lya_peak(cube, ra, dec, cluster, full_iden, nstd=2, 
                  bandwidth=1, margin=7, fband=2, search_size=5.0, 
                  plot_image=False, save_plot=False, nstruct=2, niter=1):
    """
    Finds the peak of Lyman-alpha emission in a narrowband image around the Lyman-alpha line.
    
    This function extracts a narrowband image centered on the Lyman-alpha wavelength,
    performs peak detection, and returns the coordinates of the peak closest to the
    original source position. Useful for optimizing aperture positions for sources
    detected via their Lyman-alpha emission.
    
    Parameters
    ----------
    cube : mpdaf.obj.Cube
        The MUSE data cube.
    ra : float
        Right Ascension of the source in degrees.
    dec : float
        Declination of the source in degrees.
    cluster : str
        Name of the cluster (for diagnostic output).
    full_iden : str
        Full identifier of the source (e.g., 'M1234').
    nstd : float, optional
        Number of standard deviations above the mean for peak detection threshold. Default is 2.
    bandwidth : float, optional
        Half-width of the wavelength window around Lyman-alpha in rest-frame Angstroms. Default is 1.
    margin : float, optional
        Wavelength margin between line and continuum bands in rest-frame Angstroms. Default is 7.
    fband : float, optional
        Width of continuum bands in rest-frame Angstroms. Default is 2.
    search_size : float, optional
        Size of the search region in arcseconds. Default is 5.0.
    plot_image : bool, optional
        Whether to plot a diagnostic image of the narrowband with detected peaks. Default is False.
    save_plot : bool, optional
        Whether to save the diagnostic plot. Default is False.
    nstruct : int, optional
        Additional keyword arguments for the peak_detection method (nstruct, niter). Default is 2.
    niter : int, optional
        Additional keyword arguments for the peak_detection method (nstruct, niter). Default is 1.

    Returns
    -------
    tuple
        Optimised (RA, DEC) coordinates based on Lyman-alpha peak.
    
    Notes
    -----
    The function uses MPDAF's peak_detection method with a threshold based on the
    variance image. If multiple peaks are found, the one closest to the input
    coordinates is selected.
    """
    lya_obs = aucat.get_line_peak('LYALPHA', full_iden, cluster, family='lyalpha')

    if lya_obs is None:
        print(f"Warning: No Lyman-α line peak found for source ID {full_iden}. Using original coordinates.")
        return ra, dec
    
    zlya = (lya_obs / wavedict['LYALPHA']) - 1

    # Convert rest-frame bandwidths to observed-frame
    bandwidth *= (1 + zlya)
    margin *= (1 + zlya)
    fband *= (1 + zlya)

    print(f"Searching narrowband image for Lyman-α peaks for {full_iden}"
          f" at z={zlya:.4f} (λ={lya_obs:.2f} Å) with {nstd}σ threshold")

    # Extract narrowband image around Lyman-alpha line with continuum subtraction
    lya_nb_img = cube.get_image((lya_obs - bandwidth, lya_obs + bandwidth), 
                                 subtract_off=True, margin=margin, fband=fband)

    # Limit to a region around the source to speed up peak detection
    lya_nb_img = lya_nb_img.subimage(center=(dec, ra), size=search_size)

    # Find the n sigma threshold for peak detection using the .var extension of the image
    # This creates a per-pixel threshold based on local variance, accounting for heterogeneous noise
    threshold = nstd * np.sqrt(lya_nb_img.var.data)

    # Subtract background to improve peak detection
    bkgmean, _ = lya_nb_img.background(niter=3, sigma=2.5)
    lya_nb_img.data -= bkgmean

    # Find peak positions in the narrowband image in pixel coordinates
    peak_locs = lya_nb_img.peak_detection(nstruct=nstruct, niter=niter, threshold=threshold)
    
    # If no peaks found, use original coordinates
    if len(peak_locs) == 0:
        print(f"Warning: No Lyman-α peaks found above threshold. Using original coordinates.")
        ra_opt, dec_opt = ra, dec
        peak_locs_world = np.array([])
    else:
        print(f"Found {len(peak_locs)} peak(s).")
        
        # Convert peak locations to world coordinates
        peak_locs_world = lya_nb_img.wcs.pix2sky(peak_locs)

        # Get rid of any peaks that are suspiciously close to other objects in the catalogue
        peak_locs_world = filter_contaminants(peak_locs_world, cluster, full_iden, tolerance=1.0)
        if len(peak_locs_world) == 0:
            print(f"Warning: All detected peaks excluded due to proximity to catalogue sources. Using original coordinates.")
            ra_opt, dec_opt = ra, dec
        else:
            print(f"{len(peak_locs_world)} peak(s) remain after filtering contaminants")
            peak_locs = lya_nb_img.wcs.sky2pix(peak_locs_world)

            # Select the closest peak to the original RA, DEC position
            # Calculate flux-weighted angular separations in arcseconds
            separations = np.sqrt((peak_locs_world[:,1] - ra)**2 + (peak_locs_world[:,0] - dec)**2) * 3600.0
            # Calculate fluxes at the peak locations using the sum within a 3x3 box centered on each peak
            fluxes = []
            for peak in peak_locs:
                y, x = int(peak[0]), int(peak[1])
                box = lya_nb_img.data[max(0, y-1):y+2, max(0, x-1):x+2]
                fluxes.append(np.nansum(box))
            fluxes = np.array(fluxes)
            # Weight separations by flux to prioritize brighter peaks
            weighted_separations = fluxes / separations # This ensures negative fluxes are not selected
            closest_peak_idx = np.argmax(weighted_separations)
            closest_peak = peak_locs_world[closest_peak_idx]
            ra_opt, dec_opt = closest_peak[1], closest_peak[0]

            # Check to see how much flux there is at the original position
            orig_yx = lya_nb_img.wcs.sky2pix([[dec, ra]])[0]
            oy, ox = int(np.round(orig_yx[0])), int(np.round(orig_yx[1]))
            orig_box = lya_nb_img.data[max(0, oy-1):oy+2, max(0, ox-1):ox+2]
            orig_flux = np.nansum(orig_box)
            
            peak_flux = fluxes[closest_peak_idx]
            offset_arcsec = separations[closest_peak_idx]
            
            print(f"Original position flux: {orig_flux:.2f}, Selected peak flux: {peak_flux:.2f}, Offset: {offset_arcsec:.2f}\"")
            
            # Use original position if it has significantly higher flux (>10% more)
            if orig_flux > peak_flux:
                print(f"Original position is brighter; using original coordinates.")
                ra_opt, dec_opt = ra, dec
            else:
                print(f"Using optimized position: RA={ra_opt:.6f}, DEC={dec_opt:.6f}")
                # Keep the ra_opt, dec_opt that were already set from closest_peak

    # Plot diagnostic image if requested
    if plot_image:
        plot.plot_lya_peak_detection(lya_nb_img, ra, dec, ra_opt, dec_opt, cluster, full_iden,
                                     peak_locs_world, save_plot=save_plot)

    return ra_opt, dec_opt

def filter_contaminants(peak_locs_world, cluster, full_iden, tolerance=0.2):
    """
    Filters out peak locations that are too close to other catalogue sources, unless they are within
    the specified tolerance of the catalogue position of the source being optimised.

    Parameters
    ----------
    peak_locs_world : np.ndarray
        Array of shape (N, 2) with RA and DEC of detected peaks.
    cluster : str
        Name of the cluster.
    full_iden : str
        Full source identifier to exclude from contamination check (e.g., the source being optimised).
    tolerance : float, optional
        Minimum separation in arcseconds to consider a peak uncontaminated. Default is 0.2".

    Returns
    -------
    np.ndarray
        Filtered array of peak locations.
    """
    cat = io.load_r21_catalogue(cluster)

    # Dictionary to translate idfrom prefixes to detection methods
    idfrom_map = {'M': 'MUSELET', 'X': 'EXTERN', 'P': 'PRIOR'}

    target_row = cat[(cat['idfrom'] == idfrom_map[full_iden[0]]) & (cat['iden'] == int(full_iden[1:]))][0]
    target_ra = target_row['RA']
    target_dec = target_row['DEC']

    # Now remove the target source from the catalogue for contamination checking
    cat = cat[~((cat['idfrom'] == idfrom_map[full_iden[0]]) & (cat['iden'] == int(full_iden[1:])))]

    filtered_peaks = []
    for peak in peak_locs_world:
        ra_peak, dec_peak = peak[1], peak[0]
        # Check if this peak is within tolerance of the target source position
        target_sep = np.sqrt((target_ra - ra_peak)**2 + (target_dec - dec_peak)**2) * 3600.0  # in arcseconds
        if target_sep <= tolerance:
            filtered_peaks.append(peak) # automatically allow the target source peak
            continue
        # Calculate separations to all catalogue sources
        separations = np.sqrt((cat['RA'] - ra_peak)**2 + (cat['DEC'] - dec_peak)**2) * 3600.0  # in arcseconds
        if np.all(separations > tolerance):
            filtered_peaks.append(peak)
        else:
            print(f"Excluding peak at RA={ra_peak:.6f}, DEC={dec_peak:.6f} due to proximity to catalogue source.")
    
    return np.array(filtered_peaks)

def make_quality_map(cube):
    """
    Creates a quality map for the MUSE cube to identify bad data regions.

    Parameters
    ----------
    cube : mpdaf.obj.Cube
        The MUSE data cube.

    Returns
    -------
    np.ndarray
        2D quality map where good data pixels are marked as True and bad data as False.
    """
    # Create a quality map based on the cube's mask
    quality_map = np.nanmedian(cube.mask, axis=0) != 0
    return quality_map
 
def extract_spectra(source_cat, aper_size, cluster_name, overwrite=False, optimise_apertures=None,
                    optimise_apertures_kwargs=None, plot_images=False, save_plots=False):
    """
    Extracts spectra for sources in the source catalog using aperture photometry and saves them to disk.

    Parameters
    ----------
    source_cat : list of dict
        List of source dictionaries containing source information.
    aper_size : int
        Aperture size in FWHM for spectrum extraction.
    cluster_name : str
        Name of the cluster (e.g., 'A2744').
    overwrite : bool, optional
        Whether to overwrite existing spectra. Default is False.
    optimise_apertures : str or None, optional
        Method for aperture optimisation. Options are:
        - 'auto': automatically choose method based on detection type.
        - 'none': do not optimise apertures.
        - 'lya': optimise based on Lyman-alpha peak detection.
        Default is None (no optimisation).

    Returns
    -------
    dict
        Dictionary mapping source identifiers to their extracted spectrum file paths.
    dict
        Dictionary mapping source identifiers to their optimized positions (RA, DEC).
    """
    spec_paths = {}
    positions = {}
    missing_sources = []

    for source in source_cat:
        iden = source['iden']
        idfrom = source['idfrom']
        spec_path = io.get_aper_spectra_dir(cluster_name)
        full_iden = f"{idfrom[0].replace('E', 'X')}{iden}"
        spec_name = f"{full_iden}_{aper_size}fwhm" + ("_opt" if optimise_apertures is not None else "")
        spec_file = Path(spec_path) / f"{spec_name}.fits"
        if not spec_file.exists() or overwrite:
            missing_sources.append((source, spec_file, full_iden))
        else:
            print(f"Spectrum for source ID {full_iden} already exists; skipping extraction.")
            spec_paths[full_iden] = spec_file

    if not missing_sources:
        print("All spectra already exist; skipping extraction.")
        return spec_paths

    # Load the MUSE cube for the cluster
    cube = io.load_muse_cube(cluster_name)

    # Make quality map to quickly check for bad data
    quality_map = make_quality_map(cube)

    for source, spec_file, full_iden in missing_sources:
        ra = source['RA']
        dec = source['DEC']
        # If optimise_apertures is True, center the aperture on the peak flux position within the
        # segmentation map of the source
        ra_opt, dec_opt = ra, dec
        if optimise_apertures is not None:
            try:
                print(f"Optimising aperture for source ID {full_iden} in {cluster_name}...")
                ra_opt, dec_opt = optimise_aperture(cube, ra, dec, full_iden, source['zlya'], cluster_name, 
                                                    method=optimise_apertures, kwargs=optimise_apertures_kwargs or {}, 
                                                    plot_image=plot_images, save_plot=save_plots, quality_map=quality_map)
            except BadMUSEDataError:
                print(f"WARNING! Bad data at source position. Skipping this source.")
                continue
        print(f"Extracting spectrum for source ID {full_iden} at RA: {ra_opt}, DEC: {dec_opt}")
        # Extract the spectrum using MPDAF's built in aperture method for Cube objects
        spectrum = cube.aperture((dec_opt, ra_opt), aper_size / 2)
        io.save_spectrum(spectrum, spec_file)
        spec_paths[full_iden] = spec_file
        positions[full_iden] = (ra_opt, dec_opt)

    return spec_paths, positions

def extract_spectrum_from_segmap(cluster, idlist, show_plot=False, save_plot=False, save_dir='./'):
    """
    Extracts spectra for a list of segmentation map source IDs by summing the flux within
    the segmentation map region(s) corresponding to each source. Saves the extracted spectra to disk.
    As the segmentation maps are based on HST images, the segmentation regions are expanded to 
    account for the larger MUSE PSF, and are then reprojected onto the MUSE cube WCS.

    Parameters
    ----------
    cluster : str
        Name of the cluster (e.g., 'A2744').
    idlist : list of str
        List of full source identifiers (e.g., ['M1234', 'X5678']).
    show_plot : bool, optional
        Whether to show diagnostic plots. Default is False.
    save_plot : bool, optional
        Whether to save diagnostic plots. Default is False.
    save_dir : str, optional
        Directory to save extracted spectra. Default is current directory.

    Returns
    -------
    dict
        Dictionary mapping source identifiers to their extracted spectrum file paths.
    """
    # Load the MUSE cube for the cluster
    cube = io.load_muse_cube(cluster)

