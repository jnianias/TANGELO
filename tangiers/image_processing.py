"""
Image processing utilities module.

This module provides functions for creating and processing astronomical images,
particularly from MUSE data cubes.
"""

import numpy as np
import glob
import os
from pathlib import Path
import astropy.units as u
from mpdaf.obj import Cube, Image
from . import io
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from . import spectroscopy as spectro
from astropy.io import ascii
from astropy.convolution import convolve, Gaussian2DKernel
from scipy.ndimage import label, binary_dilation

def get_muse_psf(clus):
    """
    Get the MUSE PSF FWHM for a given cluster from the PSF data file provided in the data directory.
    
    Parameters
    ----------
    clus : str
        Cluster name.
    
    Returns
    -------
    float
        PSF FWHM in arcseconds.
    """
    base_dir = io.get_data_dir()
    psf_file = Path(base_dir) / 'muse_fwhms.txt'
    # Read the psf data table using astropy ascii
    fwhmtb = np.loadtxt(psf_file, dtype={'names': ('CLUSTER', 'PSF_FWHM'), 'formats': ('U20', 'f4')}, skiprows=1)
    clusind = np.where(fwhmtb['CLUSTER'] == clus)[0]
    if len(clusind) == 0: # If the cluster is not found, raise an error
        raise ValueError(f"Cluster {clus} not found in PSF data file.")
    return fwhmtb['PSF_FWHM'][clusind[0]]


def make_muse_img(row, size, lcenter=None, width=None, cont=None, verbose=True):
    """
    Create a narrowband image from a MUSE data cube.
    
    Generates a narrowband image centered at a specific wavelength from MUSE cube data,
    with optional continuum subtraction from adjacent wavelength regions.
    
    Parameters
    ----------
    row : dict or astropy.table.Row
        Row containing target information with keys:
        - 'CLUSTER': cluster name
        - 'RA': right ascension in degrees
        - 'DEC': declination in degrees
    size : float
        Size of the image cutout in arcseconds. Will be adjusted if too close to cube edge.
    lcenter : float, optional
        Central wavelength for the narrowband image in Angstroms.
    width : float, optional
        Half-width of the wavelength window in Angstroms (image will span lcenter±width).
    cont : tuple of float, optional
        If provided, continuum will be subtracted. Should be a tuple (offset, width) where:
        - offset: wavelength offset from lcenter for continuum regions (in Angstroms)
        - width: width of the continuum regions (in Angstroms)
        Continuum is estimated from regions at lcenter±(offset±width).
        Default is None (no continuum subtraction).
    verbose : bool, optional
        If True, print progress messages. Default is True.
    
    Returns
    -------
    mpdaf.obj.Image
        Narrowband image (with continuum subtracted if cont is provided).
    
    Raises
    ------
    FileNotFoundError
        If no FITS cube files are found for the specified cluster.
    
    Notes
    -----
    The function looks for MUSE cube files in:
    $MUSE_CUBE_DIR/{cluster}/cube/*.fits
    
    or if MUSE_CUBE_DIR is not set:
    $ASTRO_DATA_DIR/muse_data/{cluster}/cube/*.fits
    
    The size is automatically adjusted to ensure it doesn't exceed the cube boundaries.
    
    Examples
    --------
    >>> # Create a simple narrowband image
    >>> img = make_muse_img(row, size=4.0, lcenter=5000.0, width=10.0)
    
    >>> # Create image with continuum subtraction
    >>> img = make_muse_img(row, size=4.0, lcenter=5000.0, width=10.0, 
    ...                     cont=(50.0, 10.0))
    """
    position = (row['DEC'], row['RA'])
    clus = row['CLUSTER']

    if verbose:
        print(f"Loading {clus} cube...")

    # Find and open the cube
    cube_dir = io.get_muse_cube_dir(clus)
    cube_files = glob.glob(str(cube_dir / clus / 'cube' / '*.fits'))
    if not cube_files:
        raise FileNotFoundError(f"No FITS cube files found for cluster {clus} in {cube_dir / clus / 'cube'}")
    
    musedata = Cube(cube_files[0])

    if verbose:
        print("Done.")
        print(f"Generating image...")

    # Get coordinate range and adjust size if needed to stay within cube boundaries
    co_range = musedata.get_range(unit_wave=u.AA, unit_wcs=u.deg)
    tightness = [
        np.nanmin([np.abs(position[0] - x) for x in [co_range[1], co_range[4]]]),
        np.nanmin([np.abs(position[1] - x) for x in [co_range[2], co_range[5]]])
    ]
    size = np.nanmin([size, np.nanmin(tightness) * 2 * 3600])

    # Get wavelength range to be used to make the image
    if lcenter is not None and width is not None:
        wl = lcenter
    else: # If no wavelength provided, use the entire wavelength range of the cube
        ends = musedata.get_range(unit_wave=u.AA)[[0,3]]
        wl = np.nanmean(ends) # central wavelength of the entire wavelength range
        width = 0.5 * (np.nanmax(ends) - np.nanmin(ends)) # half width of the entire wavelength range

    # Create narrowband image
    img_line = musedata.get_image(wave=(wl - width, wl + width))
    img_line = img_line.subimage(position, size, unit_size=u.arcsec)

    # Optionally subtract continuum from adjacent regions
    if cont:
        try:
            img_cont_u = musedata.subcube(position, size, lbda=(wl + cont[0], wl + cont[1])).mean(axis=0)
            img_cont_l = musedata.subcube(position, size, lbda=(wl - cont[1], wl - cont[0])).mean(axis=0)
            if verbose:
                print("Continuum subtracted.")
            return img_line - 0.5 * (img_cont_u + img_cont_l)
        except Exception as e:
            if verbose:
                print(f"Warning: Continuum subtraction failed: {e}")
                print("Returning image without continuum subtraction.")
            return img_line
    else:
        return img_line

def show_segmentation_mask(row, ax_in, return_cutout = False, size = 'auto', download_if_missing=False,
                           convolve_psf = None):
    """
    Overlay the R21 segmentation map on a given matplotlib axis.
    
    Parameters
    ----------
    row : dict or astropy.table.Row
        Row containing target information with keys:
        - 'CLUSTER': cluster name
        - 'RA': right ascension in degrees
        - 'DEC': declination in degrees
    ax_in : matplotlib.axes.Axes
        Matplotlib axis to overlay the segmentation map on.
    size : float or 'auto', optional
        Size of the cutout in arcseconds. If 'auto', uses the smallest box enclosing the entire segmentation map,
        down to a minimum of 2 arcseconds.
        Default is 'auto'.
    download_if_missing : bool, optional
        If True, attempts to download the segmentation map if not found locally.
        Default is False.
    convolve_psf : float or None, optional
        If provided, the segmentation map cutout will be convolved with a Gaussian PSF of this FWHM (in arcseconds).
        Default is None (no convolution).
    return_cutout : bool, optional
        If True, returns the Cutout2D object containing the segmentation map cutout.
        Default is False.
    
    Returns
    -------
    Cutout2D or None
        If return_array is True, returns the Cutout2D object containing the segmentation map cutout.
        Otherwise, returns None.
    """

    position = SkyCoord(ra=row['RA'], dec=row['DEC'], unit='deg')
    clus = row['CLUSTER'] 
    id   = row['iden'] 
    idno = int(''.join(filter(str.isdigit, id))) # Extract numeric part of id

    # Load segmentation map HDUList with proper file handling
    with io.load_segmentation_map(clus, download_if_missing=download_if_missing) as seg_hdul:
        if seg_hdul is None:
            print(f"Segmentation map for cluster {clus} could not be loaded.")
            raise FileNotFoundError(f"Segmentation map for cluster {clus} not found.")
        
        # Look for an extension called "DATA". If not present, revert to primary HDU
        if 'DATA' in seg_hdul:
            segmap = seg_hdul['DATA'].data
            segmap_header = seg_hdul['DATA'].header
        else:
            segmap = seg_hdul[0].data
            segmap_header = seg_hdul[0].header
        
        segmap_wcs = WCS(segmap_header)

        # Check to see whether the object ID exists in the segmentation map
        if idno not in segmap:
            # Check to see what the iden is at the source position
            pix_coords = segmap_wcs.world_to_pixel(position) # Convert world to pixel coordinates
            x_pix, y_pix = int(pix_coords[0]), int(pix_coords[1]) # Pixel coordinates
            id_at_position = -1
            if (0 <= x_pix < segmap.shape[1]) and (0 <= y_pix < segmap.shape[0]): # Check to make sure within array bounds
                id_at_position = segmap[y_pix, x_pix] # Value at that pixel
            else:
                raise ValueError(f"Source position {position.to_string('hmsdms')} is out of bounds for "
                                 f"segmentation map of cluster {clus}.")
            print(f"Object ID {id} not found in segmentation map for cluster {clus}. "
                  f"ID at source position is {id_at_position}. Using that instead.")
            idno = id_at_position
                
        # Determine size of cutout
        if size == 'auto':
            ys, xs = np.where(segmap == idno)
            if len(xs) == 0 or len(ys) == 0:
                print(f"Can't find segmentation region for object ID {id} in cluster {clus}.")
                return None
            # Determine the minimum enclosing box
            x_min, x_max = np.min(xs), np.max(xs)
            y_min, y_max = np.min(ys), np.max(ys)
            # Convert pixel coordinates to world coordinates
            world_min = segmap_wcs.pixel_to_world(x_min, y_min)
            world_max = segmap_wcs.pixel_to_world(x_max, y_max)
            # Calculate size in arcseconds
            size_x = np.abs(world_max.ra.arcsec - world_min.ra.arcsec)
            size_y = np.abs(world_max.dec.arcsec - world_min.dec.arcsec)
            size = np.max([size_x, size_y, 2.0]) # Minimum size of 2 arcseconds
            
        # Use the Cutout2D utility to make the cutout
        cutout = Cutout2D(segmap, position, size*u.arcsec, wcs=segmap_wcs, mode='trim')
        cutout.data = cutout.data == idno

        # Optionally convolve with Gaussian PSF
        if convolve_psf is not None:
            pixel_scale = np.abs(cutout.wcs.pixel_scale_matrix[0,0]) * 3600.0 # arcsec/pixel
            sigma_pixels = (convolve_psf / (2.0 * np.sqrt(2.0 * np.log(2.0)))) / pixel_scale
            kernel = Gaussian2DKernel(sigma_pixels)
            cutout.data = convolve(cutout.data.astype(float), kernel, fill_value=0.0)
            cutout.data = cutout.data > 0.1 # Binarize after convolution

        # Overlay contour outlining the segmentation map on the provided axis
        ax_in.contour(cutout.data, levels=[0.5], colors='red', linewidths=1.5, 
                      transform=ax_in.get_transform(WCS(cutout.wcs.to_header())))

        if return_cutout:
            return cutout # returns the cutout object containing vital WCS info
        
def get_segmap_peak(full_iden, cluster, seg_map=None, weight_map=None, search_size=20.0):
    """
    Finds the brightest pixel in the segmentation map for the given source ID. If no segmentation map is found
    for the source ID, this may be due to the source being a composite source, where multiple adjacent segmentation
    regions have been stitched together. In such cases, the function attempts to reconstruct it using the nearest 
    composite segmentation region.
    
    Parameters
    ----------
    full_iden : str
        Full identifier of the source (e.g., 'P1234' or 'M5678').
    cluster : str
        Name of the cluster.
    seg_map : astropy.io.fits.HDUList, optional
        Pre-loaded segmentation map HDUList. If None, the function will load it.
    weight_map : astropy.io.fits.HDUList, optional
        Pre-loaded weight map HDUList. If None, the function will load it.
    search_size : float, optional
        Size of the search region in arcseconds when reconstructing composite sources. Default is 10.0.
    
    Returns
    -------
    tuple
        Optimised (RA, DEC) coordinates based on segmentation map peak.

    Raises
    ------
    ValueError
        If the source detection method is not prior ('P')
    """

    if not full_iden.startswith('P'):
        raise ValueError("Segmentation map peak finding is only supported for prior-detected sources (ID starting with 'P').")

    idno = int(''.join(filter(str.isdigit, full_iden))) # Extract numeric part of id

    # Load segmentation map if not provided
    if seg_map is None:
        seg_map = io.load_segmentation_map(cluster)

    # Load the segmentation map data
    if 'DATA' in seg_map:
        segmap_data = seg_map['DATA'].data
        segmap_header = seg_map['DATA'].header
    else:
        segmap_data = seg_map[0].data
        segmap_header = seg_map[0].header
    
    wcs = WCS(segmap_header)
    
    # Load source catalog to check which IDs exist
    source_cat = io.load_r21_catalogue(cluster, type='source')
    source_cat = source_cat[source_cat['idfrom'] == 'PRIOR'] # Filter for prior-detected sources only

    # Get coordinate of the source from the catalog
    source_row = source_cat[source_cat['iden'] == idno]
    if len(source_row) == 0:
        raise ValueError(f"Source ID {full_iden} not found in R21 source catalog for cluster {cluster}.")
    ra = source_row['RA'][0]
    dec = source_row['DEC'][0]

    catalog_ids = set()
    for row in source_cat: # add all IDs in the catalog to a set
        catalog_ids.add(int(row['iden']))
    
    # Check if the ID exists in the segmentation map
    if idno > np.nanmax(segmap_data):
        # Attempt to reconstruct composite segmentation region
        print(f"Source ID {idno} not found in segmentation map (max ID: {int(np.nanmax(segmap_data))}).")
        print(f"Source appears to be a composite. Attempting to reconstruct composite region.")
        
        # Find pixel coordinates of the source position
        pix_coords = wcs.world_to_pixel(SkyCoord(ra=ra, dec=dec, unit='deg'))
        x_pix, y_pix = int(pix_coords[0]), int(pix_coords[1])

        # Check to make sure within array bounds
        if not (0 <= x_pix < segmap_data.shape[1]) or not (0 <= y_pix < segmap_data.shape[0]):
            print(f"Error: Source position {ra}, {dec} is out of bounds for segmentation map of cluster {cluster}.")
            return ra, dec
        
        search_region = Cutout2D(segmap_data, (x_pix, y_pix), search_size*u.arcsec, wcs=wcs).data
        
        # Find unique IDs in the search region
        unique_ids = np.unique(search_region[search_region > 0])
        
        # Filter for IDs that are NOT in the catalog (i.e., they were merged)
        candidate_ids = [int(uid) for uid in unique_ids if int(uid) not in catalog_ids and uid < idno]
        
        if len(candidate_ids) == 0:
            print(f"Warning: No candidate segmentation regions found near source position.")
            print(f"Using original source position.")
            return ra, dec
        
        print(f"Found {len(candidate_ids)} candidate segmentation regions: {candidate_ids}")
        
        # Find groups of adjacent regions among candidates
        # Create a binary mask for all candidate regions
        composite_mask = np.zeros_like(segmap_data, dtype=bool)
        for cid in candidate_ids:
            composite_mask |= (segmap_data == cid)
        
        # Label connected components to find groups of adjacent regions
        labeled_array, num_features = label(composite_mask)
        
        # Filter for composite regions that contain MORE THAN ONE segmentation ID
        composite_regions = []
        for label_id in range(1, num_features + 1):
            labeled_region_mask = (labeled_array == label_id)
            # Find which candidate IDs are in this labeled region
            ids_in_region = [int(uid) for uid in candidate_ids if np.any(labeled_region_mask & (segmap_data == uid))]
            
            # Only keep regions with multiple IDs (true composites)
            if len(ids_in_region) > 1:
                composite_regions.append({
                    'label_id': label_id,
                    'ids': ids_in_region,
                    'mask': labeled_region_mask
                })
        
        if len(composite_regions) == 0:
            print(f"Warning: No composite regions (with >1 ID) found near source position.")
            print(f"Using original source position.")
            return ra, dec
        
        print(f"Found {len(composite_regions)} composite regions with multiple IDs")
        
        # Find which composite region contains or is nearest to the source position
        label_at_position = labeled_array[y_pix, x_pix]
        
        # Check if the source position is in one of the composite regions
        selected_region = None
        for comp_region in composite_regions:
            if comp_region['label_id'] == label_at_position:
                selected_region = comp_region
                break
        
        if selected_region is None:
            # Source position is not in any composite region
            # Find the nearest composite region
            distances = []
            for comp_region in composite_regions:
                ys, xs = np.where(comp_region['mask'])
                # Calculate minimum distance to this region
                min_dist = np.min(np.sqrt((xs - x_pix)**2 + (ys - y_pix)**2))
                distances.append((min_dist, comp_region))
            
            if distances:
                _, selected_region = min(distances)
                print(f"Source position not in composite regions. Using nearest composite region.")
            else:
                print(f"Warning: Could not find any composite regions.")
                return ra, dec
        
        # Use the selected composite region
        composite_region = selected_region['mask']
        component_ids = selected_region['ids']
        print(f"Reconstructed composite region from IDs: {component_ids}")
        
        # Update segmap_data to use this composite region for peak finding
        segmap_data = np.where(composite_region, idno, 0)
    
    # Find the brightest pixel in the segmentation region
    # For this, we need the actual image data (not just the segmentation map)
    # Load the segmentation map's corresponding detection image if available
    # Otherwise, use the segmentation map itself (which may have weights)
    
    # Create a mask for the source region
    source_mask = (segmap_data == idno)
    
    if not np.any(source_mask):
        print(f"Warning: No pixels found for source ID {idno} in segmentation map.")
        return ra, dec
    
    # If weight map is not provided, load it
    if weight_map is None:
        weight_map = io.load_weight_map(cluster)

    if weight_map is None:
        raise FileNotFoundError(f"Weight map for cluster {cluster} could not be loaded.")

    # Load detection image data from weight map
    if 'DATA' in weight_map:
        weight_data = weight_map['DATA'].data
    else:
        weight_data = weight_map[0].data

    # Find brightest pixel within the source mask
    masked_image = np.where(source_mask, weight_data, -np.inf)
    peak_idx = np.argmax(masked_image)
    y_peak, x_peak = np.unravel_index(peak_idx, masked_image.shape)
    
    # Convert pixel coordinates to world coordinates
    peak_coord = wcs.pixel_to_world(x_peak, y_peak)
    ra_peak = peak_coord.ra.deg
    dec_peak = peak_coord.dec.deg
    
    print(f"Found peak at pixel ({x_peak}, {y_peak}) -> RA={ra_peak:.6f}, DEC={dec_peak:.6f}")
    
    return ra_peak, dec_peak

def create_circular_mask(shape, center, radius):
    """
    Create a circular mask for a 2D array.
    
    Parameters
    ----------
    shape : tuple
        Shape of the 2D array (ny, nx).
    center : tuple
        (x, y) coordinates of the center of the circle in pixel units.
    radius : float
        Radius of the circle in pixel units.
    
    Returns
    -------
    numpy.ndarray
        A boolean mask with True values inside the circle and False outside.
    """
    ny, nx = shape
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y)
    
    # Calculate distance from the center for each pixel
    distance = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)
    
    # Create mask where distance is less than or equal to radius
    mask = distance <= radius
    
    return mask