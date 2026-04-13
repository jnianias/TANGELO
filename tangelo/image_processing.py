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
from photutils import segmentation
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

def make_continuum_image(cube, wl, offset, width):
    """
    Create a continuum image from a MUSE data cube by averaging over two wavelength ranges adjacent to the line of interest.
    
    Parameters
    ----------
    cube : mpdaf.obj.Cube
        MUSE data cube object.
    wl : float
        Central wavelength of the line of interest in Angstroms.
    offset : float
        Wavelength offset from the line center for the continuum regions in Angstroms.
    width : float
        Half-width of the wavelength range for the continuum regions in Angstroms.
    
    Returns
    -------
    mpdaf.obj.Image
        Continuum image created by averaging over the two adjacent wavelength ranges.
    
    Notes
    -----
    The function assumes that the line of interest is centered at a wavelength `wl` which is provided as an argument.
    The continuum is estimated from two regions: [wl - offset - width, wl - offset + width] and [wl + offset - width, wl + offset + width].
    """
    cont_range_low = (wl - offset - width, wl - offset + width)
    cont_range_high = (wl + offset - width, wl + offset + width)

    # Generate masks that select wavelength bins within the continuum ranges
    cube_wave = cube.wave.coord() # Get the wavelength axis of the cube in Angstroms
    cont_mask_low = (cube_wave >= cont_range_low[0]) & (cube_wave <= cont_range_low[1])
    cont_mask_high = (cube_wave >= cont_range_high[0]) & (cube_wave <= cont_range_high[1])
    n_bins_low = np.sum(cont_mask_low)
    n_bins_high = np.sum(cont_mask_high)
    n_bins_total = n_bins_low + n_bins_high
    # Raise a warning if there are no bins in one of the continuum ranges, and adjust the weights accordingly
    if n_bins_low == 0:
        print(f"Warning: No wavelength bins found in lower continuum range {cont_range_low}. Using only upper continuum range.")
        n_bins_total = n_bins_high
    if n_bins_high == 0:
        print(f"Warning: No wavelength bins found in upper continuum range {cont_range_high}. Using only lower continuum range.")
        n_bins_total = n_bins_low
    if n_bins_total == 0:
        raise ValueError(f"No wavelength bins found in either continuum range. Cannot create continuum image.")
    
    # Average over the selected wavelength bins to create the continuum image
    cont_image_low = cube.get_image(wave=cont_range_low, unit_wave=u.AA, sum=True) if n_bins_low > 0 else 0
    cont_image_high = cube.get_image(wave=cont_range_high, unit_wave=u.AA, sum=True) if n_bins_high > 0 else 0

    cont_image = (cont_image_low + cont_image_high) / n_bins_total
    return cont_image


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
        - width: half-width of the continuum regions (in Angstroms)
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
    cube_files = glob.glob(str(cube_dir / '*.fits'))
    if not cube_files:
        raise FileNotFoundError(f"No FITS cube files found for cluster {clus} in {cube_dir}")
    
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
    img_line = musedata.get_image(wave=(wl - width, wl + width), unit_wave=u.AA)
    img_line = img_line.subimage(position, size, unit_size=u.arcsec)

    # Optionally subtract continuum from adjacent regions
    if cont:
        try:
            img_cont = make_continuum_image(musedata, wl, cont[0], cont[1])
            img_cont = img_cont.subimage(position, size, unit_size=u.arcsec)
            if verbose:
                print("Continuum subtracted.")
            return img_line - img_cont
        except Exception as e:
            if verbose:
                print(f"Warning: Continuum subtraction failed: {e}")
                print("Returning image without continuum subtraction.")
            return img_line
    else:
        return img_line
    
def get_segmentation_mask(cluster: str, iden: str, ra: float, dec: float, size: float = 5, 
                          download_if_missing: bool = False, connect_niter: int = 3) -> Cutout2D:
    """
    Get a cutout of the R21 segmentation map for a given source based on its RA, DEC and cluster.
    Resolves composite sources that are not in the segmentation map.
    
    Parameters
    ----------
    cluster : str
        Name of the cluster.
    ra : float
        Right ascension of the source in degrees.
    dec : float
        Declination of the source in degrees.
    size : float, optional
        Size of the cutout in arcseconds. Default is 5.
    download_if_missing : bool, optional
        If True, attempts to download the segmentation map if not found locally. Default is False.
    
    Returns
    -------
    Cutout2D
        A cutout of the R21 segmentation map centered on the source position.
    
    Raises
    ------
    FileNotFoundError
        If the segmentation map for the specified cluster cannot be loaded.
    ValueError
        If the source position is out of bounds for the segmentation map or if no segmentation region is found for the source ID.
    """
    position = SkyCoord(ra=ra, dec=dec, unit='deg')
    idno = int(''.join(filter(str.isdigit, iden))) # Extract numeric part of id

    # Load segmentation map HDUList with proper file handling
    with io.load_segmentation_map(cluster, download_if_missing=download_if_missing) as seg_hdul:
        if seg_hdul is None:
            print(f"Segmentation map for cluster {cluster} could not be loaded.")
            raise FileNotFoundError(f"Segmentation map for cluster {cluster} not found.")
        
        # Look for an extension called "DATA". If not present, revert to primary HDU
        if 'DATA' in seg_hdul:
            segmap = seg_hdul['DATA'].data
            segmap_header = seg_hdul['DATA'].header
        else:
            segmap = seg_hdul[0].data
            segmap_header = seg_hdul[0].header
        
        segmap_wcs = WCS(segmap_header)

        # Check to see whether the object ID exists in the segmentation map
        composite = False  # flag to indicate whether this source was constructed from multiple segmentation regions
        if idno not in segmap:
            composite = True # If the source wasn't in the seg map, that's because it was constructed from multiple 
                            # segmentation regions (i.e., it's a composite source)
            print(f"Object ID {iden} not found in segmentation map.")
            # Check to see what the iden is at the source position
            pix_coords = segmap_wcs.world_to_pixel(position) # Convert world to pixel coordinates
            x_pix, y_pix = int(pix_coords[0]), int(pix_coords[1]) # Pixel coordinates
            id_at_position = -1
            if (0 <= x_pix < segmap.shape[1]) and (0 <= y_pix < segmap.shape[0]): # Check to make sure within array bounds
                id_at_position = segmap[y_pix, x_pix] # Value at that pixel
            else:
                raise ValueError(f"Source position {position.to_string('hmsdms')} is out of bounds for "
                                 f"segmentation map of cluster {cluster}.")
            if id_at_position == 0:
                print(f"Source position corresponds to background. Finding closest non-zero region...")
                # Find the nearest non-zero pixel in the segmentation map to the source position
                non_zero_coords = np.argwhere(segmap > 0)
                if len(non_zero_coords) == 0:
                    print(f"Warning: No non-zero pixels found in segmentation map. Using original source position.")
                    return None
                distances = np.sqrt((non_zero_coords[:, 1] - x_pix)**2 + (non_zero_coords[:, 0] - y_pix)**2)
                nearest_idx = np.argmin(distances)
                nearest_coord = non_zero_coords[nearest_idx]
                id_at_position = segmap[nearest_coord[0], nearest_coord[1]]
                print(f"Nearest non-zero pixel found at ({nearest_coord[1]}, {nearest_coord[0]}) with ID {id_at_position}. Using that instead.")
                idno = id_at_position
            else:
                print(f"ID at source position is {id_at_position}. Using that instead.")
                idno = id_at_position
        
        # If the source is a composite, find connected regions and merge them
        if composite:
            print(f"Source appears to be a composite. Attempting to merge connected segmentation regions with ID {idno}...")
            # Get full list of ID numbers from the corresponding R21 catalogue so that we know which 
            # ones are "missing"
            source_cat = io.load_r21_catalogue(cluster, type='source')
            all_idens = source_cat['iden']
            # Create a mask for the initial ID
            for i in range(connect_niter): # Iterate a few times to ensure all connected regions are merged (adjust as needed)
                initial_mask = (segmap == idno)
                # Use binary dilation to find connected regions (this will merge adjacent regions)
                dilated_mask = binary_dilation(initial_mask, iterations=1) # Adjust iterations as needed
                # Find all unique IDs in the dilated region that are NOT in the catalogue (and therefore were merged)
                unique_ids = np.unique(segmap[dilated_mask])
                unique_ids = unique_ids[unique_ids > 0] # Exclude background
                unique_ids = [uid for uid in unique_ids if uid not in all_idens] # Exclude IDs in the catalogue
                
                if len(unique_ids) > 1:
                    print(f"Found {len(unique_ids)} connected regions with IDs: {unique_ids}. Merging them into one region.")
                    for uid in unique_ids:
                        segmap[segmap == uid] = idno # Merge all connected regions into the original ID
                else:
                    print(f"No additional connected regions found.")

        # Determine size of cutout if set to 'auto'
        if size == 'auto':
            ys, xs = np.where(segmap == idno)
            if len(xs) == 0 or len(ys) == 0:
                print(f"Can't find segmentation region for object ID {iden} in cluster {cluster}.")
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
        
        # Convert size to Quantity if it isn't already
        if isinstance(size, u.Quantity):
            size_quantity = size.to(u.arcsec)
        else:
            size_quantity = size * u.arcsec
        
        # Use the Cutout2D utility to make the cutout
        cutout = Cutout2D(segmap, position, size_quantity, wcs=segmap_wcs, mode='partial', fill_value=0)
        cutout.data = cutout.data == idno # convert to binary mask

        return cutout

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

    clus = row['CLUSTER']
    ra = row['RA']
    dec = row['DEC']
    iden = row['iden']

    cutout = get_segmentation_mask(clus, iden, ra, dec, size=size, download_if_missing=download_if_missing)    

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
        
def get_wmap_cutout(cluster: str, ra: float, dec: float, size: int = 5) -> Cutout2D:
    """
    Generate a cutout of the HST weight map for a given source in the megatable based on
    the RA, DEC and cluster.

    Parameters
    ----------
    cluster : str
        Name of the cluster.
    ra : float
        Right ascension of the source in degrees.
    dec : float
        Declination of the source in degrees.
    size : int, optional
        The size of the cutout in arcseconds (default is 5).

    Returns
    -------
    Cutout2D
        A cutout of the HST weight map centered on the source.
    """

    coord = SkyCoord(ra, dec, unit='deg')

    # Get the directory of the HST weight map based on the cluster
    with io.load_weight_map(cluster) as weight_map:
        # Convert size to Quantity if it isn't already
        if isinstance(size, u.Quantity):
            size_quantity = size.to(u.arcsec)
        else:
            size_quantity = size * u.arcsec
        # Generate cutout
        # The actual weight image is sometimes in 'DATA' and sometimes in 'SCI'
        weight_data_key = 'DATA' if 'DATA' in weight_map else 'SCI'
        cutout = Cutout2D(weight_map[weight_data_key].data, coord, size_quantity,
                          wcs = WCS(weight_map[weight_data_key].header), mode='partial', fill_value=0)
        return cutout
    
def get_convolved_weightmap(cluster: str, iden: int, ra: float, dec: float, 
                            binary: bool = False, size: int = 5, 
                            mask_thresh: float = 0.1, connect_niter: int = 3) -> Cutout2D:
    """
    Get a cutout of the HST weight map for a given source, multiplied by a binary segmentation mask
     and convolved with a Gaussian kernel to match the MUSE PSF.

    Parameters
    ----------
    cluster : str
        Name of the cluster.
    iden : int
        Numeric identifier of the source (without the 'P' prefix).
    ra : float
        Right ascension of the source in degrees.
    dec : float
        Declination of the source in degrees.
    binary : bool, optional
        If true, return only the binary mask (i.e. no weighting, just segmentation map)
    size : int, optional
        The size of the cutout in arcseconds (default is 5).

    Returns
    -------
    Cutout2D
        A cutout of the HST weight map multiplied by the segmentation mask and convolved with a Gaussian kernel.
    """
    segmap = get_segmentation_mask(cluster, iden, ra, dec, size=size, connect_niter=connect_niter)

    # Set up Gaussian kernel based on MUSE PSF FWHM for the cluster
    fwhm = get_muse_psf(cluster)
    pixel_scale = np.abs(segmap.wcs.pixel_scale_matrix[0,0]) * 3600.0 # arcsec/pixel
    sigma_pixels = (fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))) / pixel_scale
    kernel = Gaussian2DKernel(sigma_pixels)

    # Convolve segmentation mask and re-binarize
    convolved_mask = convolve(segmap.data.astype(float), kernel, fill_value=0.0)
    convolved_mask = convolved_mask > mask_thresh # Binarize after convolution

    if binary:
        segmap.data = convolved_mask.astype(float)
        return segmap

    # Convolve weight map with the same kernel
    wmap = get_wmap_cutout(cluster, ra, dec, size)
    pixel_scale_wmap = np.abs(wmap.wcs.pixel_scale_matrix[0,0]) * 3600.0 # arcsec/pixel
    sigma_pixels_wmap = (fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))) / pixel_scale_wmap
    kernel_wmap = Gaussian2DKernel(sigma_pixels_wmap)
    convolved_wmap = convolve(wmap.data.astype(float), kernel_wmap, fill_value=0.0)

    # Resample convolved segmentation mask to match the weight map grid if needed
    if convolved_mask.shape != convolved_wmap.shape:
        convolved_mask_resampled = resample_cutout_to_wcs(segmap, wmap.wcs, convolved_wmap.shape, fill_value=0)
        # Re-apply the convolution binarization threshold to the resampled mask
        convolved_mask_resampled = convolved_mask_resampled > mask_thresh
    else:
        convolved_mask_resampled = convolved_mask

    # Multiply convolved weight map by convolved segmentation mask
    weighted_mask = convolved_wmap * convolved_mask_resampled.astype(float)
    wmap.data = weighted_mask # Update the cutout data to be the weighted mask

    return wmap

def resample_cutout_to_wcs(source_cutout, target_wcs, target_shape, fill_value=0):
    """
    Resample a cutout to match a target WCS grid and shape using bilinear interpolation.
    
    Parameters
    ----------
    source_cutout : Cutout2D
        The source cutout to resample, containing `data` and `wcs` attributes.
    target_wcs : astropy.wcs.WCS
        The target WCS to resample to.
    target_shape : tuple
        The target shape (height, width) for the output array.
    fill_value : float, optional
        The fill value to use for pixels outside the source cutout region (default is 0).
    
    Returns
    -------
    np.ndarray
        A 2D array with the specified target_shape, containing the resampled data.
    """
    # Create grid of pixel coordinates for the target
    target_y = np.arange(target_shape[0])
    target_x = np.arange(target_shape[1])
    
    # Get WCS from source cutout
    source_wcs = source_cutout.wcs
    
    # Create meshgrid of target pixel coordinates
    target_xx, target_yy = np.meshgrid(target_x, target_y)
    
    # Convert target pixel coordinates to world coordinates
    target_world = target_wcs.pixel_to_world(target_xx.ravel(), target_yy.ravel())
    
    # Convert world coordinates to source cutout pixel coordinates
    source_xx, source_yy = source_wcs.world_to_pixel(target_world)
    source_xx = source_xx.reshape(target_shape)
    source_yy = source_yy.reshape(target_shape)
    
    # Create output array initialized with fill_value
    resampled = np.full(target_shape, fill_value, dtype=float)
    
    # Determine which pixels are within bounds of the source cutout
    x_min, x_max = 0, source_cutout.data.shape[1]
    y_min, y_max = 0, source_cutout.data.shape[0]
    in_bounds = ((source_xx >= x_min) & (source_xx < x_max) & 
                 (source_yy >= y_min) & (source_yy < y_max))
    
    # Interpolate only pixels within bounds using bilinear interpolation
    if np.any(in_bounds):
        xx_flat = source_xx[in_bounds]
        yy_flat = source_yy[in_bounds]
        # Bilinear interpolation
        x_floor = np.floor(xx_flat).astype(int)
        y_floor = np.floor(yy_flat).astype(int)
        x_ceil = np.minimum(x_floor + 1, source_cutout.data.shape[1] - 1)
        y_ceil = np.minimum(y_floor + 1, source_cutout.data.shape[0] - 1)
        x_frac = xx_flat - x_floor
        y_frac = yy_flat - y_floor
        
        # Bilinear interpolation formula
        v00 = source_cutout.data[y_floor, x_floor]
        v01 = source_cutout.data[y_floor, x_ceil]
        v10 = source_cutout.data[y_ceil, x_floor]
        v11 = source_cutout.data[y_ceil, x_ceil]
        
        interp_vals = (v00 * (1 - x_frac) * (1 - y_frac) + 
                       v01 * x_frac * (1 - y_frac) + 
                       v10 * (1 - x_frac) * y_frac + 
                       v11 * x_frac * y_frac)
        
        resampled[in_bounds] = interp_vals
    
    return resampled

def resample_weightmap(weight_map_cutout, image, fill_value=0):
    """
    Resample a weight map cutout to match the grid and shape of a MUSE image.
    
    Parameters
    ----------
    weight_map_cutout : Cutout2D
        The HST weight map cutout to resample, containing `data` and `wcs` attributes.
    image : mpdaf.obj.Image
        The MUSE image to match. The weight map will be resampled to match this image's grid and WCS.
    fill_value : float, optional
        The fill value to use for pixels outside the weight map region (default is 0).
    
    Returns
    -------
    np.ndarray
        A 2D array with the same shape as the image data, containing the resampled weight map.
    """
    img_wcs = WCS(image.wcs.to_header())
    img_shape = image.data.shape
    return resample_cutout_to_wcs(weight_map_cutout, img_wcs, img_shape, fill_value=fill_value)

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

def make_bb_image(cluster, bbcenter, bbwidth, save=False):
    """
    Create a broad-band image from a MUSE data cube by summing over a specified wavelength range.
    
    Parameters
    ----------
    cluster : str
        Name of the cluster.
    bbcenter : float
        Central wavelength for the broad-band image in Angstroms.
    bbwidth : float
        Half-width of the wavelength range for the broad-band image in Angstroms.
    save : bool, optional
        If True, saves the broad-band image to a FITS file in the output directory.
        Default is False.
    
    Returns
    -------
    mpdaf.obj.Image
        Broad-band image created by summing over the specified wavelength range.
    
    Raises
    ------
    FileNotFoundError
        If no FITS cube files are found for the specified cluster.
    """
    # Find and open the cube
    cube_dir = io.get_muse_cube_dir(cluster)
    cube_files = glob.glob(str(cube_dir / '*.fits'))
    if not cube_files:
        raise FileNotFoundError(f"No FITS cube files found for cluster {cluster} in {cube_dir}")
    
    musedata = Cube(cube_files[0])
    musedata.data = np.nan_to_num(musedata.data, nan=0.0) # Replace NaNs with zeros to avoid issues when summing 
                                                          # over wavelength range
    
    # Create broad-band image by summing over the specified wavelength range
    img_bb = musedata.get_image(wave=(bbcenter - bbwidth, bbcenter + bbwidth), unit_wave=u.AA)

    if save:
        misc_dir = io.get_misc_dir(cluster)
        output_file = misc_dir / f'{cluster}_bb_image_{int(bbcenter)}A_{int(bbwidth)}A.fits'
        img_bb.write(str(output_file))
        print(f"Broad-band image of {cluster} saved to {output_file}")
    
    return img_bb