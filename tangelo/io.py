"""
Input/Output utilities module.

This module provides functions to manage data directories, load spectra from different sources,
and handle environment variables for processing of MUSE spectra.
"""

import os
import glob
import re
import numpy as np
from astropy.io import fits
import astropy.table as aptb
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from mpdaf.obj import Cube

def get_r21_cat_url():
    """
    Get the base URL for the R21 catalogs from the R21_CAT_URL environment variable.
    This must be set by the user to point to the correct server location.

    Returns
    -------
    str
        Base URL for the R21 catalog.
    """
    base_url = os.environ.get('R21_CAT_URL')
    if not base_url:
        # Prompt user for the URL
        print("\nR21_CAT_URL environment variable not set.")
        print("Please provide the base URL for the R21 catalog (e.g., 'https://example.com/r21_catalogs/').")
        try:
            user_url = input("Enter R21 catalog base URL: ").strip()
            if user_url:
                os.environ['R21_CAT_URL'] = user_url
                print(f"Using: {user_url}")
                print(f"To make this permanent, add to your ~/.bashrc:")
                print(f"  export R21_CAT_URL={user_url}")
                return user_url
            else:
                raise ValueError("R21_CAT_URL is required but not provided.")
        except EOFError:
            raise ValueError("R21_CAT_URL is required but not provided.")
    return os.environ['R21_CAT_URL']
    
def download_r21_catalogue(cluster, dest_dir=None):
    """
    Download the R21 lens catalog from the server, with special case handling for the Bullet cluster.

    Parameters
    ----------
    cluster : str
        Name of the cluster.
    dest_dir : str or Path, optional
        Destination directory for the downloaded file. If None, uses the default catalog directory.

    Returns
    -------
    str or None
        Path to the downloaded FITS file, or None if not found.

    Example
    -------
    >>> download_r21_catalogue('A2744')
    '/path/to/catalog/A2744_v1.1.fits'
    """

    # Use provided destination directory or get from configuration
    if dest_dir is None:
        dest_dir = get_catalog_dir()
    else:
        dest_dir = Path(dest_dir)

    # Ensure destination directory exists
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Handle the special case for Bullet cluster
    if cluster.upper() == 'BULLET':
        server_cluster_name = 'Bullet'  # Lowercase on server
    else:
        server_cluster_name = cluster   # Normal case on server

    # Use the module-level constant for the base URL
    base_url = get_r21_cat_url()
    print(f"Scraping directory listing: {base_url}")
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Find the latest versioned FITS file
    pattern = re.compile(rf"{cluster.lower()}_v\d+\.\d+\.fits", re.IGNORECASE)
    fits_file = None
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and pattern.fullmatch(href.lower()):
            fits_file = href
            print(f"Found catalogue file: {fits_file}")
            break

    if not fits_file:
        print("No matching FITS file found in directory listing.")
        return None

    # Download the file
    file_url = base_url + fits_file
    print(f"Downloading {file_url} ...")
    file_response = requests.get(file_url)
    dest_path = dest_dir / fits_file
    with open(dest_path, "wb") as f:
        f.write(file_response.content)
    print(f"Download complete: {dest_path}")

    return str(dest_path)

def get_spectra_url():
    """
    Get the spectra base URL from the R21_URL environment variable. If not set, prompt the user 
    to enter it interactively. Note that this URL is different from the catalog URL.

    Returns
    -------
    str
        The spectra base URL.

    Raises
    ------
    ValueError
        If not set and user does not provide one.
    """
    import os
    url = os.environ.get('R21_SPEC_URL')
    if url:
        return url
    try:
        url = input("R21 spectra base URL not set. Please enter the URL: ").strip()
        if url:
            os.environ['R21_SPEC_URL'] = url # Set for this session
            return url
        else:
            raise ValueError("R21 spectra base URL is required but not provided.")
    except Exception:
        raise ValueError("R21 spectra base URL is required but not provided.")

# Directory management functions

def get_data_dir():
    """
    Get the base data directory from the MUSE_DATA_DIR environment variable.
    If not set, prompts the user to provide the path.

    Returns
    -------
    Path
        The base data directory path as a pathlib.Path object.
    """
    
    data_dir = os.environ.get('MUSE_DATA_DIR')
    if data_dir:
        return Path(data_dir)

    # Prompt user for the path
    print("\nMUSE_DATA_DIR environment variable not set.")
    print("Please provide the path to your astronomy data directory.")

    try:
        user_path = input("Enter data directory path: ").strip()
        if user_path:
            # Set the environment variable for this session
            os.environ['MUSE_DATA_DIR'] = user_path
            print(f"Using: {user_path}")
            print(f"To make this permanent, add to your ~/.bashrc:")
            print(f"  export MUSE_DATA_DIR={user_path}")
            return Path(user_path)
        else:
            raise ValueError("MUSE_DATA_DIR is required but not provided.")
    except (EOFError, KeyboardInterrupt):
        raise ValueError("\nMUSE_DATA_DIR is required but not provided.")

def get_r21_catalog_dir(cluster):
    """
    Get the directory where the R21 catalogues are stored for a given cluster.

    Parameters
    ----------
    cluster : str
        Cluster name (e.g., 'A2744', 'MACS0416', etc.)

    Returns
    -------
    Path
        The directory path where the catalogues for the specified cluster are stored as a pathlib.Path object.
    """
    data_dir = get_data_dir()
    catalog_dir = data_dir / cluster.upper() / 'catalogs' / 'R21'
    if not catalog_dir.exists(): # Raise an error if the directory does not exist (the user should have created it)
        raise FileNotFoundError(f"Catalog directory does not exist: {catalog_dir}")
    return catalog_dir

def get_fit_catalog_dir(cluster):
    """
    Get the directory where the user-generated fit catalogs are stored for a given cluster.

    Parameters
    ----------
    cluster : str
        Cluster name (e.g., 'A2744', 'MACS0416', etc.)

    Returns
    -------
    Path
        The directory path where the fit catalogues for the specified cluster are stored as a pathlib.Path object.
    """
    data_dir = get_data_dir()
    catalog_dir = data_dir / cluster.upper() / 'catalogs' / 'fit_results'
    if not catalog_dir.exists(): # Raise an error if the directory does not exist (the user should have created it)
        raise FileNotFoundError(f"Catalog directory does not exist: {catalog_dir}")
    return catalog_dir

def get_r21_spectra_dir(cluster):
    """
    Get the directory where the R21 spectra are stored for a given cluster.

    Parameters
    ----------
    cluster : str
        Cluster name (e.g., 'A2744', 'MACS0416', etc.)

    Returns
    -------
    Path
        The directory path where the spectra for the specified cluster are stored as a pathlib.Path object.
    """
    data_dir = get_data_dir()
    spectra_dir = data_dir / cluster.upper() / 'spectra' / 'R21'
    if not spectra_dir.exists(): # Raise an error if the directory does not exist (the user should have created it)
        raise FileNotFoundError(f"Spectra directory does not exist: {spectra_dir}")
    return spectra_dir

def get_aper_spectra_dir(cluster):
    """
    Get the directory where the aperture spectra are stored for a given cluster.

    Parameters
    ----------
    cluster : str
        Cluster name (e.g., 'A2744', 'MACS0416', etc.)

    Returns
    -------
    Path
        The directory path where the aperture spectra for the specified cluster are stored as a pathlib.Path object.
    """
    data_dir = get_data_dir()
    spectra_dir = data_dir / cluster.upper() / 'spectra' / 'aper'
    if not spectra_dir.exists(): # Raise an error if the directory does not exist (the user should have created it)
        raise FileNotFoundError(f"Aperture spectra directory does not exist: {spectra_dir}")
    return spectra_dir

def get_misc_dir(cluster) -> Path:
    """
    Get the directory where miscellaneous data files are stored for a given cluster.

    Parameters
    ----------
    cluster : str
        Cluster name (e.g., 'A2744', 'MACS0416', etc.)

    Returns
    -------
    Path
        The directory path where miscellaneous data files for the specified cluster are stored as a pathlib.Path object.
    """
    data_dir = get_data_dir()
    misc_dir = data_dir / cluster.upper() / 'misc'
    if not misc_dir.exists(): # Raise an error if the directory does not exist (the user should have created it)
        raise FileNotFoundError(f"Miscellaneous data directory does not exist: {misc_dir}")
    
    return misc_dir

def get_muse_cube_dir(cluster):
    """
    Get the directory where the MUSE cubes are stored for a given cluster.

    Parameters
    ----------
    cluster : str
        Cluster name (e.g., 'A2744', 'MACS0416', etc.)

    Returns
    -------
    Path
        The directory path where the MUSE cubes for the specified cluster are stored as a pathlib.Path object.
    """
    data_dir = get_data_dir()
    cube_dir = data_dir / cluster.upper() / 'cube'
    if not cube_dir.exists(): # Raise an error if the directory does not exist (the user should have created it)
        raise FileNotFoundError(f"MUSE cube directory does not exist: {cube_dir}")
    return cube_dir

# Data loading functions

def load_r21_catalogue(cluster, type='source'):
    """
    Load the R21 lensing catalog for a given cluster from local cache, or download if not present.

    Parameters
    ----------
    cluster : str
        Name of the cluster.
    type : str
        Type of catalog to load ('source' or 'line').

    Returns
    -------
    astropy.table.Table
        The R21 lensing catalog as an Astropy Table.

    Raises
    ------
    FileNotFoundError
        If the catalog cannot be found or downloaded.
    """
    # Define the path to the local cache directory
    cache_dir = get_r21_catalog_dir(cluster)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fpattern = f"{cluster}_v?.?.fits" if type == 'source' else f"{cluster}_v?.?_lines.fits"

    # Local file path
    local_path = cache_dir.glob(fpattern)
    local_path = list(local_path)

    # If not found locally, attempt to download
    if len(local_path) == 0:
        print(f"R21 catalog for {cluster} not found locally. Attempting to download...")
        success = download_r21_catalogue(cluster)
        if not success:
            raise FileNotFoundError(f"R21 catalog for {cluster} could not be downloaded.")
        
    # Check again for the local file after download attempt
    local_path = cache_dir.glob(fpattern)
    local_path = list(local_path)
    if len(local_path) == 0:
        raise FileNotFoundError(f"R21 catalog for {cluster} still not found after download attempt.")
    elif len(local_path) > 1:
        print(f"Multiple versions of R21 catalog found for {cluster}. Using the first one.")

    local_path = local_path[0]  # Use the first match if multiple found
    
    # Load and return the catalog as an astropy table
    try:
        r21_table = aptb.Table(fits.open(local_path)[1].data)
        return r21_table
    except Exception as e:
        raise FileNotFoundError(f"Failed to load R21 catalog for {cluster}: {e}")

def load_spec(clus, iden, idfrom, spec_source = 'R21', spec_type = 'weight_skysub'):
    """
    Loads a spectrum from the specified source.

    Parameters
    ----------
    clus : str
        Cluster name (e.g., 'A2744', 'MACS0416', etc.)
    iden : int
        Identifier number of the object (e.g., 1234) or str (e.g., 'X1234')
    idfrom : str
        Detection type of the source (i.e., 'MUSELET', 'PRIOR', or 'EXTERNAL')
    spec_source : str
        Source of the spectrum ('R21' for Richard et al. 2021, 'APER' for aperture spectra)
    spec_type : str
        Type of spectrum to load (either 'weight_skysub' or '2fwhm', etc.)

    Returns
    -------
    astropy.table.Table or None
        The loaded spectrum table, or None if loading failed.
    """
    if spec_source == 'R21':
        return load_r21_spec(clus, iden, idfrom, spec_type)
    elif spec_source == 'APER':
        return load_aper_spec(clus, iden, idfrom, spec_type)
    else:
        raise ValueError(f"spec_source {spec_source} not recognized. Use 'R21' or 'APER'.")
    
def download_r21_spectrum(clus, iden, spec_type):
    """
    Retrieve a spectrum from the R21 database

    Parameters
    ----------
    clus : str
        Cluster name (e.g., 'A2744', 'MACS0416', etc.)
    iden : int
        Identifier number of the object (e.g., 1234)
    spec_type : str
        Type of spectrum to download

    Returns
    -------
    bool
        True if the download was successful, False otherwise.
    """
    cluster_dir = get_r21_spectra_dir(clus)
    os.makedirs(cluster_dir, exist_ok=True)
    if clus == 'BULLET':
        os.system(f"wget --no-check-certificate {get_spectra_url()}Bullet_final_catalog/spectra/spec_{iden}_{spec_type}.fits"
                + f" -P {cluster_dir}")
    else:
        os.system(f"wget --no-check-certificate {get_spectra_url()}{clus}_final_catalog/spectra/spec_{iden}_{spec_type}.fits"
                + f" -P {cluster_dir}")

def load_r21_spec(clus, iden, idfrom, spec_type):
    """
    Loads a spectrum from the Richard et al. (2021) catalog.

    Parameters
    ----------
    clus : str
        Cluster name (e.g., 'A2744', 'MACS0416', etc.)
    iden : int
        Identifier number of the object (e.g., 1234)
    idfrom : str
        Prefix letter of the identifier (e.g., 'X' for X1234)
    spec_type : str
        Type of spectrum to load

    Returns
    -------
    astropy.table.Table or None
        The loaded spectrum table, or None if loading failed.
    """
    # Generate the full identifier
    if isinstance(iden, str):
        if iden[0].isdigit():
            identifier = (idfrom[0] + str(iden)).replace('E', 'X')
        elif iden[0].isalpha():
            identifier = iden
        else:
            raise ValueError("iden must start with a letter or digit")
    else:
        # Try to convert to int
        iden = int(iden)
        if isinstance(iden, int):
            identifier = f"{idfrom[0]}{iden}".replace('E', 'X')
        else:
            raise ValueError("iden must be an int or str")

    print(f"Loading R21 spectrum for {clus} object {identifier}...")

    # Get the spectra directory and construct the full path
    cluster_dir = get_r21_spectra_dir(clus)

    # Locate the file
    locfile = glob.glob(f"{cluster_dir}/spec_{identifier}_{spec_type}.fits")
    if len(locfile) == 0: # If the file is not found, attempt to download it
        print(f"File not found. Downloading from {get_spectra_url()}{clus}_final_catalog/spectra/spec_{identifier}_{spec_type}.fits")
        download_r21_spectrum(clus, identifier, spec_type)
        print(f"Download complete.")
        locfile = glob.glob(f"{cluster_dir}/spec_{identifier}_{spec_type}.fits")
    
    # If still not found, return None
    if len(locfile) == 0:
        print(f"File still not found after download attempt!")
        return None
    elif len(locfile) > 1:
        print(f"Multiple files found! Using the first one.")
    
    locfile = locfile[0] # Use the first match if multiple found

    # Load the spectrum as an astropy table
    def try_load(locfile):
        try:
            hdulist = fits.open(locfile)
            hdu = hdulist[1]
            varhdu = hdulist[2]
            spec = hdu.data
            errspec = np.sqrt(varhdu.data)
            wavelength = hdu.header['CRVAL1'] + np.arange(0., hdu.header['CDELT1'] * hdu.header['NAXIS1'], hdu.header['CDELT1'])
            return aptb.Table([wavelength, spec, errspec], names=('wave', 'spec', 'spec_err'))
        except Exception as e:
            print(f"Error loading FITS file: {e}")
            return None

    result = try_load(locfile)
    if result is None:
        print("File appears corrupted or truncated. Attempting to re-download...")
        # Remove the corrupted file
        try:
            os.remove(locfile)
        except Exception as e:
            print(f"Could not remove corrupted file: {e}")
        # Re-download
        download_r21_spectrum(clus, identifier, spec_type)
        print("Re-download complete.")
        # Try loading again
        result = try_load(locfile)
        if result is None:
            print("File still corrupted after re-download!")
            return None
    return result

def load_aper_spec(clus, iden, idfrom, spec_type = '2fwhm'):
    """
    Loads a spectrum from the aperture spectra files.

    Parameters
    ----------
    clus : str
        Cluster name (e.g., 'A2744', 'MACS0416', etc.)
    iden : str
        Identifier string of the object (e.g., 'X1234')
    idfrom : str
        Prefix letter of the identifier (e.g., 'E' for E1234) - not used for aperture spectra
    spec_type : str
        Type of spectrum to load ('2fwhm', '1fwhm', etc.)

    Returns
    -------
    astropy.table.Table or None
        The loaded spectrum table, or None if loading failed.
    """
    # Generate the full identifier
    if isinstance(iden, str):
        if iden[0].isdigit():
            identifier = (idfrom[0] + str(iden)).replace('E', 'X')
        elif iden[0].isalpha():
            identifier = iden
        else:
            raise ValueError("iden must start with a letter or digit")
    else:
        # Try to convert to int
        iden = int(iden)
        if isinstance(iden, int):
            identifier = f"{idfrom[0]}{iden}".replace('E', 'X')
        else:
            raise ValueError("iden must be an int or str")

    print(f"Loading aperture spectrum for {clus} object {identifier}...")

    # Get the source spectra directory and construct the full path
    cluster_dir = get_aper_spectra_dir(clus)
    
    # Locate the file
    locfile = glob.glob(f"{cluster_dir}/{identifier}_{spec_type}.fits")
    if len(locfile) == 0:
        print(f"File not found!")
        return None
    elif len(locfile) > 1:
        print(f"Multiple files found! Using the first one.")
    
    locfile = locfile[0] # Use the first match if multiple found

    # Load the spectrum as an astropy table
    try:
        spectab = aptb.Table(fits.open(locfile)[1].data)
    except AttributeError:
        print("Error: The HDU does not have a .data attribute.")
        spectab = None

    return spectab

def load_segmentation_map(clus, download_if_missing=False):
    """
    Load the R21 segmentation map for a given cluster.

    Parameters
    ----------
    clus : str
        Cluster name (e.g., 'A2744', 'MACS0416', etc.)
    download_if_missing : bool, optional
        If True, attempt to download the segmentation map if not found locally. Default is False.

    Returns
    -------
    astropy.io.fits.HDUList or None
        The segmentation map HDUList, or None if loading failed.
    """
    segmap_path = get_misc_dir(clus) / 'seg.fits'

    if not os.path.isfile(segmap_path):
        print(f"Segmentation map file not found: {segmap_path}")

        if not download_if_missing:
            return None

        # Attempt to download the segmentation map
        print(f"Attempting to download segmentation map for {clus}...")
        os.makedirs(os.path.dirname(segmap_path), exist_ok=True)
        try:
            os.system(f"wget --no-check-certificate {get_spectra_url()}{clus}_final_catalog/segmentation_maps/{clus}_segmentation.fits"
                      + f" -O {segmap_path}")
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading segmentation map: {e}")
            return None
    
    try:
        return fits.open(segmap_path)
    except Exception as e:
        print(f"Error loading segmentation map: {e}")
        return None
    
def load_weight_map(clus):
    """
    Load the R21 weight map for a given cluster.

    Parameters
    ----------
    clus : str
        Cluster name (e.g., 'A2744', 'MACS0416', etc.)
    download_if_missing : bool, optional
        If True, attempt to download the weight map if not found locally. Default is False.

    Returns
    -------
    astropy.io.fits.HDUList or None
        The weight map HDUList, or None if loading failed.
    """
    weightmap_path = get_misc_dir(clus) / 'image_weight.fits'

    if not os.path.isfile(weightmap_path):
        print(f"Weight map file not found: {weightmap_path}")
        raise FileNotFoundError(f"Weight map file not found: {weightmap_path}")
    
    try:
        return fits.open(weightmap_path)
    except Exception as e:
        print(f"Error loading weight map: {e}")
        return None

    
def load_muse_cube(clus):
    """
    Load the MUSE cube for a given cluster.

    Parameters
    ----------
    clus : str
        Cluster name (e.g., 'A2744', 'MACS0416', etc.)

    Returns
    -------
    astropy.io.fits.HDUList
        The MUSE cube HDUList.

    Raises
    ------
    FileNotFoundError
        If the MUSE cube file cannot be found.
    """
    cube_path = list(get_muse_cube_dir(clus).glob(f"DATACUBE*.fits"))

    if len(cube_path) == 0:
        raise FileNotFoundError(f"MUSE cube file not found for cluster {clus}")

    try:
        return Cube(str(cube_path[0]))
    except Exception as e:
        raise FileNotFoundError(f"Error loading MUSE cube: {e}")
    
def save_spectrum(spectrum, spec_file):
    """
    Save the extracted spectrum to a FITS file. If input spectrum is an MPDAF Spectrum object,
    processes it to a standard format before saving. Otherwise, checks column names, modifies
    if necessary, and saves the astropy Table directly.

    Parameters
    ----------
    spectrum : mpdaf.obj.Spectrum (or similar) or astropy.table.Table
        The extracted spectrum object.
    spec_file : str or Path
        The file path where the spectrum will be saved.
    """
    if 'mpdaf' in str(type(spectrum)):
        # Convert MPDAF Spectrum to astropy Table
        wave = spectrum.wave.coord()
        spec = spectrum.data
        spec_err = spectrum.var**0.5
        spec_table = aptb.Table([wave, spec, spec_err], names=('wave', 'spec', 'spec_err'))
    elif isinstance(spectrum, aptb.Table):
        spec_table = spectrum
        # Check for required columns
        required_cols = {'wave', 'spec', 'spec_err'}
        # If columns are named differently, attempt to rename them
        # Wavelength must be monotonically increasing, so look for that first
        for col in spec_table.colnames:
            if np.all(np.diff(spec_table[col]) > 0):
                spec_table.rename_column(col, 'wave')
                break
        # Next, look for flux and error columns based on typical names
        for col in spec_table.colnames:
            if col.lower() in ['flux', 'spec', 'spectrum']:
                spec_table.rename_column(col, 'spec')
            elif col.lower() in ['error', 'spec_err', 'specerror', 'uncertainty', 'err']:
                spec_table.rename_column(col, 'spec_err')
    else:
        raise TypeError("spectrum must be an MPDAF Spectrum object or an astropy Table.")

    # Save the spectrum table to FITS
    spec_table.write(spec_file, format='fits', overwrite=True)
    print(f"Spectrum saved to {spec_file}")

def get_plot_dir(cluster, iden):
    """
    Get the directory where plots are stored for a given source. If it doesn't exist, creates it.

    Parameters
    ----------
    cluster : str
        Cluster name (e.g., 'A2744', 'MACS0416', etc.)

    Returns
    -------
    Path
        The directory path where plots for the specified source are stored as a pathlib.Path object.
    """
    data_dir = get_data_dir()
    plot_dir = data_dir / cluster.upper() / 'plots' / iden
    if not plot_dir.exists(): # Create the directory if it does not exist
        plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir

from mpdaf.obj import Image

def load_bb_image(clus, bbcenter, bbwidth):
    """
    Load a broadband image cutout for a given cluster and specified center and width.

    Parameters
    ----------
    clus : str
        Cluster name (e.g., 'A2744', 'MACS0416', etc.)
    bbcenter : float
        The central wavelength of the spectral range used to make the broadband image in Angstroms.
    bbwidth : float
        The half-width of the spectral range used to make the broadband image in Angstroms.

    Returns
    -------
    mpdaf.obj.Image
        The loaded broadband image as an MPDAF Image object.
    """
    # Construct the filename based on the cluster and broadband parameters
    bb_filename = f"{clus}_bb_image_{bbcenter}A_{bbwidth}A.fits"
    bb_path = get_misc_dir(clus) / bb_filename

    if not bb_path.exists():
        raise FileNotFoundError(f"Broadband image file not found: {bb_path}")

    try:
        return Image(str(bb_path))
    except Exception as e:
        raise FileNotFoundError(f"Error loading broadband image: {e}")