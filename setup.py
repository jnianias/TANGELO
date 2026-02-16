from setuptools import setup, find_packages

setup(
    name='tangiers',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "numpy", 
        "scipy",
        "astropy",
        "error_propagation",
        "specutils",
        "mpdaf",
        "photutils",
        "requests",           # Used by catalogue_operations.py
        "beautifulsoup4"      # Used by catalogue_operations.py for web scraping
    ],
)