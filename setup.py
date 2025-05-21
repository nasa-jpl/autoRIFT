#!/usr/bin/env python3
from setuptools import setup
from setuptools.extension import Extension
from pathlib import Path
from sysconfig import get_path

import numpy as np

INCLUDE_PATHS = [
    str(Path(get_path('include')).parent),
    np.get_include(),
]
LIBRARY_PATHS = [
    str(Path(get_path('purelib')).parent.parent),
]

extensions = [
    Extension(
        name="autoRIFT/autoriftcore",
        sources=[
            'geo_autoRIFT/autoRIFT/bindings/autoriftcoremodule.cpp',
        ],
        include_dirs=INCLUDE_PATHS + [INCLUDE_PATHS[0] + '/opencv4', 'geo_autoRIFT/autoRIFT/include'],
        library_dirs=LIBRARY_PATHS,
        libraries=['opencv_core', 'opencv_imgproc'],
        extra_compile_args=['-std=c++11', '-fopenmp', '-O4'],
        language="c++",
    ),
    Extension(
        name="geogrid/geogridOptical",
        sources=[
            'geo_autoRIFT/geogrid/bindings/geogridOpticalmodule.cpp',
            'geo_autoRIFT/geogrid/src/geogridOptical.cpp',
        ],
        include_dirs=INCLUDE_PATHS + ['geo_autoRIFT/geogrid/include'],
        library_dirs=LIBRARY_PATHS,
        libraries=['gomp', 'gdal'],
        extra_compile_args=['-std=c++11', '-O4'],
        language="c++"
    ),
    Extension(
        name="geogrid/geogridRadar",
        sources=[
            'geo_autoRIFT/geogrid/bindings/geogridRadarmodule.cpp',
            'geo_autoRIFT/geogrid/src/geogridRadar.cpp',
        ],
        include_dirs=INCLUDE_PATHS + [INCLUDE_PATHS[0] + '/eigen3', 'geo_autoRIFT/geogrid/include'],
        library_dirs=LIBRARY_PATHS,
        libraries=['gomp', 'gdal', 'isce3'],
        extra_compile_args=['-std=c++17', '-O4'],
        language="c++"
    )
]

setup(name='geo_autoRIFT',
      version='2.0.0',
      description='This is the autoRIFT python package',
      python_requires='>=3.10',
      package_dir={'autoRIFT': 'geo_autoRIFT/autoRIFT', 'geogrid': 'geo_autoRIFT/geogrid'},
      packages=['autoRIFT', 'geogrid'],
      ext_modules=extensions)
