#!/usr/bin/env python3
import numpy as np
import os
 
from distutils.core import setup
from distutils.extension import Extension


##Figure out opencv paths
try:
    import cv2
except:
    raise Exception('OpenCV does not appear to be installed. Install before proceeding ... ')


##Figure out paths for headers and libraries
bldInfo = cv2.getBuildInformation().splitlines()
for line in bldInfo:
    if 'Install to:' in line:
        path = line.split()[-1]
        break

print('Open CV path: ', path)



extensions = [
     Extension(
        name="autoRIFT/autoriftcore",
        sources= ['geo_autoRIFT/autoRIFT/bindings/autoriftcoremodule.cpp'],
        include_dirs=[np.get_include()] + 
                    ['geo_autoRIFT/autoRIFT/include',
                     os.path.join(path, 'include')],
        library_dirs = [os.path.join(path, 'lib')],
        libraries=['opencv_core', 'opencv_highgui', 'opencv_imgproc'],
        extra_compile_args=['-std=c++11'],
        language="c++"
     )
]
 
setup (name = 'autoRIFT',
        version = '1.0',
        description = 'This is the autoRIFT python package',
        package_dir={'autoRIFT': 'geo_autoRIFT/autoRIFT', 'geogrid': 'geo_autoRIFT/geogrid'},
        packages=['autoRIFT', 'geogrid'],
#         scripts=['geo_autoRIFT/geogrid/GeogridOptical.py'],
        ext_modules = extensions)
