# autoRIFT (autonomous Repeat Image Feature Tracking)



[![Language](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://www.python.org/)
[![Latest version](https://img.shields.io/badge/latest%20version-v1.3.0-yellowgreen.svg)](https://github.com/leiyangleon/autoRIFT/releases)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/leiyangleon/autoRIFT/blob/master/LICENSE)
[![Citation](https://img.shields.io/badge/DOI-10.3390/rs13040749-blue)](https://doi.org/10.3390/rs13040749)

### Update Notes:

```diff
+ improved memory use (by 50%) for autoRIFT and runtime (60x) for GeogridOptical
+ parallel computing for NCC
+ support for remote input files using GDAL virtual file systems (e.g., `/vsicurl/https://...`)
+   see: https://gdal.org/user/virtual_file_systems.html
```


**A Python module of a fast and intelligent algorithm for finding the pixel displacement between two images**

**autoRIFT can be installed as a standalone Python module (only supports Cartesian coordinates) either manually or as a conda install (https://github.com/conda-forge/autorift-feedstock). To allow support for both Cartesian and radar coordinates, autoRIFT must be installed with the InSAR Scientific Computing Environment (ISCE: https://github.com/isce-framework/isce2)**

**Use cases include all dense feature tracking applications, including the measurement of surface displacements occurring between two repeat satellite images as a result of glacier flow, large earthquake displacements, and land slides**  

**autoRIFT can be used for dense feature tracking between two images over a grid defined in an arbitrary geographic Cartesian (northing/easting) coordinate projection when used in combination with the sister Geogrid Python module (https://github.com/leiyangleon/Geogrid). Example applications include searching radar-coordinate imagery on a polar stereographic grid and searching Universal Transverse Mercator (UTM) imagery at a specified geographic Cartesian (northing/easting) coordinate grid**


Copyright (C) 2019 California Institute of Technology.  Government Sponsorship Acknowledged.

Link: https://github.com/leiyangleon/autoRIFT

Citation: https://doi.org/10.3390/rs13040749


## 1. Authors

Alex Gardner (JPL/Caltech; alex.s.gardner@jpl.nasa.gov) first described the algorithm "auto-RIFT" in (Gardner et al., 2018) and developed the first version in MATLAB;

Yang Lei (GPS/Caltech; ylei@caltech.edu) translated it to Python, further optimized and incoporated to the ISCE software;

Piyush Agram (JPL/Caltech; piyush.agram@jpl.nasa.gov) set up the installation as a standalone module and further cleaned the code.

**Reference:** 

Gardner, A.S., Moholdt, G., Scambos, T., Fahnstock, M., Ligtenberg, S., Broeke, M.V.D. and Nilsson, J., 2018. [**Increased West Antarctic and unchanged East Antarctic ice discharge over the last 7 years.**](https://doi.org/10.5194/tc-12-521-2018) *The Cryosphere*, 12(2), pp.521-547. 

**[NEW]** Lei, Y., Gardner, A. and Agram, P., 2021. [**Autonomous Repeat Image Feature Tracking (autoRIFT) and Its Application for Tracking Ice Displacement.**](https://doi.org/10.3390/rs13040749) *Remote Sensing*, 13(4), p.749. 


## 2. Acknowledgement

This effort was funded by the NASA MEaSUREs program in contribution to the Inter-mission Time Series of Land Ice Velocity and Elevation (ITS_LIVE) project (https://its-live.jpl.nasa.gov/) and through Alex Gardner’s participation in the NASA NISAR Science Team
    
       
## 3. [Features](/docs/features.md)



## 4. [Possible Future Development](/docs/future.md)




## 5. Demo

### 5.1 [Optical image over regular grid in imaging coordinates](/docs/opt_img.md)




### 5.2 [Optical image over user-defined geographic Cartesian coordinate grid](/docs/opt_geo.md)




### 5.3 [Radar image over regular grid in imaging coordinates](/docs/rdr_img.md)




### 5.4 [Radar image over user-defined geographic Cartesian coordinate grid](/docs/rdr_geo.md)




## 6. [Install](/docs/install.md)





## 7. Instructions

**Note:**

* When the grid is provided in geographic Cartesian (northing/easting) coordinates (geographic coordinates lat/lon is not supported), it is required to run the "Geogrid" module (https://github.com/leiyangleon/Geogrid) first before running "autoRIFT". In other words, the outputs from "testGeogrid_ISCE.py" or "testGeogridOptical.py" (a.k.a "winlocname", "winoffname", "winsrname", "wincsminname", "wincsmaxname", "winro2vxname", "winro2vyname") will serve as optional inputs for running "autoRIFT" at a geographic grid, which is required to generate the final motion velocity maps.
* When the outputs from running the "Geogrid" module are not provided, a regular grid in the imaging coordinates will be automatically assigned

**For quick use:**

* Refer to the file "testautoRIFT.py" (standalone) and "testautoRIFT_ISCE.py" (with ISCE) for the usage of the module and modify it for your own purpose
* Input files include the reference image (required), test image (required), and the outputs from running "testGeogrid_ISCE.py" or "testGeogridOptical.py" (optional; a.k.a "winlocname", "winoffname", "winsrname", "wincsminname", "wincsmaxname", "winro2vxname", "winro2vyname"). When "winlocname" (grid location) is specified, each of the rest optional input ("win * name") can be either used or omitted.
* Output files include 1) estimated horizontal displacement (equivalent to range for radar), 2) estimated vertical displacement (equivalent to minus azimuth for radar), 3) light interpolation mask, 4) iteratively progressive chip size used. 

_Note: These four output files will be stored in a file named "offset.mat" that can be viewed in Python and MATLAB. When the grid is provided in geographic Cartesian (northing/easting) coordinates, a 4-band GeoTIFF with the same EPSG code as input grid will be created as well and named "offset.tif"; a 2-band GeoTIFF of the final converted motion velocity in geographic x- (easting) and y- (northing) coordinates will be created and named "velocity.tif". Also, it is possible to save the outputs in netCDF standard format by adding the "-nc" option to the "testautoRIFT.py" (standalone) and "testautoRIFT_ISCE.py" (with ISCE) command._

**For modular use:**

* In Python environment, type the following to import the "autoRIFT" module and initialize the "autoRIFT" object

_With ISCE:_

       import isce
       from contrib.geo_autoRIFT.autoRIFT import autoRIFT_ISCE
       obj = autoRIFT_ISCE()
       obj.configure()

_Standalone:_

       from autoRIFT import autoRIFT
       obj = autoRIFT()

* The "autoRIFT" object has several inputs that have to be assigned (listed below; can also be obtained by referring to "testautoRIFT.py"): 
       
       ------------------input------------------
       I1                                                 reference image (extracted image patches defined as "source")
       I2                                                 test image (extracted image patches defined as "template"; displacement = motion vector of I2 relative to I1)
       xGrid [units = integer image pixels]               horizontal pixel index at each grid point
       yGrid [units = integer image pixels]               vertical pixel index at each grid point
       (if xGrid and yGrid not provided, a regular grid spanning the entire image will be automatically set up, which is similar to the conventional ISCE module, "ampcor" or "denseampcor")
       Dx0 [units = integer image pixels]                 horizontal "downstream" reach location (that specifies the horizontal pixel displacement of the template's search center relative to the source's) at each grid point
       Dy0 [units = integer image pixels]                 vertical "downstream" reach location (that specifies the vertical pixel displacement of the template's search center relative to the source's) at each grid point
       (if Dx0 and Dy0 not provided, an array with zero values will be automatically assigned and there will be no offsets of the search centers)

* After the inputs are specified, run the module as below
       
       obj.preprocess_filt_XXX() or obj.preprocess_db()
       obj.uniform_data_type()
       obj.runAutorift()

where "XXX" can be "wal" for the Wallis filter, "hps" for the trivial high-pass filter, "sob" for the Sobel filter, "lap" for the Laplacian filter, and also a logarithmic operator without filtering is adopted for occasions where low-frequency components (i.e. topography) are desired, i.e. "obj.preprocess_db()".

* The "autoRIFT" object has the following four outputs: 
       
       ------------------output------------------
       Dx [units = decimal image pixels]                  estimated horizontal pixel displacement
       Dy [units = decimal image pixels]                  estimated vertical pixel displacement
       InterpMask [unitless; boolean data type]           light interpolation mask
       ChipSizeX [units = integer image pixels]           iteratively progressive chip size in horizontal direction (different chip size allowed for vertical)

* The "autoRIFT" object has many parameters that can be flexibly tweaked by the users for their own purpose (listed below; can also be obtained by referring to "geo_autoRIFT/autoRIFT/autoRIFT_ISCE.py"):

       ------------------parameter list: general function------------------
       ChipSizeMinX [units = integer image pixels]                Minimum size (in horizontal direction) of the template (chip) to correlate (default = 32; could be scalar or array with same dimension as xGrid)
       ChipSizeMaxX [units = integer image pixels]                Maximum size (in horizontal direction) of the template (chip) to correlate (default = 64; could be scalar or array with same dimension as xGrid)
       ChipSize0X [units = integer image pixels]                  Minimum acceptable size (in horizontal direction) of the template (chip) to correlate (default = 32)
       ScaleChipSizeY [unitless; integer data type]               Scaling factor to get the vertically-directed chip size in reference to the horizontally-directed size (default = 1)
       SearchLimitX [units = integer image pixels]                Range (in horizontal direction) to search for displacement in the source (default = 25; could be scalar or array with same dimension as xGrid; when provided in array, set its elements to 0 if excluded for finding displacement)
       SearchLimitY [units = integer image pixels]                Range (in vertical direction) to search for displacement in the source (default = 25; could be scalar or array with same dimension as xGrid; when provided in array, set its elements to 0 if excluded for finding displacement)
       SkipSampleX [units = integer image pixels]                 Number of samples to skip between search windows in horizontal direction for automatically-created grid if not specified by the user (default = 32)
       SkipSampleY [units = integer image pixels]                 Number of lines to skip between search windows in vertical direction for automatically-created grid if not specified by the user (default = 32)
       minSearch [units = integer image pixels]                   Minimum search limit (default = 6)
       
       ------------------parameter list: about Normalized Displacement Coherence (NDC) filter ------------------
       FracValid                   Fraction of valid displacements (default = 8/25) to be multiplied by filter window size "FiltWidth^2" and then used for thresholding the number of chip displacements that have passed the "FracSearch" disparity filtering
       FracSearch                  Fraction of search limit used as threshold for disparity filtering of the chip displacement difference that is normalized by the search limit (default = 0.25)
       FiltWidth                   Disparity filter width (default = 5)
       Iter                        Number of iterations (default = 3)
       MadScalar                   Scalar to be multiplied by Mad used as threshold for disparity filtering of the chip displacement deviation from the median (default = 4)
       
       ------------------parameter list: miscellaneous------------------
       WallisFilterWidth           Width of the filter to be used for the preprocessing (default = 21)
       fillFiltWidth               Light interpolation filling filter width (default = 3)
       sparseSearchSampleRate      downsampling rate for sparse search  (default = 4)
       BuffDistanceC               Buffer coarse correlation mask by this many pixels for use as fine search mask (default = 8)
       CoarseCorCutoff             Coarse correlation search cutoff (default = 0.01)
       OverSampleRatio             Factor for pyramid upsampling for sub-pixel level offset refinement (default = 16; can be scalar or Python dictionary for intelligent use, i.e. chip size-dependent, such as {ChipSize_1:OverSampleRatio_1,..., ChipSize_N:OverSampleRatio_N})
       DataTypeInput               Image data type: 0 -> uint8, 1 -> float32 (default = 0)
       zeroMask                    Force the margin (no data) to zeros which is useful for Wallis-filter-preprocessed images (default = None; 1 for Wallis filter)
