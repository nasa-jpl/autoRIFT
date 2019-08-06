# geoAutorift
ISCE module for finding displacement and motion velocity between two images over user-defined geographic-coordinate grid

Copyright (C) 2019 California Institute of Technology.  Government Sponsorship Acknowledged.

Citation: https://github.com/leiyangleon/geoAutorift

## Authors

### geogrid
Piyush Agram (JPL/Caltech; piyush.agram@jpl.nasa.gov), Yang Lei (GPS/Caltech; ylei@caltech.edu)

### autorift

Alex Gardner (JPL/Caltech; alex.s.gardner@jpl.nasa.gov) conceived the algorithm and developed the first version in MATLAB;
Yang Lei (GPS/Caltech; ylei@caltech.edu) translated it to Python, further optimized and incoporated to ISCE.
       
       
## Features
This module comprises two parts or sub-modules: "geogrid" and "autorift".

### geogrid
* user can define a grid in geographic coordinates provided in the form of a DEM with arbitrary EPSG code, 
* the program will extract the portion of the grid that overlaps with the given coregistered radar image pair, 
* return the range and azimuth pixel indices in the radar image pair for each grid point
* return the range and azimuth coarse displacement given the motion velocity maps and the local surface slope maps in the direction of both geographic x- and y-coordinates (they must be provided at the same grid as the DEM)
* return the matrix of conversion coefficients that can convert the range and azimuth displacement between the two radar images (precisely estimated later with the sub-module "autorift") to motion velocity in geographic x- and y-coordinates

### autorift

* fast algorithm that finds displacement between the two images using sparse search and progressive and iterative chip sizes
* faster than the conventional ampcor algorithm in ISCE by an order of magnitude
* support various preprocessing modes on the given image pair, e.g. either the raw image (both texture and topography) or the texture only (high-frequency components without the topography) can be used with various choices of the high-pass filter options 
* support data format of either unsigned integer 8 (uint8) or single-precision float (float32)
* user can adjust all of the relevant parameters, e.g. search limit, chip size range, etc
* a Normalized Displacement Coherence (NDC) Filter has been developed to filter image chip displacemnt results based on displacement difference thresholds that are scaled to the search limit
* sparse search is used to first eliminate the unreliable chip displacement results that will not be further used for fine search or following chip size iterations
* another chip size that progresses iteratively is used to determine the chip displacement results that have not been estimated from the previous iterations
* a slight interpolation is done to fill the missing (unreliable) chip displacement results using bicubic mode (that can remove pixel discrepancy when using other modes) and an interpolation mask is returned
* the core image processing is coded by calling OpenCV's Python and/or C++ functions for efficiency 
* the sub-module is not necessarily suitable for radar images, and can also be used for optical, etc


## Install

* First install ISCE
* Put the folder "geoAutorift" and the file "Sconscript" under the ISCE's source folder (where you started installing ISCE; see the snapshot "install_snapshot.png")
* run "scons install" again from command line


## Instructions

### geogrid

* It is recommended to run ISCE up to the step where coregistered SLC's are done, e.g. "mergebursts" for using topsApp.

For quick use:
* Refer to the file "testGeogrid.py" for the usage of the sub-module and modify it for your own purpose
* Input files include the master image folder (required), slave image folder (required), a DEM (required), local surface slope maps, velocity maps

For modular use:
* In Python environment, after importing ISCE "import isce", type "from components.contrib.geoAutorift.geogrid.Geogrid import Geogrid" to import the "geogrid" sub-module, and then type "obj = Geogrid()" followed by "obj.configure()" to initialize the "geogrid" object
* The "geogrid" object has several parameters that has to be set up (listed below; can also be obtained by referring to "testGeogrid.py"): 

       startingRange:       starting range
       rangePixelSize:      range pixel size
       sensingStart:        starting azimuth time
       prf:                 pulse repition frequency 
       lookSide:            look side, e.g. -1 for right looking 
       repeatTime:          time period between the acquisition of the two radar images
       numberOfLines:       number of lines (in azimuth)
       numberOfSamples:     number of samples (in range)
       orbit:               ISCE orbit data structure
       demname:             (input) name of the DEM file
       dhdxname:            (input; not required) name of the local surface slope in x-coodinate file
       dhdyname:            (input; not required) name of the local surface slope in y-coodinate file
       vxname:              (input; not required) name of the motion velocity in x-coodinate file
       vyname:              (input; not required) name of the motion velocity in y-coodinate file
       ------------------output
       winlocname:          (output) name of the range and azimuth pixel indices (at each grid point) file
       winoffname:          (output) name of the range and azimuth coarse displacement (at each grid point) file
       winro2vxname:        (output) name of the conversion coefficients from radar displacement (range and azimuth) to motion velocity in x-coordinate (at each grid point) file
       winro2vyname:        (output) name of the conversion coefficients from radar displacement (range and azimuth) to motion velocity in y-coordinate (at each grid point) file

* After the above parameters are set, run the sub-module by typing "obj.geogrid()" to create the output files


