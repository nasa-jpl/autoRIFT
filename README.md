# geoAutorift
ISCE module for finding displacement and motion velocity between two images over user-defined geographic-coordinate grid

Copyright (C) 2019 California Institute of Technology.  Government Sponsorship Acknowledged.

Citation: https://github.com/leiyangleon/geoAutorift

## Authors

### geogrid
Piyush Agram (JPL/Caltech; piyush.agram@jpl.nasa.gov), Yang Lei (GPS/Caltech; ylei@caltech.edu)

### autorift

Alex Gardner (JPL/Caltech; alex.s.gardner@jpl.nasa.gov) conceived the algorithm and developed the first version in MATLAB, and has been translated to Python, further optimized and incoporated to ISCE by Yang Lei (GPS/Caltech; ylei@caltech.edu).
       
       
## Features
This module comprises two parts or sub-modules: "geogrid" and "autorift".

### geogrid
* user can define a grid in geographic coordinates provided in the form of a DEM with arbitrary EPSG code, 
* the program will extract the portion of the grid that overlaps with the given coregistered image pair (not necessarily radar; can be optical, etc), 
* return the two-dimensional (2D) pixel indices (e.g. range and azimuth for radar) in the image pair for each grid point
* return the coarse pixel displacement given the motion velocity maps and the local surface slope maps in the directions of both geographic x- and y-coordinates (that are provided at the same grid as the DEM)
* return the matrix of conversion coefficients that can convert the displacement between the two images (can be estimated precisely later with the next sub-module "autorift") in image coordinates (2D) to motion velocity in geographic coordinates (2D)

### autorift

* fast algorithm that finds displacement between the two images using sparse search and progressive and iterative chip sizes
* faster than the conventional ampcor algorithm in ISCE by an order of magnitude
* support various preprocessing modes on the given image pair, e.g. either the raw image (both texture and topography) or the texture (high-frequency components) without the topography can be used with various choices of the high-pass filter options 
* support data format of either unsigned integer 8 (uint8) or single-precision float (float32)
* user can adjust all of the relevant parameters, e.g. search limit, chip size range, etc
* a Normalized Displacement Coherence (NDC) Filter has been developed to filter image chip displacemnt results based on displacement difference thresholds that are scaled to the search limit
* sparse search is used to first eliminate the unreliable chip displacement results that will not be not further used for fine search or following chip size iterations
* another chip size that progresses iteratively is used to determine the chip displacement results that have not been estimated from the previous iterations
* a slight interpolation is done to fill the missing (unreliable) chip displacement results using bicubic mode (that can remove pixel discrepancy when using other modes) and an interpolation mask is returned
* the core image processing is coded by calling OpenCV's Python and/or C++ functions for efficiency 


## Install

* First install ISCE
* Put the folder "geoAutorift" and the file "Sconscript" under the ISCE's source folder (where you started installing ISCE; see the snapshot "install_snapshot.png")
* run "scons install" again from command line


## Instructions

### geogrid

* When using ISCE to creat the two images, it is recommended to run ISCE up to the step where coregistered SLC's are done

For quick use:
* Refer to the file "testGeogrid.py" for the usage of the sub-module
* Input files include the master image folder (required), slave image folder (required), a DEM (required), local surface slope maps, velocity maps

For modular use:
* In Python environment, after importing ISCE "import isce", type "from components.contrib.geoAutorift.geogrid.Geogrid import Geogrid" to import the "geogrid" sub-module, and then type "obj = Geogrid()" followed by "obj.configure()" to initialize the "geogrid" object
* The "geogrid" object has several parameters that has to be set up: 

       startingRange 

       rangePixelSize
       sensingStart
prf
lookSide
repeatTime
numberOfLines
numberOfSamples
orbit
demname
dhdxname
dhdyname
vxname
vyname
winlocname
winoffname
winro2vxname
winro2vyname

* After the above parameters are set, run the sub-module by typing "obj.geogrid()" to create the output files


