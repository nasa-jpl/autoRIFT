## 7. Instructions

**Note:**

* When the grid is provided in map-projected Cartesian (northing/easting) coordinates (geographic coordinates lat/lon is not supported), it is required to run the "Geogrid" module (https://github.com/leiyangleon/Geogrid) first before running "autoRIFT". In other words, the outputs from "testGeogrid_ISCE.py" or "testGeogridOptical.py" (a.k.a "winlocname", "winoffname", "winsrname", "wincsminname", "wincsmaxname", "winro2vxname", "winro2vyname") will serve as optional inputs for running "autoRIFT" at a geographic grid, which is required to generate the final motion velocity maps.
* When the outputs from running the "Geogrid" module are not provided, a regular grid in the image coordinates will be automatically assigned

**For quick use:**

* Refer to the file "testautoRIFT.py" (standalone) and "testautoRIFT_ISCE.py" (with ISCE) for the usage of the module and modify it for your own purpose
* Refer to the Demo Section 5 for the single command line use with various options
* Input files include the reference image (required), secondary image (required), and the outputs from running "testGeogrid_ISCE.py" or "testGeogridOptical.py" (optional; a.k.a "winlocname", "winoffname", "winsrname", "wincsminname", "wincsmaxname", "winro2vxname", "winro2vyname", all of which are established on a user-defined map-projected Cartesian coordinate grid). 
* For the simplified use when the geocoded product is desired, "winlocname" (the x/y 2-band grid location represented using reference image pixel index) must be specified (otherwise it will become ungeocoded), and each of the rest optional input ("win * name") can be either used or omitted.
* For full/combinative use of these optional inputs, "winoffname" specifies the x/y (2-band) downstream search pixel displacement (in pixels), "winsrname" determines the x/y (2-band) search range in pixels, "wincsminname" and "wincsmaxname" are the x/y (2-band) minimum and maximum chip sizes in pixels, "winro2vxname" and "winro2vyname" are the x/y 2-band conversion matrix elements (transforming dx/dy to VX and dx/dy to VY respectively) as defined in https://doi.org/10.3390/rs13040749.
* Output files include 1) estimated horizontal displacement (equivalent to range for radar), 2) estimated vertical displacement (equivalent to minus azimuth for radar), 3) light interpolation mask, 4) iteratively progressive chip size used. 

_Note: These four output files will be stored in a file named "offset.mat" that can be viewed in Python and MATLAB. When the grid is provided in map-projected Cartesian (northing/easting) coordinates, a 4-band GeoTIFF with the same EPSG code as input grid will be created as well and named "offset.tif"; a 2-band GeoTIFF of the final converted motion velocity in map-projected x- (easting) and y- (northing) coordinates will be created and named "velocity.tif". Also, it is possible to save the outputs in netCDF standard format by adding the "-nc" option to the "testautoRIFT.py" (standalone) and "testautoRIFT_ISCE.py" (with ISCE) command by using a user-defined netCDF packaging routine (not included in the source code here)._

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
       I2                                                 secondary image (extracted image patches defined as "template"; displacement = motion vector of I2 relative to I1 which should be acquired earlier in our convention)
       xGrid [units = integer image pixels]               horizontal reference image pixel index at each grid point
       yGrid [units = integer image pixels]               vertical reference image pixel index at each grid point
       (if xGrid and yGrid not provided, a regular grid spanning the entire image will be automatically set up, which is similar to the conventional ISCE module, "ampcor" or "denseampcor")
       Dx0 [units = integer image pixels]                 horizontal "downstream" search location (that specifies the horizontal pixel displacement of the template's search center relative to the source's) at each grid point
       Dy0 [units = integer image pixels]                 vertical "downstream" reach location (that specifies the vertical pixel displacement of the template's search center relative to the source's) at each grid point
       (if Dx0 and Dy0 not provided, an array with zero values will be automatically assigned and there will be no offsets of the search centers)

* After the inputs are specified, run the module as below
       
       obj.preprocess_filt_XXX() or obj.preprocess_db()
       obj.uniform_data_type()
       obj.runAutorift()

where "XXX" can be "wal" for the Wallis filter, "hps" for the trivial high-pass filter, "sob" for the Sobel filter, "lap" for the Laplacian filter, and also a logarithmic operator without filtering is adopted for occasions where low-frequency components (i.e. topography) are desired, i.e. "obj.preprocess_db()".

* The "autoRIFT" object has the following four primary outputs: 
       
       ------------------output------------------
       Dx [units = decimal image pixels]                  estimated horizontal pixel displacement
       Dy [units = decimal image pixels]                  estimated vertical pixel displacement
       InterpMask [unitless; boolean data type]           light interpolation mask
       ChipSizeX [units = integer image pixels]           iteratively progressive chip size in horizontal direction (different chip size allowed for vertical, i.e. ChipSizeY, which is also an output)

* The "autoRIFT" object has many parameters that can be flexibly tweaked by the users for their own purpose (listed below; can also be obtained by referring to "geo_autoRIFT/autoRIFT/autoRIFT_ISCE.py"):

       ------------------parameter list: general function------------------
       ChipSizeMinX [units = integer image pixels]                Minimum size (in horizontal direction) of the template (chip) to correlate (default = 32; could be scalar or array with same dimension as xGrid)
       ChipSizeMaxX [units = integer image pixels]                Maximum size (in horizontal direction) of the template (chip) to correlate (default = 64; could be scalar or array with same dimension as xGrid)
       ChipSize0X [units = integer image pixels]                  Minimum acceptable size (in horizontal direction) of the template (chip) to correlate (default = 32)
       GridSpacingX [units = integer image pixels]                Grid Spacing (in horizontal direction) (default = 32; note GridSpacingX can be smaller than ChipSize0X leading to dependent chips)
       ScaleChipSizeY [unitless; integer data type]               Scaling factor to get the vertical chip size in reference to the horizontal size (default = 1)
       SearchLimitX [units = integer image pixels]                Range or limit (in horizontal direction) to search for displacement in the source (default = 25; could be scalar or array with same dimension as xGrid; when provided in array, set its elements to 0 if no search is desired in certain areas)
       SearchLimitY [units = integer image pixels]                Range or limit (in vertical direction) to search for displacement in the source (default = 25; could be scalar or array with same dimension as xGrid; when provided in array, set its elements to 0 if no search is desired in certain areas)
       SkipSampleX [units = integer image pixels]                 Number of samples to skip between search windows in horizontal direction if no grid specified by the user (default = 32)
       SkipSampleY [units = integer image pixels]                 Number of lines to skip between search windows in vertical direction if no grid specified by the user (default = 32)
       minSearch [units = integer image pixels]                   Minimum search range/limit (default = 6)
       
       ------------------parameter list: about Normalized Displacement Coherence (NDC) filter ------------------
       FracValid                   Fraction of valid displacements (default = 8/25) to be multiplied by filter window size "FiltWidth^2" and then used for thresholding the number of chip displacements that have passed the "FracSearch" disparity filtering
       FracSearch                  Fraction of search limit used as threshold for disparity filtering of the chip displacement difference that is normalized by the search limit (default = 0.20)
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
       DataTypeInput               Image data type: 0 -> uint8, 1 -> float32 (default = 0 which is much faster and does not degrade accuracy when combined with preprocessing filters such as Wallis and other high-pass ones)
       zeroMask                    Force the margin (no data) to zeros which is useful for Wallis-filter-preprocessed images (default = None; 1 for Wallis filter)
       MultiThread                 Number of threads for multi-threading (default is specified by 0, which uses the original single-core version and surpasses the multithreading routine for normalized cross-correlation)
