### 5.4 Radar image over user-defined geographic Cartesian coordinate grid

<img src="../figures/autorift1.png" width="100%">

***Output of "autoRIFT" module for a pair of Sentinel-1A/B images (20170221-20170227; same as the Demo dataset at https://github.com/leiyangleon/Geogrid) at Jakobshavn Glacier of Greenland over user-defined geographic Cartesian (northing/easting) coordinate grid (same grid used in the Demo at https://github.com/leiyangleon/Geogrid): (a) estimated range pixel displacement, (b) estimated azimuth pixel displacement, (c) light interpolation mask, (d) chip size used. Notes: all maps are established exactly over the same geographic Cartesian coordinate grid from input.***

This is done by implementing the following command line:

With ISCE:

       testautoRIFT_ISCE.py -m I1 -s I2 -g winlocname -o winoffname -sr winsrname -csmin wincsminname -csmax wincsmaxname -vx winro2vxname -vy winro2vyname

where "I1" and "I2" are the reference and test images as defined in the instructions below, and the optional inputs "winlocname", "winoffname", "winsrname", "wincsminname", "wincsmaxname", "winro2vxname", "winro2vyname" are outputs from running "testGeogrid.py" as defined at https://github.com/leiyangleon/Geogrid. When "winlocname" (grid location) is specified, each of the rest optional input ("win * name") can be either used or omitted.

**Runtime comparison for this test (on an OS X system with 2.9GHz Intel Core i7 processor and 16GB RAM):**
* __autoRIFT (single core): 10 mins__
* __Dense ampcor from ISCE (8 cores): 90 mins__


<img src="../figures/autorift2.png" width="100%">

***Final motion velocity results by combining outputs from "Geogrid" (i.e. matrix of conversion coefficients from the Demo at https://github.com/leiyangleon/Geogrid) and "autoRIFT" modules: (a) estimated motion velocity from Sentinel-1 data (x-direction; in m/yr), (b) motion velocity from input data (x-direction; in m/yr), (c) estimated motion velocity from Sentinel-1 data (y-direction; in m/yr), (d) motion velocity from input data (y-direction; in m/yr). Notes: all maps are established exactly over the same geographic Cartesian (northing/easting) coordinate grid from input.***
