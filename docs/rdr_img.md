### 5.3 Radar image over regular grid in image coordinates

<img src="../figures/regular_grid_new.png" width="100%">

***Output of "autoRIFT" module for a pair of Sentinel-1A/B images (20170221-20170227; same as the Demo dataset at https://github.com/leiyangleon/Geogrid) at Jakobshavn Glacier of Greenland over a regular-spacing grid: (a) estimated range pixel displacement, (b) estimated azimuth pixel displacement, (c) chip size used, (d) light interpolation mask.***


This is obtained by implementing the following command line:

With ISCE:

       testautoRIFT_ISCE.py -m I1 -s I2

where "I1" and "I2" are the reference and test images as defined in the instructions below. 
