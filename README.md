# geoAutorift
ISCE module for finding displacement and motion velocity between two images over user-defined geographic-coordinate grid

## Features
This module comprises two parts or sub-modules: "geogrid" and "autorift".

### * geogrid: 
* user can define a grid in geographic coordinates provided in the form of a DEM with arbitrary EPSG code, 
* the program will extract the portion of the grid that overlaps with the given coregistered image pair (not necessarily radar; can be optical, etc), 
* return the two-dimensional (2D) pixel indices (e.g. range and azimuth for radar) in the image pair for each grid point
* return the coarse pixel displacement given the motion velocity maps and the local surface slope maps in the directions of both geographic x- and y-coordinates (that are provided at the same grid as the DEM)
* return the matrix of conversion coefficients that can convert the displacement between the two images (can be estimated precisely later with the next sub-module "autorift") in image coordinates (2D) to motion velocity in geographic coordinates (2D)
