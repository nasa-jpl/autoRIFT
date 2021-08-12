    
## 3. Features

* fast algorithm that finds displacement between the two images (where source and template image patches are extracted and correlated) using sparse search and iteratively progressive chip (template) sizes
* for intelligent use, user can specify unstructured spatially-varying search centers (center of the source image patch), search offsets (center displacement of the template image patch compared to the source image patch), chip sizes (size of template image patch), search ranges (size of the source image patch)
* faster than the conventional dense feature tracking "ampcor"/"denseampcor" algorithm included in ISCE by nearly 2 orders of magnitude
* supports various preprocessing modes for specified image pair, e.g. either the raw image (both texture and topography) or the texture only (high-frequency components without the topography) can be used with various choices of the high-pass filter options 
* supports image data format of unsigned integer 8 (uint8; faster) and single-precision float (float32)
* user can adjust all of the relevant parameters, as detailed below in the instructions
* a Normalized Displacement Coherence (NDC) filter is developed (Gardner et al., 2018) to filter the chip displacemnt results based on displacement difference thresholds that are scaled to the local search range
* a sparse integer displacement search is used to narrow the template search range and to eliminate areas of "low coherence" for fine decimal displacement search and following chip size iterations
* novel nested grid design allows chip size progressively increase iteratively where a smaller chip size failed
* a light interpolation is done to fill the missing (unreliable) chip displacement results using median filtering and bicubic-mode interpolation (that can remove pixel discrepancy when using other modes) and an interpolation mask is returned
* the core displacement estimator (normalized cross correlation) and image pre-processing call OpenCV's Python and/or C++ functions for efficiency 
* sub-pixel displacement estimation using a fast Gaussian pyramid upsampling algorithm that is robust to sub-pixel locking; the intelligent (chip size-dependent) selection of oversampling ratio is also supported for a tradeoff between efficiency and accuracy
* the current version can be installed with the ISCE software (that supports both Cartesian and radar coordinates) or as a standalone Python module (Cartesian coordinates only)
* when used in combination with the Geogrid Python module (https://github.com/leiyangleon/Geogrid), autoRIFT can be used for feature tracking between image pair over a grid defined in an arbitrary geographic Cartesian (northing/easting) coordinate projection
* when the grid is provided in geographic Cartesian (northing/easting) coordinates, outputs are returned in geocoded GeoTIFF image file format with the same EPSG projection code as input search grid
* **[NEW]** For feature tracking of optical images, the program now supports fetching optical images (Landsat-8 GeoTIFF and Sentinel-2 COG formats are included) as well as other inputs (e.g. DEM, slope, etc; all in GeoTIFF format) from either local machine or remotely using [GDAL virtual file systems](https://gdal.org/user/virtual_file_systems.html) (e.g., `/vsicurl/https://...`). For feature tracking of radar images, the program now supports fetching auxiliary inputs (e.g. DEM, slope, etc; all in GeoTIFF format) from either local machine or remotely. See the changes on the Geogrid [commands](https://github.com/leiyangleon/Geogrid).
* **[NEW]** parallel computing has been added for Normalized Cross-Correlation (NCC). When using the autoRIFT commands below, users need to append a multiprocessing flag: "-mpflag $num" with "$num" being the number of threads used; if not specified, the single-core version is used. 
