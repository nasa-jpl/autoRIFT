    
## 3. Features

* fast algorithm that finds displacement between the two images (where source and template image patches are extracted and correlated) using nested grid design with  iteratively progressive chip (template) sizes, sparse/dense combinative search strategy, and a novel disparity filter
* for intelligent use, user can specify unstructured spatially-varying search centers (center of the source image patch), search offsets (center displacement of the template image patch compared to the source image patch), chip sizes (size of template image patch), search ranges (size of the source image patch)
* faster than the conventional dense feature tracking "ampcor"/"denseampcor" algorithm in ISCE by more than 2 orders of magnitude with 20% increase of accuracy
* supports various preprocessing modes for specified image pair, e.g. either the raw image (both texture and topography) or the texture only (high-frequency components without the topography) can be used with various choices of the high-pass filter options 
* supports image data format conversion to unsigned integer 8 (uint8; faster) and single-precision float (float32)
* users can adjust all of the relevant parameters, as detailed in the instructions
* a Normalized Displacement Coherence (NDC) filter is developed (Gardner et al., 2018) to filter the chip (template) displacement results based on displacement difference thresholds that are adaptively scaled to the local search range
* a sparse integer displacement search is used to narrow the template search range and to eliminate areas of "low coherence" for fine decimal displacement search and following chip size iterations
* a novel nested grid design allows chip size to iteratively progress where a smaller chip (template) size failed
* a light interpolation is done to fill the missing (unreliable) chip displacement results using median filtering and bicubic-mode interpolation (that can remove pixel discrepancy when using other modes) and an interpolation mask is returned
* the core displacement estimator (Normalized Cross-Correlation or NCC) and image pre-processing (e.g. high-pass filtering) call OpenCV's Python and/or C++ functions for efficiency 
* sub-pixel displacement estimation using a fast Gaussian pyramid upsampling algorithm that is robust to sub-pixel locking
* the intelligent (chip size-dependent) selection of sub-pixel oversampling ratio is also supported for better tradeoff between efficiency and accuracy
* the current version can be installed with the ISCE software (that supports both Cartesian and radar coordinates) or as a standalone Python module (Cartesian coordinates only)
* when used in combination with the sister Python module Geogrid (https://github.com/leiyangleon/Geogrid), autoRIFT can be used for feature tracking between image pair over a grid defined in an arbitrary map-projected Cartesian (northing/easting) coordinate system
* when the grid is provided in map-projected Cartesian (northing/easting) coordinates, outputs are returned in geocoded GeoTIFF image file format with the same EPSG map projection code as the input search grid
* **[NEW]** for combinative use of autoRIFT/Geogrid in feature tracking with optical images, the program now supports fetching optical images (Landsat-8 GeoTIFF and Sentinel-2 COG formats are included) as well as other inputs (e.g. DEM, slope, etc; all in GeoTIFF format) from either local machine or remotely using [GDAL virtual file systems](https://gdal.org/user/virtual_file_systems.html) (e.g., `/vsicurl/https://...`). For the use on radar images, the program now supports fetching auxiliary inputs (e.g. DEM, slope, etc; all in GeoTIFF format) from either local machine or remotely. See the changes on the Geogrid [commands](https://github.com/leiyangleon/Geogrid).
* **[NEW]** parallel computing has been added for Normalized Cross-Correlation (NCC). When using the autoRIFT commands in the Demo section, users need to append a multiprocessing flag: "-mpflag $num" with "$num" being the number of threads used; if not specified, the single-core version is used. 
* **[NEW]** fine grid spacing that causes overlapping (dependent) search chips (templates) is now supported with a refined NDC filter, so that spatially independent chips are no longer requested
* **[NEW]** improved memory use (by 50%) for autoRIFT and runtime (60x) for GeogridOptical using combination of Python/C++ 
* **[NEW]** the entire radar/optical workflow of using autoRIFT/Geogrid has been refined by adding a number of condition checks and fixing bugs and thus is ready for scaling the production of both optical and radar data results for large-scale (e.g. polar or global) glacier ice velocity mapping
