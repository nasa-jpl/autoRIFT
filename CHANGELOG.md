# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added docker container for development. Contributed by @AgentOxygen

### Fixed
- Fixed numpy build dependency. Contributed by @AgentOxygen

## [2.1.1]

### Fixed
- Updated numpy pad calls to work with newer numpy versions. Contributed by @AgentOxygen

## [2.1.0]

### Added
- Support for the use of pre-generated radar-geometry topographic corrections for ISCE3 when processing Sentinel-1 scenes.
- A `citation.cff` file to provide citation metadata for both humans and machines (e.g., zenodo integration).

## [2.0.0]

> [!IMPORTANT]
> This release includes major changes to Geogrid, autoRIFT, and the expected usage; please read the release notes carefully. Notably, the radar workflow has been migrated from ISCE2 to ISCE3 and autoRIFT now depends on ISCE3 instead of being built inside it.

### Added
* An `environment.yml` file for creating a developmental conda environment
* `ruff.toml` configuration files for formatting python code with [ruff](https://docs.astral.sh/ruff/) 
* `.clang-format` configuration files for formatting C/C++ code with [clang-format](https://clang.llvm.org/docs/ClangFormat.html)

* Updating the changelog will now be required for all PRs into autoRIFT's `main` branch.
* geogrid C++ extension for the radar workflow which calls core ISCE3 C++ code directly.

### Changed
* geogrid and autoRIFT now require Python 3.10 or greater
* All project documentation has been consolidated into the `README.md` and `docs/demo.md`, which have been updated to reflect new usage. 
* Python's `sysconfig` is now used to get the Python include directory and `purelib` library paths instead of predicting it from the OpenCV build info. 
* Geogrid and autoRIFT now report their version numbers from the `geo_autoRIFT` namespace package metadata
* The radar module in the Geogrid package has been changed from `Geogrid.py` to `GeogridRadar.py`.
* The radar version of Geogrid now uses [ISCE3](https://github.com/isce-framework/isce3) rather than [ISCE2](https://github.com/isce-framework/isce2).
* The `test*.py` workflows are no longer specific to optical or radar processing:
  * `testGeogridOptical.py` and `testGeogrid_ISCE.py` have been combined into a new `testGeogrid.py` script
  * `testautoRIFT.py` and `testautoRIFT_ISCE.py` workflow scripts have been merged into one `testautoRIFT.py` script
* Significant performance improvements to functions in `autoRIFT.py`:
  * `arPixDisp_s` and `arPixDisp_u` (`float32` and `uint8` pixel-wise displacement functions):
    * Moved the looping code into `autoriftmodule.cpp` 
    * Multi-threading is now done via [OpenMP](https://www.openmp.org/) in C++, rather than by using Python's `multiprocessing` library
    * Removed the `loop_unpacking_` functions that were used for multiprocessing
  * `colfilt` (column-wise filter function):
    * All the filter methods have been manually implemented with [Numba's @jit](https://numba.pydata.org/numba-doc/dev/user/jit.html)
    * The filters are now applied using SciPy's [LowLevelCallable](https://docs.scipy.org/doc/scipy/reference/generated/scipy.LowLevelCallable.html) and [generic_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generic_filter.html#scipy.ndimage.generic_filter)

### Fixed
* The `arPixDisp_s` and `arSubPixDisp_s` C++ functions now use normalized cross-covariance (`CV_TM_CCOEFF_NORMED`) in the `cv::matchTemplate` calls, rather than normalized cross-correlation (`CV_TM_CCORR_NORMED`). This matches what the `arPixDisp_u` and `arSubPixDisp_u` workflows have been using. See [here](https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html#gga3a7850640f1fe1f58fe91a2d7583695da5382c8f9df87e87cf1e9f9927dc3bc31) for the differences between the methods.

### Removed
* The `geo_autoRIFT/__init__.py` file and all the `sconscript` files have been removed with the migration to ISCE3 -- these only served to facilitate building autoRIFT as a contributed package inside ISCE2.
* `topsinsar_filename.py` has been removed with the migration to ISCE3. This functionality has been moved to the `get_topsinsar_config` function in `testautoRIFT.py`.
* `testautoRIFT.py` no longer outputs a Matlab-style `offest.mat` file
* The geocode-only workflow added to `testGeoGrid_ISCE.py` in v1.6.0 with the switch to ISCE3. A similar workflow is currently being developed for a forthcoming release.

## [1.6.0]

### Added
* autoRIFT can now successfully process Sentinel-1 burst mosaics generated with [burst2safe](https://github.com/ASFHyP3/burst2safe). We recommend processing at least 2 bursts along track, with 3-5 bursts seeing improvements in data quality.
* A geocode-only workflow was added to `testGeogrid_ISCE.py` that uses only the Sentinel-1 SAFE and orbit metadata without needing to run ISCE2 through the topo step. Thanks @leiyangleon!

### Changed
* variety of updates in `netcdf_output.py`:
  * Now following CF Conventions 1.8 and (most of) NSIDC's recommendations;  variable metadata may have changed accordingly. See the `netcdf_output.py` diff for the full list of changes: https://github.com/nasa-jpl/autoRIFT/compare/v1.5.0..v1.6.0
  * The value of the `stable_shift` netCDF attribute is now `0` instead `np.nan` if the stable shift has not been applied
  * 'satellite' global netCDF attribute will be spelled out and follow mission styling (e.g., `Landsat 9`)
  * M11/M12 now have the appropriate scale factor applied to ease use (e.g., when correcting ionosphere)
  * M11/M12 will now be stored as `float32` instead `int16` in the output netCDF file
  * Provider's image IDs will be saved as attributes in the `img_pair_info` variable
  * Ensure Landsat `satellite_img1` and `satellite_img2` attributes in the `img_pair_info` variable are strings to match the convention of other missions

### Deprecated

In the upcoming v2.0.0 autoRIFT release:
* Support for using ISCE2 for the radar workflows will be removed in favor of an ISCE3-based workflow
* The `test*.py` workflows will no longer be specific to optical or radar processing:
  * `testGeogridOptical.py` and `testGeogrid_ISCE.py` will be combined into a new `testGeogrid.py` script
  * `testautoRIFT_ISCE.py` workflow scripts will be merged into the `testautoRIFT.py script

### Fixed
* autoRIFT/Geogrid now support NumPy v2.0+, see the [migration guide](https://numpy.org/devdocs/numpy_2_0_migration_guide.html#numpy-2-migration-guide). Thanks @benhills, @gbrencher, and @sssangha!
* Fixed autoRIFT's Wallis filter to ensure non-negative variance and image masks are correctly combined

## [1.5.0]

### Added
* autoRIFT/Geogrid now support processing Landsat 4, 5, 7, and 9 scene

### Changed
* autoRIFT/Geogrid now explicitly requires scenes to be in the same projection
* autoRIFT will now use a default filter width of 5 pixels, except for Sentinel-1 scenes where it'll use the previous default of 21

## [1.4.0]

### Added
* refined the workflow and ready for scaling the production of both optical and radar data results
* netCDF packaging for production, especially improved product quality for radar images

## [1.3.1]

### Added
* add condition check with error message in Geogrid when the two images have different projections

### Fixed
* fix bug in calculating Geogrid pixel indices
* fix bug in autoRIFT column filter indices

## [1.3.0]

### Changed
* Merged PR [#29](https://github.com/nasa-jpl/autoRIFT/pull/29) for improving the memory use and runtime for both autoRIFT and Geogrid

## [1.2.0]

### Changed
* Merged the PRs ([#18](https://github.com/nasa-jpl/autoRIFT/pull/18) and [#19](https://github.com/nasa-jpl/autoRIFT/pull/19)) for using urls instead of local files & returning function outputs instead of saving as txt file
* Merged the PR ([#26](https://github.com/nasa-jpl/autoRIFT/pull/26) and [#27](https://github.com/nasa-jpl/autoRIFT/pull/25) ) for adding dt-varying search range routine
 

### Fixed
* Merged the PR ([#20](https://github.com/nasa-jpl/autoRIFT/pull/20)) for fixing the indexing problem in optical coregistration,
* Merged the PR ([#23](https://github.com/nasa-jpl/autoRIFT/pull/23)) for dealing with dependent/overlapping chips (grid spacing < chip size)
* Merged the PR ([#24](https://github.com/nasa-jpl/autoRIFT/pull/24)) for MISC formatting/checking issues
* Merged the PR ([#25](https://github.com/nasa-jpl/autoRIFT/pull/25)) for fixing bugs related to nodata values

## [1.0.0]

### Added
* parallel computing for NCC
* support of urls for input files (both images and auxiliary input files can be urls for optical data processing; only auxiliary input files can be urls for radar data processing)

### Changed
* updated netcdf packaging for production

## [1.0.8]

### Changed
* Vsicurl has been integrated to the optical data access without the need of downloading optical images (Landsat-8 and/or Sentinel-2) as well as initialization files (DEM, slope, reference velocity, etc).

## [1.0.7]

### Added
* autoRIFT was successfully compiled with OpenCV v3 and OpenCV v4

### Changed
* eased Geogrid output handling (the program will automatically check the optional inputs provided by the user and only generate the meaningful output based on the input combinations)
* netcdf output packaging (not publicly available at this moment) for autoRIFT final products is significantly improved, where error metrics are calculated at the stable surfaces wherever available, and radar range-projected velocities onto a priori flow (e.g. surface slope parallel) are also provided when ionosphere-induced azimuth error are much larger.

### Fixed
* bug of grid offset in Geogrid outputs has been fixed
* bug of erroneous large values (due to radar geometry's loss of sensitivity to geocode pixel displacement to velocity at large surface slopes) in Geogrid outputs has been fixed

## [1.0.6]

## [1.0.5]

## [1.0.4]

## [1.0.3]

### Added
* compatible to both GDAL 2 and GDAL 3

## [1.0.0]

Initial release of autoRIFT (autonomous Repeat Image Feature Tracking)
