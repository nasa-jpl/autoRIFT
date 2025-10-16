# autoRIFT (autonomous Repeat Image Feature Tracking)

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/autorift.svg)](https://anaconda.org/conda-forge/autorift)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/autorift.svg)](https://anaconda.org/conda-forge/autorift)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/nasa-jpl/autoRIFT/blob/main/LICENSE)
[![Citation](https://zenodo.org/badge/200903796.svg)](https://doi.org/10.5281/zenodo.16989574)


A Python module of a fast and intelligent algorithm for finding the pixel displacement between two images that can be used for all dense feature tracking applications, including the measurement of surface displacements occurring between two repeat satellite images as a result of glacier flow, large earthquake displacements, and land slides.

Dense feature tracking in an arbitrary map-projected Cartesian (northing/easting) coordinate systems can be done when used in combination with the sister Geogrid Python package distributed with autoRIFT. Example applications include searching radar-coordinate imagery on a polar stereographic grid and searching Universal Transverse Mercator (UTM) imagery at a user-specified map-projected Cartesian (northing/easting) coordinate grid.

> [!IMPORTANT]
> autoRIFT only returns displacement values for locations where significant feature matches are found, otherwise autoRIFT returns no data values.


Copyright (C) 2019 California Institute of Technology.  Government Sponsorship Acknowledged.

Link: https://github.com/nasa-jpl/autoRIFT

Citation: https://doi.org/10.3390/rs13040749

## Installation

Released version of autoRIFT are available from conda-forge and can be installed using [`conda`/`mamba`](https://conda-forge.org/download/):
```shell
mamba install -c conda-forge autorift
```

For a development version, we recommend using [`conda`/`mamba`](https://conda-forge.org/download/) as some dependencies (e.g., ISCE3) are only distributed on Conda-Forge and would need to be built manually otherwise. To setup a development environment, clone this repository, create a conda environment, and install an editable version of this repository:
```shell
git clone https://github.com/nasa-jpl/autoRIFT.git
cd autoRIFT
mamba env create -f environment.yml
mamba activate autoRIFT
python -m pip install --use-pep517 --no-build-isolation -e .
```

When developing files, please use `ruff` and `clang-format` to ensure files are formatted to match our style:
```shell
ruff format .
find geo_autoRIFT -iname "*.h" -exec clang-format -i {} \;
find geo_autoRIFT -iname "*.cpp" -exec clang-format -i {} \;
```

### (Optional) Development using Docker

A containerized development environment with the necessary dependencies can be created using Docker. The following commands can be used to build the container and enter it:

```
docker built --rm -t autorift .
docker run -v .:/project -it autorift /bin/bash
```

Note that `/project` is the working directory inside the container. The argument `-v .:/project` binds the local source code directory to the live installation inside the container.

## Usage

> [!WARNING]
> autoRIFT is undergoing a significant refactor during the v2.0+ series so usage is subject to change. We expect to release a stable, user-friendly version of autoRIFT with v3.0.

Currently, the expected workflow for autoRIFT is to:
1. (Optionally) Run `testGeogrid.py` for working with map-projected Cartesian (northing/easting) coordinates (lat/lon is not supported)
2. Run `testautoRIFT.py` to generate offsets and velocity maps
3. (Optionally) Run `netcdf_output.py` to package output products into an ITS_LIVE netCDF file. 

However, these files are not distributed with the package and so will need to be cloned/copied from the GitHub repository and likely modified for your use case. 

**For an example of using autoRIFT to determine the velocity of glaciers in Greenland, please see [`docs/demo.md`](docs/demo.md).**

For a full description of how to call these scripts and the options available, please use the `--help` command:
```shell
testGeogrid.py --help
testautoRIFT.py --help
netcdf_output.py --help
```

## Future Development

Notable changes to this project will be recorded in our [CHANGELOG](CHANGELOG.md) and provided as release notes.

Here are some known development opportunities:
* for radar (SAR) images, it is yet to include the complex correlation of the two images, i.e. the current version only uses the amplitude correlation
* combinative use of current SAR amplitude-based offset tracking results with InSAR phase-derived velocity maps needs to be investigated with a better data fusion
* the GPU implementation would be useful to extend 

## Authors

AutoRIFT 1.0 was initially developed by:
1. Alex Gardner (JPL/Caltech; alex.s.gardner@jpl.nasa.gov) who first described the algorithm "auto-RIFT" in (Gardner et al., 2018) and developed the first version in MATLAB and continued to refine the algorithm;
2. Yang Lei (GPS/Caltech; leiyangfrancis@gmail.com) who translated it to Python, further optimized and incorporated it into the ISCE2 software while also developed its sister module Geogrid;
3. and Piyush Agram (GPS/Caltech; piyush@gps.caltech.edu) who set up the installation as a standalone module and further cleaned the code.

Since then, autoRIFT has been further developed by a number of ITS_LIVE project members, collaborators, and contributors. Please see: 
https://github.com/nasa-jpl/autoRIFT/graphs/contributors

**Reference:** 

***1.*** Gardner, A.S., Moholdt, G., Scambos, T., Fahnstock, M., Ligtenberg, S., Broeke, M.V.D. and Nilsson, J., 2018. [**Increased West Antarctic and unchanged East Antarctic ice discharge over the last 7 years.**](https://doi.org/10.5194/tc-12-521-2018) *The Cryosphere*, 12(2), pp.521-547. 

***2.*** Lei, Y., Gardner, A. and Agram, P., 2021. [**Autonomous Repeat Image Feature Tracking (autoRIFT) and Its Application for Tracking Ice Displacement.**](https://doi.org/10.3390/rs13040749) *Remote Sensing*, 13(4), p.749. 


## Acknowledgement

This effort was funded by the NASA MEaSUREs program in contribution to the Inter-mission Time Series of Land Ice Velocity and Elevation (ITS_LIVE) project (https://its-live.jpl.nasa.gov/) and through Alex Gardner’s participation in the NASA NISAR Science Team.
