# autoRIFT (autonomous Repeat Image Feature Tracking)



[![Language](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://www.python.org/)
[![Latest version](https://img.shields.io/badge/latest%20version-v1.4.0-yellowgreen.svg)](https://github.com/leiyangleon/autoRIFT/releases)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/leiyangleon/autoRIFT/blob/master/LICENSE)
[![Citation](https://img.shields.io/badge/DOI-10.3390/rs13040749-blue)](https://doi.org/10.3390/rs13040749)

### Update Notes:

```diff
+ refined the workflow and ready for scaling the production of both optical and radar data results
+ improved memory use (by 50%) for autoRIFT and runtime (60x) for GeogridOptical
+ parallel computing for NCC
+ support for remote input files using GDAL virtual file systems (e.g., `/vsicurl/https://...`)
+   see: https://gdal.org/user/virtual_file_systems.html
```


**A Python module of a fast and intelligent algorithm for finding the pixel displacement between two images**

**autoRIFT can be installed as a standalone Python module (only supports Cartesian coordinates) either manually or as a conda install (https://github.com/conda-forge/autorift-feedstock). To allow support for both Cartesian and radar coordinates, autoRIFT must be installed with the InSAR Scientific Computing Environment (ISCE: https://github.com/isce-framework/isce2)**

**Use cases include all dense feature tracking applications, including the measurement of surface displacements occurring between two repeat satellite images as a result of glacier flow, large earthquake displacements, and land slides**  

**autoRIFT can be used for dense feature tracking between two images over a grid defined in an arbitrary geographic Cartesian (northing/easting) coordinate projection when used in combination with the sister Geogrid Python module (https://github.com/leiyangleon/Geogrid). Example applications include searching radar-coordinate imagery on a polar stereographic grid and searching Universal Transverse Mercator (UTM) imagery at a specified geographic Cartesian (northing/easting) coordinate grid**


Copyright (C) 2019 California Institute of Technology.  Government Sponsorship Acknowledged.

Link: https://github.com/leiyangleon/autoRIFT

Citation: https://doi.org/10.3390/rs13040749


## 1. Authors

Alex Gardner (JPL/Caltech; alex.s.gardner@jpl.nasa.gov) first described the algorithm "auto-RIFT" in (Gardner et al., 2018), developed the first version in MATLAB and continued to refine the algorithm;

Yang Lei (GPS/Caltech; ylei@caltech.edu; leiyangfrancis@gmail.com) translated it to Python, further optimized and incoporated to the ISCE software;

Piyush Agram (JPL/Caltech; piyush@gps.caltech.edu) set up the installation as a standalone module and further cleaned the code.

**Reference:** 

Gardner, A.S., Moholdt, G., Scambos, T., Fahnstock, M., Ligtenberg, S., Broeke, M.V.D. and Nilsson, J., 2018. [**Increased West Antarctic and unchanged East Antarctic ice discharge over the last 7 years.**](https://doi.org/10.5194/tc-12-521-2018) *The Cryosphere*, 12(2), pp.521-547. 

**[NEW]** Lei, Y., Gardner, A. and Agram, P., 2021. [**Autonomous Repeat Image Feature Tracking (autoRIFT) and Its Application for Tracking Ice Displacement.**](https://doi.org/10.3390/rs13040749) *Remote Sensing*, 13(4), p.749. 


## 2. Acknowledgement

This effort was funded by the NASA MEaSUREs program in contribution to the Inter-mission Time Series of Land Ice Velocity and Elevation (ITS_LIVE) project (https://its-live.jpl.nasa.gov/) and through Alex Gardner’s participation in the NASA NISAR Science Team
    
       
## 3. [Features](/docs/features.md)



## 4. [Possible Future Development](/docs/future.md)




## 5. [Demo](/docs/demo.md)





## 6. [Install](/docs/install.md)





## 7. [Instructions](/docs/instruction.md)

