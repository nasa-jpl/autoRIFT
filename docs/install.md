## 6. Install

**With ISCE:**

* First install ISCE (https://github.com/isce-framework/isce2)
* Put the "geo_autoRIFT" folder and the "Sconscript" file under the "contrib" folder that is one level down ISCE's source directory (denoted as "isce-version"; where you started installing ISCE), i.e. "isce-version/contrib/" (see the snapshot below)

<img src="../figures/install_ISCE.png" width="35%">

* Run "scons install" again from ISCE's source directory "isce-version" using command line
* This distribution automatically installs the "autoRIFT" module as well as the "Geogrid" module (https://github.com/leiyangleon/Geogrid).


**Standalone:**

***1) Manual Install:***

* Put the "geo_autoRIFT" folder and the "setup.py" file under some source directory (see the snapshot below)

<img src="../figures/install_standalone.png" width="35%">

* Run "python3 setup.py install" or "sudo python3 setup.py install" (if the previous failed due to permission restriction) using command line
* This distribution automatically installs the "autoRIFT" module as well as the "Geogrid" module (https://github.com/leiyangleon/Geogrid)
* The standalone version only supports Cartesian coordinate imagery. However, if the above "With ISCE" version is installed independently, the standalone version can also support radar coordinate imagery.
* If the modules cannot be imported in Python environment, please make sure the path where these modules are installed (see "setup.py") to be added to the environmental variable $PYTHONPATH.


***2) Conda Install:***

* The source code "geo_autoRIFT" folder can also be installed via conda-forge channels. Please follow the instructions on the following page (with no more than three command lines):
    https://github.com/conda-forge/autorift-feedstock
* This distribution automatically installs the "autoRIFT" module as well as the "Geogrid" module (https://github.com/leiyangleon/Geogrid)
* The standalone version only supports Cartesian coordinate imagery. However, if the above "With ISCE" version is installed independently, the standalone version can also support radar coordinate imagery.
* If the modules cannot be imported in Python environment, please make sure the path where these modules are installed (search for the modules in "anaconda3" directory) to be added to the environmental variable $PYTHONPATH.
