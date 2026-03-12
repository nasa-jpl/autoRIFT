#!/usr/bin/env python3

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2019 California Institute of Technology. ALL RIGHTS RESERVED.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# United States Government Sponsorship acknowledged. This software is subject to
# U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
# (No [Export] License Required except when exporting to an embargoed country,
# end user, or in support of a prohibited end use). By downloading this software,
# the user agrees to comply with all applicable U.S. export laws and regulations.
# The user has the responsibility to obtain export licenses, or other export
# authority as may be required before exporting this software to any 'EAR99'
# embargoed foreign country or citizen of those countries.
#
# Author: Yang Lei
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import argparse
import glob
import os
import re
import subprocess
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from autoRIFT import __version__ as version
from autoRIFT import autoRIFT
from geogrid import GeogridOptical
from osgeo import gdal
from s1reader import load_bursts

import netcdf_output as no
from testGeogrid import getPol


def get_topsinsar_config():
    """
    Input file.
    """
    orbits = glob.glob('*.EOF')
    fechas_orbits = [datetime.strptime(os.path.basename(file).split('_')[6], 'V%Y%m%dT%H%M%S') for file in orbits]
    safes = glob.glob('*.SAFE')
    if not len(safes) == 0:
        fechas_safes = [datetime.strptime(os.path.basename(file).split('_')[5], '%Y%m%dT%H%M%S') for file in safes]
    else:
        safes = glob.glob('*.zip')
        fechas_safes = [datetime.strptime(os.path.basename(file).split('_')[5], '%Y%m%dT%H%M%S') for file in safes]

    safe_ref = safes[np.argmin(fechas_safes)]  # type: ignore[arg-type]
    orbit_path_ref = orbits[np.argmin(fechas_orbits)]  # type: ignore[arg-type]

    safe_sec = safes[np.argmax(fechas_safes)]  # type: ignore[arg-type]
    orbit_path_sec = orbits[np.argmax(fechas_orbits)]  # type: ignore[arg-type]

    if len(glob.glob('*_ref*.tif')) > 0:
        swath = int(os.path.basename(glob.glob('*_ref*.tif')[0]).split('_')[2][2])
    else:
        swath = None

    safe: str
    pol = getPol(safe_ref, orbit_path_ref)
    burst = None

    config_data = {}
    for name in ['reference', 'secondary']:
        # Find the first swath with data in it
        swath_range = [swath] if swath else [1, 2, 3]
        for swath in swath_range:
            try:
                if name == 'reference':
                    burst = load_bursts(safe_ref, orbit_path_ref, swath, pol)[0]
                    safe = safe_ref
                else:
                    burst = load_bursts(safe_sec, orbit_path_sec, swath, pol)[0]
                    safe = safe_sec
            except IndexError:
                continue
            break

        assert burst
        sensing_start = burst.sensing_start
        length, width = burst.shape
        prf = 1 / burst.azimuth_time_interval

        sensing_stop = sensing_start + timedelta(seconds=(length - 1.0) / prf)

        sensing_dt = (sensing_stop - sensing_start) / 2 + sensing_start

        config_data[f'{name}_filename'] = Path(safe).name
        config_data[f'{name}_dt'] = sensing_dt.strftime('%Y%m%dT%H:%M:%S.%f').rstrip('0')

    return config_data


def runCmd(cmd):
    out = subprocess.getoutput(cmd)
    return out


def cmdLineParse():
    """
    Command line parser.
    """
    SUPPORTED_MISSIONS = ['S1', 'S2', 'L4', 'L5', 'L7', 'L8', 'L9']

    parser = argparse.ArgumentParser(description='Output geo grid')
    parser.add_argument(
        '-r',
        '--input_r',
        dest='indir_r',
        type=str,
        required=True,
        help='Input reference image file name (in ISCE format and radar coordinates) or Input reference image file name (in GeoTIFF format and Cartesian coordinates)',
    )
    parser.add_argument(
        '-t',
        '--input_t',
        dest='indir_t',
        type=str,
        required=True,
        help='Input test image file name (in ISCE format and radar coordinates) or Input test image file name (in GeoTIFF format and Cartesian coordinates)',
    )
    parser.add_argument(
        '-g', '--input_g', dest='grid_location', type=str, required=False, help='Input pixel indices file name'
    )
    parser.add_argument(
        '-o',
        '--input_o',
        dest='init_offset',
        type=str,
        required=False,
        help='Input search center offsets ("downstream" reach location) file name',
    )
    parser.add_argument(
        '-sr', '--input_sr', dest='search_range', type=str, required=False, help='Input search range file name'
    )
    parser.add_argument(
        '-csmin', '--input_csmin', dest='chip_size_min', type=str, required=False, help='Input chip size min file name'
    )
    parser.add_argument(
        '-csmax', '--input_csmax', dest='chip_size_max', type=str, required=False, help='Input chip size max file name'
    )
    parser.add_argument(
        '-vx',
        '--input_vx',
        dest='offset2vx',
        type=str,
        required=False,
        help='Input pixel offsets to vx conversion coefficients file name',
    )
    parser.add_argument(
        '-vy',
        '--input_vy',
        dest='offset2vy',
        type=str,
        required=False,
        help='Input pixel offsets to vy conversion coefficients file name',
    )
    parser.add_argument(
        '-sf',
        '--input_scale_factor',
        dest='scale_factor',
        type=str,
        required=False,
        help='Input map projection scale factor file name',
    )
    parser.add_argument(
        '-ssm',
        '--input_ssm',
        dest='stable_surface_mask',
        type=str,
        required=False,
        help='Input stable surface mask file name',
    )
    parser.add_argument(
        '-fo',
        '--flag_optical',
        dest='optical_flag',
        type=bool,
        required=False,
        default=0,
        help='flag for reading optical data (e.g. Landsat): use 1 for on and 0 (default) for off',
    )
    parser.add_argument(
        '-nc',
        '--sensor_flag_netCDF',
        dest='nc_sensor',
        type=str,
        required=False,
        default=None,
        choices=SUPPORTED_MISSIONS,
        help=f'flag for packaging output formatted for Satellite missions. Default is None; supported missions: {SUPPORTED_MISSIONS}',
    )
    parser.add_argument(
        '-mpflag',
        '--mpflag',
        dest='mpflag',
        type=int,
        required=False,
        default=0,
        help='number of threads for multiple threading (default is specified by 0, which uses the original single-core version and surpasses the multithreading routine)',
    )
    parser.add_argument(
        '-ncname',
        '--ncname',
        dest='ncname',
        type=str,
        required=False,
        default=None,
        help='User-defined filename for the NetCDF output to which the ROI percentage and the production version will be appended',
    )

    return parser.parse_args()


class Dummy(object):
    pass


def loadProduct(filename):
    ds = gdal.Open(filename, gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    img = band.ReadAsArray().astype(np.float32)
    del band
    del ds

    return img


def loadProductOptical(file_m, file_s):
    """
    Load the product using Product Manager.
    """
    obj = GeogridOptical()

    x1a, y1a, xsize1, ysize1, x2a, y2a, xsize2, ysize2, trans = obj.coregister(file_m, file_s)

    DS1 = gdal.Open(file_m)
    DS2 = gdal.Open(file_s)

    I1 = DS1.ReadAsArray(xoff=x1a, yoff=y1a, xsize=xsize1, ysize=ysize1)
    I2 = DS2.ReadAsArray(xoff=x2a, yoff=y2a, xsize=xsize2, ysize=ysize2)

    I1 = I1.astype(np.float32)
    I2 = I2.astype(np.float32)

    DS1 = None
    DS2 = None

    return I1, I2


def runAutorift(
    indir_r,
    indir_t,
    xGrid,
    yGrid,
    Dx0,
    Dy0,
    SRx0,
    SRy0,
    CSMINx0,
    CSMINy0,
    CSMAXx0,
    CSMAXy0,
    noDataMask,
    optflag,
    nodata,
    mpflag,
    geogrid_run_info=None,
    preprocessing_methods=('hps', 'hps'),
    preprocessing_filter_width=5,
):
    """
    Wire and run geogrid.
    """
    obj = autoRIFT()

    obj.WallisFilterWidth = preprocessing_filter_width
    print(f'Setting Wallis Filter Width to {preprocessing_filter_width}')

    # uncomment if starting from preprocessed images
    # I1 = I1.astype(np.uint8)
    # I2 = I2.astype(np.uint8)

    obj.MultiThread = mpflag

    if optflag == 1:
        obj.I1, obj.I2 = loadProductOptical(indir_r, indir_t)
    else:
        obj.I1 = loadProduct(indir_r)
        obj.I2 = loadProduct(indir_t)

    # create the grid if it does not exist
    if xGrid is None:
        m, n = obj.I1.shape
        xGrid = np.arange(obj.SkipSampleX + 10, n - obj.SkipSampleX, obj.SkipSampleX)
        yGrid = np.arange(obj.SkipSampleY + 10, m - obj.SkipSampleY, obj.SkipSampleY)
        nd = xGrid.__len__()
        md = yGrid.__len__()
        obj.xGrid = np.int32(np.dot(np.ones((md, 1)), np.reshape(xGrid, (1, xGrid.__len__()))))
        obj.yGrid = np.int32(np.dot(np.reshape(yGrid, (yGrid.__len__(), 1)), np.ones((1, nd))))
        noDataMask = np.logical_not(obj.xGrid)
    else:
        obj.xGrid = xGrid
        obj.yGrid = yGrid

    # NOTE: This assumes the zero values in the image are only outside the valid image "frame",
    #        but is not true for Landsat-7 after the failure of the Scan Line Corrector, May 31, 2003.
    #        We should not mask based on zero values in the L7 images as this percolates into SearchLimit{X,Y}
    #        and prevents autoRIFT from looking at large parts of the images, but untangling the logic here
    #        has proved too difficult, so lets just turn it off if `wallis_fill` preprocessing is going to be used.
    # generate the nodata mask where offset searching will be skipped based on 1) imported nodata mask and/or 2) zero values in the image
    if 'wallis_fill' not in preprocessing_methods:
        for ii in range(obj.xGrid.shape[0]):
            for jj in range(obj.xGrid.shape[1]):
                if (obj.yGrid[ii, jj] != nodata) & (obj.xGrid[ii, jj] != nodata):
                    if (obj.I1[obj.yGrid[ii, jj] - 1, obj.xGrid[ii, jj] - 1] == 0) | (
                        obj.I2[obj.yGrid[ii, jj] - 1, obj.xGrid[ii, jj] - 1] == 0
                    ):
                        noDataMask[ii, jj] = True

    # mask out nodata to skip the offset searching using the nodata mask (by setting SearchLimit to be 0)

    if SRx0 is None:
        # uncomment to customize SearchLimit based on velocity distribution (i.e. Dx0 must not be None)
        # obj.SearchLimitX = np.int32(4+(25-4)/(np.max(np.abs(Dx0[np.logical_not(noDataMask)]))-np.min(np.abs(Dx0[np.logical_not(noDataMask)])))*(np.abs(Dx0)-np.min(np.abs(Dx0[np.logical_not(noDataMask)]))))
        # obj.SearchLimitY = 5
        obj.SearchLimitX = obj.SearchLimitX * np.logical_not(noDataMask)
        obj.SearchLimitY = obj.SearchLimitY * np.logical_not(noDataMask)
    else:
        obj.SearchLimitX = SRx0
        obj.SearchLimitY = SRy0
        # add buffer to search range
        # obj.SearchLimitX[obj.SearchLimitX!=0] = obj.SearchLimitX[obj.SearchLimitX!=0] + 2
        # obj.SearchLimitY[obj.SearchLimitY!=0] = obj.SearchLimitY[obj.SearchLimitY!=0] + 2

    if CSMINx0 is not None:
        obj.ChipSizeMaxX = CSMAXx0
        obj.ChipSizeMinX = CSMINx0

        if geogrid_run_info is None:
            gridspacingx = float(str.split(runCmd('fgrep "Grid spacing in m:" testGeogrid.txt'))[-1])
            chipsizex0 = float(str.split(runCmd('fgrep "Smallest Allowable Chip Size in m:" testGeogrid.txt'))[-1])
            try:
                pixsizex = float(str.split(runCmd('fgrep "Ground range pixel size:" testGeogrid.txt'))[-1])
            except:
                pixsizex = float(str.split(runCmd('fgrep "X-direction pixel size:" testGeogrid.txt'))[-1])
        else:
            gridspacingx = geogrid_run_info['gridspacingx']
            chipsizex0 = geogrid_run_info['chipsizex0']
            pixsizex = geogrid_run_info['XPixelSize']

        obj.ChipSize0X = int(np.ceil(chipsizex0 / pixsizex / 4) * 4)
        obj.GridSpacingX = int(obj.ChipSize0X * gridspacingx / chipsizex0)

        RATIO_Y2X = CSMINy0 / CSMINx0
        obj.ScaleChipSizeY = np.median(RATIO_Y2X[(CSMINx0 != nodata) & (CSMINy0 != nodata)])
    else:
        if (optflag == 1) & (xGrid is not None):
            obj.ChipSizeMaxX = 32
            obj.ChipSizeMinX = 16
            obj.ChipSize0X = 16

    # create the downstream search offset if not provided as input
    if Dx0 is not None:
        obj.Dx0 = Dx0
        obj.Dy0 = Dy0
    else:
        obj.Dx0 = obj.Dx0 * np.logical_not(noDataMask)
        obj.Dy0 = obj.Dy0 * np.logical_not(noDataMask)

    # replace the nodata value with zero
    obj.xGrid[noDataMask] = 0
    obj.yGrid[noDataMask] = 0
    obj.Dx0[noDataMask] = 0
    obj.Dy0[noDataMask] = 0
    if SRx0 is not None:
        obj.SearchLimitX[noDataMask] = 0
        obj.SearchLimitY[noDataMask] = 0
    if CSMINx0 is not None:
        obj.ChipSizeMaxX[noDataMask] = 0
        obj.ChipSizeMinX[noDataMask] = 0

    # convert azimuth offset to vertical offset as used in autoRIFT convention
    if optflag == 0:
        obj.Dy0 = -1 * obj.Dy0

    # preprocessing
    t1 = time.time()
    print('Pre-process Start!!!')
    print(f'Using Wallis Filter Width: {obj.WallisFilterWidth}')

    # TODO: Allow different filters to be applied images independently
    # default to most stringent filtering
    if 'wallis_fill' in preprocessing_methods:
        obj.preprocess_filt_wal_nodata_fill()
    elif 'wallis' in preprocessing_methods:
        obj.preprocess_filt_wal()
    elif 'fft' in preprocessing_methods:
        # FIXME: The Landsat 4/5 FFT preprocessor looks for the image corners to
        #        determine the scene rotation, but Geogrid + autoRIFT rond the
        #        corners when co-registering and chop the non-overlapping corners
        #        when subsetting to the common image overlap. FFT filer needs to
        #        be applied to the native images before they are processed by
        #        Geogrid or autoRIFT.
        # obj.preprocess_filt_wal()
        # obj.preprocess_filt_fft()
        warnings.warn(
            'FFT filtering must be done before processing with geogrid! Be careful when using this method', UserWarning
        )
    else:
        obj.preprocess_filt_hps()
    print('Pre-process Done!!!')
    print(time.time() - t1)

    t1 = time.time()

    if obj.zeroMask is not None:
        validData = np.isfinite(obj.I1)
        S1 = np.std(obj.I1[validData]) * np.sqrt(obj.I1[validData].size / (obj.I1[validData].size - 1.0))
        M1 = np.mean(obj.I1[validData])
    else:
        S1 = np.std(obj.I1) * np.sqrt(obj.I1.size / (obj.I1.size - 1.0))
        M1 = np.mean(obj.I1)

    obj.I1 = (obj.I1 - (M1 - 3 * S1)) / (6 * S1) * (2**8 - 0)
    del S1, M1
    obj.I1 = np.round(np.clip(obj.I1, 0, 255)).astype(np.uint8)

    if obj.zeroMask is not None:
        validData = np.isfinite(obj.I2)
        S2 = np.std(obj.I2[validData]) * np.sqrt(obj.I2[validData].size / (obj.I2[validData].size - 1.0))
        M2 = np.mean(obj.I2[validData])
    else:
        S2 = np.std(obj.I2) * np.sqrt(obj.I2.size / (obj.I2.size - 1.0))
        M2 = np.mean(obj.I2)

    obj.I2 = (obj.I2 - (M2 - 3 * S2)) / (6 * S2) * (2**8 - 0)
    del S2, M2
    obj.I2 = np.round(np.clip(obj.I2, 0, 255)).astype(np.uint8)

    if obj.zeroMask is not None:
        obj.I1[obj.zeroMask] = 0
        obj.I2[obj.zeroMask] = 0
        obj.zeroMask = None

    print('Uniform Data Type Done!!!')
    print(time.time() - t1)

    obj.OverSampleRatio = 64
    # OverSampleRatio can be assigned as a scalar (such as the above line) or as a Python dictionary below for
    # intelligent use (ChipSize-dependent). Here, four chip sizes are used: ChipSize0X*[1,2,4,8] and four
    # OverSampleRatio are considered [16,32,64,128]. The intelligent selection of OverSampleRatio (as a function of
    # chip size) was determined by analyzing various combinations of (OverSampleRatio and chip size) and comparing
    # the resulting image quality and statistics with the reference scenario (where the largest OverSampleRatio of
    # 128 and chip size of ChipSize0X*8 are considered). The selection for the optical data flag is based on Landsat-8
    # data over an inland region (thus stable and not moving much) of Greenland, while that for the radar flag
    # (optflag = 0) is based on Sentinel-1 data over the same region of Greenland.
    if CSMINx0 is not None:
        if optflag == 1:
            obj.OverSampleRatio = {
                obj.ChipSize0X: 16,
                obj.ChipSize0X * 2: 32,
                obj.ChipSize0X * 4: 64,
                obj.ChipSize0X * 8: 64,
            }
        else:
            obj.OverSampleRatio = {
                obj.ChipSize0X: 32,
                obj.ChipSize0X * 2: 64,
                obj.ChipSize0X * 4: 128,
                obj.ChipSize0X * 8: 128,
            }

    # run Autorift
    t1 = time.time()
    print('AutoRIFT Start!!!')
    obj.runAutorift()
    print('AutoRIFT Done!!!')
    print(time.time() - t1)

    kernel = np.ones((3, 3), np.uint8)
    noDataMask = cv2.dilate(noDataMask.astype(np.uint8), kernel, iterations=1)
    noDataMask = noDataMask.astype(bool)

    return (
        obj.Dx,
        obj.Dy,
        obj.InterpMask,
        obj.ChipSizeX,
        obj.GridSpacingX,
        obj.ScaleChipSizeY,
        obj.SearchLimitX,
        obj.SearchLimitY,
        obj.origSize,
        noDataMask,
    )


def main():
    """
    Main driver.
    """
    inps = cmdLineParse()

    generateAutoriftProduct(
        indir_r=inps.indir_r,
        indir_t=inps.indir_t,
        grid_location=inps.grid_location,
        init_offset=inps.init_offset,
        search_range=inps.search_range,
        chip_size_min=inps.chip_size_min,
        chip_size_max=inps.chip_size_max,
        offset2vx=inps.offset2vx,
        offset2vy=inps.offset2vy,
        scale_factor=inps.scale_factor,
        stable_surface_mask=inps.stable_surface_mask,
        optical_flag=inps.optical_flag,
        nc_sensor=inps.nc_sensor,
        mpflag=inps.mpflag,
        ncname=inps.ncname,
    )


def generateAutoriftProduct(
    indir_r,
    indir_t,
    grid_location,
    init_offset,
    search_range,
    chip_size_min,
    chip_size_max,
    offset2vx,
    offset2vy,
    scale_factor,
    stable_surface_mask,
    optical_flag,
    nc_sensor,
    mpflag,
    ncname,
    geogrid_run_info=None,
):
    xGrid = None
    yGrid = None
    Dx0 = None
    Dy0 = None
    SRx0 = None
    SRy0 = None
    CSMINx0 = None
    CSMINy0 = None
    CSMAXx0 = None
    CSMAXy0 = None
    SSM = None
    noDataMask = None
    nodata = None

    if grid_location is not None:
        ds = gdal.Open(grid_location)
        tran = ds.GetGeoTransform()
        proj = ds.GetProjection()
        srs = ds.GetSpatialRef()
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        xGrid = band.ReadAsArray()
        noDataMask = xGrid == nodata
        band = ds.GetRasterBand(2)
        yGrid = band.ReadAsArray()
        band = None
        ds = None

    if init_offset is not None:
        ds = gdal.Open(init_offset)
        band = ds.GetRasterBand(1)
        Dx0 = band.ReadAsArray()
        band = ds.GetRasterBand(2)
        Dy0 = band.ReadAsArray()
        band = None
        ds = None

    if search_range is not None:
        ds = gdal.Open(search_range)
        band = ds.GetRasterBand(1)
        SRx0 = band.ReadAsArray()
        band = ds.GetRasterBand(2)
        SRy0 = band.ReadAsArray()
        band = None
        ds = None

    if chip_size_min is not None:
        ds = gdal.Open(chip_size_min)
        band = ds.GetRasterBand(1)
        CSMINx0 = band.ReadAsArray()
        band = ds.GetRasterBand(2)
        CSMINy0 = band.ReadAsArray()
        band = None
        ds = None

    if chip_size_max is not None:
        ds = gdal.Open(chip_size_max)
        band = ds.GetRasterBand(1)
        CSMAXx0 = band.ReadAsArray()
        band = ds.GetRasterBand(2)
        CSMAXy0 = band.ReadAsArray()
        band = None
        ds = None

    if stable_surface_mask is not None:
        ds = gdal.Open(stable_surface_mask)
        band = ds.GetRasterBand(1)
        SSM = band.ReadAsArray()
        SSM = SSM.astype('bool')
        band = None
        ds = None

    intermediate_nc_file = 'autoRIFT_intermediate.nc'

    if os.path.exists(intermediate_nc_file):
        (
            Dx,
            Dy,
            InterpMask,
            ChipSizeX,
            GridSpacingX,
            ScaleChipSizeY,
            SearchLimitX,
            SearchLimitY,
            origSize,
            noDataMask,
        ) = no.netCDF_read_intermediate(intermediate_nc_file)
    else:
        m_name = os.path.basename(indir_r)
        s_name = os.path.basename(indir_t)

        # FIXME: Filter width is a magic variable here and not exposed well.
        preprocessing_filter_width = 5
        if nc_sensor == 'S1':
            preprocessing_filter_width = 21

        print(f'Preprocessing filter width {preprocessing_filter_width}')

        preprocessing_methods = ['hps', 'hps']
        for ii, name in enumerate((m_name, s_name)):
            if len(re.findall('L[EO]07_', name)) > 0:
                preprocessing_methods[ii] = 'wallis_fill'
            elif len(re.findall('LT0[45]_', name)) > 0:
                preprocessing_methods[ii] = 'fft'

        print(f'Using preprocessing methods {preprocessing_methods}')

        (
            Dx,
            Dy,
            InterpMask,
            ChipSizeX,
            GridSpacingX,
            ScaleChipSizeY,
            SearchLimitX,
            SearchLimitY,
            origSize,
            noDataMask,
        ) = runAutorift(
            indir_r,
            indir_t,
            xGrid,
            yGrid,
            Dx0,
            Dy0,
            SRx0,
            SRy0,
            CSMINx0,
            CSMINy0,
            CSMAXx0,
            CSMAXy0,
            noDataMask,
            optical_flag,
            nodata,
            mpflag,
            geogrid_run_info=geogrid_run_info,
            preprocessing_methods=preprocessing_methods,
            preprocessing_filter_width=preprocessing_filter_width,
        )
        if nc_sensor is not None:
            no.netCDF_packaging_intermediate(
                Dx,
                Dy,
                InterpMask,
                ChipSizeX,
                GridSpacingX,
                ScaleChipSizeY,
                SearchLimitX,
                SearchLimitY,
                origSize,
                noDataMask,
                intermediate_nc_file,
            )

    if optical_flag == 0:
        Dy = -Dy

    DX = np.zeros(origSize, dtype=np.float32) * np.nan
    DY = np.zeros(origSize, dtype=np.float32) * np.nan
    INTERPMASK = np.zeros(origSize, dtype=np.float32)
    CHIPSIZEX = np.zeros(origSize, dtype=np.float32)
    SEARCHLIMITX = np.zeros(origSize, dtype=np.float32)
    SEARCHLIMITY = np.zeros(origSize, dtype=np.float32)

    DX[0 : Dx.shape[0], 0 : Dx.shape[1]] = Dx
    DY[0 : Dy.shape[0], 0 : Dy.shape[1]] = Dy
    INTERPMASK[0 : InterpMask.shape[0], 0 : InterpMask.shape[1]] = InterpMask
    CHIPSIZEX[0 : ChipSizeX.shape[0], 0 : ChipSizeX.shape[1]] = ChipSizeX
    SEARCHLIMITX[0 : SearchLimitX.shape[0], 0 : SearchLimitX.shape[1]] = SearchLimitX
    SEARCHLIMITY[0 : SearchLimitY.shape[0], 0 : SearchLimitY.shape[1]] = SearchLimitY

    DX[noDataMask] = np.nan
    DY[noDataMask] = np.nan
    INTERPMASK[noDataMask] = 0
    CHIPSIZEX[noDataMask] = 0
    SEARCHLIMITX[noDataMask] = 0
    SEARCHLIMITY[noDataMask] = 0
    if SSM is not None:
        SSM[noDataMask] = False

    DX[SEARCHLIMITX == 0] = np.nan
    DY[SEARCHLIMITX == 0] = np.nan
    INTERPMASK[SEARCHLIMITX == 0] = 0
    CHIPSIZEX[SEARCHLIMITX == 0] = 0
    if SSM is not None:
        SSM[SEARCHLIMITX == 0] = False

    netcdf_file = None
    if grid_location is not None:
        t1 = time.time()
        print('Write Outputs Start!!!')

        # Create the GeoTiff
        driver = gdal.GetDriverByName('GTiff')

        outRaster = driver.Create('offset.tif', int(xGrid.shape[1]), int(xGrid.shape[0]), 4, gdal.GDT_Float32)
        outRaster.SetGeoTransform(tran)
        outRaster.SetProjection(proj)
        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(DX)
        outband.FlushCache()
        outband = outRaster.GetRasterBand(2)
        outband.WriteArray(DY)
        outband.FlushCache()
        outband = outRaster.GetRasterBand(3)
        outband.WriteArray(INTERPMASK)
        outband.FlushCache()
        outband = outRaster.GetRasterBand(4)
        outband.WriteArray(CHIPSIZEX)
        outband.FlushCache()
        del outRaster

        if offset2vx is not None:
            ds = gdal.Open(scale_factor)
            band = ds.GetRasterBand(1)
            scale_factor_1 = band.ReadAsArray()
            band = ds.GetRasterBand(2)
            scale_factor_2 = band.ReadAsArray()
            band = None
            ds = None
            scale_factor_1[scale_factor_1 == nodata] = np.nan
            scale_factor_2[scale_factor_2 == nodata] = np.nan

            ds = gdal.Open(offset2vx)
            band = ds.GetRasterBand(1)
            offset2vx_1 = band.ReadAsArray()
            band = ds.GetRasterBand(2)
            offset2vx_2 = band.ReadAsArray()
            if ds.RasterCount > 2:
                band = ds.GetRasterBand(3)
                offset2vr = band.ReadAsArray()
            else:
                offset2vr = None
            band = None
            ds = None
            offset2vx_1[offset2vx_1 == nodata] = np.nan
            offset2vx_2[offset2vx_2 == nodata] = np.nan
            if offset2vr is not None:
                offset2vr[offset2vr == nodata] = np.nan

            ds = gdal.Open(offset2vy)
            band = ds.GetRasterBand(1)
            offset2vy_1 = band.ReadAsArray()
            band = ds.GetRasterBand(2)
            offset2vy_2 = band.ReadAsArray()
            if ds.RasterCount > 2:
                band = ds.GetRasterBand(3)
                offset2va = band.ReadAsArray()
            else:
                offset2va = None
            band = None
            ds = None
            offset2vy_1[offset2vy_1 == nodata] = np.nan
            offset2vy_2[offset2vy_2 == nodata] = np.nan
            if offset2va is not None:
                offset2va[offset2va == nodata] = np.nan

            VX = offset2vx_1 * (DX * scale_factor_1) + offset2vx_2 * (DY * scale_factor_2)
            VY = offset2vy_1 * (DX * scale_factor_1) + offset2vy_2 * (DY * scale_factor_2)
            VX = VX.astype(np.float32)
            VY = VY.astype(np.float32)

            # write velocity output in Geotiff format
            outRaster = driver.Create('velocity.tif', int(xGrid.shape[1]), int(xGrid.shape[0]), 2, gdal.GDT_Float32)
            outRaster.SetGeoTransform(tran)
            outRaster.SetProjection(proj)
            outband = outRaster.GetRasterBand(1)
            outband.WriteArray(VX)
            outband.FlushCache()
            outband = outRaster.GetRasterBand(2)
            outband.WriteArray(VY)
            outband.FlushCache()
            del outRaster

            # prepare for netCDF packaging
            if nc_sensor is not None:
                if nc_sensor == 'S1':
                    swath_offset_bias_ref = [-0.01, 0.019, -0.0068, 0.006]

                    DX, DY, flight_direction_m, flight_direction_s = no.cal_swath_offset_bias(
                        indir_r,
                        xGrid,
                        yGrid,
                        VX,
                        VY,
                        DX,
                        DY,
                        nodata,
                        tran,
                        proj,
                        GridSpacingX,
                        ScaleChipSizeY,
                        swath_offset_bias_ref,
                    )

                if geogrid_run_info is None:
                    vxrefname = str.split(runCmd('fgrep "Velocities:" testGeogrid.txt'))[1]
                    vyrefname = str.split(runCmd('fgrep "Velocities:" testGeogrid.txt'))[2]
                    sxname = str.split(runCmd('fgrep "Slopes:" testGeogrid.txt'))[1][:-4] + 's.tif'
                    syname = str.split(runCmd('fgrep "Slopes:" testGeogrid.txt'))[2][:-4] + 's.tif'
                    maskname = str.split(runCmd('fgrep "Slopes:" testGeogrid.txt'))[2][:-8] + 'sp.tif'
                    xoff = int(str.split(runCmd('fgrep "Origin index (in DEM) of geogrid:" testGeogrid.txt'))[6])
                    yoff = int(str.split(runCmd('fgrep "Origin index (in DEM) of geogrid:" testGeogrid.txt'))[7])
                    xcount = int(str.split(runCmd('fgrep "Dimensions of geogrid:" testGeogrid.txt'))[3])
                    ycount = int(str.split(runCmd('fgrep "Dimensions of geogrid:" testGeogrid.txt'))[5])
                    cen_lat = (
                        int(100 * float(str.split(runCmd('fgrep "Scene-center lat/lon:" testGeogrid.txt'))[2])) / 100
                    )
                    cen_lon = (
                        int(100 * float(str.split(runCmd('fgrep "Scene-center lat/lon:" testGeogrid.txt'))[3])) / 100
                    )
                else:
                    vxrefname = geogrid_run_info['vxname']
                    vyrefname = geogrid_run_info['vyname']
                    sxname = geogrid_run_info['sxname']
                    syname = geogrid_run_info['syname']
                    maskname = geogrid_run_info['maskname']
                    xoff = geogrid_run_info['xoff']
                    yoff = geogrid_run_info['yoff']
                    xcount = geogrid_run_info['xcount']
                    ycount = geogrid_run_info['ycount']
                    cen_lat = int(100 * geogrid_run_info['cen_lat']) / 100
                    cen_lon = int(100 * geogrid_run_info['cen_lon']) / 100

                ds = gdal.Open(vxrefname)
                band = ds.GetRasterBand(1)
                VXref = band.ReadAsArray(xoff, yoff, xcount, ycount)
                ds = None
                band = None

                ds = gdal.Open(vyrefname)
                band = ds.GetRasterBand(1)
                VYref = band.ReadAsArray(xoff, yoff, xcount, ycount)
                ds = None
                band = None

                ds = gdal.Open(sxname)
                band = ds.GetRasterBand(1)
                SX = band.ReadAsArray(xoff, yoff, xcount, ycount)
                ds = None
                band = None

                ds = gdal.Open(syname)
                band = ds.GetRasterBand(1)
                SY = band.ReadAsArray(xoff, yoff, xcount, ycount)
                ds = None
                band = None

                ds = gdal.Open(maskname)
                band = ds.GetRasterBand(1)
                MM = band.ReadAsArray(xoff, yoff, xcount, ycount)
                ds = None
                band = None

                DXref = (
                    offset2vy_2 / (offset2vx_1 * offset2vy_2 - offset2vx_2 * offset2vy_1) * VXref
                    - offset2vx_2 / (offset2vx_1 * offset2vy_2 - offset2vx_2 * offset2vy_1) * VYref
                )
                DYref = (
                    offset2vx_1 / (offset2vx_1 * offset2vy_2 - offset2vx_2 * offset2vy_1) * VYref
                    - offset2vy_1 / (offset2vx_1 * offset2vy_2 - offset2vx_2 * offset2vy_1) * VXref
                )
                DXref = DXref / scale_factor_1
                DYref = DYref / scale_factor_2

                stable_count = np.sum(SSM & np.logical_not(np.isnan(DX)))

                V_temp = np.sqrt(VXref**2 + VYref**2)
                try:
                    V_temp_threshold = np.percentile(V_temp[np.logical_not(np.isnan(V_temp))], 25)
                    SSM1 = V_temp <= V_temp_threshold
                except IndexError:
                    SSM1 = np.zeros(V_temp.shape).astype('bool')

                stable_count1 = np.sum(SSM1 & np.logical_not(np.isnan(DX)))

                dx_mean_shift = 0.0
                dy_mean_shift = 0.0
                dx_mean_shift1 = 0.0
                dy_mean_shift1 = 0.0

                if stable_count != 0:
                    temp = DX.copy() - DXref.copy()
                    temp[np.logical_not(SSM)] = np.nan
                    dx_mean_shift = np.median(temp[np.logical_not(np.isnan(temp))])

                    temp = DY.copy() - DYref.copy()
                    temp[np.logical_not(SSM)] = np.nan
                    dy_mean_shift = np.median(temp[np.logical_not(np.isnan(temp))])

                if stable_count1 != 0:
                    temp = DX.copy() - DXref.copy()
                    temp[np.logical_not(SSM1)] = np.nan
                    dx_mean_shift1 = np.median(temp[np.logical_not(np.isnan(temp))])

                    temp = DY.copy() - DYref.copy()
                    temp[np.logical_not(SSM1)] = np.nan
                    dy_mean_shift1 = np.median(temp[np.logical_not(np.isnan(temp))])

                if stable_count == 0:
                    if stable_count1 == 0:
                        stable_shift_applied = 0
                    else:
                        stable_shift_applied = 2
                        DX = DX - dx_mean_shift1
                        DY = DY - dy_mean_shift1
                else:
                    stable_shift_applied = 1
                    DX = DX - dx_mean_shift
                    DY = DY - dy_mean_shift

                VX = offset2vx_1 * (DX * scale_factor_1) + offset2vx_2 * (DY * scale_factor_2)
                VY = offset2vy_1 * (DX * scale_factor_1) + offset2vy_2 * (DY * scale_factor_2)
                VX = VX.astype(np.float32)
                VY = VY.astype(np.float32)

                # netCDF packaging for Sentinel and Landsat dataset; can add other sensor format as well
                if nc_sensor == 'S1':
                    if geogrid_run_info is None:
                        chipsizex0 = float(
                            str.split(runCmd('fgrep "Smallest Allowable Chip Size in m:" testGeogrid.txt'))[-1]
                        )
                        gridspacingx = float(str.split(runCmd('fgrep "Grid spacing in m:" testGeogrid.txt'))[-1])
                        rangePixelSize = float(str.split(runCmd('fgrep "Ground range pixel size:" testGeogrid.txt'))[4])
                        azimuthPixelSize = float(str.split(runCmd('fgrep "Azimuth pixel size:" testGeogrid.txt'))[3])
                        dt = float(str.split(runCmd('fgrep "Repeat Time:" testGeogrid.txt'))[2])
                        epsg = float(str.split(runCmd('fgrep "EPSG:" testGeogrid.txt'))[1])
                        #  print (str(rangePixelSize)+"      "+str(azimuthPixelSize))
                    else:
                        chipsizex0 = geogrid_run_info['chipsizex0']
                        gridspacingx = geogrid_run_info['gridspacingx']
                        rangePixelSize = geogrid_run_info['XPixelSize']
                        azimuthPixelSize = geogrid_run_info['YPixelSize']
                        dt = geogrid_run_info['dt']
                        epsg = geogrid_run_info['epsg']

                    conts = get_topsinsar_config()
                    master_filename = conts['reference_filename']
                    slave_filename = conts['secondary_filename']
                    master_dt = conts['reference_dt']
                    slave_dt = conts['secondary_dt']
                    master_split = str.split(master_filename, '_')
                    slave_split = str.split(slave_filename, '_')

                    pair_type = 'radar'
                    detection_method = 'feature'
                    coordinates = 'radar, map'
                    if np.sum(SEARCHLIMITX != 0) != 0:
                        roi_valid_percentage = (
                            int(round(np.sum(CHIPSIZEX != 0) / np.sum(SEARCHLIMITX != 0) * 1000.0)) / 1000
                        )
                    else:
                        raise Exception('Input search range is all zero everywhere, thus no search conducted')
                    PPP = roi_valid_percentage * 100
                    if ncname is None:
                        if '.zip' in master_filename:
                            out_nc_filename = (
                                f'./{master_filename[0:-4]}_X_{slave_filename[0:-4]}'
                                f'_G{gridspacingx:04.0f}V02_P{np.floor(PPP):03.0f}.nc'
                            )
                        elif '.SAFE' in master_filename:
                            out_nc_filename = (
                                f'./{master_filename[0:-5]}_X_{slave_filename[0:-5]}'
                                f'_G{gridspacingx:04.0f}V02_P{np.floor(PPP):03.0f}.nc'
                            )
                    else:
                        out_nc_filename = f'{ncname}_G{gridspacingx:04.0f}V02_P{np.floor(PPP):03.0f}.nc'
                    CHIPSIZEY = np.round(CHIPSIZEX * ScaleChipSizeY / 2) * 2

                    d0 = datetime.strptime(master_dt, '%Y%m%dT%H:%M:%S.%f')
                    d1 = datetime.strptime(slave_dt, '%Y%m%dT%H:%M:%S.%f')
                    date_dt_base = (d1 - d0).total_seconds() / timedelta(days=1).total_seconds()
                    date_dt = np.float64(date_dt_base)
                    if date_dt < 0:
                        raise Exception('Input image 1 must be older than input image 2')

                    date_ct = d0 + (d1 - d0) / 2
                    date_center = date_ct.strftime('%Y%m%dT%H:%M:%S.%f').rstrip('0')

                    IMG_INFO_DICT = {
                        'id_img1': master_filename[0:-4],
                        'id_img2': slave_filename[0:-4],
                        'absolute_orbit_number_img1': master_split[7],
                        'absolute_orbit_number_img2': slave_split[7],
                        'acquisition_date_img1': master_dt,
                        'acquisition_date_img2': slave_dt,
                        'flight_direction_img1': flight_direction_m,
                        'flight_direction_img2': flight_direction_s,
                        'mission_data_take_ID_img1': master_split[8],
                        'mission_data_take_ID_img2': slave_split[8],
                        'mission_img1': master_split[0][0],
                        'mission_img2': slave_split[0][0],
                        'product_unique_ID_img1': master_split[9][0:4],
                        'product_unique_ID_img2': slave_split[9][0:4],
                        'satellite_img1': master_split[0][1:3],
                        'satellite_img2': slave_split[0][1:3],
                        'sensor_img1': 'C',
                        'sensor_img2': 'C',
                        'time_standard_img1': 'UTC',
                        'time_standard_img2': 'UTC',
                        'date_center': date_center,
                        'date_dt': date_dt,
                        'latitude': cen_lat,
                        'longitude': cen_lon,
                        'roi_valid_percentage': PPP,
                        'autoRIFT_software_version': version,
                    }
                    error_vector = np.array(
                        [
                            [0.0356, 0.0501, 0.0266, 0.0622, 0.0357, 0.0501],
                            [0.5194, 1.1638, 0.3319, 1.3701, 0.5191, 1.1628],
                        ]
                    )

                    netcdf_file = no.netCDF_packaging(
                        VX,
                        VY,
                        DX,
                        DY,
                        INTERPMASK,
                        CHIPSIZEX,
                        CHIPSIZEY,
                        SSM,
                        SSM1,
                        SX,
                        SY,
                        offset2vx_1,
                        offset2vx_2,
                        offset2vy_1,
                        offset2vy_2,
                        offset2vr,
                        offset2va,
                        scale_factor_1,
                        scale_factor_2,
                        MM,
                        VXref,
                        VYref,
                        DXref,
                        DYref,
                        rangePixelSize,
                        azimuthPixelSize,
                        dt,
                        epsg,
                        srs,
                        tran,
                        out_nc_filename,
                        pair_type,
                        detection_method,
                        coordinates,
                        IMG_INFO_DICT,
                        stable_count,
                        stable_count1,
                        stable_shift_applied,
                        dx_mean_shift,
                        dy_mean_shift,
                        dx_mean_shift1,
                        dy_mean_shift1,
                        error_vector,
                    )

                elif nc_sensor in ('L4', 'L5', 'L7', 'L8', 'L9'):
                    if geogrid_run_info is None:
                        chipsizex0 = float(
                            str.split(runCmd('fgrep "Smallest Allowable Chip Size in m:" testGeogrid.txt'))[-1]
                        )
                        gridspacingx = float(str.split(runCmd('fgrep "Grid spacing in m:" testGeogrid.txt'))[-1])
                        XPixelSize = float(str.split(runCmd('fgrep "X-direction pixel size:" testGeogrid.txt'))[3])
                        YPixelSize = float(str.split(runCmd('fgrep "Y-direction pixel size:" testGeogrid.txt'))[3])
                        epsg = float(str.split(runCmd('fgrep "EPSG:" testGeogrid.txt'))[1])
                    else:
                        chipsizex0 = geogrid_run_info['chipsizex0']
                        gridspacingx = geogrid_run_info['gridspacingx']
                        XPixelSize = geogrid_run_info['XPixelSize']
                        YPixelSize = geogrid_run_info['YPixelSize']
                        epsg = geogrid_run_info['epsg']

                    master_path = indir_r
                    slave_path = indir_t

                    master_filename = os.path.basename(master_path)
                    slave_filename = os.path.basename(slave_path)

                    master_split = str.split(master_filename, '_')
                    slave_split = str.split(slave_filename, '_')

                    pair_type = 'optical'
                    detection_method = 'feature'
                    coordinates = 'map'
                    if np.sum(SEARCHLIMITX != 0) != 0:
                        roi_valid_percentage = (
                            int(round(np.sum(CHIPSIZEX != 0) / np.sum(SEARCHLIMITX != 0) * 1000.0)) / 1000
                        )
                    else:
                        raise Exception('Input search range is all zero everywhere, thus no search conducted')
                    PPP = roi_valid_percentage * 100
                    if ncname is None:
                        out_nc_filename = (
                            f'./{master_filename[0:-7]}_X_{slave_filename[0:-7]}'
                            f'_G{gridspacingx:04.0f}V02_P{np.floor(PPP):03.0f}.nc'
                        )
                    else:
                        out_nc_filename = f'{ncname}_G{gridspacingx:04.0f}V02_P{np.floor(PPP):03.0f}.nc'
                    CHIPSIZEY = np.round(CHIPSIZEX * ScaleChipSizeY / 2) * 2

                    d0 = datetime(int(master_split[3][0:4]), int(master_split[3][4:6]), int(master_split[3][6:8]))
                    d1 = datetime(int(slave_split[3][0:4]), int(slave_split[3][4:6]), int(slave_split[3][6:8]))
                    date_dt_base = (d1 - d0).total_seconds() / timedelta(days=1).total_seconds()
                    date_dt = np.float64(date_dt_base)
                    if date_dt < 0:
                        raise Exception('Input image 1 must be older than input image 2')

                    date_ct = d0 + (d1 - d0) / 2
                    date_center = date_ct.strftime('%Y%m%dT%H:%M:%S.%f').rstrip('0')

                    master_dt = d0.strftime('%Y%m%dT%H:%M:%S.%f').rstrip('0')
                    slave_dt = d1.strftime('%Y%m%dT%H:%M:%S.%f').rstrip('0')

                    IMG_INFO_DICT = {
                        'id_img1': master_filename[0:-7],
                        'id_img2': slave_filename[0:-7],
                        'acquisition_date_img1': master_dt,
                        'acquisition_date_img2': slave_dt,
                        'collection_category_img1': master_split[6],
                        'collection_category_img2': slave_split[6],
                        'collection_number_img1': np.float64(master_split[5]),
                        'collection_number_img2': np.float64(slave_split[5]),
                        'correction_level_img1': master_split[1],
                        'correction_level_img2': slave_split[1],
                        'mission_img1': master_split[0][0],
                        'mission_img2': slave_split[0][0],
                        'path_img1': np.float64(master_split[2][0:3]),
                        'path_img2': np.float64(slave_split[2][0:3]),
                        'processing_date_img1': master_split[4][0:8],
                        'processing_date_img2': slave_split[4][0:8],
                        'row_img1': np.float64(master_split[2][3:6]),
                        'row_img2': np.float64(slave_split[2][3:6]),
                        'satellite_img1': master_split[0][2:4].lstrip('0'),
                        'satellite_img2': slave_split[0][2:4].lstrip('0'),
                        'sensor_img1': master_split[0][1],
                        'sensor_img2': slave_split[0][1],
                        'time_standard_img1': 'UTC',
                        'time_standard_img2': 'UTC',
                        'date_center': date_center,
                        'date_dt': date_dt,
                        'latitude': cen_lat,
                        'longitude': cen_lon,
                        'roi_valid_percentage': PPP,
                        'autoRIFT_software_version': version,
                    }

                    error_vector = np.array([25.5, 25.5])

                    netcdf_file = no.netCDF_packaging(
                        VX,
                        VY,
                        DX,
                        DY,
                        INTERPMASK,
                        CHIPSIZEX,
                        CHIPSIZEY,
                        SSM,
                        SSM1,
                        SX,
                        SY,
                        offset2vx_1,
                        offset2vx_2,
                        offset2vy_1,
                        offset2vy_2,
                        None,
                        None,
                        scale_factor_1,
                        scale_factor_2,
                        MM,
                        VXref,
                        VYref,
                        None,
                        None,
                        XPixelSize,
                        YPixelSize,
                        None,
                        epsg,
                        srs,
                        tran,
                        out_nc_filename,
                        pair_type,
                        detection_method,
                        coordinates,
                        IMG_INFO_DICT,
                        stable_count,
                        stable_count1,
                        stable_shift_applied,
                        dx_mean_shift,
                        dy_mean_shift,
                        dx_mean_shift1,
                        dy_mean_shift1,
                        error_vector,
                    )

                elif nc_sensor == 'S2':
                    if geogrid_run_info is None:
                        chipsizex0 = float(
                            str.split(runCmd('fgrep "Smallest Allowable Chip Size in m:" testGeogrid.txt'))[-1]
                        )
                        gridspacingx = float(str.split(runCmd('fgrep "Grid spacing in m:" testGeogrid.txt'))[-1])
                        XPixelSize = float(str.split(runCmd('fgrep "X-direction pixel size:" testGeogrid.txt'))[3])
                        YPixelSize = float(str.split(runCmd('fgrep "Y-direction pixel size:" testGeogrid.txt'))[3])
                        epsg = float(str.split(runCmd('fgrep "EPSG:" testGeogrid.txt'))[1])
                    else:
                        chipsizex0 = geogrid_run_info['chipsizex0']
                        gridspacingx = geogrid_run_info['gridspacingx']
                        XPixelSize = geogrid_run_info['XPixelSize']
                        YPixelSize = geogrid_run_info['YPixelSize']
                        epsg = geogrid_run_info['epsg']

                    master_path = indir_r
                    slave_path = indir_t

                    master_split = master_path.split('_')
                    slave_split = slave_path.split('_')

                    if re.findall('://', master_path).__len__() > 0:
                        master_filename_full = master_path.split('/')
                        for item in master_filename_full:
                            if re.findall('S2._', item).__len__() > 0:
                                master_filename = item
                        slave_filename_full = slave_path.split('/')
                        for item in slave_filename_full:
                            if re.findall('S2._', item).__len__() > 0:
                                slave_filename = item
                    else:
                        master_filename = os.path.basename(master_path)[:-8]
                        slave_filename = os.path.basename(slave_path)[:-8]

                    pair_type = 'optical'
                    detection_method = 'feature'
                    coordinates = 'map'
                    if np.sum(SEARCHLIMITX != 0) != 0:
                        roi_valid_percentage = (
                            int(round(np.sum(CHIPSIZEX != 0) / np.sum(SEARCHLIMITX != 0) * 1000.0)) / 1000
                        )
                    else:
                        raise Exception('Input search range is all zero everywhere, thus no search conducted')
                    PPP = roi_valid_percentage * 100
                    if ncname is None:
                        out_nc_filename = (
                            f'./{master_filename}_X_{slave_filename}_G{gridspacingx:04.0f}V02_P{np.floor(PPP):03.0f}.nc'
                        )
                    else:
                        out_nc_filename = f'{ncname}_G{gridspacingx:04.0f}V02_P{np.floor(PPP):03.0f}.nc'
                    CHIPSIZEY = np.round(CHIPSIZEX * ScaleChipSizeY / 2) * 2

                    d0 = datetime(int(master_split[2][0:4]), int(master_split[2][4:6]), int(master_split[2][6:8]))
                    d1 = datetime(int(slave_split[2][0:4]), int(slave_split[2][4:6]), int(slave_split[2][6:8]))
                    date_dt_base = (d1 - d0).total_seconds() / timedelta(days=1).total_seconds()
                    date_dt = np.float64(date_dt_base)
                    if date_dt < 0:
                        raise Exception('Input image 1 must be older than input image 2')

                    date_ct = d0 + (d1 - d0) / 2
                    date_center = date_ct.strftime('%Y%m%dT%H:%M:%S.%f').rstrip('0')

                    master_dt = d0.strftime('%Y%m%dT%H:%M:%S.%f').rstrip('0')
                    slave_dt = d1.strftime('%Y%m%dT%H:%M:%S.%f').rstrip('0')

                    IMG_INFO_DICT = {
                        'id_img1': master_filename,
                        'id_img2': slave_filename,
                        'acquisition_date_img1': master_dt,
                        'acquisition_date_img2': slave_dt,
                        'correction_level_img1': master_split[4][:3],
                        'correction_level_img2': slave_split[4][:3],
                        'mission_img1': master_split[0][-3],
                        'mission_img2': slave_split[0][-3],
                        'satellite_img1': master_split[0][-2:],
                        'satellite_img2': slave_split[0][-2:],
                        'sensor_img1': 'MSI',
                        'sensor_img2': 'MSI',
                        'time_standard_img1': 'UTC',
                        'time_standard_img2': 'UTC',
                        'date_center': date_center,
                        'date_dt': date_dt,
                        'latitude': cen_lat,
                        'longitude': cen_lon,
                        'roi_valid_percentage': PPP,
                        'autoRIFT_software_version': version,
                    }

                    error_vector = np.array([25.5, 25.5])

                    netcdf_file = no.netCDF_packaging(
                        VX,
                        VY,
                        DX,
                        DY,
                        INTERPMASK,
                        CHIPSIZEX,
                        CHIPSIZEY,
                        SSM,
                        SSM1,
                        SX,
                        SY,
                        offset2vx_1,
                        offset2vx_2,
                        offset2vy_1,
                        offset2vy_2,
                        None,
                        None,
                        scale_factor_1,
                        scale_factor_2,
                        MM,
                        VXref,
                        VYref,
                        None,
                        None,
                        XPixelSize,
                        YPixelSize,
                        None,
                        epsg,
                        srs,
                        tran,
                        out_nc_filename,
                        pair_type,
                        detection_method,
                        coordinates,
                        IMG_INFO_DICT,
                        stable_count,
                        stable_count1,
                        stable_shift_applied,
                        dx_mean_shift,
                        dy_mean_shift,
                        dx_mean_shift1,
                        dy_mean_shift1,
                        error_vector,
                    )

                elif nc_sensor is None:
                    print('netCDF packaging not performed')

                else:
                    raise Exception('netCDF packaging not supported for the type "{0}"'.format(nc_sensor))

        print('Write Outputs Done!!!')
        print(time.time() - t1)

    return netcdf_file


if __name__ == '__main__':
    main()
