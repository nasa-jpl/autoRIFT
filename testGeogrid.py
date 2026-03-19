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
# Authors: Piyush Agram, Yang Lei
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import argparse
import os
import re
from datetime import date, timedelta

import isce3
import numpy as np
from geogrid import GeogridOptical, GeogridRadar
from osgeo import gdal
from s1reader import load_bursts


def cmdLineParse():
    """
    Command line parser.
    """
    parser = argparse.ArgumentParser(description='Output geo grid')
    parser.add_argument(
        '-r',
        '--input_r',
        dest='indir_r',
        type=str,
        required=True,
        help='Input reference image file name (in GeoTIFF format and Cartesian coordinates)',
    )
    parser.add_argument(
        '-t',
        '--input_t',
        dest='indir_t',
        type=str,
        required=True,
        help='Input test image file name (in GeoTIFF format and Cartesian coordinates)',
    )
    parser.add_argument('-d', '--dem', dest='demfile', type=str, required=True, help='Input DEM')
    parser.add_argument('-sx', '--dhdx', dest='dhdxfile', type=str, default='', help='Input slope in X')
    parser.add_argument('-sy', '--dhdy', dest='dhdyfile', type=str, default='', help='Input slope in Y')
    parser.add_argument('-vx', '--vx', dest='vxfile', type=str, default='', help='Input velocity in X')
    parser.add_argument('-vy', '--vy', dest='vyfile', type=str, default='', help='Input velocity in Y')
    parser.add_argument('-srx', '--srx', dest='srxfile', type=str, default='', help='Input search range in X')
    parser.add_argument('-sry', '--sry', dest='sryfile', type=str, default='', help='Input search range in Y')
    parser.add_argument('-csminx', '--csminx', dest='csminxfile', type=str, default='', help='Input chip size min in X')
    parser.add_argument('-csminy', '--csminy', dest='csminyfile', type=str, default='', help='Input chip size min in Y')
    parser.add_argument('-csmaxx', '--csmaxx', dest='csmaxxfile', type=str, default='', help='Input chip size max in X')
    parser.add_argument('-csmaxy', '--csmaxy', dest='csmaxyfile', type=str, default='', help='Input chip size max in Y')
    parser.add_argument('-ssm', '--ssm', dest='ssmfile', type=str, default='', help='Input stable surface mask')
    parser.add_argument(
        '-fo',
        '--flag_optical',
        dest='optical_flag',
        type=bool,
        required=False,
        default=0,
        help='flag for reading optical data (e.g. Landsat): use 1 for on and 0 (default) for off',
    )

    return parser.parse_args()


class Dummy(object):
    pass


def getPol(safe, orbit_path):
    pols = ['vv', 'vh', 'hh', 'hv']
    for pol in pols:
        for swath in [1, 2, 3]:
            try:
                _ = load_bursts(safe, orbit_path, swath, pol)
                print(f'Polarization {pol}')
                return pol
            except:
                pass
    raise ValueError(f'No polarization information found for {safe}.')


def getMergedOrbit(safe, orbit_path, swath):
    pol = getPol(safe, orbit_path)

    bursts = load_bursts(safe, orbit_path, swath, pol)
    burst = bursts[0]

    return burst.orbit


def loadMetadata(safe, orbit_path, swath, buffer=0, burst_shape=None):
    """
    Input file.
    """
    pol = getPol(safe, orbit_path)
    bursts = load_bursts(safe, orbit_path, swath, pol)

    for bur in bursts:
        if int(bur.burst_id.subswath[2]) == swath:
            burst = bur

    info = Dummy()

    info.prf = 1 / burst.azimuth_time_interval
    info.startingRange = burst.starting_range
    info.rangePixelSize = burst.range_pixel_spacing
    info.wavelength = burst.wavelength

    if burst_shape:
        length, width = burst_shape
    else:
        length, width = burst.shape

    info.sensingStart = burst.sensing_start
    info.aztime = float((isce3.core.DateTime(burst.sensing_start) - burst.orbit.reference_epoch).total_seconds())
    info.sensingStop = info.sensingStart + timedelta(seconds=(length - 1.0) / info.prf)
    info.orbitname = orbit_path
    info.farRange = info.startingRange + (width - 1.0) * info.rangePixelSize
    info.lookSide = isce3.core.LookSide.Right
    info.startingRange -= buffer * info.rangePixelSize
    info.farRange += buffer * info.rangePixelSize

    if burst_shape:
        info.numberOfLines = burst_shape[0]
        info.numberOfSamples = burst_shape[1]
    else:
        info.numberOfLines = int(np.round((info.sensingStop - info.sensingStart).total_seconds() * info.prf)) + 1
        info.numberOfSamples = int(np.round((info.farRange - info.startingRange) / info.rangePixelSize)) + 1 + 2 * buffer

    info.orbit = getMergedOrbit(safe, orbit_path, swath)

    return info


def loadMetadataSlc(safe, orbit_path, buffer=0, swaths=None, slc_shape=None):
    """
    Input file.
    """
    if swaths is None:
        swaths = [1, 2, 3]

    pol = getPol(safe, orbit_path)

    info = Dummy()

    orbit_file = orbit_path
    total_width = 0
    bursts = []
    for swath in swaths:
        burstst = load_bursts(safe, orbit_file, swath, pol)
        bursts += burstst
        dt = bursts[0].azimuth_time_interval
        sensingStopt = burstst[-1].sensing_start + timedelta(seconds=(burstst[-1].shape[0] - 1) * dt)
        sensingStartt = burstst[0].sensing_start
        if swath == min(swaths):
            info.prf = 1 / burstst[0].azimuth_time_interval
            info.sensingStart = sensingStartt
            info.startingRange = burstst[0].starting_range
            info.rangePixelSize = burstst[0].range_pixel_spacing
            info.wavelength = burstst[0].wavelength
            info.sensingStop = sensingStopt
        if info.sensingStart > sensingStartt:
            info.sensingStart = sensingStartt
        if info.sensingStop < sensingStopt:
            info.sensingStop = sensingStopt

    if slc_shape:
        total_width = slc_shape[1]
    else:
        total_width = (
            int(np.round((bursts[-1].starting_range - bursts[0].starting_range) / bursts[0].range_pixel_spacing))
            + bursts[-1].shape[1]
        )

    info.aztime = float((isce3.core.DateTime(info.sensingStart) - bursts[0].orbit.reference_epoch).total_seconds())
    info.orbitname = orbit_path
    info.farRange = info.startingRange + (total_width - 1.0) * info.rangePixelSize
    info.lookSide = isce3.core.LookSide.Right
    info.startingRange -= buffer * info.rangePixelSize
    info.farRange += buffer * info.rangePixelSize

    if slc_shape:
        info.numberOfLines, info.numberOfSamples = slc_shape
    else:
        info.numberOfLines = int(np.round((info.sensingStop - info.sensingStart).total_seconds() * info.prf)) + 1
        info.numberOfSamples = int(np.round((info.farRange - info.startingRange) / info.rangePixelSize)) + 1 + 2 * buffer

    print('SIZE', info.numberOfLines, info.numberOfSamples)

    info.orbit = getMergedOrbit(safe, orbit_path, swaths[0])

    return info


def coregisterLoadMetadata(indir_r, indir_t):
    """
    Input file.
    """

    obj = GeogridOptical()

    x1a, y1a, xsize1, ysize1, x2a, y2a, xsize2, ysize2, trans = obj.coregister(indir_r, indir_t)

    DS = gdal.Open(indir_r, gdal.GA_ReadOnly)

    info = Dummy()
    info.startingX = trans[0]
    info.startingY = trans[3]
    info.XSize = trans[1]
    info.YSize = trans[5]

    if re.findall('L[CO]0[89]_', DS.GetDescription()).__len__() > 0:
        nameString = os.path.basename(DS.GetDescription())
        info.time = nameString.split('_')[3]
    elif re.findall('L[EO]07_', DS.GetDescription()).__len__() > 0:
        nameString = os.path.basename(DS.GetDescription())
        info.time = nameString.split('_')[3]
    elif re.findall('LT0[45]_', DS.GetDescription()).__len__() > 0:
        nameString = os.path.basename(DS.GetDescription())
        info.time = nameString.split('_')[3]
    elif re.findall('S2._', DS.GetDescription()).__len__() > 0:
        info.time = DS.GetDescription().split('_')[2]
    else:
        raise Exception('Optical data NOT supported yet!')

    info.numberOfLines = ysize1
    info.numberOfSamples = xsize1

    info.filename = indir_r

    DS1 = gdal.Open(indir_t, gdal.GA_ReadOnly)

    info1 = Dummy()

    if re.findall('L[CO]0[89]_', DS1.GetDescription()).__len__() > 0:
        nameString1 = os.path.basename(DS1.GetDescription())
        info1.time = nameString1.split('_')[3]
    elif re.findall('L[EO]07_', DS1.GetDescription()).__len__() > 0:
        nameString1 = os.path.basename(DS1.GetDescription())
        info1.time = nameString1.split('_')[3]
    elif re.findall('LT0[45]_', DS1.GetDescription()).__len__() > 0:
        nameString1 = os.path.basename(DS1.GetDescription())
        info1.time = nameString1.split('_')[3]
    elif re.findall('S2._', DS1.GetDescription()).__len__() > 0:
        info1.time = DS1.GetDescription().split('_')[2]
    else:
        raise Exception('Optical data NOT supported yet!')

    return info, info1


def runGeogrid(
    info, info1, dem, dhdx, dhdy, vx, vy, srx, sry, csminx, csminy, csmaxx, csmaxy, ssm, optical_flag=1, **kwargs
):
    """
    Wire and run geogrid.
    """

    if optical_flag:
        dem_info = gdal.Info(dem, format='json')

        obj = GeogridOptical()

        obj.startingX = info.startingX
        obj.startingY = info.startingY
        obj.XSize = info.XSize
        obj.YSize = info.YSize

        d0 = date(int(info.time[0:4]), int(info.time[4:6]), int(info.time[6:8]))
        d1 = date(int(info1.time[0:4]), int(info1.time[4:6]), int(info1.time[6:8]))
        date_dt_base = d1 - d0
        obj.repeatTime = date_dt_base.total_seconds()
        obj.numberOfLines = info.numberOfLines
        obj.numberOfSamples = info.numberOfSamples
        obj.nodata_out = -32767
        obj.chipSizeX0 = 240
        obj.gridSpacingX = dem_info['geoTransform'][1]

        obj.dat1name = info.filename
        obj.demname = dem
        obj.dhdxname = dhdx
        obj.dhdyname = dhdy
        obj.vxname = vx
        obj.vyname = vy
        obj.srxname = srx
        obj.sryname = sry
        obj.csminxname = csminx
        obj.csminyname = csminy
        obj.csmaxxname = csmaxx
        obj.csmaxyname = csmaxy
        obj.ssmname = ssm
        obj.winlocname = 'window_location.tif'
        obj.winoffname = 'window_offset.tif'
        obj.winsrname = 'window_search_range.tif'
        obj.wincsminname = 'window_chip_size_min.tif'
        obj.wincsmaxname = 'window_chip_size_max.tif'
        obj.winssmname = 'window_stable_surface_mask.tif'
        obj.winro2vxname = 'window_rdr_off2vel_x_vec.tif'
        obj.winro2vyname = 'window_rdr_off2vel_y_vec.tif'
        obj.winsfname = 'window_scale_factor.tif'
        # dt-varying search range scale (srs) routine parameters
        # obj.srs_dt_unity = 32
        # obj.srs_max_scale = 10
        # obj.srs_max_search = 20000
        # obj.srs_min_search = 0

        obj.runGeogrid()

        run_info = {
            'chipsizex0': obj.chipSizeX0,
            'gridspacingx': obj.gridSpacingX,
            'vxname': vx,
            'vyname': vy,
            'sxname': kwargs.get('dhdxs'),
            'syname': kwargs.get('dhdys'),
            'maskname': kwargs.get('sp'),
            'xoff': obj.pOff,
            'yoff': obj.lOff,
            'xcount': obj.pCount,
            'ycount': obj.lCount,
            'dt': obj.repeatTime,
            'epsg': kwargs.get('epsg'),
            'XPixelSize': obj.X_res,
            'YPixelSize': obj.Y_res,
            'cen_lat': obj.cen_lat,
            'cen_lon': obj.cen_lon,
        }

    else:
        dem_info = gdal.Info(dem, format='json')

        obj = GeogridRadar()

        obj.startingRange = info.startingRange
        obj.rangePixelSize = info.rangePixelSize
        obj.sensingStart = info.sensingStart
        obj.sensingStop = info.sensingStop
        obj.orbitname = info.orbitname
        obj.prf = info.prf
        obj.aztime = info.aztime
        obj.wavelength = info.wavelength
        obj.lookSide = info.lookSide
        obj.repeatTime = (info1.sensingStart - info.sensingStart).total_seconds()
        obj.numberOfLines = info.numberOfLines
        obj.numberOfSamples = info.numberOfSamples
        obj.nodata_out = -32767
        obj.chipSizeX0 = 240
        obj.gridSpacingX = dem_info['geoTransform'][1]
        obj.orbit = info.orbit
        obj.demname = dem
        obj.dhdxname = dhdx
        obj.dhdyname = dhdy
        obj.vxname = vx
        obj.vyname = vy
        obj.srxname = srx
        obj.sryname = sry
        obj.csminxname = csminx
        obj.csminyname = csminy
        obj.csmaxxname = csmaxx
        obj.csmaxyname = csmaxy
        obj.ssmname = ssm
        obj.winlocname = 'window_location.tif'
        obj.winoffname = 'window_offset.tif'
        obj.winsrname = 'window_search_range.tif'
        obj.wincsminname = 'window_chip_size_min.tif'
        obj.wincsmaxname = 'window_chip_size_max.tif'
        obj.winssmname = 'window_stable_surface_mask.tif'
        obj.winro2vxname = 'window_rdr_off2vel_x_vec.tif'
        obj.winro2vyname = 'window_rdr_off2vel_y_vec.tif'
        obj.winsfname = 'window_scale_factor.tif'
        # dt-varying search range scale (srs) rountine parameters
        # obj.srs_dt_unity = 5
        # obj.srs_max_scale = 10
        # obj.srs_max_search = 20000
        # obj.srs_min_search = 0

        obj.getIncidenceAngle()
        obj.geogridRadar()

        run_info = {
            'chipsizex0': obj.chipSizeX0,
            'gridspacingx': obj.gridSpacingX,
            'vxname': vx,
            'vyname': vy,
            'sxname': kwargs.get('dhdxs'),
            'syname': kwargs.get('dhdys'),
            'maskname': kwargs.get('sp'),
            'xoff': obj.pOff,
            'yoff': obj.lOff,
            'xcount': obj.pCount,
            'ycount': obj.lCount,
            'dt': obj.repeatTime,
            'epsg': kwargs.get('epsg'),
            'XPixelSize': obj.X_res,
            'YPixelSize': obj.Y_res,
            'cen_lat': obj.cen_lat,
            'cen_lon': obj.cen_lon,
        }

    return run_info


def main():
    """
    Main driver.
    """

    inps = cmdLineParse()

    metadata_m, metadata_s = coregisterLoadMetadata(inps.indir_r, inps.indir_t)

    runGeogrid(
        metadata_m,
        metadata_s,
        inps.demfile,
        inps.dhdxfile,
        inps.dhdyfile,
        inps.vxfile,
        inps.vyfile,
        inps.srxfile,
        inps.sryfile,
        inps.csminxfile,
        inps.csminyfile,
        inps.csmaxxfile,
        inps.csmaxyfile,
        inps.ssmfile,
    )


if __name__ == '__main__':
    main()
