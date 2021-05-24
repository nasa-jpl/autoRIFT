#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




import pdb
from osgeo import gdal, osr




def runCmd(cmd):
    import subprocess
    out = subprocess.getoutput(cmd)
    return out




def cmdLineParse():
    '''
    Command line parser.
    '''
    import argparse

    parser = argparse.ArgumentParser(description='Output geo grid')
    parser.add_argument('-m', '--input_m', dest='indir_m', type=str, required=True,
            help='Input master image file name (in ISCE format and radar coordinates) or Input master image file name (in GeoTIFF format and Cartesian coordinates)')
    parser.add_argument('-s', '--input_s', dest='indir_s', type=str, required=True,
            help='Input slave image file name (in ISCE format and radar coordinates) or Input slave image file name (in GeoTIFF format and Cartesian coordinates)')
    parser.add_argument('-g', '--input_g', dest='grid_location', type=str, required=False,
            help='Input pixel indices file name')
    parser.add_argument('-o', '--input_o', dest='init_offset', type=str, required=False,
            help='Input search center offsets ("downstream" reach location) file name')
    parser.add_argument('-sr', '--input_sr', dest='search_range', type=str, required=False,
            help='Input search range file name')
    parser.add_argument('-csmin', '--input_csmin', dest='chip_size_min', type=str, required=False,
            help='Input chip size min file name')
    parser.add_argument('-csmax', '--input_csmax', dest='chip_size_max', type=str, required=False,
            help='Input chip size max file name')
    parser.add_argument('-vx', '--input_vx', dest='offset2vx', type=str, required=False,
            help='Input pixel offsets to vx conversion coefficients file name')
    parser.add_argument('-vy', '--input_vy', dest='offset2vy', type=str, required=False,
            help='Input pixel offsets to vy conversion coefficients file name')
    parser.add_argument('-ssm', '--input_ssm', dest='stable_surface_mask', type=str, required=False,
            help='Input stable surface mask file name')
    parser.add_argument('-fo', '--flag_optical', dest='optical_flag', type=bool, required=False, default=0,
            help='flag for reading optical data (e.g. Landsat): use 1 for on and 0 (default) for off')
    parser.add_argument('-nc', '--sensor_flag_netCDF', dest='nc_sensor', type=str, required=False, default=None,
            help='flag for packaging output formatted for Sentinel ("S") and Landsat ("L") dataset; default is None')
    parser.add_argument('-mpflag', '--mpflag', dest='mpflag', type=int, required=False, default=0,
            help='number of threads for multiple threading (default is specified by 0, which uses the original single-core version and surpasses the multithreading routine)')
    parser.add_argument('-ncname', '--ncname', dest='ncname', type=str, required=False, default=None,
            help='User-defined filename for the NetCDF output to which the ROI percentage and the production version will be appended')

    return parser.parse_args()

class Dummy(object):
    pass



def loadProduct(filename):
    '''
    Load the product using Product Manager.
    '''
    import isce
    import logging
    from imageMath import IML

    IMG = IML.mmapFromISCE(filename, logging)
    img = IMG.bands[0]
#    pdb.set_trace()
    return img


def loadProductOptical(file_m, file_s):
    import numpy as np
    '''
    Load the product using Product Manager.
    '''
    import isce
    from components.contrib.geo_autoRIFT.geogrid import GeogridOptical
#    from geogrid import GeogridOptical

    obj = GeogridOptical()

    x1a, y1a, xsize1, ysize1, x2a, y2a, xsize2, ysize2, trans = obj.coregister(file_m, file_s)

    DS1 = gdal.Open(file_m)
    DS2 = gdal.Open(file_s)

    I1 = DS1.ReadAsArray(xoff=x1a, yoff=y1a, xsize=xsize1, ysize=ysize1)
    I2 = DS2.ReadAsArray(xoff=x2a, yoff=y2a, xsize=xsize2, ysize=ysize2)

    I1 = I1.astype(np.float32)
    I2 = I2.astype(np.float32)

    DS1=None
    DS2=None

    return I1, I2


def runAutorift(I1, I2, xGrid, yGrid, Dx0, Dy0, SRx0, SRy0, CSMINx0, CSMINy0, CSMAXx0, CSMAXy0, noDataMask, optflag,
                nodata, mpflag, geogrid_run_info=None):
    '''
    Wire and run geogrid.
    '''

    import isce
    from components.contrib.geo_autoRIFT.autoRIFT import autoRIFT_ISCE
    import numpy as np
    import isceobj
    import time
    import subprocess


    obj = autoRIFT_ISCE()
    obj.configure()

##########     uncomment if starting from preprocessed images
#    I1 = I1.astype(np.uint8)
#    I2 = I2.astype(np.uint8)

    obj.MultiThread = mpflag

    # take the amplitude only for the radar images
    if optflag == 0:
        I1 = np.abs(I1)
        I2 = np.abs(I2)

    obj.I1 = I1
    obj.I2 = I2

    # test with lena image (533 X 533)
#    obj.ChipSizeMinX=16
#    obj.ChipSizeMaxX=32
#    obj.ChipSize0X=16
#    obj.SkipSampleX=16
#    obj.SkipSampleY=16

    # test with Venus image (407 X 407)
#    obj.ChipSizeMinX=8
#    obj.ChipSizeMaxX=16
#    obj.ChipSize0X=8
#    obj.SkipSampleX=8
#    obj.SkipSampleY=8

    # test with small tiff images
#    obj.SkipSampleX=16
#    obj.SkipSampleY=16

    # create the grid if it does not exist
    if xGrid is None:
        m,n = obj.I1.shape
        xGrid = np.arange(obj.SkipSampleX+10,n-obj.SkipSampleX,obj.SkipSampleX)
        yGrid = np.arange(obj.SkipSampleY+10,m-obj.SkipSampleY,obj.SkipSampleY)
        nd = xGrid.__len__()
        md = yGrid.__len__()
        obj.xGrid = np.int32(np.dot(np.ones((md,1)),np.reshape(xGrid,(1,xGrid.__len__()))))
        obj.yGrid = np.int32(np.dot(np.reshape(yGrid,(yGrid.__len__(),1)),np.ones((1,nd))))
        noDataMask = np.logical_not(obj.xGrid)
    else:
        obj.xGrid = xGrid
        obj.yGrid = yGrid



    # generate the nodata mask where offset searching will be skipped based on 1) imported nodata mask and/or 2) zero values in the image
    for ii in range(obj.xGrid.shape[0]):
        for jj in range(obj.xGrid.shape[1]):
            if (obj.yGrid[ii,jj] != nodata)&(obj.xGrid[ii,jj] != nodata):
                if (I1[obj.yGrid[ii,jj]-1,obj.xGrid[ii,jj]-1]==0)|(I2[obj.yGrid[ii,jj]-1,obj.xGrid[ii,jj]-1]==0):
                    noDataMask[ii,jj] = True




    ######### mask out nodata to skip the offset searching using the nodata mask (by setting SearchLimit to be 0)

    if SRx0 is None:
#        ###########     uncomment to customize SearchLimit based on velocity distribution (i.e. Dx0 must not be None)
#        obj.SearchLimitX = np.int32(4+(25-4)/(np.max(np.abs(Dx0[np.logical_not(noDataMask)]))-np.min(np.abs(Dx0[np.logical_not(noDataMask)])))*(np.abs(Dx0)-np.min(np.abs(Dx0[np.logical_not(noDataMask)]))))
#        obj.SearchLimitY = 5
#        ###########
        obj.SearchLimitX = obj.SearchLimitX * np.logical_not(noDataMask)
        obj.SearchLimitY = obj.SearchLimitY * np.logical_not(noDataMask)
    else:
        obj.SearchLimitX = SRx0
        obj.SearchLimitY = SRy0
#        ############ add buffer to search range
#        obj.SearchLimitX[obj.SearchLimitX!=0] = obj.SearchLimitX[obj.SearchLimitX!=0] + 2
#        obj.SearchLimitY[obj.SearchLimitY!=0] = obj.SearchLimitY[obj.SearchLimitY!=0] + 2

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
        
        obj.ChipSize0X = int(np.ceil(chipsizex0/pixsizex/4)*4)
        obj.GridSpacingX = int(obj.ChipSize0X*gridspacingx/chipsizex0)

#        obj.ChipSize0X = np.min(CSMINx0[CSMINx0!=nodata])
        RATIO_Y2X = CSMINy0/CSMINx0
        obj.ScaleChipSizeY = np.median(RATIO_Y2X[(CSMINx0!=nodata)&(CSMINy0!=nodata)])
#        obj.ChipSizeMaxX = obj.ChipSizeMaxX / obj.ChipSizeMaxX * 544
#        obj.ChipSizeMinX = obj.ChipSizeMinX / obj.ChipSizeMinX * 68
    else:
        if ((optflag == 1)&(xGrid is not None)):
#            print('test')
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



    ######## preprocessing
    t1 = time.time()
    print("Pre-process Start!!!")
#    obj.zeroMask = 1
    obj.preprocess_filt_hps()
#    obj.I1 = np.abs(I1)
#    obj.I2 = np.abs(I2)
    print("Pre-process Done!!!")
    print(time.time()-t1)

    t1 = time.time()
#    obj.DataType = 0
    obj.uniform_data_type()
    print("Uniform Data Type Done!!!")
    print(time.time()-t1)

#    pdb.set_trace()

#    obj.sparseSearchSampleRate = 16

    obj.OverSampleRatio = 64
#    obj.colfiltChunkSize = 4

    #   OverSampleRatio can be assigned as a scalar (such as the above line) or as a Python dictionary below for intellgient use (ChipSize-dependent).
    #   Here, four chip sizes are used: ChipSize0X*[1,2,4,8] and four OverSampleRatio are considered [16,32,64,128]. The intelligent selection of OverSampleRatio (as a function of chip size) was determined by analyzing various combinations of (OverSampleRatio and chip size) and comparing the resulting image quality and statistics with the reference scenario (where the largest OverSampleRatio of 128 and chip size of ChipSize0X*8 are considered).
    #   The selection for the optical data flag is based on Landsat-8 data over an inland region (thus stable and not moving much) of Greenland, while that for the radar flag (optflag = 0) is based on Sentinel-1 data over the same region of Greenland.
    if CSMINx0 is not None:
        if (optflag == 1):
            obj.OverSampleRatio = {obj.ChipSize0X:16,obj.ChipSize0X*2:32,obj.ChipSize0X*4:64,obj.ChipSize0X*8:64}
        else:
            obj.OverSampleRatio = {obj.ChipSize0X:32,obj.ChipSize0X*2:64,obj.ChipSize0X*4:128,obj.ChipSize0X*8:128}



#    ########## export preprocessed images to files; can be commented out if not debugging
#
#    t1 = time.time()
#
#    I1 = obj.I1
#    I2 = obj.I2
#
#    length,width = I1.shape
#
#    filename1 = 'I1_uint8_hpsnew.off'
#
#    slcFid = open(filename1, 'wb')
#
#    for yy in range(length):
#        data = I1[yy,:]
#        data.astype(np.float32).tofile(slcFid)
#
#    slcFid.close()
#
#    img = isceobj.createOffsetImage()
#    img.setFilename(filename1)
#    img.setBands(1)
#    img.setWidth(width)
#    img.setLength(length)
#    img.setAccessMode('READ')
#    img.renderHdr()
#
#
#    filename2 = 'I2_uint8_hpsnew.off'
#
#    slcFid = open(filename2, 'wb')
#
#    for yy in range(length):
#        data = I2[yy,:]
#        data.astype(np.float32).tofile(slcFid)
#
#    slcFid.close()
#
#    img = isceobj.createOffsetImage()
#    img.setFilename(filename2)
#    img.setBands(1)
#    img.setWidth(width)
#    img.setLength(length)
#    img.setAccessMode('READ')
#    img.renderHdr()
#
#    print("output Done!!!")
#    print(time.time()-t1)


    ########## run Autorift
    t1 = time.time()
    print("AutoRIFT Start!!!")
    obj.runAutorift()
    print("AutoRIFT Done!!!")
    print(time.time()-t1)

    import cv2
    kernel = np.ones((3,3),np.uint8)
    noDataMask = cv2.dilate(noDataMask.astype(np.uint8),kernel,iterations = 1)
    noDataMask = noDataMask.astype(np.bool)


    return obj.Dx, obj.Dy, obj.InterpMask, obj.ChipSizeX, obj.ScaleChipSizeY, obj.SearchLimitX, obj.SearchLimitY, obj.origSize, noDataMask



def main():
    '''
    Main driver.
    '''
    inps = cmdLineParse()

    generateAutoriftProduct(indir_m=inps.indir_m, indir_s=inps.indir_s, grid_location=inps.grid_location,
                            init_offset=inps.init_offset, search_range=inps.search_range,
                            chip_size_min=inps.chip_size_min,chip_size_max=inps.chip_size_max,
                            offset2vx=inps.offset2vx, offset2vy=inps.offset2vy,
                            stable_surface_mask=inps.stable_surface_mask, optical_flag=inps.optical_flag,
                            nc_sensor=inps.nc_sensor, mpflag=inps.mpflag, ncname=inps.ncname)


def generateAutoriftProduct(indir_m, indir_s, grid_location, init_offset, search_range, chip_size_min, chip_size_max,
                            offset2vx, offset2vy, stable_surface_mask, optical_flag, nc_sensor, mpflag, ncname,
                            geogrid_run_info=None):

    import numpy as np
    import time

    import isce
    from components.contrib.geo_autoRIFT.autoRIFT import __version__ as version
    #  from autoRIFT import __version__ as version

    if optical_flag == 1:
        data_m, data_s = loadProductOptical(indir_m, indir_s)
        # test with lena/Venus image
#        import scipy.io as sio
#        conts = sio.loadmat(indir_m)
#        data_m = conts['I']
#        data_s = conts['I1']
    else:
        data_m = loadProduct(indir_m)
        data_s = loadProduct(indir_s)



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
        noDataMask = (xGrid == nodata)
        band = ds.GetRasterBand(2)
        yGrid = band.ReadAsArray()
        band=None
        ds=None

    if init_offset is not None:
        ds = gdal.Open(init_offset)
        band = ds.GetRasterBand(1)
        Dx0 = band.ReadAsArray()
        band = ds.GetRasterBand(2)
        Dy0 = band.ReadAsArray()
        band=None
        ds=None

    if search_range is not None:
        ds = gdal.Open(search_range)
        band = ds.GetRasterBand(1)
        SRx0 = band.ReadAsArray()
        band = ds.GetRasterBand(2)
        SRy0 = band.ReadAsArray()
        band=None
        ds=None

    if chip_size_min is not None:
        ds = gdal.Open(chip_size_min)
        band = ds.GetRasterBand(1)
        CSMINx0 = band.ReadAsArray()
        band = ds.GetRasterBand(2)
        CSMINy0 = band.ReadAsArray()
        band=None
        ds=None

    if chip_size_max is not None:
        ds = gdal.Open(chip_size_max)
        band = ds.GetRasterBand(1)
        CSMAXx0 = band.ReadAsArray()
        band = ds.GetRasterBand(2)
        CSMAXy0 = band.ReadAsArray()
        band=None
        ds=None

    if stable_surface_mask is not None:
        ds = gdal.Open(stable_surface_mask)
        band = ds.GetRasterBand(1)
        SSM = band.ReadAsArray()
#        SSM = SSM * 0
        SSM = SSM.astype('bool')
        band=None
        ds=None



    Dx, Dy, InterpMask, ChipSizeX, ScaleChipSizeY, SearchLimitX, SearchLimitY, origSize, noDataMask = runAutorift(
        data_m, data_s, xGrid, yGrid, Dx0, Dy0, SRx0, SRy0, CSMINx0, CSMINy0, CSMAXx0, CSMAXy0,
        noDataMask, optical_flag, nodata, mpflag, geogrid_run_info=geogrid_run_info,
    )

    if optical_flag == 0:
        Dy = -Dy

    DX = np.zeros(origSize,dtype=np.float32) * np.nan
    DY = np.zeros(origSize,dtype=np.float32) * np.nan
    INTERPMASK = np.zeros(origSize,dtype=np.float32)
    CHIPSIZEX = np.zeros(origSize,dtype=np.float32)
    SEARCHLIMITX = np.zeros(origSize,dtype=np.float32)
    SEARCHLIMITY = np.zeros(origSize,dtype=np.float32)

    DX[0:Dx.shape[0],0:Dx.shape[1]] = Dx
    DY[0:Dy.shape[0],0:Dy.shape[1]] = Dy
    INTERPMASK[0:InterpMask.shape[0],0:InterpMask.shape[1]] = InterpMask
    CHIPSIZEX[0:ChipSizeX.shape[0],0:ChipSizeX.shape[1]] = ChipSizeX
    SEARCHLIMITX[0:SearchLimitX.shape[0],0:SearchLimitX.shape[1]] = SearchLimitX
    SEARCHLIMITY[0:SearchLimitY.shape[0],0:SearchLimitY.shape[1]] = SearchLimitY

    DX[noDataMask] = np.nan
    DY[noDataMask] = np.nan
    INTERPMASK[noDataMask] = 0
    CHIPSIZEX[noDataMask] = 0
    SEARCHLIMITX[noDataMask] = 0
    SEARCHLIMITY[noDataMask] = 0
    if SSM is not None:
        SSM[noDataMask] = False

    import scipy.io as sio
    sio.savemat('offset.mat',{'Dx':DX,'Dy':DY,'InterpMask':INTERPMASK,'ChipSizeX':CHIPSIZEX})

#    #####################  Uncomment for debug mode
#    sio.savemat('debug.mat',{'Dx':DX,'Dy':DY,'InterpMask':INTERPMASK,'ChipSizeX':CHIPSIZEX,'ScaleChipSizeY':ScaleChipSizeY,'SearchLimitX':SEARCHLIMITX,'SearchLimitY':SEARCHLIMITY})
#    conts = sio.loadmat('debug.mat')
#    DX = conts['Dx']
#    DY = conts['Dy']
#    INTERPMASK = conts['InterpMask']
#    CHIPSIZEX = conts['ChipSizeX']
#    ScaleChipSizeY = conts['ScaleChipSizeY']
#    SEARCHLIMITX = conts['SearchLimitX']
#    SEARCHLIMITY = conts['SearchLimitY']
#    #####################

    netcdf_file = None
    if grid_location is not None:


        t1 = time.time()
        print("Write Outputs Start!!!")


        # Create the GeoTiff
        driver = gdal.GetDriverByName('GTiff')

        outRaster = driver.Create("offset.tif", int(xGrid.shape[1]), int(xGrid.shape[0]), 4, gdal.GDT_Float32)
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


        if offset2vx is not None:

            ds = gdal.Open(offset2vx)
            band = ds.GetRasterBand(1)
            offset2vx_1 = band.ReadAsArray()
            band = ds.GetRasterBand(2)
            offset2vx_2 = band.ReadAsArray()
            band=None
            ds=None
            offset2vx_1[offset2vx_1 == nodata] = np.nan
            offset2vx_2[offset2vx_2 == nodata] = np.nan

            ds = gdal.Open(offset2vy)
            band = ds.GetRasterBand(1)
            offset2vy_1 = band.ReadAsArray()
            band = ds.GetRasterBand(2)
            offset2vy_2 = band.ReadAsArray()
            band=None
            ds=None
            offset2vy_1[offset2vy_1 == nodata] = np.nan
            offset2vy_2[offset2vy_2 == nodata] = np.nan

            VX = offset2vx_1 * DX + offset2vx_2 * DY
            VY = offset2vy_1 * DX + offset2vy_2 * DY
            VX = VX.astype(np.float32)
            VY = VY.astype(np.float32)

            ############ write velocity output in Geotiff format

            outRaster = driver.Create("velocity.tif", int(xGrid.shape[1]), int(xGrid.shape[0]), 2, gdal.GDT_Float32)
            outRaster.SetGeoTransform(tran)
            outRaster.SetProjection(proj)
            outband = outRaster.GetRasterBand(1)
            outband.WriteArray(VX)
            outband.FlushCache()
            outband = outRaster.GetRasterBand(2)
            outband.WriteArray(VY)
            outband.FlushCache()

            ############ prepare for netCDF packaging

            if nc_sensor is not None:
                if geogrid_run_info is None:
                    vxrefname = str.split(runCmd('fgrep "Velocities:" testGeogrid.txt'))[1]
                    vyrefname = str.split(runCmd('fgrep "Velocities:" testGeogrid.txt'))[2]
                    sxname = str.split(runCmd('fgrep "Slopes:" testGeogrid.txt'))[1][:-4]+"s.tif"
                    syname = str.split(runCmd('fgrep "Slopes:" testGeogrid.txt'))[2][:-4]+"s.tif"
                    maskname = str.split(runCmd('fgrep "Slopes:" testGeogrid.txt'))[2][:-8]+"sp.tif"
                    xoff = int(str.split(runCmd('fgrep "Origin index (in DEM) of geogrid:" testGeogrid.txt'))[6])
                    yoff = int(str.split(runCmd('fgrep "Origin index (in DEM) of geogrid:" testGeogrid.txt'))[7])
                    xcount = int(str.split(runCmd('fgrep "Dimensions of geogrid:" testGeogrid.txt'))[3])
                    ycount = int(str.split(runCmd('fgrep "Dimensions of geogrid:" testGeogrid.txt'))[5])
                    cen_lat = int(100*float(str.split(runCmd('fgrep "Scene-center lat/lon:" testGeogrid.txt'))[2]))/100
                    cen_lon = int(100*float(str.split(runCmd('fgrep "Scene-center lat/lon:" testGeogrid.txt'))[3]))/100
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
                    cen_lat = int(100*geogrid_run_info['cen_lat'])/100
                    cen_lon = int(100*geogrid_run_info['cen_lon'])/100

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

                DXref = offset2vy_2 / (offset2vx_1 * offset2vy_2 - offset2vx_2 * offset2vy_1) * VXref - offset2vx_2 / (offset2vx_1 * offset2vy_2 - offset2vx_2 * offset2vy_1) * VYref
                DYref = offset2vx_1 / (offset2vx_1 * offset2vy_2 - offset2vx_2 * offset2vy_1) * VYref - offset2vy_1 / (offset2vx_1 * offset2vy_2 - offset2vx_2 * offset2vy_1) * VXref

                stable_count = np.sum(SSM & np.logical_not(np.isnan(DX)) & (DX-DXref > -5) & (DX-DXref < 5) & (DY-DYref > -5) & (DY-DYref < 5))
                
                V_temp = np.sqrt(VX**2 + VY**2)
                try:
                    V_temp_threshold = np.percentile(V_temp[np.logical_not(np.isnan(V_temp))],25)
                    SSM1 = (V_temp <= V_temp_threshold)
                except IndexError:
                    SSM1 = np.zeros(V_temp.shape).astype('bool')
                
                stable_count1 = np.sum(SSM1 & np.logical_not(np.isnan(DX)) & (DX-DXref > -5) & (DX-DXref < 5) & (DY-DYref > -5) & (DY-DYref < 5))

                dx_mean_shift = 0.0
                dy_mean_shift = 0.0
                dx_mean_shift1 = 0.0
                dy_mean_shift1 = 0.0
                            
                if stable_count != 0:
                    temp = DX.copy() - DXref.copy()
                    temp[np.logical_not(SSM)] = np.nan
                    dx_mean_shift = np.median(temp[(temp > -5)&(temp < 5)])
                    
                    temp = DY.copy() - DYref.copy()
                    temp[np.logical_not(SSM)] = np.nan
                    dy_mean_shift = np.median(temp[(temp > -5)&(temp < 5)])
                    
                if stable_count1 != 0:
                    temp = DX.copy() - DXref.copy()
                    temp[np.logical_not(SSM1)] = np.nan
                    dx_mean_shift1 = np.median(temp[(temp > -5)&(temp < 5)])
                    
                    temp = DY.copy() - DYref.copy()
                    temp[np.logical_not(SSM1)] = np.nan
                    dy_mean_shift1 = np.median(temp[(temp > -5)&(temp < 5)])
                
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
                

                VX = offset2vx_1 * DX + offset2vx_2 * DY
                VY = offset2vy_1 * DX + offset2vy_2 * DY
                VX = VX.astype(np.float32)
                VY = VY.astype(np.float32)

            ########################################################################################
                ############   netCDF packaging for Sentinel and Landsat dataset; can add other sensor format as well
                if nc_sensor == "S":
                    if geogrid_run_info is None:
                        chipsizex0 = float(str.split(runCmd('fgrep "Smallest Allowable Chip Size in m:" testGeogrid.txt'))[-1])
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

                    runCmd('topsinsar_filename.py')
    #                import scipy.io as sio
                    conts = sio.loadmat('topsinsar_filename.mat')
                    master_filename = conts['master_filename'][0]
                    slave_filename = conts['slave_filename'][0]
                    master_dt = conts['master_dt'][0]
                    slave_dt = conts['slave_dt'][0]
                    master_split = str.split(master_filename,'_')
                    slave_split = str.split(slave_filename,'_')

                    import netcdf_output as no
                    pair_type = 'radar'
                    detection_method = 'feature'
                    coordinates = 'radar'
                    roi_valid_percentage = int(round(np.sum(CHIPSIZEX!=0)/np.sum(SEARCHLIMITX!=0)*1000.0))/1000
    #                out_nc_filename = 'Jakobshavn.nc'
                    PPP = roi_valid_percentage * 100
                    if ncname is None:
                        out_nc_filename = f"./{master_filename[0:-4]}_X_{slave_filename[0:-4]}" \
                                          f"_G{gridspacingx:04.0f}V02_P{np.floor(PPP):03.0f}.nc"
                    else:
                        out_nc_filename = f"{ncname}_G{gridspacingx:04.0f}V02_P{np.floor(PPP):03.0f}.nc"
                    CHIPSIZEY = np.round(CHIPSIZEX * ScaleChipSizeY / 2) * 2



                    from datetime import date
                    d0 = date(np.int(master_split[5][0:4]),np.int(master_split[5][4:6]),np.int(master_split[5][6:8]))
                    d1 = date(np.int(slave_split[5][0:4]),np.int(slave_split[5][4:6]),np.int(slave_split[5][6:8]))
                    date_dt_base = d1 - d0
                    date_dt = np.float64(date_dt_base.days)
                    if date_dt < 0:
                        raise Exception('Input image 1 must be older than input image 2')
                    if date_dt_base.days < 0:
                        date_ct = d1 + (d0 - d1)/2
                        date_center = date_ct.strftime("%Y%m%d")
                    else:
                        date_ct = d0 + (d1 - d0)/2
                        date_center = date_ct.strftime("%Y%m%d")

                    IMG_INFO_DICT = {'mission_img1':master_split[0][0],'sensor_img1':'C','satellite_img1':master_split[0][1:3],'acquisition_img1':master_dt,'time_standard_img1':'UTC','absolute_orbit_number_img1':master_split[7],'mission_data_take_ID_img1':master_split[8],'product_unique_ID_img1':master_split[9][0:4],'mission_img2':slave_split[0][0],'sensor_img2':'C','satellite_img2':slave_split[0][1:3],'acquisition_img2':slave_dt,'time_standard_img2':'UTC','absolute_orbit_number_img2':slave_split[7],'mission_data_take_ID_img2':slave_split[8],'product_unique_ID_img2':slave_split[9][0:4],'date_dt':date_dt,'date_center':date_center,'latitude':cen_lat,'longitude':cen_lon,'roi_valid_percentage':PPP,'autoRIFT_software_version':version}
                    error_vector = np.array([[0.0356, 0.0501, 0.0266, 0.0622, 0.0357, 0.0501],
                                             [0.5194, 1.1638, 0.3319, 1.3701, 0.5191, 1.1628]])

                    netcdf_file = no.netCDF_packaging(
                        VX, VY, DX, DY, INTERPMASK, CHIPSIZEX, CHIPSIZEY, SSM, SSM1, SX, SY,
                        offset2vx_1, offset2vx_2, offset2vy_1, offset2vy_2, MM, VXref, VYref,
                        rangePixelSize, azimuthPixelSize, dt, epsg, srs, tran, out_nc_filename, pair_type,
                        detection_method, coordinates, IMG_INFO_DICT, stable_count, stable_count1, stable_shift_applied,
                        dx_mean_shift, dy_mean_shift, dx_mean_shift1, dy_mean_shift1, error_vector
                    )

                elif nc_sensor == "L":
                    if geogrid_run_info is None:
                        chipsizex0 = float(str.split(runCmd('fgrep "Smallest Allowable Chip Size in m:" testGeogrid.txt'))[-1])
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

                    master_path = indir_m
                    slave_path = indir_s

                    import os
                    master_filename = os.path.basename(master_path)
                    slave_filename = os.path.basename(slave_path)

                    master_split = str.split(master_filename,'_')
                    slave_split = str.split(slave_filename,'_')

#                    master_MTL_path = master_path[:-6]+'MTL.txt'
#                    slave_MTL_path = slave_path[:-6]+'MTL.txt'
#
#                    master_time = str.split(str.split(runCmd('fgrep "SCENE_CENTER_TIME" '+master_MTL_path))[2][1:-2],':')
#                    slave_time = str.split(str.split(runCmd('fgrep "SCENE_CENTER_TIME" '+slave_MTL_path))[2][1:-2],':')
                    master_time = ['00','00','00']
                    slave_time = ['00','00','00']

                    from datetime import time as time1
                    master_time = time1(int(master_time[0]),int(master_time[1]),int(float(master_time[2])))
                    slave_time = time1(int(slave_time[0]),int(slave_time[1]),int(float(slave_time[2])))

                    import netcdf_output as no
                    pair_type = 'optical'
                    detection_method = 'feature'
                    coordinates = 'map'
                    roi_valid_percentage = int(round(np.sum(CHIPSIZEX!=0)/np.sum(SEARCHLIMITX!=0)*1000.0))/1000
    #                out_nc_filename = 'Jakobshavn_opt.nc'
                    PPP = roi_valid_percentage * 100
                    if ncname is None:
                        out_nc_filename = f"./{master_filename[0:-8]}_X_{slave_filename[0:-8]}" \
                                          f"_G{gridspacingx:04.0f}V02_P{np.floor(PPP):03.0f}.nc"
                    else:
                        out_nc_filename = f"{ncname}_G{gridspacingx:04.0f}V02_P{np.floor(PPP):03.0f}.nc"
                    CHIPSIZEY = np.round(CHIPSIZEX * ScaleChipSizeY / 2) * 2

                    from datetime import date
                    d0 = date(np.int(master_split[3][0:4]),np.int(master_split[3][4:6]),np.int(master_split[3][6:8]))
                    d1 = date(np.int(slave_split[3][0:4]),np.int(slave_split[3][4:6]),np.int(slave_split[3][6:8]))
                    date_dt_base = d1 - d0
                    date_dt = np.float64(date_dt_base.days)
                    if date_dt < 0:
                        raise Exception('Input image 1 must be older than input image 2')
                    if date_dt_base.days < 0:
                        date_ct = d1 + (d0 - d1)/2
                        date_center = date_ct.strftime("%Y%m%d")
                    else:
                        date_ct = d0 + (d1 - d0)/2
                        date_center = date_ct.strftime("%Y%m%d")

                    master_dt = master_split[3][0:8] + master_time.strftime("T%H:%M:%S")
                    slave_dt = slave_split[3][0:8] + slave_time.strftime("T%H:%M:%S")

                    IMG_INFO_DICT = {'mission_img1':master_split[0][0],'sensor_img1':master_split[0][1],'satellite_img1':np.float64(master_split[0][2:4]),'correction_level_img1':master_split[1],'path_img1':np.float64(master_split[2][0:3]),'row_img1':np.float64(master_split[2][3:6]),'acquisition_date_img1':master_dt,'time_standard_img1':'UTC','processing_date_img1':master_split[4][0:8],'collection_number_img1':np.float64(master_split[5]),'collection_category_img1':master_split[6],'mission_img2':slave_split[0][0],'sensor_img2':slave_split[0][1],'satellite_img2':np.float64(slave_split[0][2:4]),'correction_level_img2':slave_split[1],'path_img2':np.float64(slave_split[2][0:3]),'row_img2':np.float64(slave_split[2][3:6]),'acquisition_date_img2':slave_dt,'time_standard_img2':'UTC','processing_date_img2':slave_split[4][0:8],'collection_number_img2':np.float64(slave_split[5]),'collection_category_img2':slave_split[6],'date_dt':date_dt,'date_center':date_center,'latitude':cen_lat,'longitude':cen_lon,'roi_valid_percentage':PPP,'autoRIFT_software_version':version}

                    error_vector = np.array([25.5,25.5])

                    netcdf_file = no.netCDF_packaging(
                        VX, VY, DX, DY, INTERPMASK, CHIPSIZEX, CHIPSIZEY, SSM, SSM1, SX, SY,
                        offset2vx_1, offset2vx_2, offset2vy_1, offset2vy_2, MM, VXref, VYref,
                        XPixelSize, YPixelSize, None, epsg, srs, tran, out_nc_filename, pair_type,
                        detection_method, coordinates, IMG_INFO_DICT, stable_count, stable_count1, stable_shift_applied,
                        dx_mean_shift, dy_mean_shift, dx_mean_shift1, dy_mean_shift1, error_vector
                    )

                elif nc_sensor == "S2":
                    if geogrid_run_info is None:
                        chipsizex0 = float(str.split(runCmd('fgrep "Smallest Allowable Chip Size in m:" testGeogrid.txt'))[-1])
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

                    master_path = indir_m
                    slave_path = indir_s

                    master_split = master_path.split('_')
                    slave_split = slave_path.split('_')

                    import os

                    master_filename = master_split[0][-3:]+'_'+master_split[2]+'_'+master_split[4][:3]+'_'+os.path.basename(master_path)
                    slave_filename = slave_split[0][-3:]+'_'+slave_split[2]+'_'+slave_split[4][:3]+'_'+os.path.basename(slave_path)

                    master_time = ['00','00','00']
                    slave_time = ['00','00','00']

                    from datetime import time as time1
                    master_time = time1(int(master_time[0]),int(master_time[1]),int(float(master_time[2])))
                    slave_time = time1(int(slave_time[0]),int(slave_time[1]),int(float(slave_time[2])))

                    import netcdf_output as no
                    pair_type = 'optical'
                    detection_method = 'feature'
                    coordinates = 'map'
                    roi_valid_percentage = int(round(np.sum(CHIPSIZEX!=0)/np.sum(SEARCHLIMITX!=0)*1000.0))/1000
                    PPP = roi_valid_percentage * 100
                    if ncname is None:
                        out_nc_filename = f"./{master_filename[0:-8]}_X_{slave_filename[0:-8]}" \
                                          f"_G{gridspacingx:04.0f}V02_P{np.floor(PPP):03.0f}.nc"
                    else:
                        out_nc_filename = f"{ncname}_G{gridspacingx:04.0f}V02_P{np.floor(PPP):03.0f}.nc"
                    CHIPSIZEY = np.round(CHIPSIZEX * ScaleChipSizeY / 2) * 2

                    from datetime import date
                    d0 = date(np.int(master_split[2][0:4]),np.int(master_split[2][4:6]),np.int(master_split[2][6:8]))
                    d1 = date(np.int(slave_split[2][0:4]),np.int(slave_split[2][4:6]),np.int(slave_split[2][6:8]))
                    date_dt_base = d1 - d0
                    date_dt = np.float64(date_dt_base.days)
                    if date_dt < 0:
                        raise Exception('Input image 1 must be older than input image 2')
                    if date_dt_base.days < 0:
                        date_ct = d1 + (d0 - d1)/2
                        date_center = date_ct.strftime("%Y%m%d")
                    else:
                        date_ct = d0 + (d1 - d0)/2
                        date_center = date_ct.strftime("%Y%m%d")

                    master_dt = master_split[2] + master_time.strftime("T%H:%M:%S")
                    slave_dt = slave_split[2] + slave_time.strftime("T%H:%M:%S")

                    IMG_INFO_DICT = {'mission_img1':master_split[0][-3],'satellite_img1':master_split[0][-2:],'correction_level_img1':master_split[4][:3],'acquisition_date_img1':master_dt,'time_standard_img1':'UTC','mission_img2':slave_split[0][-3],'satellite_img2':slave_split[0][-2:],'correction_level_img2':slave_split[4][:3],'acquisition_date_img2':slave_dt,'time_standard_img2':'UTC','date_dt':date_dt,'date_center':date_center,'latitude':cen_lat,'longitude':cen_lon,'roi_valid_percentage':PPP,'autoRIFT_software_version':version}

                    error_vector = np.array([25.5,25.5])

                    netcdf_file = no.netCDF_packaging(
                        VX, VY, DX, DY, INTERPMASK, CHIPSIZEX, CHIPSIZEY, SSM, SSM1, SX, SY,
                        offset2vx_1, offset2vx_2, offset2vy_1, offset2vy_2, MM, VXref, VYref,
                        XPixelSize, YPixelSize, None, epsg, srs, tran, out_nc_filename, pair_type,
                        detection_method, coordinates, IMG_INFO_DICT, stable_count, stable_count1, stable_shift_applied,
                        dx_mean_shift, dy_mean_shift, dx_mean_shift1, dy_mean_shift1, error_vector
                    )

                elif nc_sensor is None:
                    print('netCDF packaging not performed')

                else:
                    raise Exception('netCDF packaging not supported for the type "{0}"'.format(nc_sensor))

        print("Write Outputs Done!!!")
        print(time.time()-t1)

    return netcdf_file


if __name__ == '__main__':
    main()
