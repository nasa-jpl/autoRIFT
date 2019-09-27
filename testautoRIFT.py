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

def cmdLineParse():
    '''
    Command line parser.
    '''
    import argparse

    parser = argparse.ArgumentParser(description='Output geo grid')
    parser.add_argument('-m', '--input_m', dest='indir_m', type=str, required=True,
            help='Input ISCE master image file name')
    parser.add_argument('-s', '--input_s', dest='indir_s', type=str, required=True,
            help='Input ISCE slave image file name')
    parser.add_argument('-g', '--input_g', dest='grid_location', type=str, required=False,
            help='Input pixel indices file name')
    parser.add_argument('-o', '--input_o', dest='init_offset', type=str, required=False,
            help='Input search center offsets ("downstream" reach location) file name')
    parser.add_argument('-vx', '--input_vx', dest='offset2vx', type=str, required=False,
            help='Input pixel offsets to vx conversion coefficients file name')
    parser.add_argument('-vy', '--input_vy', dest='offset2vy', type=str, required=False,
            help='Input pixel offsets to vy conversion coefficients file name')
    parser.add_argument('-fo', '--flag_optical', dest='optical_flag', type=bool, required=False, default=0,
                        help='flag for reading optical data (e.g. Landsat): use 1 for on and 0 (default) for off')


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


def loadProductOptical(filename):
    import numpy as np
    '''
    Load the product using Product Manager.
    '''
    ds = gdal.Open(filename)
#    pdb.set_trace()
    band = ds.GetRasterBand(1)
    
    img = band.ReadAsArray()
    img = img.astype(np.float32)
    
    band=None
    ds=None
    
    return img




def runAutorift(I1, I2, xGrid, yGrid, Dx0, Dy0, noDataMask, optflag):
    '''
    Wire and run geogrid.
    '''

#    import isce
    from autoRIFT import autoRIFT
    import numpy as np
#    import isceobj
    import time
    
    
    obj = autoRIFT()
#    obj.configure()

#    #uncomment if starting from preprocessed images
#    I1 = I1.astype(np.uint8)
#    I2 = I2.astype(np.uint8)



    obj.I1 = I1
    obj.I2 = I2
    
    
    
    ######### mask out nodata
    if xGrid is not None:
        xGrid[noDataMask] = 0
        yGrid[noDataMask] = 0
        obj.xGrid = xGrid
        obj.yGrid = yGrid
        obj.SearchLimitX = obj.SearchLimitX * np.logical_not(noDataMask)
        obj.SearchLimitY = obj.SearchLimitY * np.logical_not(noDataMask)

    if Dx0 is not None:
        Dx0[noDataMask] = 0
        Dy0[noDataMask] = 0
        if optflag == 0:
            Dy0 = -1 * Dy0
        obj.Dx0 = np.round(Dx0 / 1.0)
        obj.Dy0 = np.round(Dy0 / 1.0)



#    obj.zeroMask = 1


    ######## preprocessing
    t1 = time.time()
    print("Pre-process Start!!!")
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


#    ########## export preprocessed images to files
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

#    pdb.set_trace()

    ########## run Autorift
    t1 = time.time()
    print("AutoRIFT Start!!!")
    obj.runAutorift()
    print("AutoRIFT Done!!!")
    print(time.time()-t1)

#    pdb.set_trace()

    return obj.Dx, obj.Dy, obj.InterpMask, obj.ChipSizeX






if __name__ == '__main__':
    '''
    Main driver.
    '''
    import numpy as np
    import time
    
    inps = cmdLineParse()
#    pdb.set_trace()
    if inps.optical_flag == 1:
        data_m = loadProductOptical(inps.indir_m)
        data_s = loadProductOptical(inps.indir_s)
    else:
        data_m = loadProduct(inps.indir_m)
        data_s = loadProduct(inps.indir_s)



    xGrid = None
    yGrid = None
    Dx0 = None
    Dy0 = None
    noDataMask = None
    
    if inps.grid_location is not None:
        ds = gdal.Open(inps.grid_location)
        tran = ds.GetGeoTransform()
        proj = ds.GetProjection()
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        xGrid = band.ReadAsArray()
        noDataMask = (xGrid == nodata)
        band = ds.GetRasterBand(2)
        yGrid = band.ReadAsArray()
        band=None
        ds=None
    
    if inps.init_offset is not None:
        ds = gdal.Open(inps.init_offset)
        band = ds.GetRasterBand(1)
        Dx0 = band.ReadAsArray()
        band = ds.GetRasterBand(2)
        Dy0 = band.ReadAsArray()
        band=None
        ds=None




#    pdb.set_trace()

    Dx, Dy, InterpMask, ChipSizeX = runAutorift(data_m, data_s, xGrid, yGrid, Dx0, Dy0, noDataMask, inps.optical_flag)

#    pdb.set_trace()

    import scipy.io as sio
    sio.savemat('offset.mat',{'Dx':Dx,'Dy':Dy,'InterpMask':InterpMask,'ChipSizeX':ChipSizeX})


    if inps.grid_location is not None:
        
        DX = np.zeros(xGrid.shape,dtype=np.float32) * np.nan
        DY = np.zeros(xGrid.shape,dtype=np.float32) * np.nan
        INTERPMASK = np.zeros(xGrid.shape,dtype=np.float32) * np.nan
        CHIPSIZEX = np.zeros(xGrid.shape,dtype=np.float32) * np.nan

        DX[0:Dx.shape[0],0:Dx.shape[1]] = Dx
        if inps.optical_flag == 1:
            DY[0:Dy.shape[0],0:Dy.shape[1]] = Dy
        else:
            DY[0:Dy.shape[0],0:Dy.shape[1]] = -Dy
        INTERPMASK[0:InterpMask.shape[0],0:InterpMask.shape[1]] = InterpMask
        CHIPSIZEX[0:ChipSizeX.shape[0],0:ChipSizeX.shape[1]] = ChipSizeX

        t1 = time.time()
        print("Write Outputs Start!!!")


        # Create the GeoTiff
        driver = gdal.GetDriverByName('GTiff')
        #   pdb.set_trace()

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


        if inps.offset2vx is not None:
            ds = gdal.Open(inps.offset2vx)
            band = ds.GetRasterBand(1)
            offset2vx_1 = band.ReadAsArray()
            band = ds.GetRasterBand(2)
            offset2vx_2 = band.ReadAsArray()
            band=None
            ds=None

            ds = gdal.Open(inps.offset2vy)
            band = ds.GetRasterBand(1)
            offset2vy_1 = band.ReadAsArray()
            band = ds.GetRasterBand(2)
            offset2vy_2 = band.ReadAsArray()
            band=None
            ds=None

            VX = offset2vx_1 * DX + offset2vx_2 * DY
            VY = offset2vy_1 * DX + offset2vy_2 * DY
            VX = VX.astype(np.float32)
            VY = VY.astype(np.float32)

            outRaster = driver.Create("velocity.tif", int(xGrid.shape[1]), int(xGrid.shape[0]), 2, gdal.GDT_Float32)
            outRaster.SetGeoTransform(tran)
            outRaster.SetProjection(proj)
            outband = outRaster.GetRasterBand(1)
            outband.WriteArray(VX)
            outband.FlushCache()
            outband = outRaster.GetRasterBand(2)
            outband.WriteArray(VY)
            outband.FlushCache()

        print("Write Outputs Done!!!")
        print(time.time()-t1)
