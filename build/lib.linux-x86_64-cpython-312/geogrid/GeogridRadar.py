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
# Authors: Piyush Agram, Yang Lei
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#import isce
#from iscesys.Component.Component import Component
import pdb
import subprocess
import re
import string
import datetime
import numpy as np
import os

class GeogridRadar():
    '''
    Class for mapping regular geographic grid on radar imagery.
    '''

    def geogridRadar(self):
        '''
        Do the actual processing.
        '''
        #import isce
        #from geo_autoRIFT.geogrid import geogrid
        from . import geogridRadar

        ##Determine appropriate EPSG system
        #print('PROJECTION SYSTEM', self.getProjectionSystem())
        self.epsg = self.getProjectionSystem()
        
        ###Determine extent of data needed
        bbox = self.determineBbox()

        ###Load approrpriate DEM from database
        if self.demname is None:
            self.demname, self.dhdxname, self.dhdyname, self.vxname, self.vyname, self.srxname, self.sryname, self.csminxname, self.csminyname, self.csmaxxname, self.csmaxyname, self.ssmname = self.getDEM(bbox)


        ##Create and set parameters
        self.setState()
        
        ##check parameters
        self.checkState()
        
        ##Run
        geogridRadar.geogridRadar_Py(self._geogrid)
        
        #deg2rad = np.pi/180.0
        #params=[];
        #params.append(self.startingRange)
        #params.append(self.rangePixelSize)
        #params.append((self.sensingStart - self.sensingStart.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds())
        #params.append(self.prf)
        #params.append(self.numberOfSamples)
        #params.append(self.numberOfLines)
        #params.append(self.incidenceAngle)
        #params.append(self.epsg)
        #params.append(self.chipSizeX0)
        #params.append(self.gridSpacingX)
        #params.append(self.repeatTime)
        #params.append(self._xlim[0])
        #params.append(self._xlim[1])
        #params.append(self._ylim[0])
        #params.append(self._ylim[1])
        #params.append(self.demname)
        #params.append(self.dhdxname)
        #params.append(self.dhdyname)
        #params.append(self.vxname)
        #params.append(self.vyname)
        #params.append(self.srxname)
        #params.append(self.sryname)
        #params.append(self.csminxname)
        #params.append(self.csminyname)
        #params.append(self.csmaxxname)
        #params.append(self.csmaxyname)
        #params.append(self.ssmname)
        #midsensing = self.sensingStart + datetime.timedelta(seconds = (np.floor(self.numberOfLines/2)-1) / self.prf)
        #params.append(midsensing.strftime("%Y-%m-%dT%H:%M:%S.%f"))
        #params.append((midsensing-datetime.timedelta(seconds = 600)).strftime("%Y-%m-%dT%H:%M:%S.%f"))
        #params.append((midsensing+datetime.timedelta(seconds = 600)).strftime("%Y-%m-%dT%H:%M:%S.%f"))
        #params.append(self.orbitname)
        
        #params=[str(param) for param in params]
        
        #cmdline = os.path.dirname(__file__)+"/geogrid "+" ".join(params)
        #print(cmdline)
        
        #subprocess.call(cmdline,shell=True)
        
        self.get_center_latlon()
        
        ##Get parameters
        self.getState()
        #print('XOFF',geogridRadar.getXOff_Py(self._geogrid))
        output = open('output.txt')
        lines = output.readlines()
        for line in lines:
            if 'pOff' in line:
                self.pOff = int(line.split()[1])
            elif 'lOff' in line:
                self.lOff = int(line.split()[1])
            elif 'pCount' in line:
                self.pCount = int(line.split()[1])
            elif 'lCount' in line:
                self.lCount = int(line.split()[1])
            elif 'X_res' in line:
                self.X_res = float(line.split()[1])
            elif 'Y_res' in line:
                self.Y_res = float(line.split()[1])

        ##Clean up
        self.finalize()

    def get_center_latlon(self):
        '''
        Get center lat/lon of the image.
        '''
        from osgeo import gdal
        self.epsg = 4326
        self.determineBbox()
        self.cen_lat = (self._ylim[0] + self._ylim[1]) / 2
        self.cen_lon = (self._xlim[0] + self._xlim[1]) / 2
        print("Scene-center lat/lon: " + str(self.cen_lat) + "  " + str(self.cen_lon))
    

    def getProjectionSystem(self):
        '''
        Testing with Greenland.
        '''
        #print('DEM NAME',self.demname)
        if not self.demname:
            raise Exception('At least the DEM parameter must be set for geogrid')

        from osgeo import gdal, osr
        ds = gdal.Open(self.demname, gdal.GA_ReadOnly)
        srs = osr.SpatialReference()
        srs.ImportFromWkt(ds.GetProjection())
        #srs.AutoIdentifyEPSG()
        ds = None
#        pdb.set_trace()

        if srs.IsProjected():
            epsgstr = srs.GetAuthorityCode('PROJCS')
        elif srs.IsGeographic():
            raise Exception('Geographic coordinate system encountered')
        elif srs.IsLocal():
            raise Exception('Local coordinate system encountered')
        else:
            raise Exception('Non-standard coordinate system encountered')
        if not epsgstr:  #Empty string->use shell command gdalsrsinfo for last trial
            cmd = 'gdalsrsinfo -o epsg {0}'.format(self.demname)
            epsgstr = subprocess.check_output(cmd, shell=True)
#            pdb.set_trace()
            epsgstr = re.findall("EPSG:(\d+)", str(epsgstr))[0]
#            pdb.set_trace()
        if not epsgstr:  #Empty string
            raise Exception('Could not auto-identify epsg code')
#        pdb.set_trace()
        epsgcode = int(epsgstr)
#        pdb.set_trace()
        return epsgcode

    def determineBbox(self, zrange=[-200,4000]):
        '''
        Dummy.
        '''
        import numpy as np
        import datetime
        from osgeo import osr,gdal
        from isce3.geometry import rdr2geo, DEMInterpolator
        from isce3.core import Ellipsoid
        import copy
        
        refElp = Ellipsoid(a=6378137.0, e2=0.0066943799901)
        
#        import pdb
#        pdb.set_trace()

#        rng = self.startingRange + np.linspace(0, self.numberOfSamples, num=21)
        rng = self.startingRange + np.linspace(0, self.numberOfSamples-1, num=21) * self.rangePixelSize
        deltat = np.linspace(0, 1., num=21)[1:-1]

        lonlat = osr.SpatialReference()
        lonlat.ImportFromEPSG(4326)

        coord = osr.SpatialReference()
        coord.ImportFromEPSG(self.epsg)
        #print('EPSG',self.epsg)
        
        trans = osr.CoordinateTransformation(lonlat, coord)
        inv = osr.CoordinateTransformation(coord, lonlat)

        llhs = []
        xyzs = []
        
        deg2rad = np.pi/180.0

        ###First range line
        for rr in rng:
            for zz in zrange:
                #llh = rdr2geo(self.aztime, rr, ellipsoid=refElp, orbit=self.orbit, wvl=self.wavelength, side=self.lookSide, demInterpolatorHeight=zz)
                llh = rdr2geo(self.aztime, rr, self.orbit, self.lookSide, 0.0, self.wavelength, DEMInterpolator(zz), refElp)
                llht = copy.copy(llh)
                llht[0]=llh[1]/deg2rad
                llht[1]=llh[0]/deg2rad
                llhs.append(llh)
                if gdal.__version__[0] == '2':
                    x,y,z = trans.TransformPoint(llht[1], llht[0], llht[2])
                else:
                    x,y,z = trans.TransformPoint(llht[0], llht[1], llht[2])
                #llht=inv.TransformPoint(x,y,z)
                #print('LLH',np.array(llht)/deg2rad)
                #print('XYZ',x,y,z)
                xyzs.append([x,y,z])

        ##Last range line
        sensingStop = self.aztime + (self.numberOfLines-1) / self.prf
        for rr in rng:
            for zz in zrange:
                llh = rdr2geo(sensingStop, rr, self.orbit, self.lookSide, 0.0, self.wavelength, DEMInterpolator(zz), refElp)
                #llh = self.orbit.rdr2geo(sensingStop, rr, side=self.lookSide, height=zz)
                llht = copy.copy(llh)
                llht[0]=llh[1]/deg2rad
                llht[1]=llh[0]/deg2rad
                llhs.append(llh)
                if gdal.__version__[0] == '2':
                    x,y,z = trans.TransformPoint(llht[1], llht[0], llht[2])
                else:
                    x,y,z = trans.TransformPoint(llht[0], llht[1], llht[2])
                #llht=inv.TransformPoint(x,y,z)
                #print('LLH',np.array(llht)/deg2rad)
                #print('XYZ',x,y,z)
                xyzs.append([x,y,z])


        ##For each line in middle, consider the edges
        for frac in deltat:
            sensingTime = self.aztime + frac * (self.numberOfLines-1)/self.prf
#            print('sensing Time: %f %f %f'%(sensingTime.minute,sensingTime.second,sensingTime.microsecond))
            for rr in [rng[0], rng[-1]]:
                for zz in zrange:
                    llh = rdr2geo(sensingTime, rr, self.orbit, self.lookSide, 0.0, self.wavelength, DEMInterpolator(zz), refElp)
                    #llh = self.orbit.rdr2geo(sensingTime, rr, side=self.lookSide, height=zz)
                    llht = copy.copy(llh)
                    llht[0]=llh[1]/deg2rad
                    llht[1]=llh[0]/deg2rad
                    llhs.append(llh)
                    if gdal.__version__[0] == '2':
                        x,y,z = trans.TransformPoint(llht[1], llht[0], llht[2])
                    else:
                        x,y,z = trans.TransformPoint(llht[0], llht[1], llht[2])
                    #llht=inv.TransformPoint(x,y,z)
                    #print('LLH',np.array(llht))
                    #print('XYZ',x,y,z)
                    xyzs.append([x,y,z])


        llhs = np.array(llhs)
        xyzs = np.array(xyzs)
        #print('LIMS',xyzs)

        if str(self.epsg)=='4326':
            #self._xlim = [np.min(xyzs[:,0])/deg2rad, np.max(xyzs[:,0])/deg2rad]
            self._xlim = [np.min(xyzs[:,1]), np.max(xyzs[:,1])]
            #self._ylim = [np.min(xyzs[:,1])/deg2rad, np.max(xyzs[:,1])/deg2rad]
            self._ylim = [np.min(xyzs[:,0]), np.max(xyzs[:,0])]
        else:
            self._xlim = [np.min(xyzs[:,0]), np.max(xyzs[:,0])]
            self._ylim = [np.min(xyzs[:,1]), np.max(xyzs[:,1])]
        
    def getIncidenceAngle(self, zrange=[-200,4000]):
        '''
        Dummy.
        '''
        import numpy as np
        import datetime
        from osgeo import osr,gdal
        from isce3.core import Ellipsoid
        from isce3.geometry import rdr2geo, DEMInterpolator
        #from isceobj.Util.geo.ellipsoid import Ellipsoid
        
        refElp = Ellipsoid(a=6378137.0, e2=0.0066943799901)
        
        deg2rad = np.pi/180.0
        
        thetas = []
        
        midrng = self.startingRange + (np.floor(self.numberOfSamples/2)-1) * self.rangePixelSize
        midsensing = self.aztime + (np.floor(self.numberOfLines/2)-1) / self.prf
        master_pos, master_vel= self.orbit.interpolate(midsensing)
        mxyz = master_pos
        #print('MXYZ',mxyz)
        
        for zz in zrange:
            llh = rdr2geo(midsensing, midrng, self.orbit, self.lookSide, 0.0, self.wavelength, DEMInterpolator(zz), refElp)
            #print('LLH',llh)
            targxyz = np.array(refElp.lon_lat_to_xyz(llh).tolist())
            los = (mxyz-targxyz) / np.linalg.norm(mxyz-targxyz)
            n_vec = np.array([np.cos(llh[1])*np.cos(llh[0]), np.cos(llh[1])*np.sin(llh[0]), np.sin(llh[1])])
            #print('LOS',los)
            #print('N_VEC',n_vec)
            #print('DOT',np.dot(los, n_vec))
            theta = np.arccos(np.dot(los, n_vec))
            #print('THETA',theta)
            thetas.append([theta])
        
        thetas = np.array(thetas)
        print('Incidence Angle',thetas)
        self.incidenceAngle = np.mean(thetas)
        
    def getDEM(self, bbox):
        '''
        Look up database and return values.
        '''
        
        return "", "", "", "", ""

    def getState(self):
        from geogrid import geogridRadar as geogrid
        
        self.pOff = geogrid.getXOff_Py(self._geogrid)
        self.lOff = geogrid.getYOff_Py(self._geogrid)
        self.pCount = geogrid.getXCount_Py(self._geogrid)
        self.lCount = geogrid.getYCount_Py(self._geogrid)
        self.X_res = geogrid.getXPixelSize_Py(self._geogrid)
        self.Y_res = geogrid.getYPixelSize_Py(self._geogrid)
    
    def setState(self):
        '''
        Create C object and populate.
        '''
        from geogrid import geogridRadar as geogrid

        if self._geogrid is not None:
            geogrid.destroyGeoGrid_Py(self._geogrid)

        self._geogrid = geogrid.createGeoGrid_Py()
        geogrid.setRadarImageDimensions_Py( self._geogrid, self.numberOfSamples, self.numberOfLines)
        geogrid.setRangeParameters_Py( self._geogrid, self.startingRange, self.rangePixelSize)
        aztime = (self.sensingStart - self.sensingStart.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
        geogrid.setAzimuthParameters_Py( self._geogrid, aztime, self.prf)
        geogrid.setRepeatTime_Py(self._geogrid, self.repeatTime)
        
        geogrid.setDtUnity_Py( self._geogrid, self.srs_dt_unity)
        geogrid.setMaxFactor_Py( self._geogrid, self.srs_max_scale)
        geogrid.setUpperThreshold_Py( self._geogrid, self.srs_max_search)
        geogrid.setLowerThreshold_Py(self._geogrid, self.srs_min_search)

        geogrid.setEPSG_Py(self._geogrid, self.epsg)
        geogrid.setIncidenceAngle_Py(self._geogrid, self.incidenceAngle)
        geogrid.setChipSizeX0_Py(self._geogrid, self.chipSizeX0)
        geogrid.setGridSpacingX_Py(self._geogrid, self.gridSpacingX)
        
        geogrid.setXLimits_Py(self._geogrid, self._xlim[0], self._xlim[1])
        geogrid.setYLimits_Py(self._geogrid, self._ylim[0], self._ylim[1])
        
        midsensing = self.sensingStart + datetime.timedelta(seconds = (np.floor(self.numberOfLines/2)-1) / self.prf)
        tmids = midsensing.strftime("%Y-%m-%dT%H:%M:%S.%f")
        itime = (midsensing-datetime.timedelta(seconds = 600)).strftime("%Y-%m-%dT%H:%M:%S.%f")
        ftime = (midsensing+datetime.timedelta(seconds = 600)).strftime("%Y-%m-%dT%H:%M:%S.%f")
        geogrid.setTimes_Py(self._geogrid, tmids, itime, ftime)
        geogrid.setOrbit_Py(self._geogrid, self.orbitname)
        if self.demname:
            geogrid.setDEM_Py(self._geogrid, self.demname)

        if (self.dhdxname is not None) and (self.dhdyname is not None):
            geogrid.setSlopes_Py(self._geogrid, self.dhdxname, self.dhdyname)

        if (self.vxname is not None) and (self.vyname is not None):
            geogrid.setVelocities_Py(self._geogrid, self.vxname, self.vyname)
        
        if (self.srxname is not None) and (self.sryname is not None):
            geogrid.setSearchRange_Py(self._geogrid, self.srxname, self.sryname)
        
        if (self.csminxname is not None) and (self.csminyname is not None):
            geogrid.setChipSizeMin_Py(self._geogrid, self.csminxname, self.csminyname)
        
        if (self.csmaxxname is not None) and (self.csmaxyname is not None):
            geogrid.setChipSizeMax_Py(self._geogrid, self.csmaxxname, self.csmaxyname)
        
        if (self.ssmname is not None):
            geogrid.setStableSurfaceMask_Py(self._geogrid, self.ssmname)

        geogrid.setWindowLocationsFilename_Py( self._geogrid, self.winlocname)
        geogrid.setWindowOffsetsFilename_Py( self._geogrid, self.winoffname)
        geogrid.setWindowSearchRangeFilename_Py( self._geogrid, self.winsrname)
        geogrid.setWindowChipSizeMinFilename_Py( self._geogrid, self.wincsminname)
        geogrid.setWindowChipSizeMaxFilename_Py( self._geogrid, self.wincsmaxname)
        geogrid.setWindowStableSurfaceMaskFilename_Py( self._geogrid, self.winssmname)
        geogrid.setRO2VXFilename_Py( self._geogrid, self.winro2vxname)
        geogrid.setRO2VYFilename_Py( self._geogrid, self.winro2vyname)
        geogrid.setSFFilename_Py( self._geogrid, self.winsfname)
        geogrid.setLookSide_Py(self._geogrid, self.lookSide)
        geogrid.setNodataOut_Py(self._geogrid, self.nodata_out)

        #self._orbit  = self.orbit.exportToC()
        #geogrid.setOrbit_Py(self._geogrid, self._orbit)

    def checkState(self):
        '''
        Create C object and populate.
        '''
        if self.repeatTime < 0:
            raise Exception('Input image 1 must be older than input image 2')

    def finalize(self):
        '''
        Clean up all the C pointers.
        '''

        from geogrid import geogridRadar as geogrid

        geogrid.destroyGeoGrid_Py(self._geogrid)
        self._geogrid = None

    def __init__(self):
        super(GeogridRadar, self).__init__()

        ##Radar image related parameters
        self.orbit = None
        self.sensingStart = None
        self.sensingStop = None
        self.startingRange = None
        self.prf = None
        self.rangePixelSize = None
        self.numberOfSamples = None
        self.numberOfLines = None
        self.lookSide = None
        self.aztime = None
        self.repeatTime = None
        self.incidenceAngle = None
        self.chipSizeX0 = None
        self.gridSpacingX = None

        ##Input related parameters
        self.orbitname = None
        self.demname = None
        self.dhdxname = None
        self.dhdyname = None
        self.vxname = None
        self.vyname = None
        self.srxname = None
        self.sryname = None
        self.csminxname = None
        self.csminyname = None
        self.csmaxxname = None
        self.csmaxyname = None
        self.ssmname = None

        ##Output related parameters
        self.winlocname = None
        self.winoffname = None
        self.winsrname = None
        self.wincsminname = None
        self.wincsmaxname = None
        self.winssmname = None
        self.winro2vxname = None
        self.winro2vyname = None
        self.winsfname = None
        
        ##dt-varying search range scale (srs) rountine parameters
        self.srs_dt_unity = 182
        self.srs_max_scale = 5
        self.srs_max_search = 20000
        self.srs_min_search = 0

        ##Coordinate system
        self.epsg = None
        self._xlim = None
        self._ylim = None
        self.nodata_out = None

        ##Pointer to C 
        self._geogrid = None
        self._orbit = None

        ##parameters for autoRIFT
        self.pOff = None
        self.lOff = None
        self.pCount = None
        self.lCount = None
        self.X_res = None
        self.Y_res = None
