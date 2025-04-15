/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright 2019 California Institute of Technology. ALL RIGHTS RESERVED.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * United States Government Sponsorship acknowledged. This software is subject to
 * U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
 * (No [Export] License Required except when exporting to an embargoed country,
 * end user, or in support of a prohibited end use). By downloading this software,
 * the user agrees to comply with all applicable U.S. export laws and regulations.
 * The user has the responsibility to obtain export licenses, or other export
 * authority as may be required before exporting this software to any 'EAR99'
 * embargoed foreign country or citizen of those countries.
 *
 * Authors: Piyush Agram, Yang Lei
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

#include "geogridRadar.h"
#include <gdal.h>
#include <gdal_priv.h>
#include <iostream>
#include <complex>
#include <cmath>
#include <isce3/error/ErrorCode.h>
//#include "linalg3.h"
#include "isce3/geometry/geometry.h"
#include "isce3/core/Ellipsoid.h"
#include "isce3/core/detail/BuildOrbit.h"
#include "isce3/core/detail/InterpolateOrbit.h"

using isce3::error::ErrorCode;

double dot_C(isce3::core::Vec3* a, isce3::core::Vec3* b) {
   return (*a).dot(*b);
   //double product = 0.0;
   //for (int i = 0; i < 3; i++){
   //    product = product + a[i] * b[i];
   //}
   //return product;
}

double norm_C(isce3::core::Vec3* a){
   return sqrt(dot_C(a,a));

}

void cross_C(isce3::core::Vec3* a, isce3::core::Vec3* b, isce3::core::Vec3 temp) {
   std::cout << "A: " << (*a)[0] << " " << (*a)[1] << " " << (*a)[2] << "\n";
   std::cout << "B: " << (*b)[0] << " " << (*b)[1] << " " << (*b)[2] << "\n";
   temp = (*a).cross(*b);
}

void cross_C2(isce3::core::Vec3* a, isce3::core::Vec3* b, isce3::core::Vec3* temp) {
   (*temp) = (*a).cross(*b);
}

void unitvec_C(isce3::core::Vec3* a, isce3::core::Vec3 b) {
   double norma = norm_C(a);
   
   b[0] = (*a)[0] / norma;
   b[1] = (*a)[1] / norma;
   b[2] = (*a)[2] / norma;
   std::cout << "B: " << b[0] << " " << b[1] << " " << b[2] << "\n";
}

void unitvec_C2(isce3::core::Vec3* a, isce3::core::Vec3* b) {
   double norma = norm_C(a);
   
   (*b)[0] = (*a)[0] / norma;
   (*b)[1] = (*a)[1] / norma;
   (*b)[2] = (*a)[2] / norma;
}

void geoGridRadar::geogridRadar()
{
    //Some constants 
    double deg2rad = M_PI/180.0;

    //For now print inputs that were obtained
    std::cout << "\nNEW GEOGRID! \n";
    std::cout << "\nRadar parameters: \n";
    std::cout << "Range: " << startingRange << "  " << dr << "\n";
    std::cout << "Azimuth: " << sensingStart << "  " << prf << "\n";
    std::cout << "Dimensions: " << nPixels << " " << nLines << "\n";
    std::cout << "Incidence Angle: " << incidenceAngle/deg2rad << "\n";

    std::cout << "\nMap inputs: \n";
    std::cout << "EPSG: " << epsgcode << "\n";
    std::cout << "Smallest Allowable Chip Size in m: " << chipSizeX0 << "\n";
    std::cout << "Grid spacing in m: " << gridSpacingX << "\n";
    std::cout << "Repeat Time: " << dt << "\n";
    std::cout << "XLimits: " << xmin << "  " << xmax << "\n";
    std::cout << "YLimits: " << ymin << "  " << ymax << "\n";
    std::cout << "Extent in km: " << (xmax - xmin)/1000. << "  " << (ymax - ymin)/1000. << "\n";
    if (demname != "")
    {
        std::cout << "DEM: " << demname << "\n";
    }
    if (dhdxname != "")
    {
        std::cout << "Slopes: " << dhdxname << "  " << dhdyname << "\n";
    }
    if (vxname != "")
    {
        std::cout << "Velocities: " << vxname << "  " << vyname << "\n";
    }
    if (srxname != "")
    {
        std::cout << "Search Range: " << srxname << "  " << sryname << "\n";
    }
    if (csminxname != "")
    {
        std::cout << "Chip Size Min: " << csminxname << "  " << csminyname << "\n";
    }
    if (csmaxxname != "")
    {
        std::cout << "Chip Size Max: " << csmaxxname << "  " << csmaxyname << "\n";
    }
    if (ssmname != "")
    {
        std::cout << "Stable Surface Mask: " << ssmname << "\n";
    }
    
    std::cout << "\nOutputs: \n";
    std::cout << "Window locations: " << pixlinename << "\n";
    if (dhdxname != "")
    {
        if (vxname != "")
        {
            std::cout << "Window offsets: " << offsetname << "\n";
        }
        
        std::cout << "Window rdr_off2vel_x vector: " << ro2vx_name << "\n";
        std::cout << "Window rdr_off2vel_y vector: " << ro2vy_name << "\n";
        std::cout << "Window scale factor: " << sfname << "\n";
        
        if (srxname != "")
        {
            std::cout << "Window search range: " << searchrangename << "\n";
        }
    }
            
    if (csminxname != "")
    {
        std::cout << "Window chip size min: " << chipsizeminname << "\n";
    }
    if (csmaxxname != "")
    {
        std::cout << "Window chip size max: " << chipsizemaxname << "\n";
    }
    if (ssmname != "")
    {
        std::cout << "Window stable surface mask: " << stablesurfacemaskname << "\n";
    }
    
    
    std::cout << "Output Nodata Value: " << nodata_out << "\n";
    

    std::cout << "\nStarting processing .... \n";

    //Startup GDAL
    GDALAllRegister();

    //DEM related information
    GDALDataset* demDS = NULL;
    GDALDataset* sxDS = NULL;
    GDALDataset* syDS = NULL;
    GDALDataset* vxDS = NULL;
    GDALDataset* vyDS = NULL;
    GDALDataset* srxDS = NULL;
    GDALDataset* sryDS = NULL;
    GDALDataset* csminxDS = NULL;
    GDALDataset* csminyDS = NULL;
    GDALDataset* csmaxxDS = NULL;
    GDALDataset* csmaxyDS = NULL;
    GDALDataset* ssmDS = NULL;

    double geoTrans[6];

    demDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(demname.c_str(), GA_ReadOnly));
    if (dhdxname != "")
    {
        sxDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(dhdxname.c_str(), GA_ReadOnly));
        syDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(dhdyname.c_str(), GA_ReadOnly));
    }
    if (vxname != "")
    {
        vxDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(vxname.c_str(), GA_ReadOnly));
        vyDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(vyname.c_str(), GA_ReadOnly));
    }
    if (srxname != "")
    {
        srxDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(srxname.c_str(), GA_ReadOnly));
        sryDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(sryname.c_str(), GA_ReadOnly));
    }
    if (csminxname != "")
    {
        csminxDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(csminxname.c_str(), GA_ReadOnly));
        csminyDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(csminyname.c_str(), GA_ReadOnly));
    }
    if (csmaxxname != "")
    {
        csmaxxDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(csmaxxname.c_str(), GA_ReadOnly));
        csmaxyDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(csmaxyname.c_str(), GA_ReadOnly));
    }
    if (ssmname != "")
    {
        ssmDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(ssmname.c_str(), GA_ReadOnly));
    }
    if (demDS == NULL)
    {
        std::cout << "Error opening DEM file { " << demname << " }\n";
        std::cout << "Exiting with error code .... (101) \n";
        GDALDestroyDriverManager();
        exit(101);
    }
    if (dhdxname != "")
    {
        if (sxDS == NULL)
        {
            std::cout << "Error opening x-direction slope file { " << dhdxname << " }\n";
            std::cout << "Exiting with error code .... (101) \n";
            GDALDestroyDriverManager();
            exit(101);
        }
        if (syDS == NULL)
        {
            std::cout << "Error opening y-direction slope file { " << dhdyname << " }\n";
            std::cout << "Exiting with error code .... (101) \n";
            GDALDestroyDriverManager();
            exit(101);
        }
    }
    if (vxname != "")
    {
        if (vxDS == NULL)
        {
            std::cout << "Error opening x-direction velocity file { " << vxname << " }\n";
            std::cout << "Exiting with error code .... (101) \n";
            GDALDestroyDriverManager();
            exit(101);
        }
        if (vyDS == NULL)
        {
            std::cout << "Error opening y-direction velocity file { " << vyname << " }\n";
            std::cout << "Exiting with error code .... (101) \n";
            GDALDestroyDriverManager();
            exit(101);
        }
    }
    if (srxname != "")
    {
        if (srxDS == NULL)
        {
            std::cout << "Error opening x-direction search range file { " << srxname << " }\n";
            std::cout << "Exiting with error code .... (101) \n";
            GDALDestroyDriverManager();
            exit(101);
        }
        if (sryDS == NULL)
        {
            std::cout << "Error opening y-direction search range file { " << sryname << " }\n";
            std::cout << "Exiting with error code .... (101) \n";
            GDALDestroyDriverManager();
            exit(101);
        }
    }
    if (csminxname != "")
    {
        if (csminxDS == NULL)
        {
            std::cout << "Error opening x-direction chip size min file { " << csminxname << " }\n";
            std::cout << "Exiting with error code .... (101) \n";
            GDALDestroyDriverManager();
            exit(101);
        }
        if (csminyDS == NULL)
        {
            std::cout << "Error opening y-direction chip size min file { " << csminyname << " }\n";
            std::cout << "Exiting with error code .... (101) \n";
            GDALDestroyDriverManager();
            exit(101);
        }
    }
    if (csmaxxname != "")
    {
        if (csmaxxDS == NULL)
        {
            std::cout << "Error opening x-direction chip size max file { " << csmaxxname << " }\n";
            std::cout << "Exiting with error code .... (101) \n";
            GDALDestroyDriverManager();
            exit(101);
        }
        if (csmaxyDS == NULL)
        {
            std::cout << "Error opening y-direction chip size max file { " << csmaxyname << " }\n";
            std::cout << "Exiting with error code .... (101) \n";
            GDALDestroyDriverManager();
            exit(101);
        }
    }
    if (ssmname != "")
    {
        if (ssmDS == NULL)
        {
            std::cout << "Error opening stable surface mask file { " << ssmname << " }\n";
            std::cout << "Exiting with error code .... (101) \n";
            GDALDestroyDriverManager();
            exit(101);
        }
    }

    demDS->GetGeoTransform(geoTrans);
    int demXSize = demDS->GetRasterXSize();
    int demYSize = demDS->GetRasterYSize();


    //Get offsets and size to read from DEM
//    int lOff = std::max( std::floor((ymax - geoTrans[3])/geoTrans[5]), 0.);
//    int lCount = std::min( std::ceil((ymin - geoTrans[3])/geoTrans[5]), demYSize-1.) - lOff;
//
//    int pOff = std::max( std::floor((xmin - geoTrans[0])/geoTrans[1]), 0.);
//    int pCount = std::min( std::ceil((xmax - geoTrans[0])/geoTrans[1]), demXSize-1.) - pOff;
    int lOff = std::max( std::floor((ymax - geoTrans[3])/geoTrans[5]), 0.);
    int lCount = std::min( std::ceil((ymin - geoTrans[3])/geoTrans[5]), demYSize-1.) - lOff;
    
    int pOff = std::max( std::floor((xmin - geoTrans[0])/geoTrans[1]), 0.);
    int pCount = std::min( std::ceil((xmax - geoTrans[0])/geoTrans[1]), demXSize-1.) - pOff;


    std::cout << "Xlimits : " << geoTrans[0] + pOff * geoTrans[1] <<  "  " 
                             << geoTrans[0] + (pOff + pCount) * geoTrans[1] << "\n";


    std::cout << "Ylimits : " << geoTrans[3] + (lOff + lCount) * geoTrans[5] <<  "  "
                             << geoTrans[3] + lOff * geoTrans[5] << "\n";

    std::cout << "Origin index (in DEM) of geogrid: " << pOff << "   " << lOff << "\n";
    
    std::cout << "Dimensions of geogrid: " << pCount << " x " << lCount << "\n";


    //Create GDAL Transformers 
    OGRSpatialReference demSRS(nullptr);
    if (demSRS.importFromEPSG(epsgcode) != 0)
    {
        std::cout << "Could not create OGR spatial reference for EPSG code: " << epsgcode << "\n";
        GDALClose(demDS);
        GDALDestroyDriverManager();
        exit(102);
    }

    OGRSpatialReference llhSRS(nullptr);
    if (llhSRS.importFromEPSG(4326) != 0)
    {
        std::cout << "Could not create OGR spatil reference for EPSG code: 4326 \n";
        GDALClose(demDS);
        GDALDestroyDriverManager();
        exit(103);
    }

    OGRCoordinateTransformation *fwdTrans = OGRCreateCoordinateTransformation( &demSRS, &llhSRS);
    OGRCoordinateTransformation *invTrans = OGRCreateCoordinateTransformation( &llhSRS, &demSRS);

    //WGS84 ellipsoid only
    isce3::core::Ellipsoid wgs84;
    wgs84.a(6378137.0);
    wgs84.e2(0.0066943799901);

    //Initial guess for solution
    double tmid = sensingStart + 0.5 * nLines / prf;
    
    isce3::core::Vec3 satxmid;
    isce3::core::Vec3 satvmid;
    //double satxmid[3];
    //double satvmid[3];
    
    std::ifstream ifidt(forbit);
    std::string line;
    std::string aztime;
    isce3::core::Vec3 pos, vel;
    
    bool count = true;
    int num = 0;
    
    //while(count){
    //    std::getline(ifid, line);
        //std::cout << "Line: " << line << "\n";
    //    if (line.find("<List_of_OSVs count=\"") != std::string::npos)
    //    {
    //        std::string startTag = "<List_of_OSVs count=\"";
    //        int startPos = line.find(startTag)+startTag.length();
    //        std::string endTag = "\">";
    //        int endPos = line.find(endTag);
    //        num = std::stoi(line.substr(startPos,endPos-startPos));
    //        count = false;
    //    }
    //}
    while(std::getline(ifidt, line)){
        if (line.find("<UTC>") != std::string::npos)
        {
            std::string startTag = "<UTC>UTC=";
            int startPos = line.find(startTag)+startTag.length();
            std::string endTag = "</UTC>";
            int endPos = line.find(endTag);
            aztime = line.substr(startPos,endPos-startPos);
            isce3::core::DateTime t0(aztime);
            isce3::core::DateTime itimed(itime);
            isce3::core::DateTime ftimed(ftime);
            //std::cout << (t0.secondsSinceEpoch()>=itimed.secondsSinceEpoch()) << (t0.secondsSinceEpoch()<=ftimed.secondsSinceEpoch()) << "\n";
            if ((t0.secondsSinceEpoch()>=itimed.secondsSinceEpoch()) and (t0.secondsSinceEpoch()<=ftimed.secondsSinceEpoch()))
            {
                ++num;
            }
        }
    }
    ifidt.close();
    std::ifstream ifid(forbit);
    std::vector<isce3::core::StateVector> sv;
    //sv.resize(num);
    int nOrbit = 0;
    
    
    while(std::getline(ifid, line))
    {
        //std::cout << "Orbit number: " << nOrbit << "\n";
        if (line.find("<UTC>") != std::string::npos)
        {
            std::string startTag = "<UTC>UTC=";
            int startPos = line.find(startTag)+startTag.length();
            std::string endTag = "</UTC>";
            int endPos = line.find(endTag);
            aztime = line.substr(startPos,endPos-startPos);
        }
        if (line.find("<X unit") != std::string::npos)
        {
            std::string startTag = ">";
            int startPos = line.find(startTag)+startTag.length();
            std::string endTag = "</X>";
            int endPos = line.find(endTag);
            pos[0] = stod(line.substr(startPos,endPos-startPos));
        }
        if (line.find("<Y unit") != std::string::npos)
        {
            std::string startTag = ">";
            int startPos = line.find(startTag)+startTag.length();
            std::string endTag = "</Y>";
            int endPos = line.find(endTag);
            pos[1] = stod(line.substr(startPos,endPos-startPos));
        }
        if (line.find("<Z unit") != std::string::npos)
        {
            std::string startTag = ">";
            int startPos = line.find(startTag)+startTag.length();
            std::string endTag = "</Z>";
            int endPos = line.find(endTag);
            pos[2] = stod(line.substr(startPos,endPos-startPos));
        }
        if (line.find("<VX unit") != std::string::npos)
        {
            std::string startTag = ">";
            int startPos = line.find(startTag)+startTag.length();
            std::string endTag = "</VX>";
            int endPos = line.find(endTag);
            vel[0] = stod(line.substr(startPos,endPos-startPos));
        }
        if (line.find("<VY unit") != std::string::npos)
        {
            std::string startTag = ">";
            int startPos = line.find(startTag)+startTag.length();
            std::string endTag = "</VY>";
            int endPos = line.find(endTag);
            vel[1] = stod(line.substr(startPos,endPos-startPos));
        }
        if (line.find("<VZ unit") != std::string::npos)
        {
            std::string startTag = ">";
            int startPos = line.find(startTag)+startTag.length();
            std::string endTag = "</VZ>";
            int endPos = line.find(endTag);
            vel[2] = stod(line.substr(startPos,endPos-startPos));
            
            isce3::core::DateTime t0(aztime);
            isce3::core::DateTime itimed(itime);
            isce3::core::DateTime ftimed(ftime);
            //std::cout << "Tuple " << itimed << t0 << ftimed << "\n";
            //std::cout << "Tuple " << (t0.secondsSinceEpoch()>=itimed.secondsSinceEpoch()) << (t0.secondsSinceEpoch()<=ftimed.secondsSinceEpoch()) << "\n";
            if ((t0.secondsSinceEpoch()>=itimed.secondsSinceEpoch()) and (t0.secondsSinceEpoch()<=ftimed.secondsSinceEpoch()))
            {
                isce3::core::StateVector svt;
                svt.datetime = t0;
                svt.position = pos;
                svt.velocity = vel;
                sv.push_back(svt);
                ++nOrbit;
            }
            //std::cout << "Tuple " << t0 << pos << vel << "\n";
            
        }
    }
    ifid.close();
    std::cout << "Size sv: " << sv.size() << "\n";
    //isce3::core::Orbit orbit(sv,"");
    isce3::core::Orbit orbit;
    orbit.setStateVectors(sv);
    
    isce3::core::DateTime tmidd;
    tmidd.strptime(tmids);
    
    isce3::core::OrbitInterpBorderMode border_mode= isce3::core::OrbitInterpBorderMode::Error;
    if (orbit.interpolate(&satxmid, &satvmid, tmidd.secondsSinceEpoch(), border_mode) != ErrorCode::Success)
    {
        std::cout << "Error with orbit interpolation for setup. \n";
        GDALClose(demDS);
        GDALDestroyDriverManager();
        exit(104);
    }
//    std::cout << "Center Satellite Velocity: " << satvmid[0] << " " << satvmid[1] << " " << satvmid[2] << "\n";
//    std::cout << satxmid[0] << " " << satxmid[1] << " " << satxmid[2] << "\n";
    std::vector<double> demLine(pCount);
    std::vector<double> sxLine(pCount);
    std::vector<double> syLine(pCount);
    std::vector<double> vxLine(pCount);
    std::vector<double> vyLine(pCount);
    std::vector<double> srxLine(pCount);
    std::vector<double> sryLine(pCount);
    std::vector<double> csminxLine(pCount);
    std::vector<double> csminyLine(pCount);
    std::vector<double> csmaxxLine(pCount);
    std::vector<double> csmaxyLine(pCount);
    std::vector<double> ssmLine(pCount);
    
    GInt32 raster1[pCount];
    GInt32 raster2[pCount];
    GInt32 raster11[pCount];
    GInt32 raster22[pCount];
    
    GInt32 sr_raster11[pCount];
    GInt32 sr_raster22[pCount];
    GInt32 csmin_raster11[pCount];
    GInt32 csmin_raster22[pCount];
    GInt32 csmax_raster11[pCount];
    GInt32 csmax_raster22[pCount];
    GInt32 ssm_raster[pCount];
    
    double raster1a[pCount];
    double raster1b[pCount];
    double raster1c[pCount];
    
    double raster2a[pCount];
    double raster2b[pCount];
    double raster2c[pCount];
    
    double sf_raster1[pCount];
    double sf_raster2[pCount];
    

    
    GDALRasterBand *poBand1 = NULL;
    GDALRasterBand *poBand2 = NULL;
    GDALRasterBand *poBand1Off = NULL;
    GDALRasterBand *poBand2Off = NULL;
    GDALRasterBand *poBand1Sch = NULL;
    GDALRasterBand *poBand2Sch = NULL;
    GDALRasterBand *poBand1Min = NULL;
    GDALRasterBand *poBand2Min = NULL;
    GDALRasterBand *poBand1Max = NULL;
    GDALRasterBand *poBand2Max = NULL;
    GDALRasterBand *poBand1Msk = NULL;
    GDALRasterBand *poBand1RO2VX = NULL;
    GDALRasterBand *poBand1RO2VY = NULL;
    GDALRasterBand *poBand2RO2VX = NULL;
    GDALRasterBand *poBand2RO2VY = NULL;
    GDALRasterBand *poBand3RO2VX = NULL;
    GDALRasterBand *poBand3RO2VY = NULL;
    GDALRasterBand *poBand1SF = NULL;
    GDALRasterBand *poBand2SF = NULL;
    
    
    
    GDALDataset *poDstDS = NULL;
    GDALDataset *poDstDSOff = NULL;
    GDALDataset *poDstDSSch = NULL;
    GDALDataset *poDstDSMin = NULL;
    GDALDataset *poDstDSMax = NULL;
    GDALDataset *poDstDSMsk = NULL;
    GDALDataset *poDstDSRO2VX = NULL;
    GDALDataset *poDstDSRO2VY = NULL;
    GDALDataset *poDstDSSF = NULL;

    

    double nodata;
//    double nodata_out;
    if (vxname != "")
    {
        int* pbSuccess = NULL;
        nodata = vxDS->GetRasterBand(1)->GetNoDataValue(pbSuccess);
    }
//    nodata_out = -2000000000;
    
    const char *pszFormat = "GTiff";
    char **papszOptions = NULL;
    std::string str = "";
    double adfGeoTransform[6] = { geoTrans[0] + pOff * geoTrans[1], geoTrans[1], 0, geoTrans[3] + lOff * geoTrans[5], 0, geoTrans[5]};
    OGRSpatialReference oSRS;
    char *pszSRS_WKT = NULL;
    demSRS.exportToWkt( &pszSRS_WKT );
    
    
    
    GDALDriver *poDriver;
    poDriver = GetGDALDriverManager()->GetDriverByName(pszFormat);
    if( poDriver == NULL )
    exit(107);
//    GDALDataset *poDstDS;
    
    str = pixlinename;
    const char * pszDstFilename = str.c_str();
    poDstDS = poDriver->Create( pszDstFilename, pCount, lCount, 2, GDT_Int32,
                               papszOptions );
    
    
    poDstDS->SetGeoTransform( adfGeoTransform );
    poDstDS->SetProjection( pszSRS_WKT );
//    CPLFree( pszSRS_WKT );
    
    
//    GDALRasterBand *poBand1;
//    GDALRasterBand *poBand2;
    poBand1 = poDstDS->GetRasterBand(1);
    poBand2 = poDstDS->GetRasterBand(2);
    poBand1->SetNoDataValue(nodata_out);
    poBand2->SetNoDataValue(nodata_out);
    
    
    if ((dhdxname != "")&(vxname != ""))
    {

        GDALDriver *poDriverOff;
        poDriverOff = GetGDALDriverManager()->GetDriverByName(pszFormat);
        if( poDriverOff == NULL )
        exit(107);
//        GDALDataset *poDstDSOff;
    
        str = offsetname;
        const char * pszDstFilenameOff = str.c_str();
        poDstDSOff = poDriverOff->Create( pszDstFilenameOff, pCount, lCount, 2, GDT_Int32,
                                         papszOptions );
    
        poDstDSOff->SetGeoTransform( adfGeoTransform );
        poDstDSOff->SetProjection( pszSRS_WKT );
    //    CPLFree( pszSRS_WKT );
    
//        GDALRasterBand *poBand1Off;
//        GDALRasterBand *poBand2Off;
        poBand1Off = poDstDSOff->GetRasterBand(1);
        poBand2Off = poDstDSOff->GetRasterBand(2);
        poBand1Off->SetNoDataValue(nodata_out);
        poBand2Off->SetNoDataValue(nodata_out);
        
    }
    
    if ((dhdxname != "")&(srxname != ""))
    {
    
        GDALDriver *poDriverSch;
        poDriverSch = GetGDALDriverManager()->GetDriverByName(pszFormat);
        if( poDriverSch == NULL )
        exit(107);
//        GDALDataset *poDstDSSch;
        
        str = searchrangename;
        const char * pszDstFilenameSch = str.c_str();
        poDstDSSch = poDriverSch->Create( pszDstFilenameSch, pCount, lCount, 2, GDT_Int32,
                                         papszOptions );
        
        poDstDSSch->SetGeoTransform( adfGeoTransform );
        poDstDSSch->SetProjection( pszSRS_WKT );
        //    CPLFree( pszSRS_WKT );
        
//        GDALRasterBand *poBand1Sch;
//        GDALRasterBand *poBand2Sch;
        poBand1Sch = poDstDSSch->GetRasterBand(1);
        poBand2Sch = poDstDSSch->GetRasterBand(2);
        poBand1Sch->SetNoDataValue(nodata_out);
        poBand2Sch->SetNoDataValue(nodata_out);
    
    }
    
    if (csminxname != "")
    {
        
        GDALDriver *poDriverMin;
        poDriverMin = GetGDALDriverManager()->GetDriverByName(pszFormat);
        if( poDriverMin == NULL )
        exit(107);
//        GDALDataset *poDstDSMin;
        
        str = chipsizeminname;
        const char * pszDstFilenameMin = str.c_str();
        poDstDSMin = poDriverMin->Create( pszDstFilenameMin, pCount, lCount, 2, GDT_Int32,
                                         papszOptions );
        
        poDstDSMin->SetGeoTransform( adfGeoTransform );
        poDstDSMin->SetProjection( pszSRS_WKT );
        //    CPLFree( pszSRS_WKT );
        
//        GDALRasterBand *poBand1Min;
//        GDALRasterBand *poBand2Min;
        poBand1Min = poDstDSMin->GetRasterBand(1);
        poBand2Min = poDstDSMin->GetRasterBand(2);
        poBand1Min->SetNoDataValue(nodata_out);
        poBand2Min->SetNoDataValue(nodata_out);
        
    }
    
    
    if (csmaxxname != "")
    {
    
        GDALDriver *poDriverMax;
        poDriverMax = GetGDALDriverManager()->GetDriverByName(pszFormat);
        if( poDriverMax == NULL )
        exit(107);
//        GDALDataset *poDstDSMax;
        
        str = chipsizemaxname;
        const char * pszDstFilenameMax = str.c_str();
        poDstDSMax = poDriverMax->Create( pszDstFilenameMax, pCount, lCount, 2, GDT_Int32,
                                         papszOptions );
        
        poDstDSMax->SetGeoTransform( adfGeoTransform );
        poDstDSMax->SetProjection( pszSRS_WKT );
        //    CPLFree( pszSRS_WKT );
        
//        GDALRasterBand *poBand1Max;
//        GDALRasterBand *poBand2Max;
        poBand1Max = poDstDSMax->GetRasterBand(1);
        poBand2Max = poDstDSMax->GetRasterBand(2);
        poBand1Max->SetNoDataValue(nodata_out);
        poBand2Max->SetNoDataValue(nodata_out);
        
    }
    
    
    
    if (ssmname != "")
    {
    
        GDALDriver *poDriverMsk;
        poDriverMsk = GetGDALDriverManager()->GetDriverByName(pszFormat);
        if( poDriverMsk == NULL )
        exit(107);
//        GDALDataset *poDstDSMsk;
        
        str = stablesurfacemaskname;
        const char * pszDstFilenameMsk = str.c_str();
        poDstDSMsk = poDriverMsk->Create( pszDstFilenameMsk, pCount, lCount, 1, GDT_Int32,
                                         papszOptions );
        
        poDstDSMsk->SetGeoTransform( adfGeoTransform );
        poDstDSMsk->SetProjection( pszSRS_WKT );
        //    CPLFree( pszSRS_WKT );
        
//        GDALRasterBand *poBand1Msk;
        poBand1Msk = poDstDSMsk->GetRasterBand(1);
        poBand1Msk->SetNoDataValue(nodata_out);
        
    }
    
    
    if (dhdxname != "")
    {
    
        GDALDriver *poDriverRO2VX;
        poDriverRO2VX = GetGDALDriverManager()->GetDriverByName(pszFormat);
        if( poDriverRO2VX == NULL )
        exit(107);
//        GDALDataset *poDstDSRO2VX;
        
        str = ro2vx_name;
        const char * pszDstFilenameRO2VX = str.c_str();
        poDstDSRO2VX = poDriverRO2VX->Create( pszDstFilenameRO2VX, pCount, lCount, 3, GDT_Float64,
                                         papszOptions );
        
        poDstDSRO2VX->SetGeoTransform( adfGeoTransform );
        poDstDSRO2VX->SetProjection( pszSRS_WKT );
    //    CPLFree( pszSRS_WKT );
        
//        GDALRasterBand *poBand1RO2VX;
//        GDALRasterBand *poBand2RO2VX;
    //    GDALRasterBand *poBand3Los;
        poBand1RO2VX = poDstDSRO2VX->GetRasterBand(1);
        poBand2RO2VX = poDstDSRO2VX->GetRasterBand(2);
        poBand3RO2VX = poDstDSRO2VX->GetRasterBand(3);
        poBand1RO2VX->SetNoDataValue(nodata_out);
        poBand2RO2VX->SetNoDataValue(nodata_out);
        poBand3RO2VX->SetNoDataValue(nodata_out);
        

        GDALDriver *poDriverRO2VY;
        poDriverRO2VY = GetGDALDriverManager()->GetDriverByName(pszFormat);
        if( poDriverRO2VY == NULL )
        exit(107);
//        GDALDataset *poDstDSRO2VY;
        
        str = ro2vy_name;
        const char * pszDstFilenameRO2VY = str.c_str();
        poDstDSRO2VY = poDriverRO2VY->Create( pszDstFilenameRO2VY, pCount, lCount, 3, GDT_Float64,
                                         papszOptions );
        
        poDstDSRO2VY->SetGeoTransform( adfGeoTransform );
        poDstDSRO2VY->SetProjection( pszSRS_WKT );
//        CPLFree( pszSRS_WKT );
        
//        GDALRasterBand *poBand1RO2VY;
//        GDALRasterBand *poBand2RO2VY;
    //    GDALRasterBand *poBand3Alt;
        poBand1RO2VY = poDstDSRO2VY->GetRasterBand(1);
        poBand2RO2VY = poDstDSRO2VY->GetRasterBand(2);
        poBand3RO2VY = poDstDSRO2VY->GetRasterBand(3);
        poBand1RO2VY->SetNoDataValue(nodata_out);
        poBand2RO2VY->SetNoDataValue(nodata_out);
        poBand3RO2VY->SetNoDataValue(nodata_out);
        
        
        GDALDriver *poDriverSF;
        poDriverSF = GetGDALDriverManager()->GetDriverByName(pszFormat);
        if( poDriverSF == NULL )
        exit(107);
        
        str = sfname;
        const char * pszDstFilenameSF = str.c_str();
        poDstDSSF = poDriverSF->Create( pszDstFilenameSF, pCount, lCount, 2, GDT_Float64,
                                         papszOptions );
        
        poDstDSSF->SetGeoTransform( adfGeoTransform );
        poDstDSSF->SetProjection( pszSRS_WKT );
    //    CPLFree( pszSRS_WKT );
        
//        GDALRasterBand *poBand1RO2VX;
//        GDALRasterBand *poBand2RO2VX;
    //    GDALRasterBand *poBand3Los;
        poBand1SF = poDstDSSF->GetRasterBand(1);
        poBand2SF = poDstDSSF->GetRasterBand(2);
        poBand1SF->SetNoDataValue(nodata_out);
        poBand2SF->SetNoDataValue(nodata_out);
        
        
    }
    
    CPLFree( pszSRS_WKT );

    
    
    
    
    // ground range and azimuth pixel size
//    double grd_res, azm_res;
    
//    double incang = 38.0*deg2rad;
    double incang = incidenceAngle;
    double grd_res = dr / std::sin(incang);
    double azm_res = norm_C(&satvmid) / prf;
    std::cout << "Ground range pixel size: " << grd_res << "\n";
    std::cout << "Azimuth pixel size: " << azm_res << "\n";
//    int ChipSizeX0 = 240;
    double ChipSizeX0 = chipSizeX0;
    int ChipSizeX0_PIX_grd = std::ceil(ChipSizeX0 / grd_res / 4) * 4;
    int ChipSizeX0_PIX_azm = std::ceil(ChipSizeX0 / azm_res / 4) * 4;
    
    
    
    for (int ii=0; ii<lCount; ii++)
    {
        double y = geoTrans[3] + (lOff+ii+0.5) * geoTrans[5];
        int status = demDS->GetRasterBand(1)->RasterIO(GF_Read,
                        pOff, lOff+ii,
                        pCount, 1,
                        (void*) (demLine.data()),
                        pCount, 1, GDT_Float64,
                        sizeof(double), sizeof(double)*pCount, NULL);
        
        if (status != 0)
        {
            std::cout << "Error read line " << lOff + ii << " from DEM file: " << demname << "\n";
            GDALClose(demDS);
            GDALDestroyDriverManager();
            exit(105);
        }
        
        if (dhdxname != "")
        {
            status = sxDS->GetRasterBand(1)->RasterIO(GF_Read,
                                                       pOff, lOff+ii,
                                                       pCount, 1,
                                                       (void*) (sxLine.data()),
                                                       pCount, 1, GDT_Float64,
                                                       sizeof(double), sizeof(double)*pCount, NULL);
            
            if (status != 0)
            {
                std::cout << "Error read line " << lOff + ii << " from x-direction slope file: " << dhdxname << "\n";
                GDALClose(sxDS);
                GDALDestroyDriverManager();
                exit(105);
            }
            
            
            status = syDS->GetRasterBand(1)->RasterIO(GF_Read,
                                                      pOff, lOff+ii,
                                                      pCount, 1,
                                                      (void*) (syLine.data()),
                                                      pCount, 1, GDT_Float64,
                                                      sizeof(double), sizeof(double)*pCount, NULL);
            
            if (status != 0)
            {
                std::cout << "Error read line " << lOff + ii << " from y-direction slope file: " << dhdyname << "\n";
                GDALClose(syDS);
                GDALDestroyDriverManager();
                exit(105);
            }
            
        }
        
        if (vxname != "")
        {
            status = vxDS->GetRasterBand(1)->RasterIO(GF_Read,
                                                      pOff, lOff+ii,
                                                      pCount, 1,
                                                      (void*) (vxLine.data()),
                                                      pCount, 1, GDT_Float64,
                                                      sizeof(double), sizeof(double)*pCount, NULL);
            
            if (status != 0)
            {
                std::cout << "Error read line " << lOff + ii << " from x-direction velocity file: " << vxname << "\n";
                GDALClose(vxDS);
                GDALDestroyDriverManager();
                exit(105);
            }
            
            
            status = vyDS->GetRasterBand(1)->RasterIO(GF_Read,
                                                      pOff, lOff+ii,
                                                      pCount, 1,
                                                      (void*) (vyLine.data()),
                                                      pCount, 1, GDT_Float64,
                                                      sizeof(double), sizeof(double)*pCount, NULL);
            
            if (status != 0)
            {
                std::cout << "Error read line " << lOff + ii << " from y-direction velocity file: " << vyname << "\n";
                GDALClose(vyDS);
                GDALDestroyDriverManager();
                exit(105);
            }
        }
        
        if (srxname != "")
        {
            status = srxDS->GetRasterBand(1)->RasterIO(GF_Read,
                                                      pOff, lOff+ii,
                                                      pCount, 1,
                                                      (void*) (srxLine.data()),
                                                      pCount, 1, GDT_Float64,
                                                      sizeof(double), sizeof(double)*pCount, NULL);
            
            if (status != 0)
            {
                std::cout << "Error read line " << lOff + ii << " from x-direction search range file: " << srxname << "\n";
                GDALClose(srxDS);
                GDALDestroyDriverManager();
                exit(105);
            }
            
            
            status = sryDS->GetRasterBand(1)->RasterIO(GF_Read,
                                                      pOff, lOff+ii,
                                                      pCount, 1,
                                                      (void*) (sryLine.data()),
                                                      pCount, 1, GDT_Float64,
                                                      sizeof(double), sizeof(double)*pCount, NULL);
            
            if (status != 0)
            {
                std::cout << "Error read line " << lOff + ii << " from y-direction search range file: " << sryname << "\n";
                GDALClose(sryDS);
                GDALDestroyDriverManager();
                exit(105);
            }
        }
        
        
        if (csminxname != "")
        {
            status = csminxDS->GetRasterBand(1)->RasterIO(GF_Read,
                                                      pOff, lOff+ii,
                                                      pCount, 1,
                                                      (void*) (csminxLine.data()),
                                                      pCount, 1, GDT_Float64,
                                                      sizeof(double), sizeof(double)*pCount, NULL);
            
            if (status != 0)
            {
                std::cout << "Error read line " << lOff + ii << " from x-direction chip size min file: " << csminxname << "\n";
                GDALClose(csminxDS);
                GDALDestroyDriverManager();
                exit(105);
            }
            
            
            status = csminyDS->GetRasterBand(1)->RasterIO(GF_Read,
                                                      pOff, lOff+ii,
                                                      pCount, 1,
                                                      (void*) (csminyLine.data()),
                                                      pCount, 1, GDT_Float64,
                                                      sizeof(double), sizeof(double)*pCount, NULL);
            
            if (status != 0)
            {
                std::cout << "Error read line " << lOff + ii << " from y-direction chip size min file: " << csminyname << "\n";
                GDALClose(csminyDS);
                GDALDestroyDriverManager();
                exit(105);
            }
        }
        
        
        if (csmaxxname != "")
        {
            status = csmaxxDS->GetRasterBand(1)->RasterIO(GF_Read,
                                                      pOff, lOff+ii,
                                                      pCount, 1,
                                                      (void*) (csmaxxLine.data()),
                                                      pCount, 1, GDT_Float64,
                                                      sizeof(double), sizeof(double)*pCount, NULL);
            
            if (status != 0)
            {
                std::cout << "Error read line " << lOff + ii << " from x-direction chip size max file: " << csmaxxname << "\n";
                GDALClose(csmaxxDS);
                GDALDestroyDriverManager();
                exit(105);
            }
            
            
            status = csmaxyDS->GetRasterBand(1)->RasterIO(GF_Read,
                                                      pOff, lOff+ii,
                                                      pCount, 1,
                                                      (void*) (csmaxyLine.data()),
                                                      pCount, 1, GDT_Float64,
                                                      sizeof(double), sizeof(double)*pCount, NULL);
            
            if (status != 0)
            {
                std::cout << "Error read line " << lOff + ii << " from y-direction chip size max file: " << csmaxyname << "\n";
                GDALClose(csmaxyDS);
                GDALDestroyDriverManager();
                exit(105);
            }
        }
        
        
        
        if (ssmname != "")
        {
            status = ssmDS->GetRasterBand(1)->RasterIO(GF_Read,
                                                          pOff, lOff+ii,
                                                          pCount, 1,
                                                          (void*) (ssmLine.data()),
                                                          pCount, 1, GDT_Float64,
                                                          sizeof(double), sizeof(double)*pCount, NULL);
            
            if (status != 0)
            {
                std::cout << "Error read line " << lOff + ii << " from stable surface mask file: " << ssmname << "\n";
                GDALClose(ssmDS);
                GDALDestroyDriverManager();
                exit(105);
            }
            
        }
        
        
        
        
        int rgind;
        int azind;
        
        for (int jj=0; jj<pCount; jj++)
        {
            isce3::core::Vec3 xyz;
            double llh[3];
            isce3::core::Vec3 llhv;
            double targllh0[3];
            isce3::core::Vec3 llhi;
            isce3::core::Vec3 drpos;
            //double drpos[3];
            double slp[3];
            double vel[3];
            double schrng1[3];
            double schrng2[3];
            

            //Setup ENU with DEM
            llh[0] = geoTrans[0] + (jj+pOff+0.5)*geoTrans[1];
            llh[1] = y;
            llh[2] = demLine[jj];
            
            for(int pp=0; pp<3; pp++)
            {
                targllh0[pp] = llh[pp];
            }
            
            if (dhdxname != "")
            {
                slp[0] = sxLine[jj];
                slp[1] = syLine[jj];
                slp[2] = -1.0;
            }
            
            if (vxname != "")
            {
                vel[0] = vxLine[jj];
                vel[1] = vyLine[jj];
            }
            
            if (srxname != "")
            {
                schrng1[0] = srxLine[jj];
                schrng1[1] = sryLine[jj];
            
                schrng1[0] *= std::max(max_factor*((dt_unity-1)*max_factor+(max_factor-1)-(max_factor-1)*dt/24.0/3600.0)/((dt_unity-1)*max_factor),1.0);
                schrng1[0] = std::min(std::max(schrng1[0],lower_thld),upper_thld);
                schrng1[1] *= std::max(max_factor*((dt_unity-1)*max_factor+(max_factor-1)-(max_factor-1)*dt/24.0/3600.0)/((dt_unity-1)*max_factor),1.0);
                schrng1[1] = std::min(std::max(schrng1[1],lower_thld),upper_thld);
            
                schrng2[0] = -schrng1[0];
                schrng2[1] = schrng1[1];
            }
            

            //Convert from DEM coordinates to LLH inplace
            fwdTrans->Transform(1, llh, llh+1, llh+2);
            

            //Bringing it into ISCE
            llhi[0] = deg2rad * llh[1];
            llhi[1] = deg2rad * llh[0];
            llhv[0] = llh[1];
            llhv[1] = llh[0];
            
            llhi[2] = llh[2];
            llhv[2] = llh[2];
            //std::cout << llh[0] << llh[1] << "\n";
            //std::cout << "LLHV " << llhv[0] << " " << llhv[1] << " " << llhv[2] << "\n";

            //Convert to ECEF
            wgs84.lonLatToXyz(llhi,xyz);
            //std::cout << "XYZ " << xyz[0] << " " << xyz[1] << " " << xyz[2] << "\n";
            isce3::core::Vec3 llhvv;
            wgs84.xyzToLonLat(xyz,llhvv);
            //std::cout << "LLHVV " << llhvv[0]/deg2rad << " " << llhvv[1]/deg2rad << " " << llhvv[2] << "\n";
            //latlon_C(&wgs84, xyz, llhi, LLH_2_XYZ);
            
//            if ((ii == (lCount+1)/2)&(jj == pCount/2)){
//                std::cout << xyz[0] << " " << xyz[1] << " " << xyz[2] << "\n";
//            }
            
            //Start the geo2rdr algorithm
            isce3::core::Vec3 satx;
            isce3::core::Vec3 satv;
            double tprev;
            
            double tlined = tmidd.secondsSinceEpoch();
            double tline = tmid;
            double rngpix;
            isce3::core::Vec3 los;
            isce3::core::Vec3 vhat;
            isce3::core::Vec3 temp;
            isce3::core::Vec3 nhat;
            isce3::core::Vec3 chat;
            isce3::core::Vec3 that;
            isce3::core::Vec3 delta;
            isce3::core::Vec3 targVec;
            isce3::core::Vec3 targXYZ;
            isce3::core::Vec3 diffvec;
            isce3::core::Vec3 alt;
            isce3::core::Vec3 normal;
            isce3::core::Vec3 cross;
            isce3::core::Vec3 da;
            
            //double alt[3];
            //double normal[3];
            //double cross[3];
            double cross_check;
            
            double dopfact;
            double height;
            double satvc[3], altc[3];
            double vmag;
            double major, minor;
            double satDist;
            double alpha, beta, gamma;
            double radius, hgt, zsch;
            double a, b, costheta, sintheta;
            double rdiff;
            //double da[3];
            
            for(int kk=0; kk<3; kk++) 
            {
                satx[kk]  = satxmid[kk];
            }

            for(int kk=0; kk<3; kk++)
            {
                satv[kk]  = satvmid[kk];
            }

            //Iterations
            for (int kk=0; kk<51;kk++)
            {
                tprev = tline;
                
                for(int pp=0; pp<3; pp++)
                {
                    drpos[pp] = xyz[pp] - satx[pp];
                }
                
                rngpix = norm_C(&drpos);
                //std::cout << "XYZ: X: " << xyz[0] << " Y: " << xyz[1] << " Z: " << xyz[2] << "\n";
                //std::cout << "SATX: X: " << satx[0] << " Y: " << satx[1] << " Z: " << satx[2] << "\n";
                //std::cout << "DRPOS: X: " << drpos[0] << " Y: " << drpos[1] << " Z: " << drpos[2] << "\n";
                //std::cout << "RNGPIX: X: " << rngpix << "\n";
                double fn = dot_C(&drpos, &satv);
                double fnprime = -dot_C(&satv, &satv);
                
                tline = tline - fn/fnprime;
                tlined = tlined - fn/fnprime;
                //std::cout << "Tlined: " << tline << "\n";
                //if (interpolateWGS84Orbit(orbit, tline, satx, satv) != 0)
                if (orbit.interpolate(&satx, &satv, tlined, border_mode) != ErrorCode::Success)
                {
                    std::cout << "Error with orbit interpolation. \n";
                    GDALClose(demDS);
                    GDALDestroyDriverManager();
                    exit(106);
                }

            }
//            if ((ii==600)&&(jj==600))
//            {
//                std::cout << "\n" << lOff+ii << " " << pOff+jj << " " << demLine[jj] << "\n";
//            }
            rgind = std::round((rngpix - startingRange) / dr) + 0.;
            azind = std::round((tline - sensingStart) * prf) + 0.;
            
            
            //*********************Slant-range vector
            
            
            unitvec_C2(&drpos, &los);
            
            for(int pp=0; pp<3; pp++)
            {
                llh[pp]  = xyz[pp] + los[pp] * dr;
            }
            llhv[0]=llh[0];
            llhv[1]=llh[1];
            llhv[2]=llh[2];
            
            wgs84.xyzToLonLat(llhv, llhi);
            //Bringing it from ISCE into LLH
            llh[0] = llhi[1] / deg2rad;
            llh[1] = llhi[0] / deg2rad;
            
            llh[2] = llhi[2];
            
            //Convert from LLH inplace to DEM coordinates
            invTrans->Transform(1, llh, llh+1, llh+2);
            for(int pp=0; pp<3; pp++)
            {
                drpos[pp]  = llh[pp] - targllh0[pp];
            }
            unitvec_C2(&drpos, &los);
            //*********************Along-track vector
            
            tline = tline + 1/prf;
            tlined = tlined + 1/prf;
            if (orbit.interpolate(&satx, &satv, tlined, border_mode) != ErrorCode::Success)
            {
                std::cout << "Error with orbit interpolation. \n";
                GDALClose(demDS);
                GDALDestroyDriverManager();
                exit(106);
            }
            //run the topo algorithm for new tline
            dopfact = 0.0;
            height = demLine[jj];
            //std::cout << "Height" << height << "\n";
            unitvec_C2(&satv, &vhat);
            vmag = norm_C(&satv);
            
            //Convert position and velocity to local tangent plane
            major = wgs84.a();
            minor = major * std::sqrt(1 - wgs84.e2());
            //Setup ortho normal system right below satellite
            satDist = norm_C(&satx);
            temp[0] = (satx[0] / major);
            temp[1] = (satx[1] / major);
            temp[2] = (satx[2] / minor);
            alpha = 1 / norm_C(&temp);
            radius = alpha * satDist;
            hgt = (1.0 - alpha) * satDist;
            
            //Setup TCN basis - Geocentric
            unitvec_C2(&satx, &nhat);
            for(int pp=0; pp<3; pp++)
            {
                nhat[pp]  = -nhat[pp];
            }
            cross_C2(&nhat,&satv,&temp);
            unitvec_C2(&temp, &chat);
            cross_C2(&chat,&nhat,&temp);
            unitvec_C2(&temp, &that);
            
            //Solve the range doppler eqns iteratively
            //Initial guess
            zsch = height;
            int lookSide = -1;
            for (int kk=0; kk<10;kk++)
            {
                a = satDist;
                b = (radius + zsch);
                
                costheta = 0.5 * (a / rngpix + rngpix / a - (b / a) * (b / rngpix));
                sintheta = std::sqrt(1-costheta*costheta);
                
                gamma = rngpix * costheta;
                alpha = dopfact - gamma * dot_C(&nhat,&vhat) / dot_C(&vhat,&that);
                beta = -lookSide * std::sqrt(rngpix * rngpix * sintheta * sintheta - alpha * alpha);
                
                for(int pp=0; pp<3; pp++)
                {
                    delta[pp] = alpha * that[pp] + beta * chat[pp] + gamma * nhat[pp];
                }
                for(int pp=0; pp<3; pp++)
                {
                    targVec[pp] = satx[pp] + delta[pp];
                }
                wgs84.xyzToLonLat(targVec, llhi);
                llhi[2] = height;
                
                llhv[0] = llhi[0] / deg2rad;
                llhv[1] = llhi[1] / deg2rad;
            
                llhv[2] = llhi[2];
                wgs84.lonLatToXyz(llhi,targXYZ);
                
                zsch = norm_C(&targXYZ) - radius;
                
                for(int pp=0; pp<3; pp++)
                {
                    diffvec[pp] = satx[pp] - targXYZ[pp];
                }
                rdiff  = rngpix - norm_C(&diffvec);
            }
            
            //Bringing it from ISCE into LLH
            
            llh[0] = llhi[1] / deg2rad;
            llh[1] = llhi[0] / deg2rad;
            
            llh[2] = llhi[2];
            
            //wgs84.xyzToLonLat(targXYZ, llhvv);
            //Convert from LLH inplace to DEM coordinates
            invTrans->Transform(1, llh, llh+1, llh+2);
            //std::cout << "VOY BIEN targ!!!: LON: " << targllh0[0] << " LAT: " << targllh0[1] << "\n";
            //std::cout << "VOY BIEN llh!!!: LON: " << llh[0] << " LAT: " << llh[1] << "\n";
            //exit(0);
            for(int pp=0; pp<3; pp++)
            {
                alt[pp]  = llh[pp] - targllh0[pp];
            }
            unitvec_C2(&alt, &temp);
            
            
            if (dhdxname != "")
            {
                //*********************Local normal vector
                isce3::core::Vec3 slpv;
                slpv[0]=slp[0];
                slpv[1]=slp[1];
                slpv[2]=slp[2];
                unitvec_C2(&slpv, &normal);
                for(int pp=0; pp<3; pp++)
                {
                    normal[pp]  = -normal[pp];
                }
            }
            else
            {
                for(int pp=0; pp<3; pp++)
                {
                    normal[pp]  = 0.0;
                }
            }
            
            if (vxname != "")
            {
                vel[2] = -(vel[0]*normal[0]+vel[1]*normal[1])/normal[2];
            }
            
            if (srxname != "")
            {
                schrng1[2] = -(schrng1[0]*normal[0]+schrng1[1]*normal[1])/normal[2];
                schrng2[2] = -(schrng2[0]*normal[0]+schrng2[1]*normal[1])/normal[2];
            }
            
            //std::cout << "PARAMS: " << rgind << " " << azind << " "<< nLines << " "<< nPixels << "\n";
            if ((rgind > nPixels-1)|(rgind < 1-1)|(azind > nLines-1)|(azind < 1-1))
            {
                raster1[jj] = nodata_out;
                raster2[jj] = nodata_out;
                raster11[jj] = nodata_out;
                raster22[jj] = nodata_out;
                
                sr_raster11[jj] = nodata_out;
                sr_raster22[jj] = nodata_out;
                csmin_raster11[jj] = nodata_out;
                csmin_raster22[jj] = nodata_out;
                csmax_raster11[jj] = nodata_out;
                csmax_raster22[jj] = nodata_out;
                ssm_raster[jj] = nodata_out;
                
                raster1a[jj] = nodata_out;
                raster1b[jj] = nodata_out;
                raster1c[jj] = nodata_out;
                raster2a[jj] = nodata_out;
                raster2b[jj] = nodata_out;
                raster2c[jj] = nodata_out;
                
                sf_raster1[jj] = nodata_out;
                sf_raster2[jj] = nodata_out;
                
            }
            else
            {
                //std::cout << "ESTOY ESCRIBIENDO: \n";
                raster1[jj] = rgind;
                raster2[jj] = azind;
                
                if (dhdxname != "")
                {
                    
                    if (vxname != "")
                    {
                        if (vel[0] == nodata)
                        {
                            raster11[jj] = 0.;
                            raster22[jj] = 0.;
                        }
                        else
                        {
                            isce3::core::Vec3 velv;
                            velv[0]=vel[0];
                            velv[1]=vel[1];
                            velv[2]=vel[2];
                            raster11[jj] = std::round(dot_C(&velv,&los)*dt/norm_C(&drpos)/365.0/24.0/3600.0*1);
                            raster22[jj] = std::round(dot_C(&velv,&temp)*dt/norm_C(&alt)/365.0/24.0/3600.0*1);
                        }
                      
                    }
                    
                    cross_C2(&los,&temp,&cross);
                    unitvec_C2(&cross, &cross);
                    cross_check = std::abs(std::acos(dot_C(&normal,&cross))/deg2rad-90.0);
                    for(int pp=0; pp<3; pp++)
                    {
                        da[pp] = targXYZ[pp] - xyz[pp];
                    }
                    
                    if (cross_check > 1.0)
                    {
                        raster1a[jj] = normal[2]/(dt/dr/365.0/24.0/3600.0)*(normal[2]*temp[1]-normal[1]*temp[2])/((normal[2]*los[0]-normal[0]*los[2])*(normal[2]*temp[1]-normal[1]*temp[2])-(normal[2]*temp[0]-normal[0]*temp[2])*(normal[2]*los[1]-normal[1]*los[2]));
                        raster1b[jj] = -normal[2]/(dt/norm_C(&da)/365.0/24.0/3600.0)*(normal[2]*los[1]-normal[1]*los[2])/((normal[2]*los[0]-normal[0]*los[2])*(normal[2]*temp[1]-normal[1]*temp[2])-(normal[2]*temp[0]-normal[0]*temp[2])*(normal[2]*los[1]-normal[1]*los[2]));
                        raster2a[jj] = -normal[2]/(dt/dr/365.0/24.0/3600.0)*(normal[2]*temp[0]-normal[0]*temp[2])/((normal[2]*los[0]-normal[0]*los[2])*(normal[2]*temp[1]-normal[1]*temp[2])-(normal[2]*temp[0]-normal[0]*temp[2])*(normal[2]*los[1]-normal[1]*los[2]));
                        raster2b[jj] = normal[2]/(dt/norm_C(&da)/365.0/24.0/3600.0)*(normal[2]*los[0]-normal[0]*los[2])/((normal[2]*los[0]-normal[0]*los[2])*(normal[2]*temp[1]-normal[1]*temp[2])-(normal[2]*temp[0]-normal[0]*temp[2])*(normal[2]*los[1]-normal[1]*los[2]));
                    }
                    else
                    {
                        raster1a[jj] = nodata_out;
                        raster1b[jj] = nodata_out;
                        raster2a[jj] = nodata_out;
                        raster2b[jj] = nodata_out;
                    }
                    
                    // range/azimuth pixel displacement to range/azimuth velocity
                    raster1c[jj] = dr/dt*365.0*24.0*3600.0*1;
                    raster2c[jj] = norm_C(&da)/dt*365.0*24.0*3600.0*1;
                    
                    // range/azimuth distance to DEM distance scale factor
                    sf_raster1[jj] = norm_C(&drpos) / dr;
                    sf_raster2[jj] = norm_C(&alt) / norm_C(&da);
                    //std::cout << "OFFSET " << alt[0] << " " << alt[1] << " " << alt[2] << "\n";
                    //std::cout << "DA " << da[0] << " " << da[1] << " " << da[2] << "\n";
                    //std::cout << "Norm ALT " << norm_C(&alt) << "\n";
                    //std::cout << "Norm DA " << norm_C(&da) << "\n";
                    //exit(0);
                    
                    if (srxname != "")
                    {
                        if ((schrng1[0] == nodata)|(schrng1[0] == 0))
                        {
                            sr_raster11[jj] = 0;
                            sr_raster22[jj] = 0;
                        }
                        else
                        {
                            isce3::core::Vec3 schrng1v;
                            isce3::core::Vec3 schrng2v;
                            schrng1v[0]=schrng1[0];
                            schrng1v[1]=schrng1[1];
                            schrng1v[2]=schrng1[2];
                            schrng2v[0]=schrng2[0];
                            schrng2v[1]=schrng2[1];
                            schrng2v[2]=schrng2[2];
                            sr_raster11[jj] = std::abs(std::round(dot_C(&schrng1v,&los)*dt/dr/365.0/24.0/3600.0*1));
                            sr_raster22[jj] = std::abs(std::round(dot_C(&schrng1v,&temp)*dt/norm_C(&alt)/365.0/24.0/3600.0*1));
                            if (std::abs(std::round(dot_C(&schrng2v,&los)*dt/dr/365.0/24.0/3600.0*1)) > sr_raster11[jj])
                            {
                                sr_raster11[jj] = std::abs(std::round(dot_C(&schrng2v,&los)*dt/dr/365.0/24.0/3600.0*1));
                            }
                            if (std::abs(std::round(dot_C(&schrng2v,&temp)*dt/norm_C(&alt)/365.0/24.0/3600.0*1)) > sr_raster22[jj])
                            {
                                sr_raster22[jj] = std::abs(std::round(dot_C(&schrng2v,&temp)*dt/norm_C(&alt)/365.0/24.0/3600.0*1));
                            }
                            if (sr_raster11[jj] == 0)
                            {
                                sr_raster11[jj] = 1;
                            }
                            if (sr_raster22[jj] == 0)
                            {
                                sr_raster22[jj] = 1;
                            }
                        }
                    }
 
                }
                
                
                
                if (csminxname != "")
                {
                    if (csminxLine[jj] == nodata)
                    {
                        csmin_raster11[jj] = nodata_out;
                        csmin_raster22[jj] = nodata_out;
                    }
                    else
                    {
                        csmin_raster11[jj] = csminxLine[jj] / ChipSizeX0 * ChipSizeX0_PIX_grd;
                        csmin_raster22[jj] = csminyLine[jj] / ChipSizeX0 * ChipSizeX0_PIX_azm;
                    }
                }
                
                
                if (csmaxxname != "")
                {
                    if (csmaxxLine[jj] == nodata)
                    {
                        csmax_raster11[jj] = nodata_out;
                        csmax_raster22[jj] = nodata_out;
                    }
                    else
                    {
                        csmax_raster11[jj] = csmaxxLine[jj] / ChipSizeX0 * ChipSizeX0_PIX_grd;
                        csmax_raster22[jj] = csmaxyLine[jj] / ChipSizeX0 * ChipSizeX0_PIX_azm;
                    }
                }
                
                
                if (ssmname != "")
                {
                    if (ssmLine[jj] == nodata)
                    {
                        ssm_raster[jj] = nodata_out;
                    }
                    else
                    {
                        ssm_raster[jj] = ssmLine[jj];
                    }
                }
                
                
                

            }
            
            
//            std::cout << ii << " " << jj << "\n";
//            std::cout << rgind << " " << azind << "\n";
//            std::cout << raster1[jj][ii] << " " << raster2[jj][ii] << "\n";
//            std::cout << raster1[ii][jj] << "\n";
        }
        
        
        
        poBand1->RasterIO( GF_Write, 0, ii, pCount, 1,
                          raster1, pCount, 1, GDT_Int32, 0, 0 );
        poBand2->RasterIO( GF_Write, 0, ii, pCount, 1,
                          raster2, pCount, 1, GDT_Int32, 0, 0 );
        
        if ((dhdxname != "")&(vxname != ""))
        {
            poBand1Off->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 raster11, pCount, 1, GDT_Int32, 0, 0 );
            poBand2Off->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 raster22, pCount, 1, GDT_Int32, 0, 0 );
        }
        
        if ((dhdxname != "")&(srxname != ""))
        {
            poBand1Sch->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 sr_raster11, pCount, 1, GDT_Int32, 0, 0 );
            poBand2Sch->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 sr_raster22, pCount, 1, GDT_Int32, 0, 0 );
        }
        
        if (csminxname != "")
        {
            poBand1Min->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 csmin_raster11, pCount, 1, GDT_Int32, 0, 0 );
            poBand2Min->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 csmin_raster22, pCount, 1, GDT_Int32, 0, 0 );
        }
        
        if (csmaxxname != "")
        {
            poBand1Max->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 csmax_raster11, pCount, 1, GDT_Int32, 0, 0 );
            poBand2Max->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 csmax_raster22, pCount, 1, GDT_Int32, 0, 0 );
        }
        
        if (ssmname != "")
        {
            poBand1Msk->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 ssm_raster, pCount, 1, GDT_Int32, 0, 0 );
        }
        
        if (dhdxname != "")
        {
            poBand1RO2VX->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 raster1a, pCount, 1, GDT_Float64, 0, 0 );
            poBand2RO2VX->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 raster1b, pCount, 1, GDT_Float64, 0, 0 );
            poBand3RO2VX->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 raster1c, pCount, 1, GDT_Float64, 0, 0 );
            poBand1RO2VY->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 raster2a, pCount, 1, GDT_Float64, 0, 0 );
            poBand2RO2VY->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 raster2b, pCount, 1, GDT_Float64, 0, 0 );
            poBand3RO2VY->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 raster2c, pCount, 1, GDT_Float64, 0, 0 );
            poBand1SF->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 sf_raster1, pCount, 1, GDT_Float64, 0, 0 );
            poBand2SF->RasterIO( GF_Write, 0, ii, pCount, 1,
                                 sf_raster2, pCount, 1, GDT_Float64, 0, 0 );
            
        }
        
        
    }
    
    /* Once we're done, close properly the dataset */
    GDALClose( (GDALDatasetH) poDstDS );
    
    if ((dhdxname != "")&(vxname != ""))
    {
        /* Once we're done, close properly the dataset */
        GDALClose( (GDALDatasetH) poDstDSOff );
    }
    
    if ((dhdxname != "")&(srxname != ""))
    {
        /* Once we're done, close properly the dataset */
        GDALClose( (GDALDatasetH) poDstDSSch );
    }
    
    if (csminxname != "")
    {
        /* Once we're done, close properly the dataset */
        GDALClose( (GDALDatasetH) poDstDSMin );
    }
    
    if (csmaxxname != "")
    {
        /* Once we're done, close properly the dataset */
        GDALClose( (GDALDatasetH) poDstDSMax );
    }
    
    if (ssmname != "")
    {
        /* Once we're done, close properly the dataset */
        GDALClose( (GDALDatasetH) poDstDSMsk );
    }
    
    if (dhdxname != "")
    {
        /* Once we're done, close properly the dataset */
        GDALClose( (GDALDatasetH) poDstDSRO2VX );
        
        /* Once we're done, close properly the dataset */
        GDALClose( (GDALDatasetH) poDstDSRO2VY );
        
        /* Once we're done, close properly the dataset */
        GDALClose( (GDALDatasetH) poDstDSSF );
        
    }
    
    
    GDALClose(demDS);
    
    if (dhdxname != "")
    {
        GDALClose(sxDS);
        GDALClose(syDS);
    }
    
    if (vxname != "")
    {
        GDALClose(vxDS);
        GDALClose(vyDS);
    }
    
    if (srxname != "")
    {
        GDALClose(srxDS);
        GDALClose(sryDS);
    }
    
    if (csminxname != "")
    {
        GDALClose(csminxDS);
        GDALClose(csminyDS);
    }
    
    if (csmaxxname != "")
    {
        GDALClose(csmaxxDS);
        GDALClose(csmaxyDS);
    }
    
    if (ssmname != "")
    {
        GDALClose(ssmDS);
    }
    
    GDALDestroyDriverManager();
    
    std::ofstream myfile;
    myfile.open ("output.txt");
    myfile << "pOff: " << pOff << "\n";
    myfile << "lOff: " << lOff << "\n";
    myfile << "pCount: " << pCount << "\n";
    myfile << "lCount: " << lCount << "\n";
    myfile << "X_res: " << grd_res << "\n";
    myfile << "Y_res: " << azm_res << "\n";
    myfile.close();
    
    //return 0;
    
}

