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

#include "geogrid.h"
#include <gdal.h>
#include <gdal_priv.h>
#include <iostream>
#include <complex>
#include <cmath>


extern "C"
{
#include "linalg3.h"
#include "geometry.h"
}

void geoGrid::geogrid()
{
    //Some constants 
    double deg2rad = M_PI/180.0;

    //For now print inputs that were obtained
    
    std::cout << "\nRadar parameters: \n";
    std::cout << "Range: " << startingRange << "  " << dr << "\n";
    std::cout << "Azimuth: " << sensingStart << "  " << prf << "\n";
    std::cout << "Dimensions: " << nPixels << " " << nLines << "\n";

    std::cout << "\nMap inputs: \n";
    std::cout << "EPSG: " << epsgcode << "\n";
    std::cout << "Repeat Time: " << dt << "\n";
    std::cout << "XLimits: " << xmin << "  " << xmax << "\n";
    std::cout << "YLimits: " << ymin << "  " << ymax << "\n";
    std::cout << "Extent in km: " << (xmax - xmin)/1000. << "  " << (ymax - ymin)/1000. << "\n";
    std::cout << "DEM: " << demname << "\n";
    std::cout << "Velocities: " << vxname << "  " << vyname << "\n";
    std::cout << "Slopes: " << dhdxname << "  " << dhdyname << "\n";
    
    std::cout << "\nOutputs: \n";
    std::cout << "Window locations: " << pixlinename << "\n";
    std::cout << "Window offsets: " << offsetname << "\n";
    std::cout << "Window rdr_off2vel_x vector: " << ro2vx_name << "\n";
    std::cout << "Window rdr_off2vel_y vector: " << ro2vy_name << "\n";
    

    std::cout << "\nStarting processing .... \n";

    //Startup GDAL
    GDALAllRegister();

    //DEM related information
    GDALDataset* demDS = NULL;
    GDALDataset* sxDS = NULL;
    GDALDataset* syDS = NULL;
    GDALDataset* vxDS = NULL;
    GDALDataset* vyDS = NULL;

    double geoTrans[6];

    demDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(demname.c_str(), GA_ReadOnly));
    if (vxname != "")
    {
        sxDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(dhdxname.c_str(), GA_ReadOnly));
        syDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(dhdyname.c_str(), GA_ReadOnly));
        vxDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(vxname.c_str(), GA_ReadOnly));
        vyDS = reinterpret_cast<GDALDataset *>(GDALOpenShared(vyname.c_str(), GA_ReadOnly));
    }
    if (demDS == NULL)
    {
        std::cout << "Error opening DEM file { " << demname << " }\n";
        std::cout << "Exiting with error code .... (101) \n";
        GDALDestroyDriverManager();
        exit(101);
    }
    if (vxname != "")
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

    demDS->GetGeoTransform(geoTrans);
    int demXSize = demDS->GetRasterXSize();
    int demYSize = demDS->GetRasterYSize();


    //Get offsets and size to read from DEM
    int lOff = std::max( std::floor((ymax - geoTrans[3])/geoTrans[5]), 0.);
    int lCount = std::min( std::ceil((ymin - geoTrans[3])/geoTrans[5]), demYSize-1.) - lOff;

    int pOff = std::max( std::floor((xmin - geoTrans[0])/geoTrans[1]), 0.);
    int pCount = std::min( std::ceil((xmax - geoTrans[0])/geoTrans[1]), demXSize-1.) - pOff;


    std::cout << "Xlimits : " << geoTrans[0] + pOff * geoTrans[1] <<  "  " 
                             << geoTrans[0] + (pOff + pCount) * geoTrans[1] << "\n";


    std::cout << "Ylimits : " << geoTrans[3] + (lOff + lCount) * geoTrans[5] <<  "  "
                             << geoTrans[3] + lOff * geoTrans[5] << "\n";

    std::cout << "Dimensions of geogrid: " << pCount << " x " << lCount << "\n\n";


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
    cEllipsoid wgs84;
    wgs84.a = 6378137.0;
    wgs84.e2 = 0.0066943799901;

    //Initial guess for solution
    double tmid = sensingStart + 0.5 * nLines / prf;
    double satxmid[3];
    double satvmid[3];

    if (interpolateWGS84Orbit(orbit, tmid, satxmid, satvmid) != 0)
    {
        std::cout << "Error with orbit interpolation for setup. \n";
        GDALClose(demDS);
        GDALDestroyDriverManager();
        exit(104);
    }
//    std::cout << satxmid[0] << " " << satxmid[1] << " " << satxmid[2] << "\n";

    std::vector<double> demLine(pCount);
    std::vector<double> sxLine(pCount);
    std::vector<double> syLine(pCount);
    std::vector<double> vxLine(pCount);
    std::vector<double> vyLine(pCount);
    
    GInt32 raster1[pCount];
    GInt32 raster2[pCount];
    GInt32 raster11[pCount];
    GInt32 raster22[pCount];
    
    double raster1a[pCount];
    double raster1b[pCount];
//    double raster1c[pCount];
    
    double raster2a[pCount];
    double raster2b[pCount];
//    double raster2c[pCount];

    double nodata;
    double nodata_out;
    if (vxname != "")
    {
        int* pbSuccess = NULL;
        nodata = vxDS->GetRasterBand(1)->GetNoDataValue(pbSuccess);
    }
    nodata_out = -2000000000;
    
    const char *pszFormat = "GTiff";
    char **papszOptions = NULL;
    std::string str = "";
    double adfGeoTransform[6] = { xmin, (xmax-xmin)/pCount, 0, ymax, 0, (ymin-ymax)/lCount };
    OGRSpatialReference oSRS;
    char *pszSRS_WKT = NULL;
    demSRS.exportToWkt( &pszSRS_WKT );
    
    
    
    GDALDriver *poDriver;
    poDriver = GetGDALDriverManager()->GetDriverByName(pszFormat);
    if( poDriver == NULL )
    exit(107);
    GDALDataset *poDstDS;
    
    str = pixlinename;
    const char * pszDstFilename = str.c_str();
    poDstDS = poDriver->Create( pszDstFilename, pCount, lCount, 2, GDT_Int32,
                               papszOptions );
    
    
    poDstDS->SetGeoTransform( adfGeoTransform );
    poDstDS->SetProjection( pszSRS_WKT );
//    CPLFree( pszSRS_WKT );
    
    
    GDALRasterBand *poBand1;
    GDALRasterBand *poBand2;
    poBand1 = poDstDS->GetRasterBand(1);
    poBand2 = poDstDS->GetRasterBand(2);
    poBand1->SetNoDataValue(nodata_out);
    poBand2->SetNoDataValue(nodata_out);
    
    
    

    GDALDriver *poDriverOff;
    poDriverOff = GetGDALDriverManager()->GetDriverByName(pszFormat);
    if( poDriverOff == NULL )
    exit(107);
    GDALDataset *poDstDSOff;
    
    str = offsetname;
    const char * pszDstFilenameOff = str.c_str();
    poDstDSOff = poDriverOff->Create( pszDstFilenameOff, pCount, lCount, 2, GDT_Int32,
                                     papszOptions );
    
    poDstDSOff->SetGeoTransform( adfGeoTransform );
    poDstDSOff->SetProjection( pszSRS_WKT );
//    CPLFree( pszSRS_WKT );
    
    GDALRasterBand *poBand1Off;
    GDALRasterBand *poBand2Off;
    poBand1Off = poDstDSOff->GetRasterBand(1);
    poBand2Off = poDstDSOff->GetRasterBand(2);
    poBand1Off->SetNoDataValue(nodata_out);
    poBand2Off->SetNoDataValue(nodata_out);
    
    
    
    
    GDALDriver *poDriverRO2VX;
    poDriverRO2VX = GetGDALDriverManager()->GetDriverByName(pszFormat);
    if( poDriverRO2VX == NULL )
    exit(107);
    GDALDataset *poDstDSRO2VX;
    
    str = ro2vx_name;
    const char * pszDstFilenameRO2VX = str.c_str();
    poDstDSRO2VX = poDriverRO2VX->Create( pszDstFilenameRO2VX, pCount, lCount, 2, GDT_Float64,
                                     papszOptions );
    
    poDstDSRO2VX->SetGeoTransform( adfGeoTransform );
    poDstDSRO2VX->SetProjection( pszSRS_WKT );
//    CPLFree( pszSRS_WKT );
    
    GDALRasterBand *poBand1RO2VX;
    GDALRasterBand *poBand2RO2VX;
//    GDALRasterBand *poBand3Los;
    poBand1RO2VX = poDstDSRO2VX->GetRasterBand(1);
    poBand2RO2VX = poDstDSRO2VX->GetRasterBand(2);
//    poBand3Los = poDstDSLos->GetRasterBand(3);
    poBand1RO2VX->SetNoDataValue(nodata_out);
    poBand2RO2VX->SetNoDataValue(nodata_out);
//    poBand3Los->SetNoDataValue(nodata_out);
    
    

    GDALDriver *poDriverRO2VY;
    poDriverRO2VY = GetGDALDriverManager()->GetDriverByName(pszFormat);
    if( poDriverRO2VY == NULL )
    exit(107);
    GDALDataset *poDstDSRO2VY;
    
    str = ro2vy_name;
    const char * pszDstFilenameRO2VY = str.c_str();
    poDstDSRO2VY = poDriverRO2VY->Create( pszDstFilenameRO2VY, pCount, lCount, 2, GDT_Float64,
                                     papszOptions );
    
    poDstDSRO2VY->SetGeoTransform( adfGeoTransform );
    poDstDSRO2VY->SetProjection( pszSRS_WKT );
    CPLFree( pszSRS_WKT );
    
    GDALRasterBand *poBand1RO2VY;
    GDALRasterBand *poBand2RO2VY;
//    GDALRasterBand *poBand3Alt;
    poBand1RO2VY = poDstDSRO2VY->GetRasterBand(1);
    poBand2RO2VY = poDstDSRO2VY->GetRasterBand(2);
//    poBand3Alt = poDstDSAlt->GetRasterBand(3);
    poBand1RO2VY->SetNoDataValue(nodata_out);
    poBand2RO2VY->SetNoDataValue(nodata_out);
//    poBand3Alt->SetNoDataValue(nodata_out);
    
    
    
    
    
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
        
        if (vxname != "")
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
        
        int rgind;
        int azind;
        
        for (int jj=0; jj<pCount; jj++)
        {
            double xyz[3];
            double llh[3];
            double targllh0[3];
            double llhi[3];
            double drpos[3];
            double slp[3];
            double vel[3];

            //Setup ENU with DEM
            llh[0] = geoTrans[0] + (jj+pOff+0.5)*geoTrans[1];
            llh[1] = y;
            llh[2] = demLine[jj];
            
            for(int pp=0; pp<3; pp++)
            {
                targllh0[pp] = llh[pp];
            }
            
            if (vxname != "")
            {
                slp[0] = sxLine[jj];
                slp[1] = syLine[jj];
                slp[2] = -1.0;
                vel[0] = vxLine[jj];
                vel[1] = vyLine[jj];
            }
            

            //Convert from DEM coordinates to LLH inplace
            fwdTrans->Transform(1, llh, llh+1, llh+2);

            //Bringing it into ISCE
            llhi[0] = deg2rad * llh[1];
            llhi[1] = deg2rad * llh[0];
            llhi[2] = llh[2];

            //Convert to ECEF
            latlon_C(&wgs84, xyz, llhi, LLH_2_XYZ);
            
//            if ((ii == (lCount+1)/2)&(jj == pCount/2)){
//                std::cout << xyz[0] << " " << xyz[1] << " " << xyz[2] << "\n";
//            }
            
            //Start the geo2rdr algorithm
            double satx[3];
            double satv[3];
            double tprev;
            
            double tline = tmid;
            double rngpix;
            double los[3];
            double alt[3];
            double normal[3];
            
            double dopfact;
            double height;
            double vhat[3], that[3], chat[3], nhat[3], delta[3], targVec[3], targXYZ[3], diffvec[3], temp[3], satvc[3], altc[3];
            double vmag;
            double major, minor;
            double satDist;
            double alpha, beta, gamma;
            double radius, hgt, zsch;
            double a, b, costheta, sintheta;
            double rdiff;
            
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
                
                rngpix = norm_C(drpos);
                double fn = dot_C(drpos, satv);
                double fnprime = -dot_C(satv, satv);
                
                tline = tline - fn/fnprime;
                
                if (interpolateWGS84Orbit(orbit, tline, satx, satv) != 0)
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
            rgind = std::round((rngpix - startingRange) / dr) + 1.;
            azind = std::round((tline - sensingStart) * prf) + 1.;
            
            
            //*********************Slant-range vector
            
            
            unitvec_C(drpos, los);
            
            for(int pp=0; pp<3; pp++)
            {
                llh[pp]  = xyz[pp] + los[pp] * dr;
            }
            
            latlon_C(&wgs84, llh, llhi, XYZ_2_LLH);
            
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
            unitvec_C(drpos, los);
            
            //*********************Along-track vector
            
            tline = tline + 1/prf;
            
            if (interpolateWGS84Orbit(orbit, tline, satx, satv) != 0)
            {
                std::cout << "Error with orbit interpolation. \n";
                GDALClose(demDS);
                GDALDestroyDriverManager();
                exit(106);
            }
            //run the topo algorithm for new tline
            dopfact = 0.0;
            height = demLine[jj];
            unitvec_C(satv, vhat);
            vmag = norm_C(satv);
            
            //Convert position and velocity to local tangent plane
            major = wgs84.a;
            minor = major * std::sqrt(1 - wgs84.e2);
            
            //Setup ortho normal system right below satellite
            satDist = norm_C(satx);
            temp[0] = (satx[0] / major);
            temp[1] = (satx[1] / major);
            temp[2] = (satx[2] / minor);
            alpha = 1 / norm_C(temp);
            radius = alpha * satDist;
            hgt = (1.0 - alpha) * satDist;
            
            //Setup TCN basis - Geocentric
            unitvec_C(satx, nhat);
            for(int pp=0; pp<3; pp++)
            {
                nhat[pp]  = -nhat[pp];
            }
            cross_C(nhat,satv,temp);
            unitvec_C(temp, chat);
            cross_C(chat,nhat,temp);
            unitvec_C(temp, that);
            
            
            //Solve the range doppler eqns iteratively
            //Initial guess
            zsch = height;
            
            for (int kk=0; kk<10;kk++)
            {
                a = satDist;
                b = (radius + zsch);
                
                costheta = 0.5 * (a / rngpix + rngpix / a - (b / a) * (b / rngpix));
                sintheta = std::sqrt(1-costheta*costheta);
                
                gamma = rngpix * costheta;
                alpha = dopfact - gamma * dot_C(nhat,vhat) / dot_C(vhat,that);
                beta = -lookSide * std::sqrt(rngpix * rngpix * sintheta * sintheta - alpha * alpha);
                for(int pp=0; pp<3; pp++)
                {
                    delta[pp] = alpha * that[pp] + beta * chat[pp] + gamma * nhat[pp];
                }
                
                for(int pp=0; pp<3; pp++)
                {
                    targVec[pp] = satx[pp] + delta[pp];
                }
                
                latlon_C(&wgs84, targVec, llhi, XYZ_2_LLH);
                llhi[2] = height;
                latlon_C(&wgs84, targXYZ, llhi, LLH_2_XYZ);
                
                zsch = norm_C(targXYZ) - radius;
                
                for(int pp=0; pp<3; pp++)
                {
                    diffvec[pp] = satx[pp] - targXYZ[pp];
                }
                rdiff  = rngpix - norm_C(diffvec);
            }
            
            //Bringing it from ISCE into LLH
            llh[0] = llhi[1] / deg2rad;
            llh[1] = llhi[0] / deg2rad;
            llh[2] = llhi[2];
            
            //Convert from LLH inplace to DEM coordinates
            invTrans->Transform(1, llh, llh+1, llh+2);
            
            for(int pp=0; pp<3; pp++)
            {
                alt[pp]  = llh[pp] - targllh0[pp];
            }
            unitvec_C(alt, temp);
            
            
            if (vxname != "")
            {
                //*********************Local normal vector
                unitvec_C(slp, normal);
                for(int pp=0; pp<3; pp++)
                {
                    normal[pp]  = -normal[pp];
                }
                
                vel[2] = -(vel[0]*normal[0]+vel[1]*normal[1])/normal[2];
            }
            
            
            if ((rgind > nPixels)|(rgind < 1)|(azind > nLines)|(azind < 1))
            {
                raster1[jj] = nodata_out;
                raster2[jj] = nodata_out;
                raster11[jj] = nodata_out;
                raster22[jj] = nodata_out;
                raster1a[jj] = nodata_out;
                raster1b[jj] = nodata_out;
//                raster1c[jj] = nodata_out;
                raster2a[jj] = nodata_out;
                raster2b[jj] = nodata_out;
//                raster2c[jj] = nodata_out;
            }
            else
            {
                raster1[jj] = rgind;
                raster2[jj] = azind;
                if ((vxname != "")&(vel[0] != nodata))
                {
                    raster11[jj] = std::round(dot_C(vel,los)*dt/dr/365.0/24.0/3600.0*1);
                    raster22[jj] = std::round(dot_C(vel,temp)*dt/norm_C(alt)/365.0/24.0/3600.0*1);
                }
                else
                {
                    raster11[jj] = 0.;
                    raster22[jj] = 0.;
                }
                raster1a[jj] = normal[2]/(dt/dr/365.0/24.0/3600.0)*(normal[2]*temp[1]-normal[1]*temp[2])/((normal[2]*los[0]-normal[0]*los[2])*(normal[2]*temp[1]-normal[1]*temp[2])-(normal[2]*temp[0]-normal[0]*temp[2])*(normal[2]*los[1]-normal[1]*los[2]));
                raster1b[jj] = -normal[2]/(dt/norm_C(alt)/365.0/24.0/3600.0)*(normal[2]*los[1]-normal[1]*los[2])/((normal[2]*los[0]-normal[0]*los[2])*(normal[2]*temp[1]-normal[1]*temp[2])-(normal[2]*temp[0]-normal[0]*temp[2])*(normal[2]*los[1]-normal[1]*los[2]));
                raster2a[jj] = -normal[2]/(dt/dr/365.0/24.0/3600.0)*(normal[2]*temp[0]-normal[0]*temp[2])/((normal[2]*los[0]-normal[0]*los[2])*(normal[2]*temp[1]-normal[1]*temp[2])-(normal[2]*temp[0]-normal[0]*temp[2])*(normal[2]*los[1]-normal[1]*los[2]));
                raster2b[jj] = normal[2]/(dt/norm_C(alt)/365.0/24.0/3600.0)*(normal[2]*los[0]-normal[0]*los[2])/((normal[2]*los[0]-normal[0]*los[2])*(normal[2]*temp[1]-normal[1]*temp[2])-(normal[2]*temp[0]-normal[0]*temp[2])*(normal[2]*los[1]-normal[1]*los[2]));
                
//                raster1a[jj] = los[0]*dt/dr/365.0/24.0/3600.0;
//                raster1b[jj] = los[1]*dt/dr/365.0/24.0/3600.0;
//                raster1c[jj] = los[2]*dt/dr/365.0/24.0/3600.0;
//                raster2a[jj] = temp[0]*dt/norm_C(alt)/365.0/24.0/3600.0;
//                raster2b[jj] = temp[1]*dt/norm_C(alt)/365.0/24.0/3600.0;
//                raster2c[jj] = temp[2]*dt/norm_C(alt)/365.0/24.0/3600.0;
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
        poBand1Off->RasterIO( GF_Write, 0, ii, pCount, 1,
                             raster11, pCount, 1, GDT_Int32, 0, 0 );
        poBand2Off->RasterIO( GF_Write, 0, ii, pCount, 1,
                             raster22, pCount, 1, GDT_Int32, 0, 0 );
        poBand1RO2VX->RasterIO( GF_Write, 0, ii, pCount, 1,
                             raster1a, pCount, 1, GDT_Float64, 0, 0 );
        poBand2RO2VX->RasterIO( GF_Write, 0, ii, pCount, 1,
                             raster1b, pCount, 1, GDT_Float64, 0, 0 );
//        poBand3Los->RasterIO( GF_Write, 0, ii, pCount, 1,
//                             raster1c, pCount, 1, GDT_Float64, 0, 0 );
        poBand1RO2VY->RasterIO( GF_Write, 0, ii, pCount, 1,
                             raster2a, pCount, 1, GDT_Float64, 0, 0 );
        poBand2RO2VY->RasterIO( GF_Write, 0, ii, pCount, 1,
                             raster2b, pCount, 1, GDT_Float64, 0, 0 );
//        poBand3Alt->RasterIO( GF_Write, 0, ii, pCount, 1,
//                             raster2c, pCount, 1, GDT_Float64, 0, 0 );
        
    }
    
    /* Once we're done, close properly the dataset */
    GDALClose( (GDALDatasetH) poDstDS );
    
    /* Once we're done, close properly the dataset */
    GDALClose( (GDALDatasetH) poDstDSOff );
    
    /* Once we're done, close properly the dataset */
    GDALClose( (GDALDatasetH) poDstDSRO2VX );
    
    /* Once we're done, close properly the dataset */
    GDALClose( (GDALDatasetH) poDstDSRO2VY );
    
    GDALClose(demDS);
    
    if (vxname != "")
    {
        GDALClose(sxDS);
        GDALClose(syDS);
        GDALClose(vxDS);
        GDALClose(vyDS);
    }
    
    
    
    GDALDestroyDriverManager();
    
}
void geoGrid::computeBbox(double *wesn)
{
    std::cout << "\nEstimated bounding box: \n" 
              << "West: " << wesn[0] << "\n"
              << "East: " << wesn[1] << "\n"
              << "South: " << wesn[2] << "\n"
              << "North: " << wesn[3] << "\n";
}
