#!/usr/bin/env python3
# Yang Lei, Jet Propulsion Laboratory
# November 2017
import datetime
import os

import netCDF4
import numpy as np
import pandas as pd


def get_satellite_attribute(info):
    mission_mapping = {
        'L': 'Landsat ',
        'S': 'Sentinel-',
    }

    satellite_1 = f'{mission_mapping[info["mission_img1"]]}{info["satellite_img1"]}'
    satellite_2 = f'{mission_mapping[info["mission_img2"]]}{info["satellite_img2"]}'

    if satellite_1 != satellite_2:
        return f'{satellite_1} and {satellite_2}'

    return satellite_1


def v_error_cal(vx_error, vy_error):
    rng = np.random.default_rng()
    vx = rng.normal(0, vx_error, 1000000)
    vy = rng.normal(0, vy_error, 1000000)
    v = np.sqrt(vx**2 + vy**2)
    return np.std(v)


def netCDF_packaging_intermediate(Dx, Dy, InterpMask, ChipSizeX, GridSpacingX, ScaleChipSizeY, SearchLimitX,
                                  SearchLimitY, origSize, noDataMask, filename='./autoRIFT_intermediate.nc'):

    nc_outfile = netCDF4.Dataset(filename, 'w', clobber=True, format='NETCDF4')

    # First set global attributes that GDAL uses when it reads netCDF files
    nc_outfile.setncattr('date_created', datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S"))
    nc_outfile.setncattr('title', 'autoRIFT intermediate results')
    nc_outfile.setncattr('author', 'Alex S. Gardner, JPL/NASA; Yang Lei, GPS/Caltech')
    nc_outfile.setncattr('institution', 'NASA Jet Propulsion Laboratory (JPL), California Institute of Technology')

    # set dimensions
    dimidY, dimidX = Dx.shape
    nc_outfile.createDimension('x', dimidX)
    nc_outfile.createDimension('y', dimidY)

    x = np.arange(0, dimidX, 1)
    y = np.arange(0, dimidY, 1)

    chunk_lines = np.min([np.ceil(8192/dimidY)*128, dimidY])
    ChunkSize = [chunk_lines, dimidX]

    nc_outfile.createDimension('x1', noDataMask.shape[1])
    nc_outfile.createDimension('y1', noDataMask.shape[0])

    var = nc_outfile.createVariable('x', np.dtype('int32'), ('x',), fill_value=None)
    var.setncattr('standard_name', 'x_index')
    var.setncattr('description', 'x index')
    var[:] = x

    var = nc_outfile.createVariable('y', np.dtype('int32'), ('y',), fill_value=None)
    var.setncattr('standard_name', 'y_index')
    var.setncattr('description', 'y index')
    var[:] = y

    var = nc_outfile.createVariable('Dx', np.dtype('float32'), ('y', 'x'), fill_value=np.nan,
                                    zlib=True, complevel=2, shuffle=True, chunksizes=ChunkSize)
    var.setncattr('standard_name', 'x_offset')
    var.setncattr('description', 'x offset')
    var[:] = Dx.astype(np.float32)

    var = nc_outfile.createVariable('Dy', np.dtype('float32'), ('y', 'x'), fill_value=np.nan,
                                    zlib=True, complevel=2, shuffle=True, chunksizes=ChunkSize)
    var.setncattr('standard_name', 'y_offset')
    var.setncattr('description', 'y offset')
    var[:] = Dy.astype(np.float32)

    var = nc_outfile.createVariable('InterpMask', np.dtype('uint8'), ('y', 'x'), fill_value=0,
                                    zlib=True, complevel=2, shuffle=True, chunksizes=ChunkSize)
    var.setncattr('standard_name', 'interpolated_value_mask')
    var.setncattr('description', 'light interpolation mask')
    var[:] = np.round(np.clip(InterpMask, 0, 255)).astype('uint8')

    var = nc_outfile.createVariable('ChipSizeX', np.dtype('uint16'), ('y', 'x'), fill_value=0,
                                    zlib=True, complevel=2, shuffle=True, chunksizes=ChunkSize)
    var.setncattr('standard_name', 'chip_size_x')
    var.setncattr('description', 'width of search window')
    var[:] = np.round(np.clip(ChipSizeX, 0, 65535)).astype('uint16')

    var = nc_outfile.createVariable('SearchLimitX', np.dtype('uint8'), ('y', 'x'), fill_value=0,
                                    zlib=True, complevel=2, shuffle=True, chunksizes=ChunkSize)
    var.setncattr('standard_name', 'search_limit_x')
    var.setncattr('description', 'search limit x')
    var[:] = np.round(np.clip(SearchLimitX, 0, 255)).astype('uint8')

    var = nc_outfile.createVariable('SearchLimitY', np.dtype('uint8'), ('y', 'x'), fill_value=0,
                                    zlib=True, complevel=2, shuffle=True, chunksizes=ChunkSize)
    var.setncattr('standard_name', 'search_limit_y')
    var.setncattr('description', 'search limit y')
    var[:] = np.round(np.clip(SearchLimitY, 0, 255)).astype('uint8')

    var = nc_outfile.createVariable('noDataMask', np.dtype('uint8'), ('y1', 'x1'), fill_value=0,
                                    zlib=True, complevel=2, shuffle=True, chunksizes=ChunkSize)
    var.setncattr('standard_name', 'nodata_value_mask')
    var.setncattr('description', 'nodata value mask')
    var[:] = np.round(np.clip(noDataMask, 0, 255)).astype('uint8')

    var = nc_outfile.createVariable('GridSpacingX', np.dtype('uint16'), (), fill_value=None)
    var.setncattr('standard_name', 'GridSpacingX')
    var.setncattr('description', 'grid spacing x')
    var[0] = GridSpacingX

    var = nc_outfile.createVariable('ScaleChipSizeY', np.dtype('float32'), (), fill_value=None)
    var.setncattr('standard_name', 'ScaleChipSizeY')
    var.setncattr('description', 'scale of chip size in y to chip size in x')
    var[0] = ScaleChipSizeY

    var = nc_outfile.createVariable('origSizeX', np.dtype('uint16'), (), fill_value=None)
    var.setncattr('standard_name', 'origSizeX')
    var.setncattr('description', 'original array size x')
    var[0] = origSize[1]

    var = nc_outfile.createVariable('origSizeY', np.dtype('uint16'), (), fill_value=None)
    var.setncattr('standard_name', 'origSizeY')
    var.setncattr('description', 'original array size y')
    var[0] = origSize[0]

    nc_outfile.sync()  # flush data to disk
    nc_outfile.close()


def netCDF_read_intermediate(filename='./autoRIFT_intermediate.nc'):

    inter_file = netCDF4.Dataset(filename, mode='r')
    Dx = inter_file.variables['Dx'][:].data
    Dy = inter_file.variables['Dy'][:].data
    InterpMask = inter_file.variables['InterpMask'][:].data
    ChipSizeX = inter_file.variables['ChipSizeX'][:].data
    SearchLimitX = inter_file.variables['SearchLimitX'][:].data
    SearchLimitY = inter_file.variables['SearchLimitY'][:].data
    noDataMask = inter_file.variables['noDataMask'][:].data
    noDataMask = noDataMask.astype('bool')
    GridSpacingX = inter_file.variables['GridSpacingX'][:].data
    ScaleChipSizeY = inter_file.variables['ScaleChipSizeY'][:].data
    origSize = (inter_file.variables['origSizeY'][:].data, inter_file.variables['origSizeX'][:].data)

    return Dx, Dy, InterpMask, ChipSizeX, GridSpacingX, ScaleChipSizeY, SearchLimitX, SearchLimitY, origSize, noDataMask


def netCDF_packaging(VX, VY, DX, DY, INTERPMASK, CHIPSIZEX, CHIPSIZEY, SSM, SSM1, SX, SY,
                     offset2vx_1, offset2vx_2, offset2vy_1, offset2vy_2, offset2vr, offset2va, scale_factor_1, scale_factor_2, MM, VXref, VYref,
                     DXref, DYref, rangePixelSize, azimuthPixelSize, dt, epsg, srs, tran, out_nc_filename, pair_type,
                     detection_method, coordinates, IMG_INFO_DICT, stable_count, stable_count1, stable_shift_applied,
                     dx_mean_shift, dy_mean_shift, dx_mean_shift1, dy_mean_shift1, error_vector):

    vx_mean_shift = offset2vx_1 * dx_mean_shift + offset2vx_2 * dy_mean_shift
    temp = vx_mean_shift
    temp[np.logical_not(SSM)] = np.nan
    vx_mean_shift = np.median(temp[np.logical_not(np.isnan(temp))])

    vy_mean_shift = offset2vy_1 * dx_mean_shift + offset2vy_2 * dy_mean_shift
    temp = vy_mean_shift
    temp[np.logical_not(SSM)] = np.nan
    vy_mean_shift = np.median(temp[np.logical_not(np.isnan(temp))])

    vx_mean_shift1 = offset2vx_1 * dx_mean_shift1 + offset2vx_2 * dy_mean_shift1
    temp = vx_mean_shift1
    temp[np.logical_not(SSM1)] = np.nan
    vx_mean_shift1 = np.median(temp[np.logical_not(np.isnan(temp))])

    vy_mean_shift1 = offset2vy_1 * dx_mean_shift1 + offset2vy_2 * dy_mean_shift1
    temp = vy_mean_shift1
    temp[np.logical_not(SSM1)] = np.nan
    vy_mean_shift1 = np.median(temp[np.logical_not(np.isnan(temp))])

    V = np.sqrt(VX**2+VY**2)

    if pair_type == 'radar':
        dr_2_vr_factor = np.median(offset2vr[np.logical_not(np.isnan(offset2vr))])
        SlantRangePixelSize = np.median(offset2vr[np.logical_not(np.isnan(offset2vr))]) * dt/365.0/24.0/3600.0
        azimuthPixelSize = np.median(offset2va[np.logical_not(np.isnan(offset2va))]) * dt/365.0/24.0/3600.0

        VR = DX * offset2vr
        VR = VR.astype(np.float32)

        VA = DY * offset2va
        VA = VA.astype(np.float32)

        VRref = DXref * offset2vr
        VRref = VRref.astype(np.float32)

        VAref = DYref * offset2va
        VAref = VAref.astype(np.float32)

        vr_mean_shift = dx_mean_shift * offset2vr
        vr_mean_shift = np.median(vr_mean_shift[np.logical_not(np.isnan(vr_mean_shift))])

        va_mean_shift = dy_mean_shift * offset2va
        va_mean_shift = np.median(va_mean_shift[np.logical_not(np.isnan(va_mean_shift))])

        vr_mean_shift1 = dx_mean_shift1 * offset2vr
        vr_mean_shift1 = np.median(vr_mean_shift1[np.logical_not(np.isnan(vr_mean_shift1))])

        va_mean_shift1 = dy_mean_shift1 * offset2va
        va_mean_shift1 = np.median(va_mean_shift1[np.logical_not(np.isnan(va_mean_shift1))])

        # create the (slope parallel & reference) flow-based range-projected result
        alpha_sp = (DX * scale_factor_1) / (offset2vy_2 / (offset2vx_1 * offset2vy_2 - offset2vx_2 * offset2vy_1) * (-SX) - offset2vx_2 / (offset2vx_1 * offset2vy_2 - offset2vx_2 * offset2vy_1) * (-SY))
        alpha_ref = (DX * scale_factor_1) / (offset2vy_2 / (offset2vx_1 * offset2vy_2 - offset2vx_2 * offset2vy_1) * VXref - offset2vx_2 / (offset2vx_1 * offset2vy_2 - offset2vx_2 * offset2vy_1) * VYref)
        VXS = alpha_sp * (-SX)
        VYS = alpha_sp * (-SY)
        VXR = alpha_ref * VXref
        VYR = alpha_ref * VYref

        zero_flag_sp = (SX == 0) & (SY == 0)
        zero_flag_ref = (VXref == 0) & (VYref == 0)
        VXS[zero_flag_sp] = np.nan
        VYS[zero_flag_sp] = np.nan
        VXR[zero_flag_ref] = np.nan
        VYR[zero_flag_ref] = np.nan

        rngX = offset2vx_1
        rngY = offset2vy_1
        angle_df_S = np.arccos((-SX * rngX - SY * rngY) / (np.sqrt(SX**2 + SY**2) * np.sqrt(rngX**2+rngY**2)))
        angle_df_S = np.abs(np.real(angle_df_S) - np.pi / 2)
        angle_df_R = np.arccos((VXref * rngX + VYref * rngY) / (np.sqrt(VXref**2 + VYref**2) * np.sqrt(rngX**2+rngY**2)))
        angle_df_R = np.abs(np.real(angle_df_R) - np.pi / 2)

        angle_threshold_S = 0.75
        angle_threshold_R = 0.75

        VXS[angle_df_S < angle_threshold_S] = np.nan
        VYS[angle_df_S < angle_threshold_S] = np.nan
        VXR[angle_df_R < angle_threshold_R] = np.nan
        VYR[angle_df_R < angle_threshold_R] = np.nan


        # FIXME: Switch to using the updated (better) estimates of velocity fields when available
        # Use the updated dhdxs and dhdys input files that combine the velocity fields and smoothed slopes
        VXP = VXS
        VYP = VYS
        # use the updated (better) estimates of velocity fields
        # VXP = VXR
        # VYP = VYR

        VXP = VXP.astype(np.float32)
        VYP = VYP.astype(np.float32)
        VP = np.sqrt(VXP**2+VYP**2)

        VXPP = VX.copy()
        VYPP = VY.copy()

        stable_count_p = np.sum(SSM & np.logical_not(np.isnan(VXP)))
        stable_count1_p = np.sum(SSM1 & np.logical_not(np.isnan(VXP)))

        vxp_mean_shift = 0.0
        vxp_mean_shift1 = 0.0

        vyp_mean_shift = 0.0
        vyp_mean_shift1 = 0.0

        if stable_count_p != 0:
            temp = VXP.copy() - VX.copy()
            temp[np.logical_not(SSM)] = np.nan
            bias_mean_shift = np.median(temp[np.logical_not(np.isnan(temp))])
            vxp_mean_shift = vx_mean_shift + bias_mean_shift / 1

            temp = VYP.copy() - VY.copy()
            temp[np.logical_not(SSM)] = np.nan
            bias_mean_shift = np.median(temp[np.logical_not(np.isnan(temp))])
            vyp_mean_shift = vy_mean_shift + bias_mean_shift / 1

        if stable_count1_p != 0:
            temp = VXP.copy() - VX.copy()
            temp[np.logical_not(SSM1)] = np.nan
            bias_mean_shift1 = np.median(temp[np.logical_not(np.isnan(temp))])
            vxp_mean_shift1 = vx_mean_shift1 + bias_mean_shift1 / 1

            temp = VYP.copy() - VY.copy()
            temp[np.logical_not(SSM1)] = np.nan
            bias_mean_shift1 = np.median(temp[np.logical_not(np.isnan(temp))])
            vyp_mean_shift1 = vy_mean_shift1 + bias_mean_shift1 / 1

        if stable_count_p == 0:
            if stable_count1_p == 0:
                stable_shift_applied_p = 0
            else:
                stable_shift_applied_p = 2
        else:
            stable_shift_applied_p = 1

    CHIPSIZEX = CHIPSIZEX * rangePixelSize
    CHIPSIZEY = CHIPSIZEY * azimuthPixelSize

    NoDataValue = -32767
    noDataMask = np.isnan(VX) | np.isnan(VY)

    CHIPSIZEX[noDataMask] = 0
    CHIPSIZEY[noDataMask] = 0
    INTERPMASK[noDataMask] = 0

    title = 'autoRIFT surface velocities'
    author = 'Alex S. Gardner, JPL/NASA; Yang Lei, GPS/Caltech'
    institution = 'NASA Jet Propulsion Laboratory (JPL), California Institute of Technology'

    source = f'NASA MEaSUREs ITS_LIVE project. Processed with autoRIFT version ' \
             f'{IMG_INFO_DICT["autoRIFT_software_version"]}'
    if pair_type == 'radar':
        import isce3
        source += f' using ISCE3 version {isce3.__version__}'
    if IMG_INFO_DICT['mission_img1'].startswith('S'):
        source += f'. Contains modified Copernicus Sentinel data {IMG_INFO_DICT["date_center"][0:4]}, processed by ESA'
    if IMG_INFO_DICT['mission_img1'].startswith('L'):
        source += f'. Landsat-{IMG_INFO_DICT["satellite_img1"]} images courtesy of the U.S. Geological Survey'

    references = 'When using this data, please acknowledge the source (see global source attribute) and cite:\n' \
                 '* Gardner, A. S., Moholdt, G., Scambos, T., Fahnestock, M., Ligtenberg, S., van den Broeke, M.,\n' \
                 '  and Nilsson, J., 2018. Increased West Antarctic and unchanged East Antarctic ice discharge over\n' \
                 '  the last 7 years. The Cryosphere, 12, p.521. https://doi.org/10.5194/tc-12-521-2018\n' \
                 '* Lei, Y., Gardner, A. and Agram, P., 2021. Autonomous Repeat Image Feature Tracking (autoRIFT)\n' \
                 '  and Its Application for Tracking Ice Displacement. Remote Sensing, 13(4), p.749.\n' \
                 '  https://doi.org/10.3390/rs13040749\n' \
                 '\n' \
                 'Additionally, a DOI is provided for the software used to generate this data:\n' \
                 '* autoRIFT: https://doi.org/10.5281/zenodo.4025445\n' \

    tran = [tran[0] + tran[1]/2, tran[1], 0.0, tran[3] + tran[5]/2, 0.0, tran[5]]

    clobber = True     # overwrite existing output nc file

    nc_outfile = netCDF4.Dataset(out_nc_filename, 'w', clobber=clobber, format='NETCDF4')

    # First set global attributes that GDAL uses when it reads netCFDF files
    nc_outfile.setncattr('GDAL_AREA_OR_POINT', 'Area')
    nc_outfile.setncattr('Conventions', 'CF-1.8')
    nc_outfile.setncattr('date_created', datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S"))
    nc_outfile.setncattr('title', title)
    nc_outfile.setncattr('autoRIFT_software_version', IMG_INFO_DICT["autoRIFT_software_version"])
    nc_outfile.setncattr('scene_pair_type', pair_type)
    nc_outfile.setncattr('satellite', get_satellite_attribute(IMG_INFO_DICT))
    nc_outfile.setncattr('motion_detection_method', detection_method)
    nc_outfile.setncattr('motion_coordinates', coordinates)
    nc_outfile.setncattr('author', author)
    nc_outfile.setncattr('institution', institution)
    nc_outfile.setncattr('source', source)
    nc_outfile.setncattr('references', references)


    var = nc_outfile.createVariable('img_pair_info', 'U1', (), fill_value=None)
    var.setncattr('standard_name', 'image_pair_information')
    for key in IMG_INFO_DICT:
        if key == 'autoRIFT_software_version':
            continue
        var.setncattr(key, IMG_INFO_DICT[key])

    # set dimensions
    dimidY, dimidX = VX.shape
    nc_outfile.createDimension('x', dimidX)
    nc_outfile.createDimension('y', dimidY)

    x = np.arange(tran[0], tran[0] + tran[1] * dimidX, tran[1])
    y = np.arange(tran[3], tran[3] + tran[5] * dimidY, tran[5])

    chunk_lines = np.min([np.ceil(8192/dimidY)*128, dimidY])
    ChunkSize = [chunk_lines, dimidX]


    var = nc_outfile.createVariable('x', np.dtype('float64'), 'x', fill_value=None)
    var.setncattr('standard_name', 'projection_x_coordinate')
    var.setncattr('description', 'x coordinate of projection')
    var.setncattr('units', 'm')
    var[:] = x


    var = nc_outfile.createVariable('y', np.dtype('float64'), 'y', fill_value=None)
    var.setncattr('standard_name', 'projection_y_coordinate')
    var.setncattr('description', 'y coordinate of projection')
    var.setncattr('units', 'm')
    var[:] = y


    mapping_var_name = 'mapping'  # need to set this as an attribute for the image variables
    var = nc_outfile.createVariable(mapping_var_name, 'U1', (), fill_value=None)
    if srs.GetAttrValue('PROJECTION') == 'Polar_Stereographic':
        var.setncattr('grid_mapping_name', 'polar_stereographic')
        var.setncattr('straight_vertical_longitude_from_pole', srs.GetProjParm('central_meridian'))
        var.setncattr('false_easting', srs.GetProjParm('false_easting'))
        var.setncattr('false_northing', srs.GetProjParm('false_northing'))
        var.setncattr('latitude_of_projection_origin', np.sign(srs.GetProjParm('latitude_of_origin'))*90.0)  # could hardcode this to be -90 for landsat - just making it more general, maybe...
        var.setncattr('latitude_of_origin', srs.GetProjParm('latitude_of_origin'))
        var.setncattr('semi_major_axis', float(srs.GetAttrValue('GEOGCS|SPHEROID', 1)))
        var.setncattr('scale_factor_at_projection_origin', 1)
        var.setncattr('inverse_flattening', float(srs.GetAttrValue('GEOGCS|SPHEROID', 2)))
        var.setncattr('spatial_ref', srs.ExportToWkt())
        var.setncattr('crs_wkt', srs.ExportToWkt())
        var.setncattr('proj4text', srs.ExportToProj4())
        var.setncattr('spatial_epsg', epsg)
        var.setncattr('GeoTransform', ' '.join(str(x) for x in tran))  # note this has pixel size in it - set  explicitly above

    elif srs.GetAttrValue('PROJECTION') == 'Transverse_Mercator':
        var.setncattr('grid_mapping_name', 'universal_transverse_mercator')
        zone = epsg - np.floor(epsg/100)*100
        var.setncattr('utm_zone_number', zone)
        var.setncattr('longitude_of_central_meridian', srs.GetProjParm('central_meridian'))
        var.setncattr('false_easting', srs.GetProjParm('false_easting'))
        var.setncattr('false_northing', srs.GetProjParm('false_northing'))
        var.setncattr('latitude_of_projection_origin', srs.GetProjParm('latitude_of_origin'))
        var.setncattr('semi_major_axis', float(srs.GetAttrValue('GEOGCS|SPHEROID', 1)))
        var.setncattr('scale_factor_at_central_meridian', srs.GetProjParm('scale_factor'))
        var.setncattr('inverse_flattening', float(srs.GetAttrValue('GEOGCS|SPHEROID', 2)))
        var.setncattr('spatial_ref', srs.ExportToWkt())
        var.setncattr('crs_wkt', srs.ExportToWkt())
        var.setncattr('proj4text', srs.ExportToProj4())
        var.setncattr('spatial_epsg', epsg)
        var.setncattr('GeoTransform', ' '.join(str(x) for x in tran))  # note this has pixel size in it - set  explicitly above

    else:
        raise Exception('Projection {0} not recognized for this program'.format(srs.GetAttrValue('PROJECTION')))


    var = nc_outfile.createVariable('vx', np.dtype('int16'), ('y', 'x'), fill_value=NoDataValue,
                                    zlib=True, complevel=2, shuffle=True, chunksizes=ChunkSize)
    var.setncattr('standard_name', 'land_ice_surface_x_velocity')
    if pair_type == 'radar':
        var.setncattr('description', 'velocity component in x direction from radar range and azimuth measurements')
    else:
        var.setncattr('description', 'velocity component in x direction')
    var.setncattr('units', 'meter/year')
    var.setncattr('grid_mapping', mapping_var_name)

    if stable_count != 0:
        temp = VX.copy() - VXref.copy()
        temp[np.logical_not(SSM)] = np.nan
        vx_error_mask = np.std(temp[np.logical_not(np.isnan(temp))])
    else:
        vx_error_mask = np.nan
    if stable_count1 != 0:
        temp = VX.copy() - VXref.copy()
        temp[np.logical_not(SSM1)] = np.nan
        vx_error_slow = np.std(temp[np.logical_not(np.isnan(temp))])
    else:
        vx_error_slow = np.nan
    if pair_type == 'radar':
        vx_error_mod = (error_vector[0][0]*IMG_INFO_DICT['date_dt']+error_vector[1][0])/IMG_INFO_DICT['date_dt']*365
    else:
        vx_error_mod = error_vector[0]/IMG_INFO_DICT['date_dt']*365
    if stable_shift_applied == 1:
        vx_error = vx_error_mask
    elif stable_shift_applied == 2:
        vx_error = vx_error_slow
    else:
        vx_error = vx_error_mod

    var.setncattr('error', int(round(vx_error*10))/10)
    var.setncattr('error_description', 'best estimate of x_velocity error: vx_error is populated '
                                       'according to the approach used for the velocity bias '
                                       'correction as indicated in "stable_shift_flag"')

    if stable_count != 0:
        var.setncattr('error_stationary', int(round(vx_error_mask*10))/10)
    else:
        var.setncattr('error_stationary', np.nan)
    var.setncattr('error_stationary_description', 'RMSE over stable surfaces, stationary or slow-flowing '
                                            'surfaces with velocity < 15 meter/year identified from an external mask')

    var.setncattr('error_modeled', int(round(vx_error_mod*10))/10)
    var.setncattr('error_modeled_description', '1-sigma error calculated using a modeled error-dt relationship')

    if stable_count1 != 0:
        var.setncattr('error_slow', int(round(vx_error_slow*10))/10)
    else:
        var.setncattr('error_slow', np.nan)
    var.setncattr('error_slow_description', 'RMSE over slowest 25% of retrieved velocities')

    if stable_shift_applied == 2:
        var.setncattr('stable_shift', int(round(vx_mean_shift1*10))/10)
    elif stable_shift_applied == 1:
        var.setncattr('stable_shift', int(round(vx_mean_shift*10))/10)
    else:
        var.setncattr('stable_shift', 0)
    var.setncattr('stable_shift_flag', stable_shift_applied)
    var.setncattr('stable_shift_flag_description', 'flag for applying velocity bias correction: 0 = no correction; '
                                                   '1 = correction from overlapping stable surface mask (stationary '
                                                   'or slow-flowing surfaces with velocity < 15 meter/year)(top priority); '
                                                   '2 = correction from slowest 25% of overlapping velocities '
                                                   '(second priority)')

    if stable_count != 0:
        var.setncattr('stable_shift_stationary', int(round(vx_mean_shift*10))/10)
    else:
        var.setncattr('stable_shift_stationary', np.nan)
    var.setncattr('stable_count_stationary', stable_count)

    if stable_count1 != 0:
        var.setncattr('stable_shift_slow', int(round(vx_mean_shift1*10))/10)
    else:
        var.setncattr('stable_shift_slow', np.nan)
    var.setncattr('stable_count_slow', stable_count1)

    VX[noDataMask] = NoDataValue
    var[:] = np.round(np.clip(VX, -32768, 32767)).astype(np.int16)
    # var.setncattr('_FillValue', np.int16(FillValue))


    var = nc_outfile.createVariable('vy', np.dtype('int16'), ('y', 'x'), fill_value=NoDataValue,
                                    zlib=True, complevel=2, shuffle=True, chunksizes=ChunkSize)
    var.setncattr('standard_name', 'land_ice_surface_y_velocity')
    if pair_type == 'radar':
        var.setncattr('description', 'velocity component in y direction from radar range and azimuth measurements')
    else:
        var.setncattr('description', 'velocity component in y direction')
    var.setncattr('units', 'meter/year')
    var.setncattr('grid_mapping', mapping_var_name)

    if stable_count != 0:
        temp = VY.copy() - VYref.copy()
        temp[np.logical_not(SSM)] = np.nan
        vy_error_mask = np.std(temp[np.logical_not(np.isnan(temp))])
    else:
        vy_error_mask = np.nan
    if stable_count1 != 0:
        temp = VY.copy() - VYref.copy()
        temp[np.logical_not(SSM1)] = np.nan
        vy_error_slow = np.std(temp[np.logical_not(np.isnan(temp))])
    else:
        vy_error_slow = np.nan
    if pair_type == 'radar':
        vy_error_mod = (error_vector[0][1]*IMG_INFO_DICT['date_dt']+error_vector[1][1])/IMG_INFO_DICT['date_dt']*365
    else:
        vy_error_mod = error_vector[1]/IMG_INFO_DICT['date_dt']*365
    if stable_shift_applied == 1:
        vy_error = vy_error_mask
    elif stable_shift_applied == 2:
        vy_error = vy_error_slow
    else:
        vy_error = vy_error_mod
    var.setncattr('error', int(round(vy_error*10))/10)
    var.setncattr('error_description', 'best estimate of y_velocity error: vy_error is populated according '
                                       'to the approach used for the velocity bias correction as indicated '
                                       'in "stable_shift_flag"')

    if stable_count != 0:
        var.setncattr('error_stationary', int(round(vy_error_mask*10))/10)
    else:
        var.setncattr('error_stationary', np.nan)
    var.setncattr('error_stationary_description', 'RMSE over stable surfaces, stationary or slow-flowing surfaces '
                                            'with velocity < 15 meter/year identified from an external mask')

    var.setncattr('error_modeled', int(round(vy_error_mod * 10)) / 10)
    var.setncattr('error_modeled_description', '1-sigma error calculated using a modeled error-dt relationship')

    if stable_count1 != 0:
        var.setncattr('error_slow', int(round(vy_error_slow * 10)) / 10)
    else:
        var.setncattr('error_slow', np.nan)
    var.setncattr('error_slow_description', 'RMSE over slowest 25% of retrieved velocities')

    if stable_shift_applied == 2:
        var.setncattr('stable_shift', int(round(vy_mean_shift1*10))/10)
    elif stable_shift_applied == 1:
        var.setncattr('stable_shift', int(round(vy_mean_shift*10))/10)
    else:
        var.setncattr('stable_shift', 0)

    var.setncattr('stable_shift_flag', stable_shift_applied)
    var.setncattr('stable_shift_flag_description', 'flag for applying velocity bias correction: 0 = no correction; '
                                                   '1 = correction from overlapping stable surface mask (stationary '
                                                   'or slow-flowing surfaces with velocity < 15 meter/year)(top priority); '
                                                   '2 = correction from slowest 25% of overlapping velocities '
                                                   '(second priority)')

    if stable_count != 0:
        var.setncattr('stable_shift_stationary', int(round(vy_mean_shift*10))/10)
    else:
        var.setncattr('stable_shift_stationary', np.nan)
    var.setncattr('stable_count_stationary', stable_count)

    if stable_count1 != 0:
        var.setncattr('stable_shift_slow', int(round(vy_mean_shift1*10))/10)
    else:
        var.setncattr('stable_shift_slow', np.nan)
    var.setncattr('stable_count_slow', stable_count1)

    VY[noDataMask] = NoDataValue
    var[:] = np.round(np.clip(VY, -32768, 32767)).astype(np.int16)


    var = nc_outfile.createVariable('v', np.dtype('int16'), ('y', 'x'), fill_value=NoDataValue,
                                    zlib=True, complevel=2, shuffle=True, chunksizes=ChunkSize)
    var.setncattr('standard_name', 'land_ice_surface_velocity')
    var.setncattr('description', 'velocity magnitude')
    var.setncattr('units', 'meter/year')
    var.setncattr('grid_mapping', mapping_var_name)

    V[noDataMask] = NoDataValue
    var[:] = np.round(np.clip(V, -32768, 32767)).astype(np.int16)


    var = nc_outfile.createVariable('v_error', np.dtype('int16'), ('y', 'x'), fill_value=NoDataValue,
                                    zlib=True, complevel=2, shuffle=True, chunksizes=ChunkSize)
    var.setncattr('standard_name', 'velocity_error')
    if pair_type == 'radar':
        var.setncattr('description', 'velocity magnitude error from radar range and azimuth measurements')
    else:
        var.setncattr('description', 'velocity magnitude error')
    var.setncattr('units', 'meter/year')
    var.setncattr('grid_mapping', mapping_var_name)

    v_error = v_error_cal(vx_error, vy_error)
    V_error = np.sqrt((vx_error * VX / V)**2 + (vy_error * VY / V)**2)
    V_error[V == 0] = v_error
    V_error[noDataMask] = NoDataValue
    var[:] = np.round(np.clip(V_error, -32768, 32767)).astype(np.int16)


    if pair_type == 'radar':
        var = nc_outfile.createVariable('vr', np.dtype('int16'), ('y', 'x'), fill_value=NoDataValue,
                                        zlib=True, complevel=2, shuffle=True, chunksizes=ChunkSize)

        var.setncattr('standard_name', 'range_velocity')
        var.setncattr('description', 'velocity in radar range direction')
        var.setncattr('units', 'meter/year')
        var.setncattr('grid_mapping', mapping_var_name)

        if stable_count != 0:
            temp = VR.copy() - VRref.copy()
            temp[np.logical_not(SSM)] = np.nan
            vr_error_mask = np.std(temp[np.logical_not(np.isnan(temp))])
        else:
            vr_error_mask = np.nan
        if stable_count1 != 0:
            temp = VR.copy() - VRref.copy()
            temp[np.logical_not(SSM1)] = np.nan
            vr_error_slow = np.std(temp[np.logical_not(np.isnan(temp))])
        else:
            vr_error_slow = np.nan
        vr_error_mod = (error_vector[0][2]*IMG_INFO_DICT['date_dt']+error_vector[1][2])/IMG_INFO_DICT['date_dt']*365
        if stable_shift_applied == 1:
            vr_error = vr_error_mask
        elif stable_shift_applied == 2:
            vr_error = vr_error_slow
        else:
            vr_error = vr_error_mod

        var.setncattr('error', int(round(vr_error*10))/10)
        var.setncattr('error_description', 'best estimate of range_velocity error: vr_error is populated '
                                           'according to the approach used for the velocity bias correction '
                                           'as indicated in "stable_shift_flag"')

        if stable_count != 0:
            var.setncattr('error_stationary', int(round(vr_error_mask*10))/10)
        else:
            var.setncattr('error_stationary', np.nan)
        var.setncattr('error_stationary_description', 'RMSE over stable surfaces, stationary or slow-flowing '
                                                'surfaces with velocity < 15 meter/year identified from an external mask')

        var.setncattr('error_modeled', int(round(vr_error_mod*10))/10)
        var.setncattr('error_modeled_description', '1-sigma error calculated using a modeled error-dt relationship')

        if stable_count1 != 0:
            var.setncattr('error_slow', int(round(vr_error_slow*10))/10)
        else:
            var.setncattr('error_slow', np.nan)
        var.setncattr('error_slow_description', 'RMSE over slowest 25% of retrieved velocities')

        if stable_shift_applied == 2:
            var.setncattr('stable_shift', int(round(vr_mean_shift1*10))/10)
        elif stable_shift_applied == 1:
            var.setncattr('stable_shift', int(round(vr_mean_shift*10))/10)
        else:
            var.setncattr('stable_shift', 0)
        var.setncattr('stable_shift_flag', stable_shift_applied)
        var.setncattr('stable_shift_flag_description', 'flag for applying velocity bias correction: 0 = no correction; '
                                                       '1 = correction from overlapping stable surface mask '
                                                       '(stationary or slow-flowing surfaces with velocity < 15 meter/year)'
                                                       '(top priority); 2 = correction from slowest 25% of overlapping '
                                                       'velocities (second priority)')

        if stable_count != 0:
            var.setncattr('stable_shift_stationary', int(round(vr_mean_shift*10))/10)
        else:
            var.setncattr('stable_shift_stationary', np.nan)
        var.setncattr('stable_count_stationary', stable_count)

        if stable_count1 != 0:
            var.setncattr('stable_shift_slow', int(round(vr_mean_shift1*10))/10)
        else:
            var.setncattr('stable_shift_slow', np.nan)
        var.setncattr('stable_count_slow', stable_count1)

        VR[noDataMask] = NoDataValue
        var[:] = np.round(np.clip(VR, -32768, 32767)).astype(np.int16)


        var = nc_outfile.createVariable('va', np.dtype('int16'), ('y', 'x'), fill_value=NoDataValue,
                                        zlib=True, complevel=2, shuffle=True, chunksizes=ChunkSize)
        var.setncattr('standard_name', 'azimuth_velocity')
        var.setncattr('description', 'velocity in radar azimuth direction')
        var.setncattr('units', 'meter/year')
        var.setncattr('grid_mapping', mapping_var_name)

        if stable_count != 0:
            temp = VA.copy() - VAref.copy()
            temp[np.logical_not(SSM)] = np.nan
            va_error_mask = np.std(temp[np.logical_not(np.isnan(temp))])
        else:
            va_error_mask = np.nan
        if stable_count1 != 0:
            temp = VA.copy() - VAref.copy()
            temp[np.logical_not(SSM1)] = np.nan
            va_error_slow = np.std(temp[np.logical_not(np.isnan(temp))])
        else:
            va_error_slow = np.nan
        va_error_mod = (error_vector[0][3]*IMG_INFO_DICT['date_dt']+error_vector[1][3])/IMG_INFO_DICT['date_dt']*365
        if stable_shift_applied == 1:
            va_error = va_error_mask
        elif stable_shift_applied == 2:
            va_error = va_error_slow
        else:
            va_error = va_error_mod

        var.setncattr('error', int(round(va_error*10))/10)
        var.setncattr('error_description', 'best estimate of azimuth_velocity error: va_error is populated '
                                           'according to the approach used for the velocity bias correction '
                                           'as indicated in "stable_shift_flag"')

        if stable_count != 0:
            var.setncattr('error_stationary', int(round(va_error_mask*10))/10)
        else:
            var.setncattr('error_stationary', np.nan)
        var.setncattr('error_stationary_description', 'RMSE over stable surfaces, stationary or slow-flowing surfaces with velocity < 15 meter/year identified from an external mask')

        var.setncattr('error_modeled', int(round(va_error_mod*10))/10)
        var.setncattr('error_modeled_description', '1-sigma error calculated using a modeled error-dt relationship')

        if stable_count1 != 0:
            var.setncattr('error_slow', int(round(va_error_slow*10))/10)
        else:
            var.setncattr('error_slow', np.nan)
        var.setncattr('error_slow_description', 'RMSE over slowest 25% of retrieved velocities')

        if stable_shift_applied == 2:
            var.setncattr('stable_shift', int(round(va_mean_shift1*10))/10)
        elif stable_shift_applied == 1:
            var.setncattr('stable_shift', int(round(va_mean_shift*10))/10)
        else:
            var.setncattr('stable_shift', 0)
        var.setncattr('stable_shift_flag', stable_shift_applied)
        var.setncattr('stable_shift_flag_description', 'flag for applying velocity bias correction: 0 = no correction; '
                                                       '1 = correction from overlapping stable surface mask '
                                                       '(stationary or slow-flowing surfaces with velocity < 15 meter/year)'
                                                       '(top priority); 2 = correction from slowest 25% of overlapping '
                                                       'velocities (second priority)')

        if stable_count != 0:
            var.setncattr('stable_shift_stationary', int(round(va_mean_shift*10))/10)
        else:
            var.setncattr('stable_shift_stationary', np.nan)
        var.setncattr('stable_count_stationary', stable_count)

        if stable_count1 != 0:
            var.setncattr('stable_shift_slow', int(round(va_mean_shift1*10))/10)
        else:
            var.setncattr('stable_shift_slow', np.nan)
        var.setncattr('stable_count_slow', stable_count1)

        VA[noDataMask] = NoDataValue
        var[:] = np.round(np.clip(VA, -32768, 32767)).astype(np.int16)

        # fuse the (slope parallel & reference) flow-based range-projected result with the raw observed range/azimuth-based result
        if stable_count_p != 0:
            temp = VXP.copy() - VXref.copy()
            temp[np.logical_not(SSM)] = np.nan
            vxp_error_mask = np.std(temp[np.logical_not(np.isnan(temp))])

            temp = VYP.copy() - VYref.copy()
            temp[np.logical_not(SSM)] = np.nan
            vyp_error_mask = np.std(temp[np.logical_not(np.isnan(temp))])
        if stable_count1_p != 0:
            temp = VXP.copy() - VXref.copy()
            temp[np.logical_not(SSM1)] = np.nan
            vxp_error_slow = np.std(temp[np.logical_not(np.isnan(temp))])

            temp = VYP.copy() - VYref.copy()
            temp[np.logical_not(SSM1)] = np.nan
            vyp_error_slow = np.std(temp[np.logical_not(np.isnan(temp))])
        vxp_error_mod = (error_vector[0][4]*IMG_INFO_DICT['date_dt']+error_vector[1][4])/IMG_INFO_DICT['date_dt']*365
        vyp_error_mod = (error_vector[0][5]*IMG_INFO_DICT['date_dt']+error_vector[1][5])/IMG_INFO_DICT['date_dt']*365

        if stable_shift_applied_p == 1:
            vxp_error = vxp_error_mask
            vyp_error = vyp_error_mask
        elif stable_shift_applied_p == 2:
            vxp_error = vxp_error_slow
            vyp_error = vyp_error_slow
        else:
            vxp_error = vxp_error_mod
            vyp_error = vyp_error_mod

        VP_error = np.sqrt((vxp_error * VXP / VP)**2 + (vyp_error * VYP / VP)**2)

        VXPP[V_error > VP_error] = VXP[V_error > VP_error]
        VYPP[V_error > VP_error] = VYP[V_error > VP_error]
        VXP = VXPP.astype(np.float32)
        VYP = VYPP.astype(np.float32)
        VP = np.sqrt(VXP**2+VYP**2)

        stable_count_p = np.sum(SSM & np.logical_not(np.isnan(VXP)))
        stable_count1_p = np.sum(SSM1 & np.logical_not(np.isnan(VXP)))

        vxp_mean_shift = 0.0
        vyp_mean_shift = 0.0
        vxp_mean_shift1 = 0.0
        vyp_mean_shift1 = 0.0

        if stable_count_p != 0:
            temp = VXP.copy() - VX.copy()
            temp[np.logical_not(SSM)] = np.nan
            bias_mean_shift = np.median(temp[np.logical_not(np.isnan(temp))])
            vxp_mean_shift = vx_mean_shift + bias_mean_shift / 1

            temp = VYP.copy() - VY.copy()
            temp[np.logical_not(SSM)] = np.nan
            bias_mean_shift = np.median(temp[np.logical_not(np.isnan(temp))])
            vyp_mean_shift = vy_mean_shift + bias_mean_shift / 1

        if stable_count1_p != 0:
            temp = VXP.copy() - VX.copy()
            temp[np.logical_not(SSM1)] = np.nan
            bias_mean_shift1 = np.median(temp[np.logical_not(np.isnan(temp))])
            vxp_mean_shift1 = vx_mean_shift1 + bias_mean_shift1 / 1

            temp = VYP.copy() - VY.copy()
            temp[np.logical_not(SSM1)] = np.nan
            bias_mean_shift1 = np.median(temp[np.logical_not(np.isnan(temp))])
            vyp_mean_shift1 = vy_mean_shift1 + bias_mean_shift1 / 1

        if stable_count_p == 0:
            if stable_count1_p == 0:
                stable_shift_applied_p = 0
            else:
                stable_shift_applied_p = 2
        else:
            stable_shift_applied_p = 1

        var = nc_outfile.createVariable('M11', np.dtype('float32'), ('y', 'x'), fill_value=NoDataValue,
                                        zlib=True, complevel=2, shuffle=True, chunksizes=ChunkSize)
        var.setncattr('standard_name', 'conversion_matrix_element_11')
        var.setncattr('description', 'conversion matrix element (1st row, 1st column) that can be multiplied with vx '
                                     'to give range pixel displacement dr (see Eq. A18 in https://www.mdpi.com/2072-4292/13/4/749)')
        var.setncattr('units', 'pixel/(meter/year)')
        var.setncattr('grid_mapping', mapping_var_name)
        var.setncattr('dr_to_vr_factor', dr_2_vr_factor)
        var.setncattr('dr_to_vr_factor_description', 'multiplicative factor that converts slant range '
                                                     'pixel displacement dr to slant range velocity vr')

        M11 = offset2vy_2 / (offset2vx_1 * offset2vy_2 - offset2vx_2 * offset2vy_1) / scale_factor_1

        M11[noDataMask] = NoDataValue
        var[:] = M11

        var = nc_outfile.createVariable('M12', np.dtype('float32'), ('y', 'x'), fill_value=NoDataValue,
                                        zlib=True, complevel=2, shuffle=True, chunksizes=ChunkSize)
        var.setncattr('standard_name', 'conversion_matrix_element_12')
        var.setncattr('description', 'conversion matrix element (1st row, 2nd column) that can be multiplied with vy '
                                     'to give range pixel displacement dr (see Eq. A18 in https://www.mdpi.com/2072-4292/13/4/749)')
        var.setncattr('units', 'pixel/(meter/year)')
        var.setncattr('grid_mapping', mapping_var_name)

        var.setncattr('dr_to_vr_factor', dr_2_vr_factor)
        var.setncattr('dr_to_vr_factor_description', 'multiplicative factor that converts slant range '
                                                     'pixel displacement dr to slant range velocity vr')

        M12 = -offset2vx_2 / (offset2vx_1 * offset2vy_2 - offset2vx_2 * offset2vy_1) / scale_factor_1

        M12[noDataMask] = NoDataValue
        var[:] = M12

    var = nc_outfile.createVariable('chip_size_width', np.dtype('uint16'), ('y', 'x'), fill_value=0,
                                    zlib=True, complevel=2, shuffle=True, chunksizes=ChunkSize)
    var.setncattr('standard_name', 'chip_size_width')
    var.setncattr('description', 'width of search template (chip)')
    var.setncattr('units', 'm')
    var.setncattr('grid_mapping', mapping_var_name)

    if pair_type == 'radar':
        var.setncattr('range_pixel_size', rangePixelSize)
        var.setncattr('chip_size_coordinates', 'radar geometry: width = range, height = azimuth')
    else:
        var.setncattr('x_pixel_size', rangePixelSize)
        var.setncattr('chip_size_coordinates', 'image projection geometry: width = x, height = y')

    var[:] = np.round(np.clip(CHIPSIZEX, 0, 65535)).astype('uint16')

    var = nc_outfile.createVariable('chip_size_height', np.dtype('uint16'), ('y', 'x'), fill_value=0,
                                    zlib=True, complevel=2, shuffle=True, chunksizes=ChunkSize)
    var.setncattr('standard_name', 'chip_size_height')
    var.setncattr('description', 'height of search template (chip)')
    var.setncattr('units', 'm')
    var.setncattr('grid_mapping', mapping_var_name)

    if pair_type == 'radar':
        var.setncattr('azimuth_pixel_size', azimuthPixelSize)
        var.setncattr('chip_size_coordinates', 'radar geometry: width = range, height = azimuth')
    else:
        var.setncattr('y_pixel_size', azimuthPixelSize)
        var.setncattr('chip_size_coordinates', 'image projection geometry: width = x, height = y')

    var[:] = np.round(np.clip(CHIPSIZEY, 0, 65535)).astype('uint16')

    var = nc_outfile.createVariable('interp_mask', np.dtype('uint8'), ('y', 'x'), fill_value=0,
                                    zlib=True, complevel=2, shuffle=True, chunksizes=ChunkSize)
    var.setncattr('standard_name', 'interpolated_value_mask')
    var.setncattr('description', 'true where values have been interpolated')
    var.setncattr('flag_values', [np.uint8(0), np.uint8(1)])
    var.setncattr('flag_meanings', 'measured interpolated')
    var.setncattr('grid_mapping', mapping_var_name)

    var[:] = np.round(np.clip(INTERPMASK, 0, 255)).astype('uint8')


    nc_outfile.sync()  # flush data to disk
    nc_outfile.close()

    return out_nc_filename


def rotate_vel2radar(rngind, azmind, vel_x, vel_y, swath_border, swath_border_full, GridSpacingX, ScaleChipSizeY, flag):
    ncols = np.nanmax(rngind)+1
    nrows = np.nanmax(azmind)+1

    skipSample_x = GridSpacingX
    skipSample_y = int(np.round(GridSpacingX*ScaleChipSizeY))
    xGrid = np.arange(0, ncols, skipSample_x)
    yGrid = np.arange(0, nrows, skipSample_y)
    nd = xGrid.__len__()
    md = yGrid.__len__()
    rngind1 = np.int32(np.dot(np.ones((md, 1)), np.reshape(xGrid, (1, xGrid.__len__()))))
    azmind1 = np.int32(np.dot(np.reshape(yGrid, (yGrid.__len__(), 1)), np.ones((1, nd))))

    rngind1 = rngind1.astype(np.float32)
    azmind1 = azmind1.astype(np.float32)

    output_vel_x = np.zeros(rngind1.shape)*np.nan
    output_vel_y = np.zeros(rngind1.shape)*np.nan

    for irow in range(rngind.shape[0]):
        for icol in range(rngind.shape[1]):
            tempcol = rngind[irow, icol]
            temprow = azmind[irow, icol]
            if (~np.isnan(tempcol)) & (~np.isnan(temprow)):
                tempcol = np.argmin(np.abs(xGrid-rngind[irow, icol]))
                temprow = np.argmin(np.abs(yGrid-azmind[irow, icol]))
                output_vel_x[temprow, tempcol] = vel_x[irow, icol]
                output_vel_y[temprow, tempcol] = vel_y[irow, icol]

    shift = 500

    if flag == 0:
        mask = (rngind1 > swath_border[0]-shift) & (rngind1 < swath_border[0]+shift)
        output_vel_x[mask] = np.nan
        output_vel_y[mask] = np.nan

    if flag == 1:
        mask = (rngind1 > swath_border[1]-shift) & (rngind1 < swath_border[1]+shift)
        output_vel_x[mask] = np.nan
        output_vel_y[mask] = np.nan

    df = pd.DataFrame(output_vel_x)
    df = df.interpolate(method='linear', axis=1)
    output_vel_x = df.to_numpy()

    df = pd.DataFrame(output_vel_y)
    df = df.interpolate(method='linear', axis=1)
    output_vel_y = df.to_numpy()

    output_vel_x = output_vel_x.astype('float32')
    output_vel_y = output_vel_y.astype('float32')

    output_vel_x1 = vel_x.copy()
    output_vel_y1 = vel_y.copy()

    for irow in range(rngind.shape[0]):
        for icol in range(rngind.shape[1]):
            tempcol = rngind[irow, icol]
            temprow = azmind[irow, icol]
            if (~np.isnan(tempcol)) & (~np.isnan(temprow)):
                tempcol = np.argmin(np.abs(xGrid-rngind[irow, icol]))
                temprow = np.argmin(np.abs(yGrid-azmind[irow, icol]))
                output_vel_x1[irow, icol] = output_vel_x[temprow, tempcol]
                output_vel_y1[irow, icol] = output_vel_y[temprow, tempcol]

    output_vel_x1[np.isnan(vel_x)] = np.nan
    output_vel_y1[np.isnan(vel_y)] = np.nan

    return output_vel_x1, output_vel_y1


def getPol(safe, orbit_path):
    from s1reader import load_bursts

    pols = ['vv', 'vh', 'hh', 'hv']
    for pol in pols:
        try:
            bursts = load_bursts(safe,orbit_path,1,pol)
            print('Polarization '+pol)
            return pol
        except:
            pass


def loadMetadata(indir):
    """
    Input file.
    """
    import os
    import numpy as np
    from datetime import datetime
    from s1reader import load_bursts
    import glob
    orbits = glob.glob('*.EOF')
    fechas_orbits = [datetime.strptime(os.path.basename(file).split('_')[6], 'V%Y%m%dT%H%M%S') for file in orbits]
    safes = glob.glob('*.SAFE')
    if len(safes)==0:
        safes = glob.glob('*.zip')
    fechas_safes = [datetime.strptime(os.path.basename(file).split('_')[5], '%Y%m%dT%H%M%S') for file in safes]

    if 'ref' in indir:
        safe = safes[np.argmin(fechas_safes)]
        orbit_path = orbits[np.argmin(fechas_orbits)]
    else:
        safe = safes[np.argmax(fechas_safes)]
        orbit_path = orbits[np.argmax(fechas_orbits)]

    pol = getPol(safe, orbit_path)

    swaths = []

    if '_' in indir:
        swath = int(indir.split('_')[2][2])
        bursts = load_bursts(safe, orbit_path, swath, pol)
        for bur in bursts:
            if int(bur.burst_id.subswath[2])==swath:
                burst = bur
    else:
        # Find the first swath containing data
        for swath in [1, 2, 3]:
            try:
                bursts = load_bursts(safe, orbit_path, swath, pol)
                burst = bursts[0]
                swaths.append(bursts[0])
            except:
                continue

    return swaths, burst.orbit_direction


def cal_swath_offset_bias(indir_m, rngind, azmind, VX, VY, DX, DY, nodata,
                          tran, proj, GridSpacingX, ScaleChipSizeY, output_ref=[0.0, 0.0, 0.0, 0.0]):

    swaths, flight_direction = loadMetadata(os.path.basename(indir_m))
    swaths_s, flight_direction_s = loadMetadata(os.path.basename(indir_m.replace('reference','secondary')))

    flight_direction = flight_direction.lower()
    flight_direction_s = flight_direction_s.lower()

    if swaths[0].platform_id == swaths_s[0].platform_id:
        print('subswath offset bias correction not performed for non-S1A/B combination')
        return DX, DY, flight_direction, flight_direction_s
    else:
        if swaths[0].platform_id[2] == 'B':
            output_ref = [-output_ref[0], -output_ref[1], -output_ref[2], -output_ref[3]]

    output = []
    swath_border = []
    swath_border_full = []
    flag21 = 0
    flag32 = 0

    rngind = rngind.astype(np.float32)
    azmind = azmind.astype(np.float32)
    rngind[rngind == nodata] = np.nan
    azmind[azmind == nodata] = np.nan

    IW1_shape = swaths[0].shape
    IW1_far_range = swaths[0].starting_range + (IW1_shape[1] - 1.0) * swaths[0].range_pixel_spacing

    IW2_shape = swaths[1].shape
    IW2_far_range = swaths[1].starting_range + (IW2_shape[1] - 1.0) * swaths[1].range_pixel_spacing

    IW3_shape = swaths[2].shape
    IW3_far_range = swaths[2].starting_range + (IW3_shape[1] - 1.0) * swaths[2].range_pixel_spacing

    if len(swaths) > 1:
        ncols = int(np.round((IW3_far_range - swaths[0].starting_range) / swaths[0].range_pixel_spacing))
        ind2 = int(np.round((IW1_far_range - swaths[0].starting_range) / swaths[0].range_pixel_spacing))
        ind1 = int(np.round((swaths[1].starting_range - swaths[0].starting_range) / swaths[0].range_pixel_spacing))
    else:
        ncols = int(np.round((IW1_far_range - swaths[0].starting_range) / swaths[0].range_pixel_spacing))
        ind2 = ncols
        ind1 = 0

    swath_border.append((ind1+ind2)/2)
    swath_border_full.append(ind1)
    swath_border_full.append(ind2)

    mask1 = (rngind > ind1-500) & (rngind < ind1)
    mask2 = (rngind > ind2) & (rngind < ind2+500)

    if (np.nanstd(VX[mask2]) < 25) & (np.nanstd(VX[mask1]) < 25) \
            & (np.nanstd(VY[mask2]) < 25) & (np.nanstd(VY[mask1]) < 25):
        output.append(output_ref[0])
        output.append(output_ref[1])
        flag21 = 1
    else:
        output.append(output_ref[0])
        output.append(output_ref[1])

    if len(swaths) > 1:
        ind2 = int(np.round((IW2_far_range - swaths[0].starting_range) / swaths[0].range_pixel_spacing))
        ind1 = int(np.round((swaths[2].starting_range - swaths[0].starting_range) / swaths[0].range_pixel_spacing))

    swath_border.append((ind1+ind2)/2)
    swath_border_full.append(ind1)
    swath_border_full.append(ind2)

    mask1 = (rngind > ind1-500) & (rngind < ind1)
    mask2 = (rngind > ind2) & (rngind < ind2+500)

    if (np.nanstd(VX[mask2]) < 25) & (np.nanstd(VX[mask1]) < 25) \
            & (np.nanstd(VY[mask2]) < 25) & (np.nanstd(VY[mask1]) < 25):
        output.append(output_ref[2])
        output.append(output_ref[3])
        flag32 = 1
    else:
        output.append(output_ref[2])
        output.append(output_ref[3])

    mask1 = (rngind > swath_border[1]) & (rngind < ncols)
    DX[mask1] = DX[mask1] - output[2]
    DY[mask1] = DY[mask1] - output[3]

    mask2 = (rngind > swath_border[0]) & (rngind < ncols)
    DX[mask2] = DX[mask2] - output[0]
    DY[mask2] = DY[mask2] - output[1]

    if flag21 == 1:
        DX, DY = rotate_vel2radar(rngind, azmind, DX, DY, swath_border, swath_border_full, GridSpacingX, ScaleChipSizeY, 0)
        print('sharp peaks corrected at subswath 2 and 1 borders')
    if flag32 == 1:
        DX, DY = rotate_vel2radar(rngind, azmind, DX, DY, swath_border, swath_border_full, GridSpacingX, ScaleChipSizeY, 1)
        print('sharp peaks corrected at subswath 3 and 2 borders')

    print('subswath offset bias between subswath 2 and 1: {0} {1}'.format(output[0], output[1]))
    print('subswath offset bias between subswath 3 and 2: {0} {1}'.format(output[2], output[3]))
    print('subswath border index: {0} {1}'.format(swath_border[0], swath_border[1]))

    return DX, DY, flight_direction, flight_direction_s
