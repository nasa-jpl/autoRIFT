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
# Author: Yang Lei, Alex S. Gardner
#
# Note: this is based on the MATLAB code, "auto-RIFT", written by Alex S. Gardner,
#       and has been translated to Python and further optimized.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import cv2
import sys
import numpy as np
import numpy.fft as fft
from numba import cfunc, carray, jit
from numba.types import intc, CPointer, float64, intp, voidptr
from scipy import LowLevelCallable
from scipy.ndimage import generic_filter

def _remove_local_mean(image, kernel):
    mean = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return image - mean


def _preprocess_filt_std(image, kernel):
    n = np.prod(kernel.shape)
    conv_sum = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)
    conv_squared_sum = cv2.filter2D(image**2, -1, kernel, borderType=cv2.BORDER_REFLECT)

    variance = conv_squared_sum - (conv_sum**2)
    variance = np.clip(variance, 0., None)

    std = np.sqrt(variance) * np.sqrt(n / (n - 1))
    return std


def _wallis_filter(image, filter_width):
    kernel = np.ones((filter_width, filter_width), dtype=np.float32)
    kernel = kernel / np.sum(kernel)

    shifted = _remove_local_mean(image, kernel)
    std = _preprocess_filt_std(image, kernel)
    std[np.isclose(std, 0.)] = np.nan
    return shifted / std


def _wallis_filter_fill(image, filter_width, std_cutoff):
    invalid_data = np.isclose(image, 0.0)
    buff = np.sqrt(2 * ((filter_width - 1) / 2) ** 2) + 0.01

    # find edges of image, this makes missing scan lines valid and will
    # later be filled with random white noise
    potential_data = cv2.distanceTransform(invalid_data.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE) < 30
    missing_data = potential_data & invalid_data
    missing_data = cv2.distanceTransform((~missing_data).astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE) <= buff

    # trying to frame out the image
    valid_domain = ~invalid_data | missing_data
    zero_mask = ~valid_domain

    kernel = np.ones((filter_width, filter_width), dtype=np.float32)
    kernel = kernel / np.sum(kernel)

    shifted = _remove_local_mean(image, kernel)
    std = _preprocess_filt_std(image, kernel)

    low_std = std < std_cutoff
    low_std = cv2.distanceTransform((~low_std).astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE) <= buff
    missing_data = (missing_data | low_std) & valid_domain

    std[missing_data] = np.nan
    image = shifted / std

    valid_data = valid_domain & ~missing_data
    invalid_data |= ~valid_data

    # wallis filter normalizes the imagery to have a mean=0 and std=1;
    # fill with random values from a normal distribution with same mean and std
    fill_data = valid_domain & missing_data
    random_fill = np.random.normal(size=(fill_data.sum(),))
    image[fill_data] = random_fill

    return image, zero_mask


def _find_largest_region(binary_arr):
    n_labels, label_arr, stats, centroids = cv2.connectedComponentsWithStats(binary_arr)
    area = stats[:, cv2.CC_STAT_AREA]
    max_label = area[1:].argmax() + 1
    label_arr[label_arr != max_label] = 0
    return label_arr


def _calculate_slope(point1, point2):
    slope = np.rad2deg(np.arctan((point1[1] - point2[1]) / (point1[0] - point2[0])))
    return slope


def _get_slopes(tl, tr, bl, br):
    slope1 = _calculate_slope(bl, br)
    slope2 = _calculate_slope(tl, tr)
    slope3 = _calculate_slope(br, tr)
    slope4 = _calculate_slope(bl, tl)
    along_track_angle = np.nanmax([slope1, slope2])
    cross_track_angle = np.nanmax([slope3, slope4])
    return along_track_angle, cross_track_angle


def _fft_filter(Ix, valid_domain, power_threshold=500):
    y, x = valid_domain.shape
    center_y = y / 2
    center_y_int = np.floor(y / 2).astype(int)
    center_x = x / 2
    center_x_int = np.floor(x / 2).astype(int)

    regions = (valid_domain != 0).astype("uint8") * 255
    single_region = _find_largest_region(regions)
    single_region = np.uint8(single_region * 255)
    contours, hierarchy = cv2.findContours(single_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy.shape[1] > 1:
        raise ValueError(f"{hierarchy.shape[1]} external objects founds, only expecting 1.")
    contour = contours[0]
    moment = cv2.moments(contour)

    centroid_y = moment["m01"] / moment["m00"]
    centroid_x = moment["m10"] / moment["m00"]
    centroid_y_int = np.floor(centroid_y).astype(int)
    centroid_x_int = np.floor(centroid_x).astype(int)
    angle = cv2.minAreaRect(contour)[2]
    quadrants = np.ones(single_region.shape).astype("uint8")
    quadrants[:, centroid_x_int:] += 1
    quadrants[centroid_y_int:, :] += 2

    rotation = cv2.getRotationMatrix2D(center=(centroid_x, centroid_y), angle=-angle, scale=1)
    rotated_quadrants = cv2.warpAffine(src=quadrants, M=rotation, dsize=(x, y))

    centroid_array = np.ones(single_region.shape).astype("uint8")
    centroid_array[centroid_y_int, centroid_x_int] = 0
    distance_from_centroid = cv2.distanceTransform(centroid_array, cv2.DIST_L2, 5)
    distance_from_centroid[single_region != 255] = 0
    slices = {
        "tl": 1,
        "tr": 2,
        "bl": 3,
        "br": 4,
    }
    corners = {}
    for s in slices:
        window = np.zeros(distance_from_centroid.shape)
        roi = rotated_quadrants == slices[s]
        window[roi] = distance_from_centroid[roi]
        max_index_of_flattened = np.argmax(window)
        max_point = np.unravel_index(max_index_of_flattened, window.shape)
        corners[s] = max_point

    along_track, cross_track = _get_slopes(**corners)
    print(f"Along track angle is {along_track:.2f} degrees")
    print(f"Cross track angle is {cross_track:.2f} degrees")

    filter_base = np.zeros((y, x))
    filter_base[center_y_int - 70 : center_y_int + 70, :] = 1
    filter_base[:, center_x_int - 100 : center_x_int + 100] = 0

    rotation_a = cv2.getRotationMatrix2D(center=(center_x, center_y), angle=cross_track, scale=1)
    rotation_b = cv2.getRotationMatrix2D(center=(center_x, center_y), angle=along_track, scale=1)
    filter_a = cv2.warpAffine(src=filter_base, M=rotation_a, dsize=(x, y))
    filter_b = cv2.warpAffine(src=filter_base, M=rotation_b, dsize=(x, y))

    # Alex's code appears not to use this shift
    # y_shift = centroid_y - center_y
    # x_shift = centroid_x - center_x
    # print(f"shift = ({x_shift:.1f},{y_shift:.1f})")

    # translation = np.array([[1, 0, x_shift],
    #                         [0, 1, y_shift]],
    #                        dtype=np.float32)
    # filter_a = cv2.warpAffine(src=filter_a, M=translation, dsize=(x, y))
    # filter_b = cv2.warpAffine(src=filter_b, M=translation, dsize=(x, y))

    image = Ix.copy()
    image[image > 3] = 3
    image[image < -3] = -3
    image[np.isnan(image)] = 0

    fft_image = fft.fftshift(fft.fft2(image))
    P = abs(fft_image)
    mP = np.mean(P)
    stdP = np.std(P)
    P = (P - mP) > (10 * stdP)

    sA = np.nansum(P[filter_a == 1])
    sB = np.nansum(P[filter_b == 1])
    print(f"Along track power is {sA:.0f} degrees")
    print(f"Cross track power is {sB:.0f} degrees")
    if ((sA / sB >= 2) | (sB / sA >= 2)) & ((sA > power_threshold) | (sB > power_threshold)):
        if sA > sB:
            final_filter = filter_a.copy()
        else:
            final_filter = filter_b.copy()

        filtered_image = np.real(fft.ifft2(fft.ifftshift(fft_image * (1 - (final_filter)))))
        filtered_image[~valid_domain] = 0
    else:
        print(
            f"Power along flight direction ({max(sB, sA)}) does not exceed banding threshold ({power_threshold}). "
            f"No banding filter applied.")
        return image

    return filtered_image


class autoRIFT:
    """
    Class for mapping regular geographic grid on radar imagery.
    """

    def preprocess_filt_wal_nodata_fill(self):
        """
        Wallis filter with nodata infill for L7 SLC Off preprocessing
        """
        image_1, zero_mask_1 = _wallis_filter_fill(self.I1, self.WallisFilterWidth, self.StandardDeviationCutoff)
        image_2, zero_mask_2 = _wallis_filter_fill(self.I2, self.WallisFilterWidth, self.StandardDeviationCutoff)

        self.I1 = image_1
        self.I2 = image_2

        self.zeroMask = zero_mask_1 | zero_mask_2

    def preprocess_filt_wal(self):
        """
        Do the preprocessing using wallis filter (10 min vs 15 min in Matlab).
        """
        self.I1zeroMask = self.I1 == 0
        self.I2zeroMask = self.I2 == 0
        print(f"Wallis filter width is {self.WallisFilterWidth}")

        self.I1 = _wallis_filter(self.I1, self.WallisFilterWidth)
        nan_mask = np.isnan(self.I1)
        self.I1zeroMask = self.I1zeroMask | nan_mask
        self.I1[self.I1zeroMask] = 0

        self.I2 = _wallis_filter(self.I2, self.WallisFilterWidth)
        nan_mask = np.isnan(self.I2)
        self.I2zeroMask = self.I2zeroMask | nan_mask
        self.I2[self.I2zeroMask] = 0

        self.zeroMask = self.I1zeroMask | self.I2zeroMask

    def preprocess_filt_hps(self):
        """
        Do the pre processing using (orig - low-pass filter) = high-pass filter filter (3.9/5.3 min).
        """
        import cv2
        import numpy as np

        kernel = -np.ones((self.WallisFilterWidth, self.WallisFilterWidth), dtype=np.float32)

        kernel[int((self.WallisFilterWidth - 1) / 2), int((self.WallisFilterWidth - 1) / 2)] = kernel.size - 1

        kernel = kernel / kernel.size

        #        pdb.set_trace()

        self.I1 = cv2.filter2D(self.I1, -1, kernel, borderType=cv2.BORDER_CONSTANT)

        self.I2 = cv2.filter2D(self.I2, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    def preprocess_filt_fft(self):
        """
        Preprocess images to remove banding perpendicular to the along flight direction by masking in frequency space
        """
        self.I1 = _fft_filter(self.I1, ~self.I1zeroMask, power_threshold=500)
        self.I2 = _fft_filter(self.I2, ~self.I2zeroMask, power_threshold=500)
        print("fft done")

    def preprocess_db(self):
        """
        Do the pre processing using db scale (4 min).
        """
        import cv2
        import numpy as np

        self.zeroMask = self.I1 == 0

        #        pdb.set_trace()

        self.I1 = 20.0 * np.log10(self.I1)

        self.I2 = 20.0 * np.log10(self.I2)

    def preprocess_filt_sob(self):
        """
        Do the pre processing using sobel filter (4.5/5.8 min).
        """
        import cv2
        import numpy as np

        sobelx = cv2.getDerivKernels(1, 0, self.WallisFilterWidth)

        kernelx = np.outer(sobelx[0], sobelx[1])

        sobely = cv2.getDerivKernels(0, 1, self.WallisFilterWidth)

        kernely = np.outer(sobely[0], sobely[1])

        kernel = kernelx + kernely

        self.I1 = cv2.filter2D(self.I1, -1, kernel, borderType=cv2.BORDER_CONSTANT)

        self.I2 = cv2.filter2D(self.I2, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    def preprocess_filt_lap(self):
        """
        Do the pre processing using Laplacian filter (2.5 min / 4 min).
        """
        import cv2
        import numpy as np

        self.zeroMask = self.I1 == 0

        self.I1 = 20.0 * np.log10(self.I1)
        self.I1 = cv2.Laplacian(self.I1, -1, ksize=self.WallisFilterWidth, borderType=cv2.BORDER_CONSTANT)

        self.I2 = 20.0 * np.log10(self.I2)
        self.I2 = cv2.Laplacian(self.I2, -1, ksize=self.WallisFilterWidth, borderType=cv2.BORDER_CONSTANT)

    def uniform_data_type(self):

        import numpy as np

        if self.DataType == 0:
            if self.zeroMask is not None:
                #            validData = np.logical_not(np.isnan(self.I1))
                validData = np.isfinite(self.I1)
                temp = self.I1[validData]
            else:
                temp = self.I1
            S1 = np.std(temp) * np.sqrt(temp.size / (temp.size - 1.0))
            M1 = np.mean(temp)
            temp = None
            self.I1 = (self.I1 - (M1 - 3 * S1)) / (6 * S1) * (2**8 - 0)
            del S1, M1, temp

            #            self.I1[np.logical_not(np.isfinite(self.I1))] = 0
            self.I1 = np.round(np.clip(self.I1, 0, 255)).astype(np.uint8)

            if self.zeroMask is not None:
                #            validData = np.logical_not(np.isnan(self.I2))
                validData = np.isfinite(self.I2)
                temp = self.I2[validData]
            else:
                temp = self.I2
            S2 = np.std(temp) * np.sqrt(temp.size / (temp.size - 1.0))
            M2 = np.mean(temp)
            temp = None
            self.I2 = (self.I2 - (M2 - 3 * S2)) / (6 * S2) * (2**8 - 0)
            del S2, M2, temp

            #            self.I2[np.logical_not(np.isfinite(self.I2))] = 0
            self.I2 = np.round(np.clip(self.I2, 0, 255)).astype(np.uint8)

            if self.zeroMask is not None:
                self.I1[self.zeroMask] = 0
                self.I2[self.zeroMask] = 0
                self.zeroMask = None

        elif self.DataType == 1:

            if self.zeroMask is not None:
                self.I1[np.logical_not(np.isfinite(self.I1))] = 0
                self.I2[np.logical_not(np.isfinite(self.I2))] = 0

            self.I1 = self.I1.astype(np.float32)
            self.I2 = self.I2.astype(np.float32)

            if self.zeroMask is not None:
                self.I1[self.zeroMask] = 0
                self.I2[self.zeroMask] = 0
                self.zeroMask = None

        else:
            sys.exit("invalid data type for the image pair which must be unsigned integer 8 or 32-bit float")

    def autorift(self):
        """
        Do the actual processing.
        """
        import numpy as np
        import cv2
        from scipy import ndimage

        ChipSizeUniX = np.unique(np.append(np.unique(self.ChipSizeMinX), np.unique(self.ChipSizeMaxX)))
        ChipSizeUniX = np.delete(ChipSizeUniX, np.where(ChipSizeUniX == 0)[0])

        if np.any(np.mod(ChipSizeUniX, self.ChipSize0X) != 0):
            sys.exit("chip sizes must be even integers of ChipSize0")

        ChipRangeX = self.ChipSize0X * np.array([1, 2, 4, 8, 16, 32, 64], np.float32)
        #        ChipRangeX = ChipRangeX[ChipRangeX < (2**8 - 1)]
        if np.max(ChipSizeUniX) > np.max(ChipRangeX):
            sys.exit("max each chip size is out of range")

        ChipSizeUniX = ChipRangeX[(ChipRangeX >= np.min(ChipSizeUniX)) & (ChipRangeX <= np.max(ChipSizeUniX))]

        maxScale = np.max(ChipSizeUniX) / self.ChipSize0X

        if (np.mod(self.xGrid.shape[0], maxScale) != 0) | (np.mod(self.xGrid.shape[1], maxScale) != 0):
            message = (
                "xgrid and ygrid have an incorect size "
                + str(self.xGrid.shape)
                + " for nested search, they must have dimensions that an interger multiple of "
                + str(maxScale)
            )
            sys.exit(message)

        self.xGrid = self.xGrid.astype(np.float32)
        self.yGrid = self.yGrid.astype(np.float32)

        if np.size(self.Dx0) == 1:
            self.Dx0 = np.ones(self.xGrid.shape, np.float32) * np.round(self.Dx0)
        else:
            self.Dx0 = self.Dx0.astype(np.float32)
        if np.size(self.Dy0) == 1:
            self.Dy0 = np.ones(self.xGrid.shape, np.float32) * np.round(self.Dy0)
        else:
            self.Dy0 = self.Dy0.astype(np.float32)
        if np.size(self.SearchLimitX) == 1:
            self.SearchLimitX = np.ones(self.xGrid.shape, np.float32) * np.round(self.SearchLimitX)
        else:
            self.SearchLimitX = self.SearchLimitX.astype(np.float32)
        if np.size(self.SearchLimitY) == 1:
            self.SearchLimitY = np.ones(self.xGrid.shape, np.float32) * np.round(self.SearchLimitY)
        else:
            self.SearchLimitY = self.SearchLimitY.astype(np.float32)
        if np.size(self.ChipSizeMinX) == 1:
            self.ChipSizeMinX = np.ones(self.xGrid.shape, np.float32) * np.round(self.ChipSizeMinX)
        else:
            self.ChipSizeMinX = self.ChipSizeMinX.astype(np.float32)
        if np.size(self.ChipSizeMaxX) == 1:
            self.ChipSizeMaxX = np.ones(self.xGrid.shape, np.float32) * np.round(self.ChipSizeMaxX)
        else:
            self.ChipSizeMaxX = self.ChipSizeMaxX.astype(np.float32)

        ChipSizeX = np.zeros(self.xGrid.shape, np.float32)
        InterpMask = np.zeros(self.xGrid.shape, bool)
        Dx = np.empty(self.xGrid.shape, dtype=np.float32)
        Dx.fill(np.nan)
        Dy = np.empty(self.xGrid.shape, dtype=np.float32)
        Dy.fill(np.nan)

        Flag = 3

        if self.ChipSize0X > self.GridSpacingX:
            if np.mod(self.ChipSize0X, self.GridSpacingX) != 0:
                sys.exit(
                    "when GridSpacing < smallest allowable chip size (ChipSize0), ChipSize0 must be integer multiples of GridSpacing"
                )
            else:
                ChipSize0_GridSpacing_oversample_ratio = int(self.ChipSize0X / self.GridSpacingX)
        else:
            ChipSize0_GridSpacing_oversample_ratio = 1

        DispFiltC = DISP_FILT()
        overlap_c = np.max((1 - self.sparseSearchSampleRate / ChipSize0_GridSpacing_oversample_ratio, 0))
        DispFiltC.FracValid = self.FracValid * (1 - overlap_c) + overlap_c**2
        DispFiltC.FracSearch = self.FracSearch
        DispFiltC.FiltWidth = (self.FiltWidth - 1) * ChipSize0_GridSpacing_oversample_ratio + 1
        DispFiltC.Iter = self.Iter - 1
        DispFiltC.MadScalar = self.MadScalar
        DispFiltC.colfiltChunkSize = self.colfiltChunkSize

        DispFiltF = DISP_FILT()
        overlap_f = 1 - 1 / ChipSize0_GridSpacing_oversample_ratio
        DispFiltF.FracValid = self.FracValid * (1 - overlap_f) + overlap_f**2
        DispFiltF.FracSearch = self.FracSearch
        DispFiltF.FiltWidth = (self.FiltWidth - 1) * ChipSize0_GridSpacing_oversample_ratio + 1
        DispFiltF.Iter = self.Iter
        DispFiltF.MadScalar = self.MadScalar
        DispFiltF.colfiltChunkSize = self.colfiltChunkSize

        for i in range(ChipSizeUniX.__len__()):

            # Nested grid setup: chip size being ChipSize0X no need to resize, otherwise has to resize the arrays
            if self.ChipSize0X != ChipSizeUniX[i]:
                Scale = self.ChipSize0X / ChipSizeUniX[i]
                dstShape = (int(self.xGrid.shape[0] * Scale), int(self.xGrid.shape[1] * Scale))
                xGrid0 = cv2.resize(self.xGrid.astype(np.float32), dstShape[::-1], interpolation=cv2.INTER_AREA)
                yGrid0 = cv2.resize(self.yGrid.astype(np.float32), dstShape[::-1], interpolation=cv2.INTER_AREA)

                if np.mod(ChipSizeUniX[i], 2) == 0:
                    xGrid0 = np.round(xGrid0 + 0.5) - 0.5
                    yGrid0 = np.round(yGrid0 + 0.5) - 0.5
                else:
                    xGrid0 = np.round(xGrid0)
                    yGrid0 = np.round(yGrid0)

                M0 = (ChipSizeX == 0) & (self.ChipSizeMinX <= ChipSizeUniX[i]) & (self.ChipSizeMaxX >= ChipSizeUniX[i])
                M0 = colfilt(M0.copy(), (int(1 / Scale * 6), int(1 / Scale * 6)), 0, self.colfiltChunkSize)
                M0 = cv2.resize(
                    np.logical_not(M0).astype(np.uint8), dstShape[::-1], interpolation=cv2.INTER_NEAREST
                ).astype(bool)

                SearchLimitX0 = colfilt(
                    self.SearchLimitX.copy(), (int(1 / Scale), int(1 / Scale)), 0, self.colfiltChunkSize
                ) + colfilt(self.Dx0.copy(), (int(1 / Scale), int(1 / Scale)), 4, self.colfiltChunkSize)
                SearchLimitY0 = colfilt(
                    self.SearchLimitY.copy(), (int(1 / Scale), int(1 / Scale)), 0, self.colfiltChunkSize
                ) + colfilt(self.Dy0.copy(), (int(1 / Scale), int(1 / Scale)), 4, self.colfiltChunkSize)
                Dx00 = colfilt(self.Dx0.copy(), (int(1 / Scale), int(1 / Scale)), 2, self.colfiltChunkSize)
                Dy00 = colfilt(self.Dy0.copy(), (int(1 / Scale), int(1 / Scale)), 2, self.colfiltChunkSize)

                SearchLimitX0 = np.ceil(cv2.resize(SearchLimitX0, dstShape[::-1]))
                SearchLimitY0 = np.ceil(cv2.resize(SearchLimitY0, dstShape[::-1]))
                SearchLimitX0[M0] = 0
                SearchLimitY0[M0] = 0
                Dx00 = np.round(cv2.resize(Dx00, dstShape[::-1], interpolation=cv2.INTER_NEAREST))
                Dy00 = np.round(cv2.resize(Dy00, dstShape[::-1], interpolation=cv2.INTER_NEAREST))
            #                pdb.set_trace()
            else:
                SearchLimitX0 = self.SearchLimitX.copy()
                SearchLimitY0 = self.SearchLimitY.copy()
                Dx00 = self.Dx0.copy()
                Dy00 = self.Dy0.copy()
                xGrid0 = self.xGrid.copy()
                yGrid0 = self.yGrid.copy()
            #                M0 = (ChipSizeX == 0) & (self.ChipSizeMinX <= ChipSizeUniX[i]) & (self.ChipSizeMaxX >= ChipSizeUniX[i])
            #                SearchLimitX0[np.logical_not(M0)] = 0
            #                SearchLimitY0[np.logical_not(M0)] = 0

            if np.logical_not(np.any(SearchLimitX0 != 0)):
                continue

            idxZero = (SearchLimitX0 <= 0) | (SearchLimitY0 <= 0)
            SearchLimitX0[idxZero] = 0
            SearchLimitY0[idxZero] = 0
            SearchLimitX0[(np.logical_not(idxZero)) & (SearchLimitX0 < self.minSearch)] = self.minSearch
            SearchLimitY0[(np.logical_not(idxZero)) & (SearchLimitY0 < self.minSearch)] = self.minSearch

            # Setup for coarse search: sparse sampling / resize
            rIdxC = slice(
                (self.sparseSearchSampleRate * ChipSize0_GridSpacing_oversample_ratio) - 1,
                xGrid0.shape[0],
                (self.sparseSearchSampleRate * ChipSize0_GridSpacing_oversample_ratio),
            )
            cIdxC = slice(
                (self.sparseSearchSampleRate * ChipSize0_GridSpacing_oversample_ratio) - 1,
                xGrid0.shape[1],
                (self.sparseSearchSampleRate * ChipSize0_GridSpacing_oversample_ratio),
            )
            xGrid0C = xGrid0[rIdxC, cIdxC]
            yGrid0C = yGrid0[rIdxC, cIdxC]

            #            pdb.set_trace()

            if np.remainder((self.sparseSearchSampleRate * ChipSize0_GridSpacing_oversample_ratio), 2) == 0:
                filtWidth = (self.sparseSearchSampleRate * ChipSize0_GridSpacing_oversample_ratio) + 1
            else:
                filtWidth = self.sparseSearchSampleRate * ChipSize0_GridSpacing_oversample_ratio

            SearchLimitX0C = colfilt(SearchLimitX0.copy(), (int(filtWidth), int(filtWidth)), 0, self.colfiltChunkSize)
            SearchLimitY0C = colfilt(SearchLimitY0.copy(), (int(filtWidth), int(filtWidth)), 0, self.colfiltChunkSize)
            SearchLimitX0C = SearchLimitX0C[rIdxC, cIdxC]
            SearchLimitY0C = SearchLimitY0C[rIdxC, cIdxC]

            Dx0C = Dx00[rIdxC, cIdxC]
            Dy0C = Dy00[rIdxC, cIdxC]

            # Coarse search
            SubPixFlag = False
            ChipSizeXC = ChipSizeUniX[i]
            ChipSizeYC = np.float32(np.round(ChipSizeXC * self.ScaleChipSizeY / 2) * 2)

            if type(self.OverSampleRatio) is dict:
                overSampleRatio = self.OverSampleRatio[ChipSizeUniX[i]]
            else:
                overSampleRatio = self.OverSampleRatio

            #            pdb.set_trace()

            if self.I1.dtype == np.uint8:
                DxC, DyC = arImgDisp_u(
                    self.I2.copy(),
                    self.I1.copy(),
                    xGrid0C.copy(),
                    yGrid0C.copy(),
                    ChipSizeXC,
                    ChipSizeYC,
                    SearchLimitX0C.copy(),
                    SearchLimitY0C.copy(),
                    Dx0C.copy(),
                    Dy0C.copy(),
                    SubPixFlag,
                    overSampleRatio,
                )
            elif self.I1.dtype == np.float32:
                DxC, DyC = arImgDisp_s(
                    self.I2.copy(),
                    self.I1.copy(),
                    xGrid0C.copy(),
                    yGrid0C.copy(),
                    ChipSizeXC,
                    ChipSizeYC,
                    SearchLimitX0C.copy(),
                    SearchLimitY0C.copy(),
                    Dx0C.copy(),
                    Dy0C.copy(),
                    SubPixFlag,
                    overSampleRatio,
                )
            else:
                sys.exit("invalid data type for the image pair which must be unsigned integer 8 or 32-bit float")

            #            pdb.set_trace()

            # M0C is the mask for reliable estimates after coarse search, MC is the mask after disparity filtering, MC2 is the mask after area closing for fine search
            M0C = np.logical_not(np.isnan(DxC))

            MC = DispFiltC.filtDisp(
                DxC.copy(), DyC.copy(), SearchLimitX0C.copy(), SearchLimitY0C.copy(), M0C.copy(), overSampleRatio
            )

            MC[np.logical_not(M0C)] = False

            ROIC = SearchLimitX0C > 0
            CoarseCorValidFac = np.sum(MC[ROIC]) / np.sum(M0C[ROIC])
            if CoarseCorValidFac < self.CoarseCorCutoff:
                continue

            MC2 = ndimage.distance_transform_edt(np.logical_not(MC)) < self.BuffDistanceC
            dstShape = (
                int(MC2.shape[0] * (self.sparseSearchSampleRate * ChipSize0_GridSpacing_oversample_ratio)),
                int(MC2.shape[1] * (self.sparseSearchSampleRate * ChipSize0_GridSpacing_oversample_ratio)),
            )

            MC2 = cv2.resize(MC2.astype(np.uint8), dstShape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
            #            pdb.set_trace()
            if np.logical_not(np.all(MC2.shape == SearchLimitX0.shape)):
                rowAdd = SearchLimitX0.shape[0] - MC2.shape[0]
                colAdd = SearchLimitX0.shape[1] - MC2.shape[1]
                if rowAdd > 0:
                    MC2 = np.append(MC2, MC2[-rowAdd:, :], axis=0)
                if colAdd > 0:
                    MC2 = np.append(MC2, MC2[:, -colAdd:], axis=1)

            SearchLimitX0[np.logical_not(MC2)] = 0
            SearchLimitY0[np.logical_not(MC2)] = 0

            # Fine Search
            SubPixFlag = True
            ChipSizeXF = ChipSizeUniX[i]
            ChipSizeYF = np.float32(np.round(ChipSizeXF * self.ScaleChipSizeY / 2) * 2)
            #            pdb.set_trace()
            if self.I1.dtype == np.uint8:
                DxF, DyF = arImgDisp_u(
                    self.I2.copy(),
                    self.I1.copy(),
                    xGrid0.copy(),
                    yGrid0.copy(),
                    ChipSizeXF,
                    ChipSizeYF,
                    SearchLimitX0.copy(),
                    SearchLimitY0.copy(),
                    Dx00.copy(),
                    Dy00.copy(),
                    SubPixFlag,
                    overSampleRatio,
                )
            elif self.I1.dtype == np.float32:
                DxF, DyF = arImgDisp_s(
                    self.I2.copy(),
                    self.I1.copy(),
                    xGrid0.copy(),
                    yGrid0.copy(),
                    ChipSizeXF,
                    ChipSizeYF,
                    SearchLimitX0.copy(),
                    SearchLimitY0.copy(),
                    Dx00.copy(),
                    Dy00.copy(),
                    SubPixFlag,
                    overSampleRatio,
                )
            else:
                sys.exit("invalid data type for the image pair which must be unsigned integer 8 or 32-bit float")

            #            pdb.set_trace()

            M0 = DispFiltF.filtDisp(
                DxF.copy(),
                DyF.copy(),
                SearchLimitX0.copy(),
                SearchLimitY0.copy(),
                np.logical_not(np.isnan(DxF)),
                overSampleRatio,
            )
            #            pdb.set_trace()
            DxF[np.logical_not(M0)] = np.nan
            DyF[np.logical_not(M0)] = np.nan

            # Light interpolation with median filtered values: DxFM (filtered) and DxF (unfiltered)
            DxFM = colfilt(DxF.copy(), (self.fillFiltWidth, self.fillFiltWidth), 3, self.colfiltChunkSize)
            DyFM = colfilt(DyF.copy(), (self.fillFiltWidth, self.fillFiltWidth), 3, self.colfiltChunkSize)

            # M0 is mask for original valid estimates, MF is mask for filled ones, MM is mask where filtered ones exist for filling
            MF = np.zeros(M0.shape, dtype=bool)
            MM = np.logical_not(np.isnan(DxFM))

            for j in range(3):
                foo = MF | M0  # initial valid estimates
                foo1 = (
                    cv2.filter2D(foo.astype(np.float32), -1, np.ones((3, 3)), borderType=cv2.BORDER_CONSTANT) >= 6
                ) | foo  # 1st area closing followed by the 2nd (part of the next line calling OpenCV)
                #                pdb.set_trace()
                fillIdx = (
                    np.logical_not(bwareaopen(np.logical_not(foo1).astype(np.uint8), 5)) & np.logical_not(foo) & MM
                )
                MF[fillIdx] = True
                DxF[fillIdx] = DxFM[fillIdx]
                DyF[fillIdx] = DyFM[fillIdx]

            # Below is for replacing the valid estimates with the bicubic filtered values for robust and accurate estimation
            if self.ChipSize0X == ChipSizeUniX[i]:
                Dx = DxF
                Dy = DyF
                ChipSizeX[M0 | MF] = ChipSizeUniX[i]
                InterpMask[MF] = True
            #                pdb.set_trace()
            else:
                #                pdb.set_trace()
                Scale = ChipSizeUniX[i] / self.ChipSize0X
                dstShape = (int(Dx.shape[0] / Scale), int(Dx.shape[1] / Scale))

                # DxF0 (filtered) / Dx (unfiltered) is the result from earlier iterations, DxFM (filtered) / DxF (unfiltered) is that of the current iteration
                # first colfilt nans within 2-by-2 area (otherwise 1 nan will contaminate all 4 points)
                DxF0 = colfilt(Dx.copy(), (int(Scale + 1), int(Scale + 1)), 2, self.colfiltChunkSize)
                # then resize to half size using area (similar to averaging) to match the current iteration
                DxF0 = cv2.resize(DxF0, dstShape[::-1], interpolation=cv2.INTER_AREA)
                DyF0 = colfilt(Dy.copy(), (int(Scale + 1), int(Scale + 1)), 2, self.colfiltChunkSize)
                DyF0 = cv2.resize(DyF0, dstShape[::-1], interpolation=cv2.INTER_AREA)

                # Note this DxFM is almost the same as DxFM (same variable) in the light interpolation (only slightly better); however, only small portion of it will be used later at locations specified by M0 and MF that are determined in the light interpolation. So even without the following two lines, the final Dx and Dy result is still the same.
                # to fill out all of the missing values in DxF
                DxFM = colfilt(DxF.copy(), (5, 5), 3, self.colfiltChunkSize)
                DyFM = colfilt(DyF.copy(), (5, 5), 3, self.colfiltChunkSize)

                # fill the current-iteration result with previously determined reliable estimates that are not searched in the current iteration
                idx = np.isnan(DxF) & np.logical_not(np.isnan(DxF0))
                DxFM[idx] = DxF0[idx]
                DyFM[idx] = DyF0[idx]

                # Strong interpolation: use filtered estimates wherever the unfiltered estimates do not exist
                idx = np.isnan(DxF) & np.logical_not(np.isnan(DxFM))
                DxF[idx] = DxFM[idx]
                DyF[idx] = DyFM[idx]

                dstShape = (Dx.shape[0], Dx.shape[1])
                DxF = cv2.resize(DxF, dstShape[::-1], interpolation=cv2.INTER_CUBIC)
                DyF = cv2.resize(DyF, dstShape[::-1], interpolation=cv2.INTER_CUBIC)
                MF = cv2.resize(MF.astype(np.uint8), dstShape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
                M0 = cv2.resize(M0.astype(np.uint8), dstShape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)

                idxRaw = M0 & (ChipSizeX == 0)
                idxFill = MF & (ChipSizeX == 0)
                ChipSizeX[idxRaw | idxFill] = ChipSizeUniX[i]
                InterpMask[idxFill] = True
                Dx[idxRaw | idxFill] = DxF[idxRaw | idxFill]
                Dy[idxRaw | idxFill] = DyF[idxRaw | idxFill]

        Flag = 1
        ChipSizeY = np.round(ChipSizeX * self.ScaleChipSizeY / 2) * 2
        self.Dx = Dx
        self.Dy = Dy
        self.InterpMask = InterpMask
        self.Flag = Flag
        self.ChipSizeX = ChipSizeX
        self.ChipSizeY = ChipSizeY

    def runAutorift(self):
        """
        quick processing routine which calls autorift main function (user can define their own way by mimicing the workflow here).
        """
        import numpy as np

        # truncate the grid to fit the nested grid
        if np.size(self.ChipSizeMaxX) == 1:
            chopFactor = self.ChipSizeMaxX / self.ChipSize0X
        else:
            chopFactor = np.max(self.ChipSizeMaxX) / self.ChipSize0X
        rlim = int(np.floor(self.xGrid.shape[0] / chopFactor) * chopFactor)
        clim = int(np.floor(self.xGrid.shape[1] / chopFactor) * chopFactor)
        self.origSize = self.xGrid.shape
        #        pdb.set_trace()
        self.xGrid = np.round(self.xGrid[0:rlim, 0:clim]) + 0.5
        self.yGrid = np.round(self.yGrid[0:rlim, 0:clim]) + 0.5

        # truncate the initial offset as well if they exist
        if np.size(self.Dx0) != 1:
            self.Dx0 = self.Dx0[0:rlim, 0:clim]
            self.Dy0 = self.Dy0[0:rlim, 0:clim]

        # truncate the search limits as well if they exist
        if np.size(self.SearchLimitX) != 1:
            self.SearchLimitX = self.SearchLimitX[0:rlim, 0:clim]
            self.SearchLimitY = self.SearchLimitY[0:rlim, 0:clim]

        # truncate the chip sizes as well if they exist
        if np.size(self.ChipSizeMaxX) != 1:
            self.ChipSizeMaxX = self.ChipSizeMaxX[0:rlim, 0:clim]
            self.ChipSizeMinX = self.ChipSizeMinX[0:rlim, 0:clim]

        # call autoRIFT main function
        self.autorift()

    def __init__(self):

        super(autoRIFT, self).__init__()

        ##Input related parameters
        self.I1 = None
        self.I2 = None
        self.xGrid = None
        self.yGrid = None
        self.Dx0 = 0
        self.Dy0 = 0
        self.origSize = None
        self.zeroMask = None
        self.I1zeroMask = None
        self.I2zeroMask = None

        ##Output file
        self.Dx = None
        self.Dy = None
        self.InterpMask = None
        self.Flag = None
        self.ChipSizeX = None
        self.ChipSizeY = None

        ##Parameter list
        self.WallisFilterWidth = 5
        self.StandardDeviationCutoff = 0.25
        self.ChipSizeMinX = 32
        self.ChipSizeMaxX = 64
        self.ChipSize0X = 32
        self.GridSpacingX = 32
        self.ScaleChipSizeY = 1
        self.SearchLimitX = 25
        self.SearchLimitY = 25
        self.SkipSampleX = 32
        self.SkipSampleY = 32
        self.fillFiltWidth = 3
        self.minSearch = 6
        self.sparseSearchSampleRate = 4
        self.FracValid = 8 / 25
        self.FracSearch = 0.20
        self.FiltWidth = 5
        self.Iter = 3
        self.MadScalar = 4
        self.colfiltChunkSize = 4
        self.BuffDistanceC = 8
        self.CoarseCorCutoff = 0.01
        self.OverSampleRatio = 16
        self.DataType = 0
        self.MultiThread = 0


class AUTO_RIFT_CORE:
    def __init__(self):
        ##Pointer to C
        self._autoriftcore = None


var_dict = {}


def initializer(I1, I2, xGrid, yGrid, SearchLimitX, SearchLimitY, ChipSizeX, ChipSizeY, Dx0, Dy0):
    var_dict["I1"] = I1
    var_dict["I2"] = I2
    var_dict["xGrid"] = xGrid
    var_dict["yGrid"] = yGrid
    var_dict["SearchLimitX"] = SearchLimitX
    var_dict["SearchLimitY"] = SearchLimitY
    var_dict["ChipSizeX"] = ChipSizeX
    var_dict["ChipSizeY"] = ChipSizeY
    var_dict["Dx0"] = Dx0
    var_dict["Dy0"] = Dy0


def arImgDisp_u(
    I1,
    I2,
    xGrid,
    yGrid,
    ChipSizeX,
    ChipSizeY,
    SearchLimitX,
    SearchLimitY,
    Dx0,
    Dy0,
    SubPixFlag,
    oversample,
):
    import numpy as np
    from . import autoriftcore
    import multiprocessing as mp

    core = AUTO_RIFT_CORE()
    if core._autoriftcore is not None:
        autoriftcore.destroyAutoRiftCore_Py(core._autoriftcore)

    core._autoriftcore = autoriftcore.createAutoRiftCore_Py()

    if np.size(SearchLimitX) == 1:
        if np.logical_not(isinstance(SearchLimitX, np.float32) & isinstance(SearchLimitY, np.float32)):
            sys.exit("SearchLimit must be float")
    else:
        if np.logical_not((SearchLimitX.dtype == np.float32) & (SearchLimitY.dtype == np.float32)):
            sys.exit("SearchLimit must be float")

    if np.size(Dx0) == 1:
        if np.logical_not(isinstance(Dx0, np.float32) & isinstance(Dy0, np.float32)):
            sys.exit("Search offsets must be float")
    else:
        if np.logical_not((Dx0.dtype == np.float32) & (Dy0.dtype == np.float32)):
            sys.exit("Search offsets must be float")

    if np.size(ChipSizeX) == 1:
        if np.logical_not(isinstance(ChipSizeX, np.float32) & isinstance(ChipSizeY, np.float32)):
            sys.exit("ChipSize must be float")
    else:
        if np.logical_not((ChipSizeX.dtype == np.float32) & (ChipSizeY.dtype == np.float32)):
            sys.exit("ChipSize must be float")

    if np.any(np.mod(ChipSizeX, 2) != 0) | np.any(np.mod(ChipSizeY, 2) != 0):
        sys.exit("it is better to have ChipSize = even number")

    if np.any(np.mod(SearchLimitX, 1) != 0) | np.any(np.mod(SearchLimitY, 1) != 0):
        sys.exit("SearchLimit must be an integar value")

    if np.any(SearchLimitX < 0) | np.any(SearchLimitY < 0):
        sys.exit("SearchLimit cannot be negative")

    if np.any(np.mod(ChipSizeX, 4) != 0) | np.any(np.mod(ChipSizeY, 4) != 0):
        sys.exit("ChipSize should be evenly divisible by 4")

    if np.size(Dx0) == 1:
        Dx0 = np.ones(xGrid.shape, dtype=np.float32) * Dx0

    if np.size(Dy0) == 1:
        Dy0 = np.ones(xGrid.shape, dtype=np.float32) * Dy0

    if np.size(SearchLimitX) == 1:
        SearchLimitX = np.ones(xGrid.shape, dtype=np.float32) * SearchLimitX

    if np.size(SearchLimitY) == 1:
        SearchLimitY = np.ones(xGrid.shape, dtype=np.float32) * SearchLimitY

    if np.size(ChipSizeX) == 1:
        ChipSizeX = np.ones(xGrid.shape, dtype=np.float32) * ChipSizeX

    if np.size(ChipSizeY) == 1:
        ChipSizeY = np.ones(xGrid.shape, dtype=np.float32) * ChipSizeY

    # convert from cartesian X-Y to matrix X-Y: X no change, Y from up being positive to down being positive
    Dy0 = -Dy0

    SLx_max = np.max(SearchLimitX + np.abs(Dx0))
    Px = int(np.max(ChipSizeX) / 2 + SLx_max + 2)
    SLy_max = np.max(SearchLimitY + np.abs(Dy0))
    Py = int(np.max(ChipSizeY) / 2 + SLy_max + 2)

    I1 = np.lib.pad(I1, ((Py, Py), (Px, Px)), "constant")
    I2 = np.lib.pad(I2, ((Py, Py), (Px, Px)), "constant")

    # adjust center location by the padarray size and 0.5 is added because we need to extract the chip centered at X+1 with -chipsize/2:chipsize/2-1, which equivalently centers at X+0.5 (X is the original grid point location). So for even chipsize, always returns offset estimates at (X+0.5).
    xGrid += Px + 0.5
    yGrid += Py + 0.5

    Dx = np.empty(xGrid.shape, dtype=np.float32)
    Dx.fill(np.nan)
    Dy = Dx.copy()

    # Call C++
    if not SubPixFlag:
        Dx, Dy = np.float32(
            autoriftcore.arPixDisp_u_Py(
                core._autoriftcore,
                I2.shape[1],
                I2.shape[0],
                I2.ravel(),
                I1.shape[1],
                I1.shape[0],
                I1.ravel(),
                xGrid.shape[1],
                xGrid.shape[0],
                xGrid.ravel(),
                yGrid.shape[1],
                yGrid.shape[0],
                yGrid.ravel(),
                SearchLimitX.ravel(),
                SearchLimitY.ravel(),
                ChipSizeX.ravel(),
                ChipSizeY.ravel(),
                Dx.ravel(),
                Dy.ravel(),
                Dx0.ravel(),
                Dy0.ravel()
            )
        )
    else:
        Dx, Dy = np.float32(
            autoriftcore.arSubPixDisp_u_Py(
                core._autoriftcore,
                I2.shape[1],
                I2.shape[0],
                I2.ravel(),
                I1.shape[1],
                I1.shape[0],
                I1.ravel(),
                xGrid.shape[1],
                xGrid.shape[0],
                xGrid.ravel(),
                yGrid.shape[1],
                yGrid.shape[0],
                yGrid.ravel(),
                SearchLimitX.ravel(),
                SearchLimitY.ravel(),
                ChipSizeX.ravel(),
                ChipSizeY.ravel(),
                Dx.ravel(),
                Dy.ravel(),
                Dx0.ravel(),
                Dy0.ravel(),
                oversample
            )
        )

    Dx = Dx.reshape(xGrid.shape)
    Dy = Dy.reshape(yGrid.shape)

    # add back 1) I1 (RefI) relative to I2 (ChipI) initial offset Dx0 and Dy0, and
    #          2) RefI relative to ChipI has a left/top boundary offset of -SearchLimitX and -SearchLimitY
    idx = np.logical_not(np.isnan(Dx))
    Dx[idx] += Dx0[idx] - SearchLimitX[idx]
    Dy[idx] += Dy0[idx] - SearchLimitY[idx]

    # convert from matrix X-Y to cartesian X-Y: X no change, Y from down being positive to up being positive
    Dy = -Dy

    autoriftcore.destroyAutoRiftCore_Py(core._autoriftcore)
    core._autoriftcore = None

    return Dx, Dy


def arImgDisp_s(
    I1,
    I2,
    xGrid,
    yGrid,
    ChipSizeX,
    ChipSizeY,
    SearchLimitX,
    SearchLimitY,
    Dx0,
    Dy0,
    SubPixFlag,
    oversample,
):
    import numpy as np
    from . import autoriftcore
    import multiprocessing as mp

    core = AUTO_RIFT_CORE()
    if core._autoriftcore is not None:
        autoriftcore.destroyAutoRiftCore_Py(core._autoriftcore)

    core._autoriftcore = autoriftcore.createAutoRiftCore_Py()

    if np.size(SearchLimitX) == 1:
        if np.logical_not(isinstance(SearchLimitX, np.float32) & isinstance(SearchLimitY, np.float32)):
            sys.exit("SearchLimit must be float")
    else:
        if np.logical_not((SearchLimitX.dtype == np.float32) & (SearchLimitY.dtype == np.float32)):
            sys.exit("SearchLimit must be float")

    if np.size(Dx0) == 1:
        if np.logical_not(isinstance(Dx0, np.float32) & isinstance(Dy0, np.float32)):
            sys.exit("Search offsets must be float")
    else:
        if np.logical_not((Dx0.dtype == np.float32) & (Dy0.dtype == np.float32)):
            sys.exit("Search offsets must be float")

    if np.size(ChipSizeX) == 1:
        if np.logical_not(isinstance(ChipSizeX, np.float32) & isinstance(ChipSizeY, np.float32)):
            sys.exit("ChipSize must be float")
    else:
        if np.logical_not((ChipSizeX.dtype == np.float32) & (ChipSizeY.dtype == np.float32)):
            sys.exit("ChipSize must be float")

    if np.any(np.mod(ChipSizeX, 2) != 0) | np.any(np.mod(ChipSizeY, 2) != 0):
        sys.exit("it is better to have ChipSize = even number")

    if np.any(np.mod(SearchLimitX, 1) != 0) | np.any(np.mod(SearchLimitY, 1) != 0):
        sys.exit("SearchLimit must be an integar value")

    if np.any(SearchLimitX < 0) | np.any(SearchLimitY < 0):
        sys.exit("SearchLimit cannot be negative")

    if np.any(np.mod(ChipSizeX, 4) != 0) | np.any(np.mod(ChipSizeY, 4) != 0):
        sys.exit("ChipSize should be evenly divisible by 4")

    if np.size(Dx0) == 1:
        Dx0 = np.ones(xGrid.shape, dtype=np.float32) * Dx0

    if np.size(Dy0) == 1:
        Dy0 = np.ones(xGrid.shape, dtype=np.float32) * Dy0

    if np.size(SearchLimitX) == 1:
        SearchLimitX = np.ones(xGrid.shape, dtype=np.float32) * SearchLimitX

    if np.size(SearchLimitY) == 1:
        SearchLimitY = np.ones(xGrid.shape, dtype=np.float32) * SearchLimitY

    if np.size(ChipSizeX) == 1:
        ChipSizeX = np.ones(xGrid.shape, dtype=np.float32) * ChipSizeX

    if np.size(ChipSizeY) == 1:
        ChipSizeY = np.ones(xGrid.shape, dtype=np.float32) * ChipSizeY

    # convert from cartesian X-Y to matrix X-Y: X no change, Y from up being positive to down being positive
    Dy0 = -Dy0

    SLx_max = np.max(SearchLimitX + np.abs(Dx0))
    Px = int(np.max(ChipSizeX) / 2 + SLx_max + 2)
    SLy_max = np.max(SearchLimitY + np.abs(Dy0))
    Py = int(np.max(ChipSizeY) / 2 + SLy_max + 2)

    I1 = np.lib.pad(I1, ((Py, Py), (Px, Px)), "constant")
    I2 = np.lib.pad(I2, ((Py, Py), (Px, Px)), "constant")

    # adjust center location by the padarray size and 0.5 is added because we need to extract the chip centered at X+1 with -chipsize/2:chipsize/2-1, which equivalently centers at X+0.5 (X is the original grid point location). So for even chipsize, always returns offset estimates at (X+0.5).
    xGrid += Px + 0.5
    yGrid += Py + 0.5

    Dx = np.empty(xGrid.shape, dtype=np.float32)
    Dx.fill(np.nan)
    Dy = Dx.copy()

    # Call C++
    if not SubPixFlag:
        Dx, Dy = np.float32(
            autoriftcore.arPixDisp_s_Py(
                core._autoriftcore,
                I2.shape[1],
                I2.shape[0],
                I2.ravel(),
                I1.shape[1],
                I1.shape[0],
                I1.ravel(),
                xGrid.shape[1],
                xGrid.shape[0],
                xGrid.ravel(),
                yGrid.shape[1],
                yGrid.shape[0],
                yGrid.ravel(),
                SearchLimitX.ravel(),
                SearchLimitY.ravel(),
                ChipSizeX.ravel(),
                ChipSizeY.ravel(),
                Dx.ravel(),
                Dy.ravel(),
                Dx0.ravel(),
                Dy0.ravel()
            )
        )
    else:
        Dx, Dy = np.float32(
            autoriftcore.arSubPixDisp_s_Py(
                core._autoriftcore,
                I2.shape[1],
                I2.shape[0],
                I2.ravel(),
                I1.shape[1],
                I1.shape[0],
                I1.ravel(),
                xGrid.shape[1],
                xGrid.shape[0],
                xGrid.ravel(),
                yGrid.shape[1],
                yGrid.shape[0],
                yGrid.ravel(),
                SearchLimitX.ravel(),
                SearchLimitY.ravel(),
                ChipSizeX.ravel(),
                ChipSizeY.ravel(),
                Dx.ravel(),
                Dy.ravel(),
                Dx0.ravel(),
                Dy0.ravel(),
                oversample
            )
        )

    Dx = Dx.reshape(xGrid.shape)
    Dy = Dy.reshape(yGrid.shape)

    idx = np.logical_not(np.isnan(Dx))
    Dx[idx] += Dx0[idx] - SearchLimitX[idx]
    Dy[idx] += Dy0[idx] - SearchLimitY[idx]

    # convert from matrix X-Y to cartesian X-Y: X no change, Y from down being positive to up being positive
    Dy = -Dy

    autoriftcore.destroyAutoRiftCore_Py(core._autoriftcore)
    core._autoriftcore = None

    return Dx, Dy


################## Chunked version of column filter
def jit_filter_function(filter_function):
    """Decorator for use with scipy.ndimage.generic_filter."""
    jitted_function = jit(filter_function, nopython=True)

    @cfunc(intc(CPointer(float64), intp, CPointer(float64), voidptr))
    def wrapped(values_ptr, len_values, result, data):
        values = carray(values_ptr, (len_values,), dtype=float64)
        result[0] = jitted_function(values)
        return 1
    return LowLevelCallable(wrapped.ctypes, signature="int (double *, npy_intp, double *, void *)")


@jit_filter_function
def fmax(values):
    """colfilt max function"""
    result = -np.inf
    for v in values:
        if v > result:
            result = v
    return result


@jit_filter_function
def fmin(values):
    """colfilt min function"""
    result = np.inf
    for v in values:
        if v < result:
            result = v
    return result


@jit_filter_function
def fmean(values):
    """colfilt mean function"""
    result = 0
    count = 0
    for v in values:
        if v==v:            # if not a nan
            result += v
            count += 1
        else:
            pass
    if count == 0:
        return np.nan
    return result/count


@jit
def partition(values, low, high):
    pivot = values[high]
    i = low - 1
    for j in range(low, high):
        if values[j] <= pivot:
            i += 1
            values[i], values[j] = values[j], values[i]
            
    values[i + 1], values[high] = values[high], values[i + 1]
    return i + 1


@jit
def quickselect_non_recursive(values, k):
        low = 0
        high = len(values) - 1
        while low <= high:
            pivot_index = partition(values, low, high)
            if pivot_index < k:
                low = pivot_index + 1
            elif pivot_index > k:
                high = pivot_index - 1
            else:
                return values[pivot_index]
        return None


@jit
def quickselect_non_recursive_duo(values, k):
        # this function returns (values[k-1]+values[k])/2
        low = 0
        high = len(values) - 1
        while low <= high:
            pivot_index = partition(values, low, high)
            if pivot_index < k-1:
                low = pivot_index + 1
            elif pivot_index > k:
                high = pivot_index - 1
            elif pivot_index == k:
                temp_max = values[0]
                for v in values[1:k]:
                    if v > temp_max:
                        temp_max = v
                return (values[pivot_index]+temp_max)/2
            else:                               # pivot_index == k-1:
                temp_min = values[k]
                for v in values[k+1:]:
                    if v < temp_min:
                        temp_min = v
                return (values[pivot_index]+temp_min)/2
        return None


@jit_filter_function
def fmedian_quickSelect(values):
    """colfilt median function that propagates nans"""
    values = [v for v in values if v==v]  # remove nans
    if len(values) < 3:
        if len(values) == 1:
            return values[0]
        elif len(values) == 2:
            return (values[0] + values[1]) / 2
        else:
            return np.nan
    else:
        if len(values)%2:
            return quickselect_non_recursive(values, len(values)//2)
        else:
            return quickselect_non_recursive_duo(values, len(values)//2)


@jit_filter_function
def frange(values):
    """colfilt range function"""
    result_max = -np.inf
    result_min = np.inf
    for v in values:
        if v < result_min:
            result_min = v
        if v > result_max:
            result_max = v
    return result_max-result_min


@jit
def median_quickSelect(values):
    """MAD median function that does not support nans"""
    if len(values) < 3:
        if len(values) == 1:
            return values[0]
        elif len(values) == 2:
            return (values[0]+values[1])/2
        else:
            return np.nan
    else:
        if len(values)%2:
            return quickselect_non_recursive(values, len(values)//2)
        else:
            return quickselect_non_recursive_duo(values, len(values)//2)


@jit_filter_function
def fMAD(values):
    """colfilt mean absolute deviation function"""
    values = [v for v in values if v==v]  # remove nans
    if len(values):
        med = median_quickSelect(values)
        values = [abs(v-med) for v in values]
        return median_quickSelect(values)
    else:
        return np.nan


def colfilt(A, kernelSize, option, chunkSize=4):
    kernelSize = (min(kernelSize[0],A.shape[0]), min(kernelSize[1],A.shape[1]))
    N_winsInCols = A.shape[1] - kernelSize[1] + 1
    N_chunks = min(chunkSize,N_winsInCols)
    N_winsInChunk_vec = np.full(N_chunks,N_winsInCols//N_chunks)
    N_winsInChunk_vec[:(N_winsInCols % N_chunks)] += 1
    cs = np.cumsum(N_winsInChunk_vec)
    ind_startCols_chunks = cs - N_winsInChunk_vec
    ind_stopCols_chunks = cs + kernelSize[1] - 1
    m = (kernelSize[1]-1)//2    # margin due to kernel size (left)
    m2 = kernelSize[1] - m - 1  # margin due to kernel size (right)
    ind_out_start = ind_startCols_chunks + m
    ind_out_start[0] = 0
    ind_out_stop = ind_stopCols_chunks - m2
    ind_out_stop[-1] = A.shape[1]
    relInd_start = np.full(N_chunks,m)
    relInd_stop = relInd_start + N_winsInChunk_vec
    relInd_start[0] = 0
    relInd_stop[-1] = N_winsInChunk_vec[-1] + kernelSize[1] - 1
    out = np.full(A.shape,np.float32(np.nan))   # pre-allocate output

    if option == 0:  # max
        for ii in np.arange(N_chunks):
            out[:,ind_out_start[ii]:ind_out_stop[ii]] \
                = generic_filter(A[:,ind_startCols_chunks[ii]:ind_stopCols_chunks[ii]],fmax,size=kernelSize)[:,relInd_start[ii]:relInd_stop[ii]]
        out[np.isneginf(out)] = np.nan

    elif option == 1:  # min
        for ii in np.arange(N_chunks):
            out[:,ind_out_start[ii]:ind_out_stop[ii]] \
                = generic_filter(A[:,ind_startCols_chunks[ii]:ind_stopCols_chunks[ii]],fmin,size=kernelSize)[:,relInd_start[ii]:relInd_stop[ii]]
        out[np.isposinf(out)] = np.nan

    elif option == 2:  # mean
        for ii in np.arange(N_chunks):
            out[:,ind_out_start[ii]:ind_out_stop[ii]] \
                = generic_filter(A[:,ind_startCols_chunks[ii]:ind_stopCols_chunks[ii]],fmean,size=kernelSize,mode='constant',cval=np.nan)[:,relInd_start[ii]:relInd_stop[ii]]

    elif option == 3:  # median
        for ii in np.arange(N_chunks):
            out[:,ind_out_start[ii]:ind_out_stop[ii]] \
                = generic_filter(A[:,ind_startCols_chunks[ii]:ind_stopCols_chunks[ii]],fmedian_quickSelect,size=kernelSize,mode='constant',cval=np.nan)[:,relInd_start[ii]:relInd_stop[ii]]

    elif option == 4:  # range
        for ii in np.arange(N_chunks):
            out[:,ind_out_start[ii]:ind_out_stop[ii]] \
                = generic_filter(A[:,ind_startCols_chunks[ii]:ind_stopCols_chunks[ii]],frange,size=kernelSize)[:,relInd_start[ii]:relInd_stop[ii]]

    elif option == 6:  # MAD (Median Absolute Deviation)
        for ii in np.arange(N_chunks):
            out[:,ind_out_start[ii]:ind_out_stop[ii]] \
                = generic_filter(A[:,ind_startCols_chunks[ii]:ind_stopCols_chunks[ii]],fMAD,size=kernelSize,mode='constant',cval=np.nan)[:,relInd_start[ii]:relInd_stop[ii]]

    elif option[0] == 5:  # displacement distance count with option[1] being the threshold
        center_ind = int(round((kernelSize[0]*kernelSize[1] + 1) / 2) - 1)

        @jit_filter_function
        def fDDC(values):
            count = 0
            for v in values:
                count += abs(v - values[center_ind]) < option[1]
            return count
    
        for ii in np.arange(N_chunks):
            out[:,ind_out_start[ii]:ind_out_stop[ii]] \
                = generic_filter(A[:,ind_startCols_chunks[ii]:ind_stopCols_chunks[ii]],fDDC,size=kernelSize,mode='constant',cval=np.nan)[:,relInd_start[ii]:relInd_stop[ii]]
    else:
        sys.exit("invalid option for columnwise neighborhood filtering")
        pass
    
    return out


class DISP_FILT:
    def __init__(self):
        ##filter parameters; try different parameters to decide how much fine-resolution estimates we keep, which can make the final images smoother

        self.FracValid = 8 / 25
        self.FracSearch = 0.20
        self.FiltWidth = 5
        self.Iter = 3
        self.MadScalar = 4
        self.colfiltChunkSize = 4

    def filtDisp(self, Dx, Dy, SearchLimitX, SearchLimitY, M, OverSampleRatio):

        import numpy as np

        if np.mod(self.FiltWidth, 2) == 0:
            sys.exit("NDC filter width must be an odd number")

        dToleranceX = self.FracValid * self.FiltWidth**2
        dToleranceY = self.FracValid * self.FiltWidth**2
        #        pdb.set_trace()
        Dx = Dx / SearchLimitX
        Dy = Dy / SearchLimitY

        DxMadmin = np.ones(Dx.shape) / OverSampleRatio / SearchLimitX * 2
        DyMadmin = np.ones(Dy.shape) / OverSampleRatio / SearchLimitY * 2

        for i in range(self.Iter):
            Dx[np.logical_not(M)] = np.nan
            Dy[np.logical_not(M)] = np.nan
            M = (
                colfilt(Dx.copy(), (self.FiltWidth, self.FiltWidth), (5, self.FracSearch), self.colfiltChunkSize)
                >= dToleranceX
            ) & (
                colfilt(Dy.copy(), (self.FiltWidth, self.FiltWidth), (5, self.FracSearch), self.colfiltChunkSize)
                >= dToleranceY
            )

        #        if self.Iter == 3:
        #            pdb.set_trace()

        for i in range(np.max([self.Iter - 1, 1])):
            Dx[np.logical_not(M)] = np.nan
            Dy[np.logical_not(M)] = np.nan

            DxMad = colfilt(Dx.copy(), (self.FiltWidth, self.FiltWidth), 6, self.colfiltChunkSize)
            DyMad = colfilt(Dy.copy(), (self.FiltWidth, self.FiltWidth), 6, self.colfiltChunkSize)

            DxM = colfilt(Dx.copy(), (self.FiltWidth, self.FiltWidth), 3, self.colfiltChunkSize)
            DyM = colfilt(Dy.copy(), (self.FiltWidth, self.FiltWidth), 3, self.colfiltChunkSize)

            M = (
                (np.abs(Dx - DxM) <= np.maximum(self.MadScalar * DxMad, DxMadmin))
                & (np.abs(Dy - DyM) <= np.maximum(self.MadScalar * DyMad, DyMadmin))
                & M
            )

        return M


def bwareaopen(image, size1):
    import numpy as np
    from skimage import measure

    # now identify the objects and remove those above a threshold
    labels, N = measure.label(image, connectivity=2, return_num=True)
    label_size = [(labels == label).sum() for label in range(N + 1)]

    # now remove the labels
    for label, size in enumerate(label_size):
        if size < size1:
            image[labels == label] = 0

    return image
