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
 * Author: Yang Lei
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */






#include <Python.h>
#include <string>
//#include "autoriftcore.h"
#include "autoriftcoremodule.h"


#include "stdio.h"
#include "iostream"
#include "numpy/arrayobject.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/core/core.hpp"


using namespace cv;
using namespace std;

struct autoRiftCore{
//  This empty structure "autoRiftCore" in C++ is assgined to "self._autoriftcore" in python, which can take a set of variables in this file (declare here or in "autoriftcore.h" and set states below). For example,
//    ((autoRiftCore*)(ptr))->widC = widC;
//    ((autoRiftCore*)(ptr))->arPixDisp()
//  If taking all the variables here in the structure, the complicated computation can be performed in another C++ file, "autoriftcore.cpp" (that includes functions like void autoRiftCore::arPixDisp()).
};


static const char * const __doc__ = "Python extension for autoriftcore";


PyModuleDef moduledef = {
    //header
    PyModuleDef_HEAD_INIT,
    //name of the module
    "autoriftcore",
    //module documentation string
    __doc__,
    //size of the per-interpreter state of the module;
    -1,
    autoriftcore_methods,
};

//Initialization function for the module
PyMODINIT_FUNC
PyInit_autoriftcore()
{
    PyObject* module = PyModule_Create(&moduledef);
    if (!module)
    {
        return module;
    }
    return module;
}

PyObject* createAutoRiftCore(PyObject* self, PyObject *args)
{
    autoRiftCore* ptr = new autoRiftCore;
    return Py_BuildValue("K", (uint64_t) ptr);
}

PyObject* destroyAutoRiftCore(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    if (!PyArg_ParseTuple(args, "K", &ptr))
    {
        return NULL;
    }

    if (((autoRiftCore*)(ptr))!=NULL)
    {
        delete ((autoRiftCore*)(ptr));
    }
    return Py_BuildValue("i", 0);
}

PyObject* arPixDisp_u(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    PyArrayObject *I1, *I2, *Dx, *Dy, *Dx0, *Dy0, *xGrid, *yGrid, *SearchLimitX, *SearchLimitY, *ChipSizeX, *ChipSizeY;
    int widX, lenX;
    int widY, lenY;
    int widC, lenC;
    int widR, lenR;

    if (!PyArg_ParseTuple(args, "KiiOiiOiiOiiOOOOOOOOO", &ptr, &widC, &lenC, &I1, &widR, &lenR, &I2, &widX, &lenX, &xGrid, &widY, &lenY, &yGrid, &SearchLimitX, &SearchLimitY, &ChipSizeX, &ChipSizeY, &Dx, &Dy, &Dx0, &Dy0))
    {
        return NULL;
    }

    cv::Mat sec_img = cv::Mat(cv::Size(widC, lenC), CV_8UC1, reinterpret_cast<uint8_t*>(PyArray_DATA(I1)));
    cv::Mat ref_img = cv::Mat(cv::Size(widR, lenR), CV_8UC1, reinterpret_cast<uint8_t*>(PyArray_DATA(I2)));
    cv::Mat x_grid = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(xGrid)));
    cv::Mat y_grid = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(yGrid)));
    cv::Mat dx = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(Dx)));
    cv::Mat dy = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(Dy)));
    cv::Mat dx0 = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(Dx0)));
    cv::Mat dy0 = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(Dy0)));
    cv::Mat search_limit_x = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(SearchLimitX)));
    cv::Mat search_limit_y = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(SearchLimitY)));
    cv::Mat chip_size_x = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(ChipSizeX)));
    cv::Mat chip_size_y = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(ChipSizeY)));

    #pragma omp parallel for collapse(2)
    for(int j = 0; j < widX; j++)
    {
        for(int i = 0; i < lenX; i++)
        {
            if(search_limit_x.at<float>(i, j) == 0 and search_limit_y.at<float>(i, j) == 0)
            {
                continue;
            }

            float chip_limit_x = floor(chip_size_x.at<float>(i, j) / 2.0);
            float chip_limit_y = floor(chip_size_y.at<float>(i, j) / 2.0);

            int chip_x_start = int(-chip_limit_x - dx0.at<float>(i, j) + x_grid.at<float>(i, j));
            int chip_x_end = int(chip_limit_x - dx0.at<float>(i, j) + x_grid.at<float>(i, j));
            int chip_y_start = int(-chip_limit_y - dy0.at<float>(i, j) + y_grid.at<float>(i, j));
            int chip_y_end = int(chip_limit_y - dy0.at<float>(i, j) + y_grid.at<float>(i, j));
            int search_x_start = int(-chip_limit_x - search_limit_x.at<float>(i, j) + x_grid.at<float>(i, j));
            int search_x_end = int(chip_limit_x + search_limit_x.at<float>(i, j) - 1 + x_grid.at<float>(i, j));
            int search_y_start = int(-chip_limit_y - search_limit_y.at<float>(i, j) + y_grid.at<float>(i, j));
            int search_y_end = int(chip_limit_y + search_limit_y.at<float>(i, j) - 1 + y_grid.at<float>(i, j));

            cv::Range chip_x_range = cv::Range(chip_x_start, chip_x_end);
            cv::Range chip_y_range = cv::Range(chip_y_start, chip_y_end);
            cv::Range search_x_range = cv::Range(search_x_start, search_x_end);
            cv::Range search_y_range = cv::Range(search_y_start, search_y_end);

            cv::Mat chip = sec_img(chip_y_range, chip_x_range);
            cv::Mat ref = ref_img(search_y_range, search_x_range);

            cv::Point ref_min_loc;
            cv::Point chip_min_loc;
            cv::minMaxLoc(ref, NULL, NULL, &ref_min_loc, NULL);
            cv::minMaxLoc(chip, NULL, NULL, &chip_min_loc, NULL);

            uint8_t ref_min = ref.at<uint8_t>(ref_min_loc.y, ref_min_loc.x);
            uint8_t chip_min = chip.at<uint8_t>(chip_min_loc.y, chip_min_loc.x);

            ref =  ref_min < 0 ? ref.clone() - ref_min : ref;
            chip =  chip_min < 0 ? chip.clone() - chip_min : chip;

            int chip_width = chip_x_end - chip_x_start;
            int chip_length = chip_y_end - chip_y_start;
            int search_width = search_x_end - search_x_start;
            int search_length = search_y_end - search_y_start;

            int result_cols =  search_width - chip_width + 1;
            int result_rows = search_length - chip_length + 1;

            cv::Mat result;
            result.create( result_rows, result_cols, CV_32FC1 );

            cv::matchTemplate( ref, chip, result, CV_TM_CCOEFF_NORMED );

            cv::Point maxLoc;
            cv::minMaxLoc(result, NULL, NULL, NULL, &maxLoc);

            dx.col(j).at<float>(i) = maxLoc.x;
            dy.col(j).at<float>(i) = maxLoc.y;
        }
    }

    return Py_BuildValue("OO", Dx, Dy);
}

PyObject* arSubPixDisp_u(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    PyArrayObject *I1, *I2, *Dx, *Dy, *Dx0, *Dy0, *xGrid, *yGrid, *SearchLimitX, *SearchLimitY, *ChipSizeX, *ChipSizeY;
    int widX, lenX;
    int widY, lenY;
    int widC, lenC;
    int widR, lenR;
    int overSampleNC;

    if (!PyArg_ParseTuple(args, "KiiOiiOiiOiiOOOOOOOOOi", &ptr, &widC, &lenC, &I1, &widR, &lenR, &I2, &widX, &lenX, &xGrid, &widY, &lenY, &yGrid, &SearchLimitX, &SearchLimitY, &ChipSizeX, &ChipSizeY, &Dx, &Dy, &Dx0, &Dy0, &overSampleNC))
    {
        return NULL;
    }

    cv::Mat sec_img = cv::Mat(cv::Size(widC, lenC), CV_8UC1, reinterpret_cast<uint8_t*>(PyArray_DATA(I1)));
    cv::Mat ref_img = cv::Mat(cv::Size(widR, lenR), CV_8UC1, reinterpret_cast<uint8_t*>(PyArray_DATA(I2)));
    cv::Mat x_grid = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(xGrid)));
    cv::Mat y_grid = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(yGrid)));
    cv::Mat dx = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(Dx)));
    cv::Mat dy = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(Dy)));
    cv::Mat dx0 = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(Dx0)));
    cv::Mat dy0 = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(Dy0)));
    cv::Mat search_limit_x = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(SearchLimitX)));
    cv::Mat search_limit_y = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(SearchLimitY)));
    cv::Mat chip_size_x = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(ChipSizeX)));
    cv::Mat chip_size_y = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(ChipSizeY)));

    #pragma omp parallel for collapse(2)
    for(int j = 0; j < widX; j++)
    {
        for(int i = 0; i < lenX; i++)
        {
            if(search_limit_x.at<float>(i, j) == 0 and search_limit_y.at<float>(i, j) == 0)
            {
                continue;
            }

            float chip_limit_x = floor(chip_size_x.at<float>(i, j) / 2.0);
            float chip_limit_y = floor(chip_size_y.at<float>(i, j) / 2.0);

            int chip_x_start = int(-chip_limit_x - dx0.at<float>(i, j) + x_grid.at<float>(i, j));
            int chip_x_end = int(chip_limit_x - dx0.at<float>(i, j) + x_grid.at<float>(i, j));
            int chip_y_start = int(-chip_limit_y - dy0.at<float>(i, j) + y_grid.at<float>(i, j));
            int chip_y_end = int(chip_limit_y - dy0.at<float>(i, j) + y_grid.at<float>(i, j));
            int search_x_start = int(-chip_limit_x - search_limit_x.at<float>(i, j) + x_grid.at<float>(i, j));
            int search_x_end = int(chip_limit_x + search_limit_x.at<float>(i, j) - 1 + x_grid.at<float>(i, j));
            int search_y_start = int(-chip_limit_y - search_limit_y.at<float>(i, j) + y_grid.at<float>(i, j));
            int search_y_end = int(chip_limit_y + search_limit_y.at<float>(i, j) - 1 + y_grid.at<float>(i, j));

            cv::Range chip_x_range = cv::Range(chip_x_start, chip_x_end);
            cv::Range chip_y_range = cv::Range(chip_y_start, chip_y_end);
            cv::Range search_x_range = cv::Range(search_x_start, search_x_end);
            cv::Range search_y_range = cv::Range(search_y_start, search_y_end);

            cv::Mat chip = sec_img(chip_y_range, chip_x_range).clone();
            cv::Mat ref = ref_img(search_y_range, search_x_range).clone();

            cv::Point ref_min_loc;
            cv::Point chip_min_loc;
            cv::minMaxLoc(ref, NULL, NULL, &ref_min_loc, NULL);
            cv::minMaxLoc(chip, NULL, NULL, &chip_min_loc, NULL);

            uint8_t ref_min = ref.at<uint8_t>(ref_min_loc.y, ref_min_loc.x);
            uint8_t chip_min = chip.at<uint8_t>(chip_min_loc.y, chip_min_loc.x);

            ref =  ref_min < 0 ? ref.clone() - ref_min : ref;
            chip =  chip_min < 0 ? chip.clone() - chip_min : chip;

            int chip_width = chip_x_end - chip_x_start;
            int chip_length = chip_y_end - chip_y_start;
            int search_width = search_x_end - search_x_start;
            int search_length = search_y_end - search_y_start;

            int result_cols =  search_width - chip_width + 1;
            int result_rows = search_length - chip_length + 1;

            cv::Mat result;
            result.create( result_rows, result_cols, CV_32FC1 );

            cv::matchTemplate( ref, chip, result, CV_TM_CCOEFF_NORMED);

            cv::Point maxLoc;
            cv::minMaxLoc(result, NULL, NULL, NULL, &maxLoc);

            int x_start, y_start, x_count, y_count;

            x_start = cv::max(maxLoc.x-2, 0);
            x_start = cv::min(x_start, result_cols-5);
            x_count = 5;

            y_start = cv::max(maxLoc.y-2, 0);
            y_start = cv::min(y_start, result_rows-5);
            y_count = 5;

            cv::Mat result_small (result, cv::Rect(x_start, y_start, x_count, y_count));

            int cols = result_small.cols;
            int rows = result_small.rows;
            int overSampleFlag = 1;

            cv::Mat predecessor_small = result_small;
            cv::Mat foo;

            while (overSampleFlag < overSampleNC){
                cols *= 2;
                rows *= 2;
                overSampleFlag *= 2;
                foo.create(cols, rows, CV_32FC1);
                cv::pyrUp(predecessor_small, foo, cv::Size(cols, rows));
                predecessor_small = foo;
            }

            cv::Point maxLoc_small;
            cv::minMaxLoc(foo, NULL, NULL, NULL, &maxLoc_small);

            dx.col(j).at<float>(i) = ((maxLoc_small.x + 0.0)/overSampleNC + x_start);
            dy.col(j).at<float>(i) = ((maxLoc_small.y + 0.0)/overSampleNC + y_start);
        }
    }

    return Py_BuildValue("OO", Dx, Dy);
}

PyObject* arPixDisp_s(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    PyArrayObject *I1, *I2, *Dx, *Dy, *Dx0, *Dy0, *xGrid, *yGrid, *SearchLimitX, *SearchLimitY, *ChipSizeX, *ChipSizeY;
    int widX, lenX;
    int widY, lenY;
    int widC, lenC;
    int widR, lenR;

    if (!PyArg_ParseTuple(args, "KiiOiiOiiOiiOOOOOOOOO", &ptr, &widC, &lenC, &I1, &widR, &lenR, &I2, &widX, &lenX, &xGrid, &widY, &lenY, &yGrid, &SearchLimitX, &SearchLimitY, &ChipSizeX, &ChipSizeY, &Dx, &Dy, &Dx0, &Dy0))
    {
        return NULL;
    }

    cv::Mat sec_img = cv::Mat(cv::Size(widC, lenC), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(I1)));
    cv::Mat ref_img = cv::Mat(cv::Size(widR, lenR), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(I2)));
    cv::Mat x_grid = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(xGrid)));
    cv::Mat y_grid = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(yGrid)));
    cv::Mat dx = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(Dx)));
    cv::Mat dy = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(Dy)));
    cv::Mat dx0 = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(Dx0)));
    cv::Mat dy0 = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(Dy0)));
    cv::Mat search_limit_x = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(SearchLimitX)));
    cv::Mat search_limit_y = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(SearchLimitY)));
    cv::Mat chip_size_x = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(ChipSizeX)));
    cv::Mat chip_size_y = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(ChipSizeY)));

    #pragma omp parallel for collapse(2)
    for(int j = 0; j < widX; j++)
    {
        for(int i = 0; i < lenX; i++)
        {
            if(search_limit_x.at<float>(i, j) == 0 and search_limit_y.at<float>(i, j) == 0)
            {
                continue;
            }

            float chip_limit_x = floor(chip_size_x.at<float>(i, j) / 2.0);
            float chip_limit_y = floor(chip_size_y.at<float>(i, j) / 2.0);

            int chip_x_start = int(-chip_limit_x - dx0.at<float>(i, j) + x_grid.at<float>(i, j));
            int chip_x_end = int(chip_limit_x - dx0.at<float>(i, j) + x_grid.at<float>(i, j));
            int chip_y_start = int(-chip_limit_y - dy0.at<float>(i, j) + y_grid.at<float>(i, j));
            int chip_y_end = int(chip_limit_y - dy0.at<float>(i, j) + y_grid.at<float>(i, j));
            int search_x_start = int(-chip_limit_x - search_limit_x.at<float>(i, j) + x_grid.at<float>(i, j));
            int search_x_end = int(chip_limit_x + search_limit_x.at<float>(i, j) - 1 + x_grid.at<float>(i, j));
            int search_y_start = int(-chip_limit_y - search_limit_y.at<float>(i, j) + y_grid.at<float>(i, j));
            int search_y_end = int(chip_limit_y + search_limit_y.at<float>(i, j) - 1 + y_grid.at<float>(i, j));

            cv::Range chip_x_range = cv::Range(chip_x_start, chip_x_end);
            cv::Range chip_y_range = cv::Range(chip_y_start, chip_y_end);
            cv::Range search_x_range = cv::Range(search_x_start, search_x_end);
            cv::Range search_y_range = cv::Range(search_y_start, search_y_end);

            cv::Mat chip = sec_img(chip_y_range, chip_x_range).clone();
            cv::Mat ref = ref_img(search_y_range, search_x_range).clone();

            cv::Point ref_min_loc;
            cv::Point chip_min_loc;
            cv::minMaxLoc(ref, NULL, NULL, &ref_min_loc, NULL);
            cv::minMaxLoc(chip, NULL, NULL, &chip_min_loc, NULL);

            float ref_min = ref.at<float>(ref_min_loc.y, ref_min_loc.x);
            float chip_min = chip.at<float>(chip_min_loc.y, chip_min_loc.x);

            ref =  ref_min < 0 ? ref.clone() - ref_min : ref;
            chip =  chip_min < 0 ? chip.clone() - chip_min : chip;

            int chip_width = chip_x_end - chip_x_start;
            int chip_length = chip_y_end - chip_y_start;
            int search_width = search_x_end - search_x_start;
            int search_length = search_y_end - search_y_start;

            int result_cols =  search_width - chip_width + 1;
            int result_rows = search_length - chip_length + 1;

            cv::Mat result;
            result.create( result_rows, result_cols, CV_32FC1 );

            cv::matchTemplate( ref, chip, result, CV_TM_CCOEFF_NORMED );

            cv::Point maxLoc;
            cv::minMaxLoc(result, NULL, NULL, NULL, &maxLoc);

            dx.col(j).at<float>(i) = maxLoc.x;
            dy.col(j).at<float>(i) = maxLoc.y;
        }
    }

    return Py_BuildValue("OO", Dx, Dy);
}

PyObject* arSubPixDisp_s(PyObject *self, PyObject *args)
{
    uint64_t ptr;
    PyArrayObject *I1, *I2, *Dx, *Dy, *Dx0, *Dy0, *xGrid, *yGrid, *SearchLimitX, *SearchLimitY, *ChipSizeX, *ChipSizeY;
    int widX, lenX;
    int widY, lenY;
    int widC, lenC;
    int widR, lenR;
    int overSampleNC;

    if (!PyArg_ParseTuple(args, "KiiOiiOiiOiiOOOOOOOOOi", &ptr, &widC, &lenC, &I1, &widR, &lenR, &I2, &widX, &lenX, &xGrid, &widY, &lenY, &yGrid, &SearchLimitX, &SearchLimitY, &ChipSizeX, &ChipSizeY, &Dx, &Dy, &Dx0, &Dy0, &overSampleNC))
    {
        return NULL;
    }

    cv::Mat sec_img = cv::Mat(cv::Size(widC, lenC), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(I1)));
    cv::Mat ref_img = cv::Mat(cv::Size(widR, lenR), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(I2)));
    cv::Mat x_grid = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(xGrid)));
    cv::Mat y_grid = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(yGrid)));
    cv::Mat dx = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(Dx)));
    cv::Mat dy = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(Dy)));
    cv::Mat dx0 = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(Dx0)));
    cv::Mat dy0 = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(Dy0)));
    cv::Mat search_limit_x = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(SearchLimitX)));
    cv::Mat search_limit_y = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(SearchLimitY)));
    cv::Mat chip_size_x = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(ChipSizeX)));
    cv::Mat chip_size_y = cv::Mat(cv::Size(widX, lenX), CV_32FC1, reinterpret_cast<float*>(PyArray_DATA(ChipSizeY)));

    #pragma omp parallel for collapse(2)
    for(int j = 0; j < widX; j++)
    {
        for(int i = 0; i < lenX; i++)
        {
            if(search_limit_x.at<float>(i, j) == 0 and search_limit_y.at<float>(i, j) == 0)
            {
                continue;
            }

            float chip_limit_x = floor(chip_size_x.at<float>(i, j) / 2.0);
            float chip_limit_y = floor(chip_size_y.at<float>(i, j) / 2.0);

            int chip_x_start = int(-chip_limit_x - dx0.at<float>(i, j) + x_grid.at<float>(i, j));
            int chip_x_end = int(chip_limit_x - dx0.at<float>(i, j) + x_grid.at<float>(i, j));
            int chip_y_start = int(-chip_limit_y - dy0.at<float>(i, j) + y_grid.at<float>(i, j));
            int chip_y_end = int(chip_limit_y - dy0.at<float>(i, j) + y_grid.at<float>(i, j));
            int search_x_start = int(-chip_limit_x - search_limit_x.at<float>(i, j) + x_grid.at<float>(i, j));
            int search_x_end = int(chip_limit_x + search_limit_x.at<float>(i, j) - 1 + x_grid.at<float>(i, j));
            int search_y_start = int(-chip_limit_y - search_limit_y.at<float>(i, j) + y_grid.at<float>(i, j));
            int search_y_end = int(chip_limit_y + search_limit_y.at<float>(i, j) - 1 + y_grid.at<float>(i, j));

            cv::Range chip_x_range = cv::Range(chip_x_start, chip_x_end);
            cv::Range chip_y_range = cv::Range(chip_y_start, chip_y_end);
            cv::Range search_x_range = cv::Range(search_x_start, search_x_end);
            cv::Range search_y_range = cv::Range(search_y_start, search_y_end);

            cv::Mat chip = sec_img(chip_y_range, chip_x_range).clone();
            cv::Mat ref = ref_img(search_y_range, search_x_range).clone();

            cv::Point ref_min_loc;
            cv::Point chip_min_loc;
            cv::minMaxLoc(ref, NULL, NULL, &ref_min_loc, NULL);
            cv::minMaxLoc(chip, NULL, NULL, &chip_min_loc, NULL);

            float ref_min = ref.at<float>(ref_min_loc.y, ref_min_loc.x);
            float chip_min = chip.at<float>(chip_min_loc.y, chip_min_loc.x);

            ref =  ref_min < 0 ? ref.clone() - ref_min : ref;
            chip =  chip_min < 0 ? chip.clone() - chip_min : chip;

            int chip_width = chip_x_end - chip_x_start;
            int chip_length = chip_y_end - chip_y_start;
            int search_width = search_x_end - search_x_start;
            int search_length = search_y_end - search_y_start;

            int result_cols =  search_width - chip_width + 1;
            int result_rows = search_length - chip_length + 1;

            cv::Mat result;
            result.create( result_rows, result_cols, CV_32FC1 );

            cv::matchTemplate( ref, chip, result, CV_TM_CCOEFF_NORMED );

            cv::Point maxLoc;
            cv::minMaxLoc(result, NULL, NULL, NULL, &maxLoc);

            int x_start, y_start, x_count, y_count;

            x_start = cv::max(maxLoc.x-2, 0);
            x_start = cv::min(x_start, result_cols-5);
            x_count = 5;

            y_start = cv::max(maxLoc.y-2, 0);
            y_start = cv::min(y_start, result_rows-5);
            y_count = 5;

            cv::Mat result_small (result, cv::Rect(x_start, y_start, x_count, y_count));

            int cols = result_small.cols;
            int rows = result_small.rows;
            int overSampleFlag = 1;

            cv::Mat predecessor_small = result_small;
            cv::Mat foo;

            while (overSampleFlag < overSampleNC){
                cols *= 2;
                rows *= 2;
                overSampleFlag *= 2;
                foo.create(cols, rows, CV_32FC1);
                cv::pyrUp(predecessor_small, foo, cv::Size(cols, rows));
                predecessor_small = foo;
            }

            cv::Point maxLoc_small;
            cv::minMaxLoc(foo, NULL, NULL, NULL, &maxLoc_small);

            dx.col(j).at<float>(i) = ((maxLoc_small.x + 0.0)/overSampleNC + x_start);
            dy.col(j).at<float>(i) = ((maxLoc_small.y + 0.0)/overSampleNC + y_start);
        }
    }

    return Py_BuildValue("OO", Dx, Dy);
}
