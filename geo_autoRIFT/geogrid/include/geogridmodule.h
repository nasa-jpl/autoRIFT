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


#ifndef geogridmodule_h
#define geogridmodule_h

#include <Python.h>
#include <stdint.h>

extern "C"
{
        PyObject * createGeoGrid(PyObject*, PyObject*);
        PyObject * destroyGeoGrid(PyObject*, PyObject*);
        PyObject * geogrid(PyObject *, PyObject *);
        PyObject * setRadarImageDimensions(PyObject *, PyObject *);
        PyObject * setRangeParameters(PyObject *, PyObject *);
        PyObject * setAzimuthParameters(PyObject*, PyObject *);
        PyObject * setRepeatTime(PyObject *, PyObject *);
    
        PyObject * setDEM(PyObject *, PyObject *);
        PyObject * setVelocities(PyObject*, PyObject*);
        PyObject * setSlopes(PyObject*, PyObject*);
        PyObject * setOrbit(PyObject *, PyObject *);
        PyObject * setLookSide(PyObject *, PyObject *); 

        PyObject * setWindowLocationsFilename(PyObject *, PyObject *);
        PyObject * setWindowOffsetsFilename(PyObject *, PyObject *);
        PyObject * setRO2VXFilename(PyObject *, PyObject *);
        PyObject * setRO2VYFilename(PyObject *, PyObject *);
        PyObject * setEPSG(PyObject *, PyObject *);
        PyObject * setXLimits(PyObject *, PyObject *);
        PyObject * setYLimits(PyObject *, PyObject *);
}

static PyMethodDef geogrid_methods[] =
{
        {"createGeoGrid_Py", createGeoGrid, METH_VARARGS, " "},
        {"destroyGeoGrid_Py", destroyGeoGrid, METH_VARARGS, " "},
        {"geogrid_Py", geogrid, METH_VARARGS, " "},
        {"setRadarImageDimensions_Py", setRadarImageDimensions, METH_VARARGS, " "},
        {"setRangeParameters_Py", setRangeParameters, METH_VARARGS, " "},
        {"setAzimuthParameters_Py", setAzimuthParameters, METH_VARARGS, " "},
        {"setRepeatTime_Py", setRepeatTime, METH_VARARGS, " "},
        {"setDEM_Py", setDEM, METH_VARARGS, " "},
        {"setEPSG_Py", setEPSG, METH_VARARGS, " "},
        {"setVelocities_Py", setVelocities, METH_VARARGS, " "},
        {"setSlopes_Py", setSlopes, METH_VARARGS, " "},
        {"setOrbit_Py", setOrbit, METH_VARARGS, " "},
        {"setLookSide_Py", setLookSide, METH_VARARGS, " "},
        {"setXLimits_Py", setXLimits, METH_VARARGS, " "},
        {"setYLimits_Py", setYLimits, METH_VARARGS, " "},
        {"setWindowLocationsFilename_Py", setWindowLocationsFilename, METH_VARARGS, " "},
        {"setWindowOffsetsFilename_Py", setWindowOffsetsFilename, METH_VARARGS, " "},
        {"setRO2VXFilename_Py", setRO2VXFilename, METH_VARARGS, " "},
        {"setRO2VYFilename_Py", setRO2VYFilename, METH_VARARGS, " "},
        {NULL, NULL, 0, NULL}
};
#endif //geoGridmodule_h

