#!/usr/bin/env python

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2011, by the California Institute of Technology. ALL RIGHTS RESERVED.
# United States Government Sponsorship acknowledged. Any commercial use must be
# negotiated with the Office of Technology Transfer at the California Institute of
# Technology.  This software is subject to U.S. export control laws and regulations
# and has been classified as EAR99.  By accepting this software, the user agrees to
# comply with all applicable U.S. export laws and regulations.  User has the
# responsibility to obtain export licenses, or other export authority as may be
# required before exporting such information to foreign countries or providing
# access to foreign persons.
#
# Author: Eric Gurrola
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





import os

Import('env')
Import('envapplications')
Import('envcomponents')
envcontrib = env.Clone()
package = 'contrib'
envcontrib['PACKAGE'] = os.path.join(envcomponents['PACKAGE'],package)
envcontrib['INSTALL_COMPS'] = os.path.join(envcomponents['INSTALL_PATH'],package)
envcontrib['INSTALL_APPS'] = envapplications['INSTALL_PATH']
envcontrib['INSTALL_PATH'] = envcontrib['INSTALL_COMPS']
install = envcontrib['INSTALL_PATH']

initFile = '__init__.py'
if not os.path.exists(initFile):
	fout = open(initFile,"w")
	fout.write("#!/usr/bin/env python")
	fout.close()

listFiles = [initFile]
envcontrib.Install(install,listFiles)
envcontrib.Alias('install',install)
Export('envcontrib')

issi = os.path.join('issi','SConscript')
SConscript(issi)
snaphu = os.path.join('Snaphu','SConscript')
SConscript(snaphu)
demUtils = os.path.join('demUtils','SConscript')
SConscript(demUtils)
frameUtils = os.path.join('frameUtils','SConscript')
SConscript(frameUtils)
unwUtils = os.path.join('UnwrapComp','SConscript')
SConscript(unwUtils)

if 'MOTIFLIBPATH' in envcontrib.Dictionary():
    mdx = os.path.join('mdx','SConscript')
    SConscript(mdx)

rfi = os.path.join('rfi', 'SConscript')
SConscript(rfi)

SConscript('PyCuAmpcor/SConscript')
SConscript('splitSpectrum/SConscript')

SConscript('geo_autoRIFT/SConscript')
