#!/usr/bin/env python3

########
#Yang Lei, Jet Propulsion Laboratory
#November 2017

def loadXml():
    import os
    from topsApp import TopsInSAR
    insar = TopsInSAR(name="topsApp")
    insar.configure()
    master_filename = os.path.basename(insar.master.safe[0])
    slave_filename = os.path.basename(insar.slave.safe[0])
    return master_filename, slave_filename, insar.master.safe[0], insar.slave.safe[0]

def loadXml_new():
    import os
    from topsApp import TopsInSAR
    insar = TopsInSAR(name="topsApp")
    insar.configure()
    master_filename = os.path.basename(insar.reference.safe[0])
    slave_filename = os.path.basename(insar.secondary.safe[0])
    return master_filename, slave_filename, insar.reference.safe[0], insar.secondary.safe[0]

def loadParsedata(indir):
    '''
        Input file.
        '''
    import os
    import numpy as np
    import isce
    from isceobj.Sensor.TOPS.Sentinel1 import Sentinel1
    
    
    frames = []
    for swath in range(1,4):
        rdr=Sentinel1()
        rdr.configure()
#        rdr.safe=['./S1A_IW_SLC__1SDH_20180401T100057_20180401T100124_021272_024972_8CAF.zip']
        rdr.safe=[indir]
        rdr.output='master'
        rdr.orbitDir='/Users/yanglei/orbit/S1A/precise'
        rdr.auxDir='/Users/yanglei/orbit/S1A/aux'
        rdr.swathNumber=swath
        rdr.polarization='hh'
        rdr.parse()
        frames.append(rdr.product)
    
    sensingStart = min([x.sensingStart for x in frames])
    sensingStop = max([x.sensingStop for x in frames])
    
    info = (sensingStop - sensingStart) / 2 + sensingStart
    
#    info = info.strftime("%Y%m%dT%H:%M:%S")
    info = info.strftime("%Y%m%dT%H:%M:%S.%f").rstrip('0')
    
    return info

def cmdLineParse():
    '''
        Command line parser.
        '''
    import argparse
    parser = argparse.ArgumentParser(description="Single-pair InSAR processing of Sentinel-1 data using ISCE modules")
    
    return parser.parse_args()


if __name__ == '__main__':
    import scipy.io as sio
    try:
        master_filename, slave_filename, master_path, slave_path = loadXml()
    except:
        master_filename, slave_filename, master_path, slave_path = loadXml_new()
    print(master_filename)
    print(slave_filename)
    
    master_dt = loadParsedata(master_path)
    slave_dt = loadParsedata(slave_path)
    
    print(master_dt)
    print(slave_dt)
    
    sio.savemat('topsinsar_filename.mat',{'master_filename':master_filename,'slave_filename':slave_filename,'master_dt':master_dt,'slave_dt':slave_dt})
