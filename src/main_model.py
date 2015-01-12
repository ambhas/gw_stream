#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 14:25:08 2014

@author: Sat Kumar Tomer
@email: satkumartomer@gmail.com
@website: www.ambhas.com
"""

from __future__ import division
import numpy as np
from ambhas.gis import read_ascii_grid
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve 
from scipy.sparse import lil_matrix
from matplotlib.mlab import find
import datetime 
from ambhas.time import doy2md, ymd2doy
from osgeo.gdalconst import *
import gdal
import os
import csv

def ambhas_gw_2d_xy(watershed, hini, T, Sy, dx, dt, hmin, par_discharge, net_recharge):
    """
    this class performs the 2 dimensinoal groundwater modelling in horizonal 
    plane using the 2 dimensional groundwater flow equation

    The 2D groundwater euqation is solved using generic implicit-explicit method 
    
    Input:
        watershed:  map of watershed; 1 means inside and 0 means outside watershed
        hini:       initial groundwater level
        T:          transmissivity 
        Sy:         specific yield
        dx:         spatial resolution
        dt:         time step 
        hmin:       groundwater level corresponding to zero discharge
        par_discharge: parameter controlling the discharge (range is 0 to 1) 
        net_recharge:   net recharge to the groundwater (recharge-groundwater pumping)
    
    Output:
        hnew:       groundwater level at next time step
        discharge:  baseflow (volume/time)
        
    """
    d = dx*1.0  # spatial resolution of the model
    D = T/Sy
    
    #neumann_criterion = 0.5*d**2/D
    neumann_criterion = 2*D*dt/d**2
    #max_neumann_criterion = neumann_criterion.max()
    
    # update the model for recharge 
    hini = hini + net_recharge/Sy
    
    # take the discharge out from each cell
    discharge = (1-par_discharge)*(hini-hmin)*Sy
    discharge[hini<hmin] = 0
    hini = hini - discharge/Sy 
   
    # spatial computing    
    n = int(np.sum(watershed))
    foo = np.cumsum(watershed.flatten())
    foo.shape = watershed.shape
    foo[watershed==0] = np.nan
    foo = foo-1 # indices start from 0

    
    ih, jh = np.unravel_index(find(~np.isnan(foo)), watershed.shape) # indices of non nan
    alpha = 1.0 # implicit method
    
    # setup A and b matrix
    A = lil_matrix((n, n))
    b = np.zeros(n,)
    for i in xrange(n):
        a1 = alpha/d**2
        a2 = -(4*alpha/d**2 + Sy/T[ih[i],jh[i]]/dt)
    
        # i,j
        A[i,i] = a2
        b[i] = (-4*(1-alpha)/d**2 - Sy/T[ih[i],jh[i]]/dt)*hini[ih[i],jh[i]]
    
        # i-1,j
        ind_h = foo[ih[i]-1,jh[i]]
        if np.isnan(ind_h):
            A[i,i] = A[i,i] + a1
            b[i] = b[i] + (1-alpha)/d**2*hini[ih[i],jh[i]]
        else:
            A[i,int(ind_h)] = a1
            b[i] = b[i] + (1-alpha)/d**2*hini[ih[i]-1,jh[i]]
        
        # i+1, j
        ind_h = foo[ih[i]+1,jh[i]]
        if np.isnan(ind_h):
            A[i,i] = A[i,i] + a1
            b[i] = b[i] + (1-alpha)/d**2*hini[ih[i],jh[i]]
        else:
            A[i,int(ind_h)] = a1
            b[i] = b[i] + (1-alpha)/d**2*hini[ih[i]+1,jh[i]]
        
        # i, j-1
        ind_h = foo[ih[i],jh[i]-1]
        if np.isnan(ind_h):
            A[i,i] = A[i,i] + a1
            b[i] = b[i] + (1-alpha)/d**2*hini[ih[i],jh[i]]
        else:
            A[i,int(ind_h)] = a1
            b[i] = b[i] + (1-alpha)/d**2*hini[ih[i],jh[i]-1]
        
        # i, j+1
        ind_h = foo[ih[i],jh[i]+1]
        if np.isnan(ind_h):
            A[i,i] = A[i,i] + a1
            b[i] = b[i] + (1-alpha)/d**2*hini[ih[i],jh[i]]
        else:
            A[i,int(ind_h)] = a1
            b[i] = b[i] + (1-alpha)/d**2*hini[ih[i],jh[i]+1]
    
    # solve
    tmp = spsolve(A.tocsr(),b)
    hnew = np.zeros(watershed.shape)
    hnew[ih,jh] = tmp
    hnew[watershed==0] = np.nan
    
    return hnew, np.nansum(discharge)*d**2

def ambhas_stream_2d_xy(dem, stream_depth, stream_k, stream_area, stream_m, h_gw):
    h_stream_bed = dem - stream_depth
    Ql = (stream_k*stream_area/stream_m)*(h_gw-h_stream_bed)
    return Ql

def run_model(input_file_name):
    
    # read input file
    print('Started reading input file %s'%input_file_name)
    
    par = {}
    with open(input_file_name) as f:
        for line in f:
            line = line.split('#', 1)[0]
            line = line.rstrip()
            if len(line)>1:
                key, value = line.split()
                par[key.strip()] = value.strip()
    
    
    start_year = int(par['start_year'])
    end_year = int(par['end_year'])
    start_month, start_day = doy2md(int(par['start_doy']), start_year)    
    end_month, end_day = doy2md(int(par['end_doy']), end_year)    
    
    start_dt = datetime.date(start_year, start_month, start_day)    
    end_dt = datetime.date(end_year, end_month, end_day)   
    n_time = end_dt.toordinal()-start_dt.toordinal()+1
    print('Read temporal parameters')
    
    hini_file = par['hini_file']
    dataset = gdal.Open(hini_file, GA_ReadOnly)
    hini = dataset.GetRasterBand(1).ReadAsArray()
    geotransform = dataset.GetGeoTransform()
    RasterXSize = dataset.RasterXSize
    RasterYSize = dataset.RasterYSize    
    dataset = None
    print('Read hini')   
    
        
    dem_file = par['dem_file']
    dataset = gdal.Open(dem_file, GA_ReadOnly)
    dem = dataset.GetRasterBand(1).ReadAsArray()
    dataset = None
    print('Read DEM')
    
    stream_file = par['stream_file']
    dataset = gdal.Open(stream_file, GA_ReadOnly)
    stream = dataset.GetRasterBand(1).ReadAsArray()
    dataset = None
    print('Read stream')
    
    watershed_file = par['watershed_file']
    dataset = gdal.Open(watershed_file, GA_ReadOnly)
    watershed = dataset.GetRasterBand(1).ReadAsArray()
    dataset = None
    
    watershed[watershed>0] = 1
    watershed[np.isnan(watershed)] = 0
    watershed[0,:] = 0
    watershed[-1,:] = 0
    watershed[:,0] = 0
    watershed[:,-1] = 0   
    
    print('Read watershed')
    
    T = np.empty(hini.shape)
    T[:] = float(par['T'])
    Sy = float(par['Sy'])
    hmin = float(par['hmin'])
    par_discharge = float(par['par_discharge'])
    recharge_factor = float(par['recharge_factor'])
    rain_multiplier = float(par['rain_multiplier'])
    stream_depth = float(par['stream_depth'])
    stream_k = float(par['stream_k'])
    stream_m = float(par['stream_m'])
    print('Read groundwater parameters')
    
    print('Finished reading input file %s'%input_file_name)
    
        
    if not os.path.exists(par['h_dir_tif']):
        os.makedirs(par['h_dir_tif'])
    
    if not os.path.exists(par['h_dir_png']):
        os.makedirs(par['h_dir_png'])
        
    if not os.path.exists(par['ql_dir_tif']):
        os.makedirs(par['ql_dir_tif'])
    
    if not os.path.exists(par['ql_dir_png']):
        os.makedirs(par['ql_dir_png'])
    
    dx = geotransform[1]
    dt = 1.0
    stream_area = stream*dx**2
    
    plt.ioff()
    
    out_file = par['discharge_file']
    with open(out_file, 'w') as the_file:
        header = ['year', 'month', 'day', 'doy', 'mean_h_gw', 'discharge']
        writer = csv.writer(the_file, quotechar='"')
        writer.writerow(header)
            
        for t in range(n_time):
            t_ordinal = datetime.date.fromordinal(start_dt.toordinal()+t)
            t_day = t_ordinal.day
            t_month = t_ordinal.month
            t_year = t_ordinal.year
            t_doy = ymd2doy([t_year], [t_month], [t_day])[0]
            
            # read rainfall file
            rain_dir = par['rain_dir']
            rain_file = os.path.join(rain_dir, '%i%03d.tif'%(t_year, t_doy))
            dataset = gdal.Open(rain_file, GA_ReadOnly)
            rain = dataset.GetRasterBand(1).ReadAsArray()
            dataset = None        
            
            # compute recharge
            net_recharge = recharge_factor*rain*rain_multiplier
            
            # groundwater model          
            h_gw, discharge = ambhas_gw_2d_xy(watershed, hini, T, Sy, dx, dt, hmin, par_discharge, net_recharge)
            
            # stream model
            Ql = ambhas_stream_2d_xy(dem, stream_depth, stream_k, stream_area, stream_m, h_gw)     
            hini = h_gw - Ql/(dx**2)
            #discharge_stream = np.nansum(Ql)
            
            mean_h_gw = np.nanmean(hini[watershed>0])
            
            foo = ['%d'%t_year, '%d'%t_month, '%d'%t_day, '%d'%t_doy, '%.2f'%mean_h_gw, '%.2f'%discharge]                    
            writer.writerow(foo)
    
            # save the gw level as Gtiff
            out_file = os.path.join(par['h_dir_tif'], '%i%03d.tif'%(t_year, t_doy))
            driver = gdal.GetDriverByName('GTiff')
            output_dataset = driver.Create(out_file, RasterXSize, RasterYSize, 1, gdal.GDT_Float32)
            output_dataset.SetGeoTransform(geotransform)
            output_dataset.GetRasterBand(1).WriteArray(hini, 0, 0)
            output_dataset = None       
            
            # save the gw level as png            
            plt.matshow(hini)
            plt.colorbar(shrink=0.7)
            fig_png = os.path.join(par['h_dir_png'], '%i%03d.png'%(t_year, t_doy))
            plt.savefig(fig_png)
            plt.close()
            
            # save the Ql as Gtiff
            out_file = os.path.join(par['ql_dir_tif'], '%i%03d.tif'%(t_year, t_doy))
            driver = gdal.GetDriverByName('GTiff')
            output_dataset = driver.Create(out_file, RasterXSize, RasterYSize, 1, gdal.GDT_Float32)
            output_dataset.SetGeoTransform(geotransform)
            output_dataset.GetRasterBand(1).WriteArray(Ql, 0, 0)
            output_dataset = None      
            
            # save the gw level as png
            plt.matshow(Ql)
            plt.colorbar(shrink=0.7)
            fig_png = os.path.join(par['ql_dir_png'], '%i%03d.png'%(t_year, t_doy))
            plt.savefig(fig_png)
            plt.close()
            
            print('year=%i, doy=%03d'%(t_year, t_doy))
    
    
if __name__ == "__main__":
    run_model('../input/input.txt')

    
    
    