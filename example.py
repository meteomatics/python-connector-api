#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding:=utf-8

import argparse
import python_query_api_v1_4 as api
import datetime as dt
import sys

'''
    For further information on available parameters, models etc. please visit
    api.meteomatics.com

    In case of questions just write a mail to:
    support@meteomatics.com
'''

###Credentials:

username = 'python-community'  # TODO
password = 'Erumukete173'  # TODO

def example():
    ###Input timeseries:
    now = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    startdate_ts = now
    enddate_ts = startdate_ts + dt.timedelta(days=7)
    interval_ts = dt.timedelta(hours=1)
    coordinates_ts = [(47.249297, 9.342854), (50., 10.)]
    parameters_ts = ['t_2m:C', 'rr_1h:mm']
    model = 'ecmwf-ifs'
    ens_select = None  # e.g. 'median'
    interp_select = 'gradient_interpolation'
    
    ###Input grid / grid unpivoted:
    lat_N = 50
    lon_W = -15
    lat_S = 20
    lon_E = 10
    res_lat = 0.1
    res_lon = 0.1
    startdate_grid = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    parameter_grid = 'evapotranspiration_1h:mm'  # 't_2m:C'
    
    parameters_grid_unpiv = ['t_2m:C', 'rr_1h:mm']
    valid_dates_unpiv = [dt.datetime.utcnow(), dt.datetime.utcnow()+dt.timedelta(days=1)]
    
    ###input grid png
    filename_png = "grid_target.png"
    startdate_png = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    parameter_png = 't_2m:C'
    
    ###input lightning
    startdate_l = dt.datetime(2017, 8, 6, 00, 00, 00)
    enddate_l = dt.datetime(2017, 8, 6, 00, 20, 00)
    lat_N_l = 50
    lon_W_l = 10
    lat_S_l = 40
    lon_E_l = 20
    
    ###input grads
    filename_grads = "path_grads/grads_target.png"
    startdate_grads = now
    parameters_grads = ['wind_speed_u_100m:ms', 'wind_speed_v_100m:ms']  # ['t_500hPa:C','gh_500hPa:m']
    model_grads = 'ecmwf-ifs'
    area_grads = 'europe'
    
    ###input netcdf
    filename_nc = "path_netcdf/netcdf_target.nc"
    startdate_nc = now
    enddate_nc = startdate_nc+dt.timedelta(days=1)
    interval_nc = dt.timedelta(days=1)
    parameter_nc = 't_2m:C'
    
    ###input png timeseries
    prefixpath_png_ts = 'path/to/directory' #TODO
    startdate_png_ts = now
    enddate_png_ts = startdate_png_ts+dt.timedelta(days=2)
    interval_png_ts = dt.timedelta(hours=12)
    parameter_png_ts = 't_2m:C'
    
    ###input grads timeseries
    # prefixpath_grads_ts = 'path/to/directory' #TODO
    prefixpath_grads_ts = '' #TODO
    startdate_grads_ts = now
    enddate_grads_ts= startdate_grads_ts + dt.timedelta(days=2)
    interval_grads_ts= dt.timedelta(hours=24)
    parameters_grads_ts = ['t_500hPa:C','gh_500hPa:m']
    model_grads_ts = 'ecmwf-ifs'
    area_grads_ts = 'australia' #For Lat/Lon setting: None
 
    print("\ntime series:")
    try:
        df_ts = api.query_time_series(coordinates_ts, startdate_ts, enddate_ts, interval_ts, parameters_ts, username, password, model, ens_select, interp_select)
        print (df_ts.head())
    except Exception as e:
        print("Failed, the exception is {}".format(e))
    
    print("\ngrid:")
    try:
        df_grid = api.query_grid(startdate_grid, parameter_grid, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon, username, password)
        print (df_grid.head())
    except Exception as e:
        print("Failed, the exception is {}".format(e))
    
    print("\nunpivoted grid:")
    try:
        df_grid_unpivoted = api.query_grid_unpivoted(valid_dates_unpiv, parameters_grid_unpiv, lat_N, lon_W, lat_S, lon_E, res_lat, username, password, res_lon)
        print (df_grid_unpivoted.head())
    except Exception as e:
        print("Failed, the exception is {}".format(e))
     
    
    print("\nlighning strokes as csv:")
    try:
        df_lightning = api.query_lightnings(startdate_l, enddate_l,lat_N_l, lon_W_l, lat_S_l, lon_E_l,  username, password)
        print(df_lightning.head())
    except Exception as e:
        print("Failed, the exception is {}".format(e))
    
    print("\ngrid as a png:")
    try:
        api.query_grid_png(filename_png, startdate_png, parameter_png, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon, username, password)
        print("filename = {}".format(filename_png))
    except Exception as e:
        print("Failed, the exception is {}".format(e))
    
    print("\nnetCDF file:")
    try:
        api.query_netcdf(filename_nc, startdate_nc, enddate_nc, interval_nc, parameter_nc, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon, username, password)
        print("filename = {}".format(filename_nc))
    except Exception as e:
        print("Failed, the exception is {}".format(e))
    
    print("\nGrads plot:")
    try:
        api.query_grads(filename_grads, startdate_grads, parameters_grads, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon, username, password, model_grads, area=area_grads)
        print("filename = {}".format(filename_grads))
    except Exception as e:
        print("Failed, the exception is {}".format(e))
    
    print("\npng timeseries:")
    try:
        api.query_png_timeseries(prefixpath_png_ts, startdate_png_ts, enddate_png_ts, interval_png_ts, parameter_png_ts, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon, username, password)
        print("filename = {}".format(png_ts))
    except Exception as e:
        print("Failed, the exception is {}".format(e))
    
    print("\ngrads timeseries:")
    try:
        api.query_grads_timeseries(prefixpath_grads_ts, startdate_grads_ts, enddate_grads_ts, interval_grads_ts,parameters_grads_ts, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon, username, password, model=model_grads_ts, area=area_grads_ts)
        print("prefix = {}".format(prefixpath_grads_ts))
    except Exception as e:
        print("Failed, the exception is {}".format(e))
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', default=username)
    parser.add_argument('--password', default=password)
    arguments = parser.parse_args()
    
    username = arguments.username
    password = arguments.password

    if username is None or password is None:
        print("You need to provide a username and a password, either on the command line or by inserting them in the script")
        sys.exit()

    example()
