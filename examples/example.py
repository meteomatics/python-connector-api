#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding:=utf-8

import argparse
import datetime as dt
import sys

import meteomatics.api as api
from meteomatics.logger import create_log_handler

'''
    For further information on available parameters, models etc. please visit
    api.meteomatics.com

    In case of questions just write a mail to:
    support@meteomatics.com
'''

###Credentials:

username = 'python-community'
password = 'Umivipawe179'


def example():
    ###Input timeseries:
    now = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    startdate_ts = now
    enddate_ts = startdate_ts + dt.timedelta(days=1)
    interval_ts = dt.timedelta(hours=1)
    coordinates_ts = [(47.249297, 9.342854), (50., 10.)]
    parameters_ts = ['t_2m:C', 'rr_1h:mm']
    model = 'mix'
    ens_select = None  # e.g. 'median'
    cluster_select = None  # e.g. "cluster:1", see http://api.meteomatics.com/API-Request.html#cluster-selection
    interp_select = 'gradient_interpolation'

    ###Input grid / grid unpivoted:
    lat_N = 50
    lon_W = -15
    lat_S = 20
    lon_E = 10
    res_lat = 3
    res_lon = 3
    startdate_grid = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    parameter_grid = 'evapotranspiration_1h:mm'  # 't_2m:C'

    parameters_grid_unpiv = ['t_2m:C', 'rr_1h:mm']
    valid_dates_unpiv = [dt.datetime.utcnow(), dt.datetime.utcnow() + dt.timedelta(days=1)]

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
    enddate_nc = startdate_nc + dt.timedelta(days=1)
    interval_nc = dt.timedelta(days=1)
    parameter_nc = 't_2m:C'

    ###input png timeseries
    # prefixpath_png_ts = 'path/to/directory' #TODO
    prefixpath_png_ts = ''  # TODO
    startdate_png_ts = now
    enddate_png_ts = startdate_png_ts + dt.timedelta(days=2)
    interval_png_ts = dt.timedelta(hours=12)
    parameter_png_ts = 't_2m:C'

    ###input grads timeseries
    # prefixpath_grads_ts = 'path/to/directory' #TODO
    prefixpath_grads_ts = ''  # TODO
    startdate_grads_ts = now
    enddate_grads_ts = startdate_grads_ts + dt.timedelta(days=2)
    interval_grads_ts = dt.timedelta(hours=24)
    parameters_grads_ts = ['t_500hPa:C', 'gh_500hPa:m']
    model_grads_ts = 'ecmwf-ifs'
    area_grads_ts = 'australia'  # For Lat/Lon setting: None

    ###input station data timeseries
    startdate_station_ts = startdate_grads_ts - dt.timedelta(days=2)
    enddate_station_ts = startdate_grads_ts - dt.timedelta(hours=3)
    interval_station_ts = dt.timedelta(hours=1)
    parameters_station_ts = ['t_2m:C', 'wind_speed_10m:ms', 'precip_1h:mm']
    model_station_ts = 'mix-obs'
    coordinates_station_ts = [(47.43, 9.4), (50.03, 8.52)]  # St. Gallen / Frankfurt/Main
    wmo_stations = ['066810']  # St. Gallen
    metar_stations = ['EDDF']  # Frankfurt/Main
    mch_stations = ['STG']  # MeteoSchweiz Station St. Gallen

    limits = api.query_user_features(username, password)

    print("\ntime series:")
    try:
        df_ts = api.query_time_series(coordinates_ts, startdate_ts, enddate_ts, interval_ts, parameters_ts,
                                      username, password, model, ens_select, interp_select,
                                      cluster_select=cluster_select)
        print (df_ts.head())
    except Exception as e:
        print("Failed, the exception is {}".format(e))

    print("\npng timeseries:")
    try:
        api.query_png_timeseries(prefixpath_png_ts, startdate_png_ts, enddate_png_ts, interval_png_ts, parameter_png_ts,
                                 lat_N, lon_W, lat_S, lon_E, res_lat, res_lon, username, password)
    except Exception as e:
        print("Failed, the exception is {}".format(e))

    if limits['area request option']:
        print("\ngrid:")
        try:
            df_grid = api.query_grid(startdate_grid, parameter_grid, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon,
                                     username, password)
            print (df_grid.head())
        except Exception as e:
            print("Failed, the exception is {}".format(e))

        print("\nunpivoted grid:")
        try:
            df_grid_unpivoted = api.query_grid_unpivoted(valid_dates_unpiv, parameters_grid_unpiv, lat_N, lon_W, lat_S,
                                                         lon_E, res_lat, res_lon, username, password)
            print (df_grid_unpivoted.head())
        except Exception as e:
            print("Failed, the exception is {}".format(e))

        print("\ngrid timeseries:")
        try:
            df_grid_timeseries = api.query_grid_timeseries(startdate_ts, enddate_ts, interval_ts, parameters_ts, lat_N,
                                                           lon_W, lat_S, lon_E, res_lat, res_lon, username, password)
            print (df_grid_timeseries.head())
        except Exception as e:
            print("Failed, the exception is {}".format(e))

        print("\ngrid as a png:")
        try:
            api.query_grid_png(filename_png, startdate_png, parameter_png, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon,
                               username, password)
            print("filename = {}".format(filename_png))
        except Exception as e:
            print("Failed, the exception is {}".format(e))

    else:
        print("""
Your account '{}' does not include area requests.
With the corresponding upgrade you could query whole grids of data at once or even time series of grids.
Please check http://shop.meteomatics.com or contact us at shop@meteomatics.com for an individual offer.
""".format(username)
              )

    if limits['historic request option'] and limits['area request option']:
        print("\nlighning strokes as csv:")
        try:
            df_lightning = api.query_lightnings(startdate_l, enddate_l, lat_N_l, lon_W_l, lat_S_l, lon_E_l, username,
                                                password)
            print(df_lightning.head())
        except Exception as e:
            print("Failed, the exception is {}".format(e))
    else:
        print("""
Your account '{}' does not include historic requests.
With the corresponding upgrade you could query data from the past as well as forecasts.
Please check http://shop.meteomatics.com or contact us at shop@meteomatics.com for an individual offer.
""".format(username)
              )
    if limits['model select option']:
        print("\nnetCDF file:")
        try:
            api.query_netcdf(filename_nc, startdate_nc, enddate_nc, interval_nc, parameter_nc, lat_N, lon_W, lat_S,
                             lon_E,
                             res_lat, res_lon, username, password)
            print("filename = {}".format(filename_nc))
        except Exception as e:
            print("Failed, the exception is {}".format(e))

        print("\nGrads plot:")
        try:
            api.query_grads(filename_grads, startdate_grads, parameters_grads, lat_N, lon_W, lat_S, lon_E, res_lat,
                            res_lon,
                            username, password, model_grads, area=area_grads)
            print("filename = {}".format(filename_grads))
        except Exception as e:
            print("Failed, the exception is {}".format(e))

        print("\ngrads timeseries:")
        try:
            api.query_grads_timeseries(prefixpath_grads_ts, startdate_grads_ts, enddate_grads_ts, interval_grads_ts,
                                       parameters_grads_ts, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon, username,
                                       password, model=model_grads_ts, area=area_grads_ts)
            print("prefix = {}".format(prefixpath_grads_ts))
        except Exception as e:
            print("Failed, the exception is {}".format(e))

        print("\nfind stations:")
        try:
            met = api.query_station_list(username, password, startdate=startdate_station_ts, enddate=enddate_station_ts,
                                         parameters=parameters_station_ts)
            print(met.head())
        except Exception as e:
            print("Failed, the exception is {}".format(e))

        print("\nstation coordinates timeseries:")
        try:
            df_sd_coord = api.query_station_timeseries(startdate_station_ts, enddate_station_ts, interval_station_ts,
                                                       parameters_station_ts, username, password,
                                                       model=model_station_ts,
                                                       latlon_tuple_list=coordinates_station_ts,
                                                       on_invalid='fill_with_invalid', request_type="POST",
                                                       temporal_interpolation='none')
            print(df_sd_coord.head())
        except Exception as e:
            print("Failed, the exception is {}".format(e))

        print("\nstation wmo + metar ids timeseries:")
        try:
            df_sd_ids = api.query_station_timeseries(startdate_station_ts, enddate_station_ts, interval_station_ts,
                                                     parameters_station_ts, username, password, model=model_station_ts,
                                                     wmo_ids=wmo_stations, metar_ids=metar_stations,
                                                     mch_ids=mch_stations, on_invalid='fill_with_invalid',
                                                     request_type="POST", temporal_interpolation='none')
            print(df_sd_ids.head())
        except Exception as e:
            print("Failed, the exception is {}".format(e))

        print("\nget init dates:")
        try:
            df_init_dates = api.query_init_date(now, now + dt.timedelta(days=2), dt.timedelta(hours=3), 't_2m:C',
                                                username,
                                                password, 'ecmwf-ens')
            print(df_init_dates.head())
        except Exception as e:
            print("Failed, the exception is {}".format(e))

        print("\nget available time ranges:")
        try:
            df_time_ranges = api.query_available_time_ranges(['t_2m:C', 'precip_6h:mm'], username, password,
                                                             'ukmo-euro4')
            print(df_time_ranges.head())
        except Exception as e:
            print("Failed, the exception is {}".format(e))

    else:
        print("""
Your account '{}' does not include model selection.
With the corresponding upgrade you could query data from stations and request your data in netcdf or grads format.
Please check http://shop.meteomatics.com or contact us at shop@meteomatics.com for an individual offer. 
""".format(username)
              )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', default=username)
    parser.add_argument('--password', default=password)
    arguments = parser.parse_args()

    username = arguments.username
    password = arguments.password

    if username is None or password is None:
        print(
        "You need to provide a username and a password, either on the command line or by inserting them in the script")
        sys.exit()

    create_log_handler()
    example()
