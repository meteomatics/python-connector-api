#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding:=utf-8

import argparse
import datetime as dt
import logging
import sys
from examples.credentials import username as username_default, password as password_default

import meteomatics.api as api
from meteomatics.logger import create_log_handler
from meteomatics._constants_ import LOGGERNAME

def example():
    _logger = logging.getLogger(LOGGERNAME)

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
    startdate_l = dt.datetime.utcnow() - dt.timedelta(days=1)
    enddate_l = dt.datetime.utcnow() - dt.timedelta(minutes=5)
    lat_N_l = 90
    lon_W_l = -180
    lat_S_l = -90
    lon_E_l = 180

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

    ###input station data timeseries
    startdate_station_ts = now - dt.timedelta(days=2)
    enddate_station_ts = now - dt.timedelta(hours=3)
    interval_station_ts = dt.timedelta(hours=1)
    parameters_station_ts = ['t_2m:C', 'wind_speed_10m:ms', 'precip_1h:mm']
    model_station_ts = 'mix-obs'
    coordinates_station_ts = [(47.43, 9.4), (50.03, 8.52)]  # St. Gallen / Frankfurt/Main
    wmo_stations = ['066810']  # St. Gallen
    metar_stations = ['EDDF']  # Frankfurt/Main
    mch_stations = ['STG']  # MeteoSchweiz Station St. Gallen

    limits = api.query_user_features(username, password)

    _logger.info("\ntime series:")
    try:
        df_ts = api.query_time_series(coordinates_ts, startdate_ts, enddate_ts, interval_ts, parameters_ts,
                                      username, password, model, ens_select, interp_select,
                                      cluster_select=cluster_select)
        _logger.info("Dataframe head \n" + df_ts.head().to_string())
    except Exception as e:
        _logger.info("Failed, the exception is {}".format(e))

    _logger.info("\npng timeseries:")
    try:
        api.query_png_timeseries(prefixpath_png_ts, startdate_png_ts, enddate_png_ts, interval_png_ts, parameter_png_ts,
                                 lat_N, lon_W, lat_S, lon_E, res_lat, res_lon, username, password)
    except Exception as e:
        _logger.error("Failed, the exception is {}".format(e))

    if limits['area request option']:
        _logger.info("\ngrid:")
        try:
            df_grid = api.query_grid(startdate_grid, parameter_grid, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon,
                                     username, password)
            _logger.info ("Dataframe head \n" + df_grid.head().to_string())
        except Exception as e:
            _logger.error("Failed, the exception is {}".format(e))

        _logger.info("\nunpivoted grid:")
        try:
            df_grid_unpivoted = api.query_grid_unpivoted(valid_dates_unpiv, parameters_grid_unpiv, lat_N, lon_W, lat_S,
                                                         lon_E, res_lat, res_lon, username, password)
            _logger.info ("Dataframe head \n" + df_grid_unpivoted.head().to_string())
        except Exception as e:
            _logger.error("Failed, the exception is {}".format(e))

        _logger.info("\ngrid timeseries:")
        try:
            df_grid_timeseries = api.query_grid_timeseries(startdate_ts, enddate_ts, interval_ts, parameters_ts, lat_N,
                                                           lon_W, lat_S, lon_E, res_lat, res_lon, username, password)
            _logger.info ("Dataframe head \n" + df_grid_timeseries.head().to_string())
        except Exception as e:
            _logger.error("Failed, the exception is {}".format(e))

        _logger.info("\ngrid as a png:")
        try:
            api.query_grid_png(filename_png, startdate_png, parameter_png, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon,
                               username, password)
            _logger.info("filename = {}".format(filename_png))
        except Exception as e:
            _logger.error("Failed, the exception is {}".format(e))

    else:
        _logger.error("""
Your account '{}' does not include area requests.
With the corresponding upgrade you could query whole grids of data at once or even time series of grids.
Please check http://shop.meteomatics.com or contact us at shop@meteomatics.com for an individual offer.
""".format(username)
              )

    if limits['historic request option'] and limits['area request option']:
        _logger.info("\nlighning strokes as csv:")
        try:
            df_lightning = api.query_lightnings(startdate_l, enddate_l, lat_N_l, lon_W_l, lat_S_l, lon_E_l, username,
                                                password)
            _logger.info("Dataframe head \n" + df_lightning.head().to_string())
        except Exception as e:
            _logger.error("Failed, the exception is {}".format(e))
    else:
        _logger.error("""
Your account '{}' does not include historic requests.
With the corresponding upgrade you could query data from the past as well as forecasts.
Please check http://shop.meteomatics.com or contact us at shop@meteomatics.com for an individual offer.
""".format(username)
              )
    if limits['model select option']:
        _logger.info("\nnetCDF file:")
        try:
            api.query_netcdf(filename_nc, startdate_nc, enddate_nc, interval_nc, parameter_nc, lat_N, lon_W, lat_S,
                             lon_E,
                             res_lat, res_lon, username, password)
            _logger.info("filename = {}".format(filename_nc))
        except Exception as e:
            _logger.error("Failed, the exception is {}".format(e))

        _logger.info("\nfind stations:")
        try:
            met = api.query_station_list(username, password, startdate=startdate_station_ts, enddate=enddate_station_ts,
                                         parameters=parameters_station_ts)
            _logger.info("Dataframe head \n" +  met.head().to_string())
        except Exception as e:
            _logger.error("Failed, the exception is {}".format(e))

        _logger.info("\nstation coordinates timeseries:")
        try:
            df_sd_coord = api.query_station_timeseries(startdate_station_ts, enddate_station_ts, interval_station_ts,
                                                       parameters_station_ts, username, password,
                                                       model=model_station_ts,
                                                       latlon_tuple_list=coordinates_station_ts,
                                                       on_invalid='fill_with_invalid', request_type="POST",
                                                       temporal_interpolation='none')
            _logger.info("Dataframe head \n" + df_sd_coord.head().to_string())
        except Exception as e:
            _logger.error("Failed, the exception is {}".format(e))

        _logger.info("\nstation wmo + metar ids timeseries:")
        try:
            df_sd_ids = api.query_station_timeseries(startdate_station_ts, enddate_station_ts, interval_station_ts,
                                                     parameters_station_ts, username, password, model=model_station_ts,
                                                     wmo_ids=wmo_stations, metar_ids=metar_stations,
                                                     mch_ids=mch_stations, on_invalid='fill_with_invalid',
                                                     request_type="POST", temporal_interpolation='none')
            _logger.info("Dataframe head \n" + df_sd_ids.head().to_string())
        except Exception as e:
            _logger.error("Failed, the exception is {}".format(e))

        _logger.info("\nget init dates:")
        try:
            df_init_dates = api.query_init_date(now, now + dt.timedelta(days=2), dt.timedelta(hours=3), 't_2m:C',
                                                username,
                                                password, 'ecmwf-ens')
            _logger.info("Dataframe head \n" + df_init_dates.head().to_string())
        except Exception as e:
            _logger.error("Failed, the exception is {}".format(e))

        _logger.info("\nget available time ranges:")
        try:
            df_time_ranges = api.query_available_time_ranges(['t_2m:C', 'precip_6h:mm'], username, password,
                                                             'ukmo-euro4')
            _logger.info("Dataframe head \n" + df_time_ranges.head().to_string())
        except Exception as e:
            _logger.error("Failed, the exception is {}".format(e))

    else:
        _logger.error("""
Your account '{}' does not include model selection.
With the corresponding upgrade you could query data from stations and request your data in netcdf.
Please check http://shop.meteomatics.com or contact us at shop@meteomatics.com for an individual offer. 
""".format(username)
              )
    _logger.info("Checking the limits:")
    result = api.query_user_limits(username, password)
    _logger.info(result)


def run_example(example_lambda):
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', default=username_default)
    parser.add_argument('--password', default=password_default)
    arguments = parser.parse_args()

    username = arguments.username
    password = arguments.password

    create_log_handler()
    logging.getLogger(LOGGERNAME).setLevel(logging.INFO)
    _logger = logging.getLogger(LOGGERNAME)

    if username is None or password is None:
        _logger.info(
        "You need to provide a username and a password, either on the command line or by inserting them in the script")
        sys.exit()

    example_lambda(username, password, _logger)
