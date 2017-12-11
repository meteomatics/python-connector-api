# !/usr/bin/env python
# -*- coding: utf-8 -*-
# coding:=utf-8

"""Meteomatics Weather API Connector"""

import isodate
import requests
import pandas as pd
import datetime as dt
import pytz
import sys
import os
from io import StringIO

__author__ = 'Jonas Lauer (jlauer@meteomatics.com)'
__copyright__ = 'Copyright (c) 2017 Meteomatics'
__license__ = 'Meteomatics Internal License'
__version__ = '1.4'

logdepth = 0


def log(lvl, msg, depth=-1):
    global logdepth
    if depth == -1:
        depth = logdepth
    else:
        logdepth = depth

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = "   " * depth
    print (now, "|", lvl, "|", prefix + msg)
    sys.stdout.flush()


def log_info(msg, depth=-1):
    log("INFO ", msg, depth)


def CreatePath(_file):
    _path = os.path.dirname(_file)
    if (os.path.exists(_path) == False) & (len(_path) > 0):
        log_info("Create Path: {}".format(_path))
        os.makedirs(_path)


DEFAULT_API_BASE_URL = "https://api.meteomatics.com"
VERSION = 'python_1.4'

# Templates
TIME_SERIES_TEMPLATE = "{api_base_url}/{startdate}--{enddate}:{interval}/{parameters}/{coordinates}/csv?{urlParams}"
GRID_TEMPLATE = "{api_base_url}/{startdate}/{parameter_grid}/{lat_N},{lon_W}_{lat_S},{lon_E}:{res_lat},{res_lon}/csv?{urlParams}"
GRID_PNG_TEMPLATE = "{api_base_url}/{startdate}/{parameter_grid}/{lat_N},{lon_W}_{lat_S},{lon_E}:{res_lat},{res_lon}/png?{urlParams}"
LIGHTNING_TEMPLATE = "{api_base_url}/get_lightning_list?time_range={startdate}--{enddate}&bounding_box={lat_N},{lon_W}_{lat_S},{lon_E}&format=csv"
GRADS_TEMPLATE = "{api_base_url}/{startdate}/{parameters}/{area}/grads?model={model}&{urlParams}"
NETCDF_TEMPLATE = "{api_base_url}/{startdate}--{enddate}:{interval}/{parameter_netcdf}/{lat_N},{lon_W}_{lat_S},{lon_E}:{res_lat},{res_lon}/netcdf?{urlParams}"
STATIONS_LIST_TEMPLATE = "{api_base_url}/find_station?{urlParams}"


class WeatherApiException(Exception):
    def __init__(self, message):
        super(WeatherApiException, self).__init__(message)


def convert_time_series_response_to_df(input, latlon_tuple_list, station=False):
    try:
        is_str = isinstance(input, basestring)  # python 2
    except NameError:
        is_str = isinstance(input, str)  # python 3
    finally:
        if is_str:
            input = StringIO(input)

    # parse response
    try:
        df = pd.read_csv(
            input,
            sep=";",
            header=0,
            encoding="utf-8",
            parse_dates=['validdate'],
            index_col='validdate',
            na_values=["-999"]
        )

        # mark index as UTC timezone
        df.index = df.index.tz_localize("UTC")

    except:
        raise WeatherApiException(input.getvalue())

    if not station:
        parameters = [c for c in df.columns if c not in ['lat', 'lon']]
        # extract coordinates
        if 'lat' not in df.columns:
            df['lat'] = latlon_tuple_list[0][0]
            df['lon'] = latlon_tuple_list[0][1]
        # set multiindex
        df = df.reset_index().set_index(['lat', 'lon', 'validdate'])
    else:
        parameters = [c for c in df.columns if c not in ['station_id']]
        # extract coordinates
        if 'station_id' not in df.columns:
            df['station_id'] = latlon_tuple_list
        # set multiindex
        df = df.reset_index().set_index(['station_id', 'validdate'])

    return df[parameters]


def query_station_list(username, password, source=None, parameters=None, enddate=None, location=None,
                       api_base_url=DEFAULT_API_BASE_URL, request_type='GET'):
    '''Function to query available stations in API
    source as string
    parameters as list
    enddate as datetime object
    location as string (e.g. "40,10")
    request_type is one of 'GET'/'POST'
    '''
    urlParams = {}
    if source is not None:
        urlParams['source'] = source

    if parameters is not None:
        urlParams['parameter'] = ",".join(parameters)

    if enddate is not None:
        urlParams['enddate'] = dt.datetime.strftime(enddate, "%Y-%m-%dT%HZ")

    if location is not None:
        urlParams['location'] = location

    url = STATIONS_LIST_TEMPLATE.format(
        api_base_url=api_base_url,
        urlParams="&".join(["{}={}".format(k, v) for k, v in urlParams.items()])
    )

    log_info("Calling URL: {} (username = {})".format(url, username))
    # determine request type
    if request_type.lower() == 'get':
        request = requests.get
    elif request_type.lower() == 'post':
        request = requests.post
    else:
        raise ValueError('Unknown request_type: {}.'.format(request_type))

    # fire request
    try:
        response = request(
            url,
            auth=requests.auth.HTTPBasicAuth(username, password),
            headers={'Accept': 'text/csv'}
        )

        if response.status_code != requests.codes.ok:
            raise WeatherApiException(response.text)

    except requests.ConnectionError as e:
        raise WeatherApiException(e)

    sl = pd.read_csv(StringIO(response.text), sep=";")
    sl['lat'] = sl['Location Lat,Lon'].apply(lambda x: float(x.split(",")[0]))
    sl['lon'] = sl['Location Lat,Lon'].apply(lambda x: float(x.split(",")[1]))
    sl.drop('Location Lat,Lon', 1, inplace=True)

    return sl


def query_station_timeseries(startdate, enddate, interval, parameters, username, password, model='station_mix',
                             latlon_tuple_list=None, wmo_ids=None,
                             metar_ids=None, temporal_interpolation=None, spatial_interpolation=None, on_invalid=None,
                             api_base_url=DEFAULT_API_BASE_URL, request_type='GET'):
    """Retrieve a time series from the Meteomatics Weather API.
    Requested can be by WMO ID, Metar ID or coordinates.
    Start and End dates have to be in UTC.
    Returns a Pandas `DataFrame` with a `DateTimeIndex`.
    request_type is one of 'GET/POST'
    """

    # set time zone info to UTC if necessary
    if startdate.tzinfo is None:
        startdate = startdate.replace(tzinfo=pytz.UTC)
    if enddate.tzinfo is None:
        enddate = enddate.replace(tzinfo=pytz.UTC)

    # build URL

    coordinates = ""
    urlParams = {}
    urlParams['connector'] = VERSION
    if latlon_tuple_list is not None:
        coordinates += "+" + ("+".join(["{},{}".format(*latlon_tuple) for latlon_tuple in latlon_tuple_list]))

    if wmo_ids is not None:
        if len(coordinates) > 0:
            coordinates += "+" + "+".join(['wmo_' + s for s in wmo_ids])
        else:
            coordinates += "+".join(['wmo_' + s for s in wmo_ids])
    if metar_ids is not None:
        if len(coordinates) > 0:
            coordinates += "+" + "+".join(['metar_' + s for s in metar_ids])
        else:
            coordinates += "+".join(['metar_' + s for s in metar_ids])

    if model is not None:
        urlParams['model'] = model

    if on_invalid is not None:
        urlParams['on_invalid'] = on_invalid

    if temporal_interpolation is not None:
        urlParams['temporal_interpolation'] = temporal_interpolation

    if spatial_interpolation is not None:
        urlParams['spatial_interpolation'] = spatial_interpolation

    url = TIME_SERIES_TEMPLATE.format(
        api_base_url=api_base_url,
        coordinates=coordinates,
        startdate=startdate.isoformat(),
        enddate=enddate.isoformat(),
        interval=isodate.duration_isoformat(interval),
        parameters=",".join(parameters),
        urlParams="&".join(["{}={}".format(k, v) for k, v in urlParams.items()])
    )

    log_info("Calling URL: {} (username = {})".format(url, username))
    # determine request type
    if request_type.lower() == 'get':
        request = requests.get
    elif request_type.lower() == 'post':
        request = requests.post
    else:
        raise ValueError('Unknown request_type: {}.'.format(request_type))

    # fire request
    try:
        response = request(
            url,
            auth=requests.auth.HTTPBasicAuth(username, password),
            headers={'Accept': 'text/csv'}
        )

        if response.status_code != requests.codes.ok:
            raise WeatherApiException(response.text)

    except requests.ConnectionError as e:
        raise WeatherApiException(e)

    return convert_time_series_response_to_df(StringIO(response.text), coordinates, station=True)


def query_time_series(latlon_tuple_list, startdate, enddate, interval, parameters, username, password, model=None,
                      ens_select=None, interp_select=None, api_base_url=DEFAULT_API_BASE_URL, request_type='GET'):
    """Retrieve a time series from the Meteomatics Weather API.
    Start and End dates have to be in UTC.
    Returns a Pandas `DataFrame` with a `DateTimeIndex`.
    request_type is one of 'GET'/'POST'
    """

    # set time zone info to UTC if necessary
    if startdate.tzinfo is None:
        startdate = startdate.replace(tzinfo=pytz.UTC)
    if enddate.tzinfo is None:
        enddate = enddate.replace(tzinfo=pytz.UTC)

    # build URL

    urlParams = {}
    urlParams['connector'] = VERSION
    if model is not None:
        urlParams['model'] = model

    if ens_select is not None:
        urlParams['ens_select'] = ens_select

    if interp_select is not None:
        urlParams['interp_select'] = interp_select

    url = TIME_SERIES_TEMPLATE.format(
        api_base_url=api_base_url,
        coordinates="+".join(["{},{}".format(*latlon_tuple) for latlon_tuple in latlon_tuple_list]),
        startdate=startdate.isoformat(),
        enddate=enddate.isoformat(),
        interval=isodate.duration_isoformat(interval),
        parameters=",".join(parameters),
        urlParams="&".join(["{}={}".format(k, v) for k, v in urlParams.items()])
    )

    log_info("Calling URL: {} (username = {})".format(url, username))
    # determine request type
    if request_type.lower() == 'get':
        request = requests.get
    elif request_type.lower() == 'post':
        request = requests.post
    else:
        raise ValueError('Unknown request_type: {}.'.format(request_type))

    # fire request
    try:
        response = request(
            url,
            auth=requests.auth.HTTPBasicAuth(username, password),
            headers={'Accept': 'text/csv'}
        )

        if response.status_code != requests.codes.ok:
            raise WeatherApiException(response.text)

    except requests.ConnectionError as e:
        raise WeatherApiException(e)

    return convert_time_series_response_to_df(StringIO(response.text), latlon_tuple_list)


def convert_grid_response_to_df(input):
    try:
        is_str = isinstance(input, basestring)  # python 2
    except NameError:
        is_str = isinstance(input, str)  # python 3
    finally:
        if is_str:
            input = StringIO(input)

    # parse response
    try:
        df = pd.read_csv(
            input,
            sep=";",
            skiprows=[0, 1],
            header=0,
            encoding="utf-8",
            index_col=0,
            na_values=["-999"]
        )

        df.index.name = 'lat'
        df.columns.name = 'lon'

    except:
        raise WeatherApiException(input.getvalue())

    parameter_grid = [c for c in df.columns if c not in ['lat', 'lon']]

    # omit name and id column
    return df[parameter_grid]


def query_grid(startdate, parameter_grid, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon, username, password, model=None,
               ens_select=None, interp_select=None, api_base_url=DEFAULT_API_BASE_URL, request_type='GET'):
    # interpret time as UTC
    if startdate.tzinfo is None:
        startdate = startdate.replace(tzinfo=pytz.UTC)

    # build URL

    urlParams = {}
    urlParams['connector'] = VERSION
    if model is not None:
        urlParams['model'] = model

    if ens_select is not None:
        urlParams['ens_select'] = ens_select

    if interp_select is not None:
        urlParams['interp_select'] = interp_select

    url = GRID_TEMPLATE.format(
        api_base_url=api_base_url,
        startdate=startdate.isoformat(),
        parameter_grid=parameter_grid,
        lat_N=lat_N,
        lon_W=lon_W,
        lat_S=lat_S,
        lon_E=lon_E,
        res_lat=res_lat,
        res_lon=res_lon,
        urlParams="&".join(["{}={}".format(k, v) for k, v in urlParams.items()])
    )

    log_info("Calling URL: {} (username = {})".format(url, username))
    # determine request type
    if request_type.lower() == 'get':
        request = requests.get
    elif request_type.lower() == 'post':
        request = requests.post
    else:
        raise ValueError('Unknown request_type: {}.'.format(request_type))

    # fire request
    try:
        response = request(url,
                           auth=requests.auth.HTTPBasicAuth(username, password),
                           headers={'Accept': 'text/csv'}
                           )

        if response.status_code != requests.codes.ok:
            raise WeatherApiException(response.text)

    except requests.ConnectionError as e:
        raise WeatherApiException(e)

    return convert_grid_response_to_df(response.text)


def query_grid_unpivoted(valid_dates, parameters, lat_N, lon_W, lat_S, lon_E, res_lat, username, password, res_lon,
                         model=None, ens_select=None, interp_select=None, request_type='GET'):
    idxcols = ['valid_date', 'lat', 'lon']
    vd_dfs = []

    for valid_date in valid_dates:
        vd_df = None
        for parameter in parameters:

            dmo = query_grid(valid_date, parameter, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon, username, password,
                             model, ens_select, interp_select, request_type=request_type)

            df = pd.melt(dmo.reset_index(), id_vars='lat', var_name='lon', value_name=parameter)
            df['valid_date'] = valid_date
            df.lat = df.lat.apply(float)
            df.lon = df.lon.apply(float)

            if vd_df is None:
                vd_df = df
            else:
                vd_df = vd_df.merge(df, on=idxcols)

        vd_dfs.append(vd_df)

    data = pd.concat(vd_dfs)

    # sort_values might not available in older pandas versions
    try:
        data.sort_values(idxcols, inplace=True)
    except AttributeError as e:
        data.sort(idxcols, inplace=True)

    data.set_index(idxcols, inplace=True)

    return data


def convert_lightning_response_to_df(input):
    """converts the response of the query of query_lightnings to a pandas DataFrame."""

    try:
        is_str = isinstance(input, basestring)  # python 2
    except NameError:
        is_str = isinstance(input, str)  # python 3
    finally:
        if is_str:
            input = StringIO(input)

        # parse response
        try:
            df = pd.read_csv(
                input,
                sep=";",
                header=0,
                encoding="utf-8",
                parse_dates=['stroke_time:sql'],
                index_col='stroke_time:sql'
            )

            # mark index as UTC timezone
            df.index = df.index.tz_localize("UTC")

        except:
            raise WeatherApiException(input.getvalue())

        # rename columns to make consistent with other csv file headers
        df = df.reset_index().rename(
            columns={'stroke_time:sql': 'validdate', 'stroke_lat:d': 'lat', 'stroke_lon:d': 'lon'})
        df.set_index(['validdate', 'lat', 'lon'], inplace=True)

        return df


def query_lightnings(startdate, enddate, lat_N, lon_W, lat_S, lon_E, username, password,
                     api_base_url=DEFAULT_API_BASE_URL, request_type='GET'):
    """Queries lightning strokes in the specified area during the specified time via the Meteomatics API.
    Returns a Pandas 'DataFrame'.
    request_type is one of 'GET'/'POST'
    """
    # interpret time as UTC
    if startdate.tzinfo is None:
        startdate = startdate.replace(tzinfo=pytz.UTC)

    if enddate.tzinfo is None:
        enddate = enddate.replace(tzinfo=pytz.UTC)

    # build URL
    url = LIGHTNING_TEMPLATE.format(
        api_base_url=api_base_url,
        startdate=startdate.isoformat(),
        enddate=enddate.isoformat(),
        lat_N=lat_N,
        lon_W=lon_W,
        lat_S=lat_S,
        lon_E=lon_E
    )

    log_info("Calling URL: {} (username = {})".format(url, username))
    # determine request type
    if request_type.lower() == 'get':
        request = requests.get
    elif request_type.lower() == 'post':
        request = requests.post
    else:
        raise ValueError('Unknown request_type: {}.'.format(request_type))

    # fire request
    try:
        response = request(
            url,
            auth=requests.auth.HTTPBasicAuth(username, password),
            headers={'Accept': 'text/csv'}
        )

        if response.status_code != requests.codes.ok:
            raise WeatherApiException(response.text)

    except requests.ConnectionError as e:
        raise WeatherApiException(e)

    return convert_lightning_response_to_df(response.text)


def query_netcdf(filename, startdate, enddate, interval, parameter_netcdf, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon,
                 username, password, model=None, ens_select=None, interp_select=None,
                 api_base_url=DEFAULT_API_BASE_URL, request_type='GET'):
    """Queries a netCDF file form the Meteomatics API and stores it in filename.
    request_type is one of 'GET'/'POST'
    """

    # set time zone info to UTC if necessary
    if startdate.tzinfo is None:
        startdate = startdate.replace(tzinfo=pytz.UTC)
    if enddate.tzinfo is None:
        enddate = enddate.replace(tzinfo=pytz.UTC)

    # build URL

    urlParams = {}
    urlParams['connector'] = VERSION
    if model is not None:
        urlParams['model'] = model

    if ens_select is not None:
        urlParams['ens_select'] = ens_select

    if interp_select is not None:
        urlParams['interp_select'] = interp_select

    url = NETCDF_TEMPLATE.format(
        api_base_url=api_base_url,
        startdate=startdate.isoformat(),
        enddate=enddate.isoformat(),
        interval=isodate.duration_isoformat(interval),
        parameter_netcdf=parameter_netcdf,
        lat_N=lat_N,
        lon_W=lon_W,
        lat_S=lat_S,
        lon_E=lon_E,
        res_lat=res_lat,
        res_lon=res_lon,
        urlParams="&".join(["{}={}".format(k, v) for k, v in urlParams.items()])
    )

    log_info("Calling URL: {} (username = {})".format(url, username))
    # determine request type
    if request_type.lower() == 'get':
        request = requests.get
    elif request_type.lower() == 'post':
        request = requests.post
    else:
        raise ValueError('Unknown request_type: {}.'.format(request_type))

    # fire request
    try:
        response = request(url,
                           auth=requests.auth.HTTPBasicAuth(username, password),
                           headers={'Accept': 'text/netcdf'}, stream=True,
                           )

        if response.status_code != requests.codes.ok:
            raise WeatherApiException(response.text)

    except requests.ConnectionError as e:
        raise WeatherApiException(e)

    # Check if target directory exists
    CreatePath(filename)

    # save to the specified filename
    with open(filename, 'wb') as f:
        log_info('Create File {}'.format(filename))
        for chunk in response.iter_content():
            f.write(chunk)

    return


def query_grid_png(filename, startdate, parameter_grid, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon, username,
                   password, model=None, ens_select=None, interp_select=None, api_base_url=DEFAULT_API_BASE_URL,
                   request_type='GET'):
    """Gets a png image generated by the Meteomatics API from grid data (see method query_grid) and saves it to the specified filename.
    request_type is one of 'GET'/'POST'
    """

    # interpret time as UTC
    if startdate.tzinfo is None:
        startdate = startdate.replace(tzinfo=pytz.UTC)

    # build URL

    urlParams = {}
    urlParams['connector'] = VERSION
    if model is not None:
        urlParams['model'] = model

    if ens_select is not None:
        urlParams['ens_select'] = ens_select

    if interp_select is not None:
        urlParams['interp_select'] = interp_select

    url = GRID_PNG_TEMPLATE.format(
        api_base_url=api_base_url,
        startdate=startdate.isoformat(),
        parameter_grid=parameter_grid,
        lat_N=lat_N,
        lon_W=lon_W,
        lat_S=lat_S,
        lon_E=lon_E,
        res_lat=res_lat,
        res_lon=res_lon,
        urlParams="&".join(["{}={}".format(k, v) for k, v in urlParams.items()])
    )

    log_info("Calling URL: {} (username = {})".format(url, username))
    # determine request type
    if request_type.lower() == 'get':
        request = requests.get
    elif request_type.lower() == 'post':
        request = requests.post
    else:
        raise ValueError('Unknown request_type: {}.'.format(request_type))

    # fire request
    try:
        response = request(url,
                           auth=requests.auth.HTTPBasicAuth(username, password),
                           headers={'Accept': 'image/png'}, stream=True,
                           )

        if response.status_code != requests.codes.ok:
            raise WeatherApiException(response.text)

    except requests.ConnectionError as e:
        raise WeatherApiException(e)

    # save to the specified filename
    with open(filename, 'wb') as f:
        log_info('Create File {}'.format(filename))
        for chunk in response.iter_content():
            f.write(chunk)

    return


def query_grads(filename, startdate, parameters, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon, username,
                password, model, ens_select=None, interp_select=None, api_base_url=DEFAULT_API_BASE_URL, area=None,
                request_type='GET'):
    """Queries grads plots from the Meteomatics API and saves it to the specified filename.
    Aside from the regular grid specifiers (lat, lon, ...) a predefined area can be specified (ex: 'north-america',
     'europe', 'australia', 'asia', 'africa'). In this case the latlon specifiers are ignored and the grads plot
     for this area is retrieved.
    Grads plots can be created for the following parameters:
         - temperature and geopotential height (e.g. parameters = ['t_500hPa:C','gh_500hPa:m'])
         - temperature at 2m above ground level (parameters = ['t_2m:C'])
         - precipitation (e.g. parameters = ['precip_3h:mm'])
         - cloud cover (e.g. parameters = ['low_cloud_cover:p'])
         - wind speed and direction (e.g. parameters = ['wind_speed_u_100m:ms','wind_speed_v_100m:ms'])
         - wind power (e.g. parameters = ['wind_power_turbine_aaer_a1000_1000_hub_height_110m:MW'])
         - significant wave height and mean sea level pressure (parameters = ['significant_wave_height:m','msl_pressure:hPa'], requires model='ecmwf-wam')
         - mean wave period and mean sea level pressure (parameters = ['mean_wave_period:s','msl_pressure:hPa'] , requires model='ecmwf-wam')
    request_type is one of 'GET'/'POST'
    """

    # interpret time as UTC
    if startdate.tzinfo is None:
        startdate = startdate.replace(tzinfo=pytz.UTC)

    # build URL
    urlParams = {}
    urlParams['connector'] = VERSION
    if ens_select is not None:
        urlParams['ens_select'] = ens_select

    if interp_select is not None:
        urlParams['interp_select'] = interp_select

    # construct the area from latlon specifiers if area is not one of the predefined ones.
    if area is None:
        area_template = "{lat_N},{lon_W}_{lat_S},{lon_E}:{res_lat},{res_lon}"
        area = area_template.format(lat_N=lat_N, lon_W=lon_W, lat_S=lat_S, lon_E=lon_E, res_lat=res_lat,
                                    res_lon=res_lon, )

    url = GRADS_TEMPLATE.format(
        api_base_url=api_base_url,
        startdate=startdate.isoformat(),
        parameters=",".join(parameters),
        area=area,
        model=model,
        urlParams="&".join(["{}={}".format(k, v) for k, v in urlParams.items()])
    )

    log_info("Calling URL: {} (username = {})".format(url, username))
    # determine request type
    if request_type.lower() == 'get':
        request = requests.get
    elif request_type.lower() == 'post':
        request = requests.post
    else:
        raise ValueError('Unknown request_type: {}.'.format(request_type))

    # fire request
    try:
        response = request(url,
                           auth=requests.auth.HTTPBasicAuth(username, password),
                           headers={'Accept': 'image/grads'}, stream=True,
                           )

        if response.status_code != requests.codes.ok:
            raise WeatherApiException(response.text)

    except requests.ConnectionError as e:
        raise WeatherApiException(e)

    # save to the specified filename
    CreatePath(filename)
    with open(filename, 'wb') as f:
        log_info('Create File {}'.format(filename))
        for chunk in response.iter_content():
            f.write(chunk)

    return


def query_png_timeseries(prefixpath, startdate, enddate, interval, parameter, lat_N, lon_W, lat_S, lon_E, res_lat,
                         res_lon, username, password, model=None, ens_select=None, interp_select=None,
                         api_base_url=DEFAULT_API_BASE_URL, request_type='GET'):
    """Queries a series of png's for the requested time period and area from the Meteomatics API. The retrieved png's
    are saved to the directory prefixpath.
    request_type is one of 'GET'/'POST'
    """

    # iterate over all requested dates
    this_date = startdate
    while this_date <= enddate:
        # construct filename
        if len(prefixpath) > 0:
            filename = prefixpath + '/' + parameter.replace(':', '_') + '_' + this_date.strftime(
                '%Y%m%d_%H%M%S') + '.png'
        else:
            filename = parameter.replace(':', '_') + '_' + this_date.strftime(
                '%Y%m%d_%H%M%S') + '.png'

        # query base method
        query_grid_png(filename, this_date, parameter, lat_N, lon_W, lat_S, lon_E, res_lat,
                       res_lon, username, password, model, ens_select, interp_select,
                       api_base_url, request_type=request_type)

        this_date = this_date + interval

    return


def query_grads_timeseries(prefixpath, startdate, enddate, interval, parameters, lat_N, lon_W, lat_S, lon_E, res_lat,
                           res_lon, username, password, model=None, ens_select=None, interp_select=None,
                           api_base_url=DEFAULT_API_BASE_URL, area=None, request_type='GET'):
    """Queries several grad plots from the Meteomatics API and saves them in the directory prefixpath (filenames are
    generated automatically based upon parameter and time values).
    Aside from the regular grid specifiers (lat, lon, ...) a predefined area can be specified (ex: 'north-america',
    'europe', 'australia', 'asia', 'africa'). In this case the latlon specifiers are ignored and the grads plot
    for this area is retrieved.
    Grads plots can be created for the following parameters:
         - temperature and geopotential height (e.g. parameters = ['t_500hPa:C','gh_500hPa:m'])
         - temperature at 2m above ground level (parameters = ['t_2m:C'])
         - precipitation (e.g. parameters = ['precip_3h:mm'])
         - cloud cover (e.g. parameters = ['low_cloud_cover:p'])
         - wind speed and direction (e.g. parameters = ['wind_speed_u_100m:ms','wind_speed_v_100m:ms'])
         - wind power (e.g. parameters = ['wind_power_turbine_aaer_a1000_1000_hub_height_110m:MW'])
         - significant wave height and mean sea level pressure (parameters = ['significant_wave_height:m','msl_pressure:hPa'], requires model='ecmwf-wam')
         - mean wave period and mean sea level pressure (parameters = ['mean_wave_period:s','msl_pressure:hPa'] , requires model='ecmwf-wam')
    request_type is one of 'GET'/'POST'
    """

    # iterate over all requested dates
    this_date = startdate
    while this_date <= enddate:
        # construct filename
        if len(prefixpath) > 0:
            filename = prefixpath + '/' + '_'.join(parameters).replace(':', '_') + '_' + this_date.strftime(
                '%Y%m%d_%H%M%S') + '.png'
        else:
            filename = '_'.join(parameters).replace(':', '_') + '_' + this_date.strftime(
                '%Y%m%d_%H%M%S') + '.png'

        # query base method
        query_grads(filename, this_date, parameters, lat_N, lon_W, lat_S, lon_E, res_lat,
                    res_lon, username, password, model, ens_select, interp_select,
                    api_base_url, area, request_type=request_type)

        this_date = this_date + interval

    return
