# -*- coding: utf-8 -*-
"""Meteomatics Weather API Connector

Visit https://www.meteomatics.com/en/api/overview/ for an overview of the API.
Checkout the examples!
If necessary, you can open an issue at https://github.com/meteomatics/python-connector-api or
write an email to support@meteomatics.com if you need further assistance.
"""
# Python 2 compatibility
from __future__ import print_function

import itertools
import logging
import os
import warnings
from functools import wraps
from io import StringIO

import isodate
import pandas as pd
import requests
from urllib3.exceptions import InsecureRequestWarning
from meteomatics.deprecated import deprecated

from ._constants_ import DEFAULT_API_BASE_URL, VERSION, TIME_SERIES_TEMPLATE, GRID_TEMPLATE, POLYGON_TEMPLATE, \
    GRID_TIME_SERIES_TEMPLATE, GRID_PNG_TEMPLATE, LIGHTNING_TEMPLATE, NETCDF_TEMPLATE, STATIONS_LIST_TEMPLATE, \
    INIT_DATE_TEMPLATE, AVAILABLE_TIME_RANGES_TEMPLATE, NA_VALUES, LOGGERNAME
from .binary_parser import BinaryParser
from .binary_reader import BinaryReader
from .exceptions import API_EXCEPTIONS, WeatherApiException
from .parsing_util import all_entries_postal, build_coordinates_str_for_polygon, build_coordinates_str, \
    build_coordinates_str_from_postal_codes, \
    build_response_params, convert_grid_binary_response_to_df, convert_lightning_response_to_df, \
    convert_polygon_response_to_df, \
    parse_date_num, extract_user_statistics, parse_ens, parse_query_station_params, \
    parse_query_station_timeseries_params, \
    parse_time_series_params, parse_url_for_post_data, localize_datenum, sanitize_datetime, set_index_for_ts, \
    extract_user_limits

_logger = logging.getLogger(LOGGERNAME)


class Config:
    _config = {
        "VERIFY_SSL": True,  # Disable SSL verification. This setting is useful for corporate environments where
        # "secure" proxies are deployed.
        "PROXIES": {}  # proxies – (optional) Dictionary mapping protocol to the URL of the proxy.
    }

    @staticmethod
    def get(item):
        return Config._config[item]

    @staticmethod
    def set(key, value):
        if key not in Config._config.keys():
            raise KeyError("Key '{}' does not exist.".format(key))
        Config._config[key] = value


def handle_ssl(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not Config.get("VERIFY_SSL"):
            # Disable InsecureRequestWarnings if VERIFY_SSL is disabled.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', InsecureRequestWarning)
                return func(*args, verify=False, **kwargs)
        return func(*args, **kwargs)

    return wrapper


def handle_proxy(func):
    """Passing the proxies dictionary to requests proxies optional argument."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not len(Config.get("PROXIES")) == 0:
            return func(*args, proxies=Config.get("PROXIES"), **kwargs)
        return func(*args, **kwargs)

    return wrapper


@handle_ssl
@handle_proxy
def get_request(*args, **kwargs):
    return requests.get(*args, **kwargs)


@handle_ssl
@handle_proxy
def post_request(*args, **kwargs):
    return requests.post(*args, **kwargs)


def create_path(_file):
    _path = os.path.dirname(_file)
    if not os.path.exists(_path) and len(_path) > 0:
        _logger.info("Create Path: {}".format(_path))
        os.makedirs(_path)


def query_api(url, username, password, request_type="GET", timeout_seconds=330,
              headers={'Accept': 'application/octet-stream'}):
    if request_type.lower() == "get":
        _logger.debug("Calling URL: {} (username = {})".format(url, username))
        response = get_request(url, timeout=timeout_seconds, auth=(username, password), headers=headers)
    elif request_type.lower() == "post":
        url, data = parse_url_for_post_data(url)
        _logger.debug("Calling URL: {} (username = {})".format(url, username))
        headers['Content-Type'] = "text/plain"
        response = post_request(url, timeout=timeout_seconds, auth=(username, password), headers=headers, data=data)
    else:
        raise ValueError('Unknown request_type: {}.'.format(request_type))

    if response.status_code != requests.codes.ok:
        exc = API_EXCEPTIONS[response.status_code]
        raise exc(response.text)

    return response


@deprecated("Do not programmatically rely on user features since the returned keys can change over time.")
def query_user_features(username, password):
    """Get user features"""
    response = get_request(DEFAULT_API_BASE_URL + '/user_stats_json', auth=(username, password))
    if response.status_code != requests.codes.ok:
        exc = API_EXCEPTIONS[response.status_code]
        raise exc(response.text)
    return extract_user_statistics(response)


def query_user_limits(username, password):
    """Get users usage and limits

    returns {limit[name]: (current_count, limit[value]) for limit in defined_limits}
    """
    response = get_request(DEFAULT_API_BASE_URL + '/user_stats_json', auth=(username, password))
    if response.status_code != requests.codes.ok:
        exc = API_EXCEPTIONS[response.status_code]
        raise exc(response.text)
    return extract_user_limits(response)


def convert_time_series_binary_response_to_df(bin_input, coordinate_list, parameters, station=False,
                                              na_values=NA_VALUES):
    df = raw_df_from_bin(bin_input, coordinate_list, parameters, na_values, station)
    # parse parameters which are queried as sql dates but arrive as date_num
    df = df.apply(lambda col: parse_date_num(col) if col.name.endswith(":sql") else col)
    df = set_index_for_ts(df, station, coordinate_list)
    return df


def raw_df_from_bin(bin_input, coordinate_list, parameters, na_values, station):
    binary_parser = BinaryParser(BinaryReader(bin_input), na_values)
    df = binary_parser.parse(parameters, station, coordinate_list)
    return df


def query_station_list(username, password, source=None, parameters=None, startdate=None, enddate=None, location=None,
                       api_base_url=DEFAULT_API_BASE_URL, request_type='GET', elevation=None, id=None):
    """Function to query available stations in API
    source as string
    parameters as list
    enddate as datetime object
    location as string (e.g. "40,10")
    request_type is one of 'GET'/'POST'
    elevation as integer/float (e.g. 1050 ; 0.5)
    """
    url_params_dict = parse_query_station_params(source, parameters, startdate, enddate, location, elevation, id)

    url = STATIONS_LIST_TEMPLATE.format(
        api_base_url=api_base_url,
        urlParams="&".join(["{}={}".format(k, v) for k, v in url_params_dict.items()])
    )
    response = query_api(url, username, password, request_type=request_type)

    sl = pd.read_csv(StringIO(response.text), sep=";")
    sl['lat'] = sl['Location Lat,Lon'].apply(lambda x: float(x.split(",")[0]))
    sl['lon'] = sl['Location Lat,Lon'].apply(lambda x: float(x.split(",")[1]))
    sl.drop('Location Lat,Lon', axis=1, inplace=True)

    return sl


def query_station_timeseries(startdate, enddate, interval, parameters, username, password, model='mix-obs',
                             latlon_tuple_list=None, wmo_ids=None, mch_ids=None, general_ids=None, hash_ids=None,
                             metar_ids=None, temporal_interpolation=None, spatial_interpolation=None, on_invalid=None,
                             api_base_url=DEFAULT_API_BASE_URL, request_type='GET', na_values=NA_VALUES):
    """Retrieve a time series from the Meteomatics Weather API.
    Requested can be by WMO ID, Metar ID or coordinates.
    Start and End dates have to be in UTC.
    Returns a Pandas `DataFrame` with a `DateTimeIndex`.
    request_type is one of 'GET/POST'
    na_values: list of special Values that get converted to NaN.
        Default = [-666, -777, -888, -999]
        See also https://www.meteomatics.com/en/api/response/#reservedvalues
    """

    # set time zone info to UTC if necessary
    startdate = sanitize_datetime(startdate)
    enddate = sanitize_datetime(enddate)

    # build URL
    coordinates = build_coordinates_str(latlon_tuple_list, wmo_ids, metar_ids, mch_ids, general_ids, hash_ids)
    url_params_dict = parse_query_station_timeseries_params(model, on_invalid, temporal_interpolation,
                                                            spatial_interpolation)
    url = TIME_SERIES_TEMPLATE.format(
        api_base_url=api_base_url,
        coordinates=coordinates,
        startdate=startdate.isoformat(),
        enddate=enddate.isoformat(),
        interval=isodate.duration_isoformat(interval),
        parameters=",".join(parameters),
        urlParams="&".join(["{}={}".format(k, v) for k, v in url_params_dict.items()])
    )

    headers = {'Accept': 'text/csv'}
    response = query_api(url, username, password, request_type=request_type, headers=headers)
    coordinates_list = coordinates.split("+")
    return convert_time_series_binary_response_to_df(response.content, coordinates_list, parameters,
                                                     station=True, na_values=na_values)


def query_special_locations_timeseries(startdate, enddate, interval, parameters, username, password, model='mix',
                                       postal_codes=None, temporal_interpolation=None, spatial_interpolation=None,
                                       on_invalid=None, api_base_url=DEFAULT_API_BASE_URL, request_type='GET',
                                       na_values=NA_VALUES):
    """Retrieve a time series from the Meteomatics Weather API.
    Requested locations can also be specified by Postal Codes;
        Input as dictionary, e.g.: postal_codes={'DE': [71679,70173], ...}.
    Start and End dates have to be in UTC.
    Returns a Pandas `DataFrame` with a `DateTimeIndex`.
    request_type is one of 'GET/POST'
    na_values: list of special Values that get converted to NaN.
        Default = [-666, -777, -888, -999]
        See also https://www.meteomatics.com/en/api/response/#reservedvalues
    """

    # set time zone info to UTC if necessary
    startdate = sanitize_datetime(startdate)
    enddate = sanitize_datetime(enddate)

    # build URL
    coordinates = build_coordinates_str_from_postal_codes(postal_codes)
    url_params = parse_query_station_timeseries_params(model, on_invalid, temporal_interpolation, spatial_interpolation)
    url = TIME_SERIES_TEMPLATE.format(
        api_base_url=api_base_url,
        coordinates=coordinates,
        startdate=startdate.isoformat(),
        enddate=enddate.isoformat(),
        interval=isodate.duration_isoformat(interval),
        parameters=",".join(parameters),
        urlParams="&".join(["{}={}".format(k, v) for k, v in url_params.items()])
    )

    headers = {'Accept': 'text/csv'}
    response = query_api(url, username, password, request_type=request_type, headers=headers)
    coordinates_list = coordinates.split("+")
    return convert_time_series_binary_response_to_df(response.content, coordinates_list, parameters, station=True,
                                                     na_values=na_values)


def query_time_series(coordinate_list, startdate, enddate, interval, parameters, username, password, model=None,
                      ens_select=None, interp_select=None, on_invalid=None, api_base_url=DEFAULT_API_BASE_URL,
                      request_type='GET', cluster_select=None, na_values=NA_VALUES,
                      **kwargs):
    """Retrieve a time series from the Meteomatics Weather API.
    Start and End dates have to be in UTC.
    Returns a Pandas `DataFrame` with a `DateTimeIndex`.
    request_type is one of 'GET'/'POST'
    na_values: list of special Values that get converted to NaN.
        Default = [-666, -777, -888, -999]
        See also https://www.meteomatics.com/en/api/response/#reservedvalues
    """

    # set time zone info to UTC if necessary
    startdate = sanitize_datetime(startdate)
    enddate = sanitize_datetime(enddate)

    # build URL
    extended_params = parameters if ens_select is None else build_response_params(parameters, parse_ens(ens_select))
    url_params = parse_time_series_params(model, ens_select, cluster_select, interp_select, on_invalid, kwargs)

    is_postal = all_entries_postal(coordinate_list)
    coordinate_list_str = '+'.join(coordinate_list) if is_postal else "+".join(
        ["{},{}".format(*latlon_tuple) for latlon_tuple in coordinate_list])

    url = TIME_SERIES_TEMPLATE.format(
        api_base_url=api_base_url,
        coordinates=coordinate_list_str,
        startdate=startdate.isoformat(),
        enddate=enddate.isoformat(),
        interval=isodate.duration_isoformat(interval),
        parameters=",".join(parameters),
        urlParams="&".join(["{}={}".format(k, v) for k, v in url_params.items()])
    )

    response = query_api(url, username, password, request_type=request_type)
    df = convert_time_series_binary_response_to_df(response.content, coordinate_list, extended_params,
                                                   na_values=na_values)

    return df


def query_grid(startdate, parameter_grid, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon, username, password, model=None,
               ens_select=None, interp_select=None, on_invalid=None, api_base_url=DEFAULT_API_BASE_URL, request_type='GET',
               na_values=NA_VALUES,
               **kwargs):
    # interpret time as UTC
    startdate = sanitize_datetime(startdate)

    # build URL
    url_params = parse_time_series_params(model=model, ens_select=ens_select, cluster_select=None,
                                          interp_select=interp_select, on_invalid=on_invalid, kwargs=kwargs)
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
        urlParams="&".join(["{}={}".format(k, v) for k, v in url_params.items()])
    )

    response = query_api(url, username, password, request_type=request_type)
    return convert_grid_binary_response_to_df(response.content, parameter_grid, na_values=na_values)


def query_grid_unpivoted(valid_dates, parameters, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon, username, password,
                         model=None, ens_select=None, interp_select=None, on_invalid=None, request_type='GET', na_values=NA_VALUES):
    idxcols = ['valid_date', 'lat', 'lon']
    vd_dfs = []

    for valid_date in valid_dates:
        vd_df = None
        for parameter in parameters:

            dmo = query_grid(valid_date, parameter, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon, username, password,
                             model, ens_select, interp_select, on_invalid=on_invalid, request_type=request_type, na_values=na_values)

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
    except AttributeError:
        data.sort(idxcols, inplace=True)

    data.set_index(idxcols, inplace=True)

    return data


def query_grid_timeseries(startdate, enddate, interval, parameters, lat_N, lon_W, lat_S, lon_E,
                          res_lat, res_lon, username, password, model=None, ens_select=None, interp_select=None,
                          on_invalid=None, api_base_url=DEFAULT_API_BASE_URL, request_type='GET', na_values=NA_VALUES,
                          **kwargs):
    """Retrieve a grid time series from the Meteomatics Weather API.
       Start and End dates have to be in UTC.
       Returns a Pandas `DataFrame` with a `DateTimeIndex`.
       request_type is one of 'GET'/'POST'
       na_values: list of special Values that get converted to NaN.
        Default = [-666, -777, -888, -999]
        See also https://www.meteomatics.com/en/api/response/#reservedvalues
       """

    # set time zone info to UTC if necessary
    startdate = sanitize_datetime(startdate)
    enddate = sanitize_datetime(enddate)

    # build URL
    url_params = parse_time_series_params(model=model, ens_select=ens_select, cluster_select=None,
                                          interp_select=interp_select, on_invalid=on_invalid, kwargs=kwargs)
    url = GRID_TIME_SERIES_TEMPLATE.format(
        api_base_url=api_base_url,
        startdate=startdate.isoformat(),
        enddate=enddate.isoformat(),
        interval=isodate.duration_isoformat(interval),
        lat_N=lat_N,
        lon_W=lon_W,
        lat_S=lat_S,
        lon_E=lon_E,
        res_lat=res_lat,
        res_lon=res_lon,
        parameters=",".join(parameters),
        urlParams="&".join(["{}={}".format(k, v) for k, v in url_params.items()])
    )

    response = query_api(url, username, password, request_type=request_type)

    lats = arange(lat_S, lat_N, res_lat)
    lons = arange(lon_W, lon_E, res_lon)

    latlon_tuple_list = list(itertools.product(lats, lons))
    df = convert_time_series_binary_response_to_df(response.content, latlon_tuple_list, parameters, na_values=na_values)

    return df


def query_polygon(latlon_tuple_lists, startdate, enddate, interval, parameters, aggregation, username,
                  password, operator=None, model=None, ens_select=None, interp_select=None, on_invalid=None,
                  api_base_url=DEFAULT_API_BASE_URL, request_type='GET', cluster_select=None, **kwargs):
    """Retrieve a time series from the Meteomatics Weather API for a selected polygon.
    Start and End dates have to be in UTC.
    Returns a Pandas `DataFrame` with a `DateTimeIndex`.
    request_type is one of 'GET'/'POST'

    Polygons have to be supplied in lists containing lat/lon tuples. For example, input of 2 polygons:
    [[(45.1, 8.2), (45.2, 8.0), (46.2, 7.5)], [(55.1, 8.2), (55.2, 8.0), (56.2, 7.5)]]
    If more than 1 polygon is supplied, then the operator key has to be defined!

    The aggregation parameter can be chosen from: mean, max, min, median, mode. Input format is a list of strings.
    In case of multiple polygons with different aggregators the number of aggregators and polygons must match
    and the operator has to be set to None!

    The operator can be either D (difference) or U (union). Input format is a string.
    """

    # set time zone info to UTC if necessary
    startdate = sanitize_datetime(startdate)
    enddate = sanitize_datetime(enddate)

    # build URL
    url_params_dict = parse_time_series_params(model=model, ens_select=ens_select, cluster_select=cluster_select,
                                               interp_select=interp_select, on_invalid=on_invalid, kwargs=kwargs)
    coordinates = build_coordinates_str_for_polygon(latlon_tuple_lists, aggregation, operator)
    url = POLYGON_TEMPLATE.format(
        api_base_url=api_base_url,
        coordinates_aggregation=coordinates,
        startdate=startdate.isoformat(),
        enddate=enddate.isoformat(),
        interval=isodate.duration_isoformat(interval),
        parameters=",".join(parameters),
        urlParams="&".join(["{}={}".format(k, v) for k, v in url_params_dict.items()])
    )

    response = query_api(url, username, password, request_type=request_type)
    df = convert_polygon_response_to_df(response.text)
    return df


def query_lightnings(startdate, enddate, lat_N, lon_W, lat_S, lon_E, username, password,
                     api_base_url=DEFAULT_API_BASE_URL, request_type='GET', model='mix'):
    """Queries lightning strokes in the specified area during the specified time via the Meteomatics API.
    Returns a Pandas 'DataFrame'.
    request_type is one of 'GET'/'POST'
    """
    # interpret time as UTC
    startdate = sanitize_datetime(startdate)
    enddate = sanitize_datetime(enddate)

    # build URL
    url = LIGHTNING_TEMPLATE.format(
        api_base_url=api_base_url,
        startdate=startdate.isoformat(),
        enddate=enddate.isoformat(),
        lat_N=lat_N,
        lon_W=lon_W,
        lat_S=lat_S,
        lon_E=lon_E,
        source=model
    )

    headers = {'Accept': 'text/csv'}
    response = query_api(url, username, password, request_type=request_type, headers=headers)

    return convert_lightning_response_to_df(response.text)


def query_netcdf(filename, startdate, enddate, interval, parameter_netcdf, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon,
                 username, password, model=None, ens_select=None, interp_select=None,
                 api_base_url=DEFAULT_API_BASE_URL, request_type='GET', cluster_select=None):
    """Queries a netCDF file form the Meteomatics API and stores it in filename.
    request_type is one of 'GET'/'POST'
    """

    # set time zone info to UTC if necessary
    startdate = sanitize_datetime(startdate)
    enddate = sanitize_datetime(enddate)

    # build URL
    url_params_dict = parse_time_series_params(model=model, ens_select=ens_select, cluster_select=cluster_select,
                                               interp_select=interp_select)
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
        urlParams="&".join(["{}={}".format(k, v) for k, v in url_params_dict.items()])
    )

    headers = {'Accept': 'application/netcdf'}
    response = query_api(url, username, password, request_type=request_type, headers=headers)

    # Check if target directory exists
    create_path(filename)

    # save to the specified filename
    with open(filename, 'wb') as f:
        _logger.debug('Create File {}'.format(filename))
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)

    return


def query_init_date(startdate, enddate, interval, parameter, username, password, model,
                    api_base_url=DEFAULT_API_BASE_URL):
    # set time zone info to UTC if necessary
    startdate = sanitize_datetime(startdate)
    enddate = sanitize_datetime(enddate)

    interval_string = "{}--{}:{}".format(startdate.isoformat(),
                                         enddate.isoformat(),
                                         isodate.duration_isoformat(interval))

    url = INIT_DATE_TEMPLATE.format(api_base_url=api_base_url,
                                    model=model, interval_string=interval_string, parameter=parameter)

    headers = {'Accept': 'text/csv'}
    response = query_api(url, username, password, request_type='GET', headers=headers)

    try:
        df = pd.read_csv(
            StringIO(response.text),
            sep=";",
            header=0,
            encoding="utf-8",
            index_col=0,
            na_values=["0000-00-00T00:00:00Z"],
            parse_dates=[0, 1]
        )
    except Exception:
        raise WeatherApiException(response.text)

    df = localize_datenum(df)

    return df


def query_available_time_ranges(parameters, username, password, model, api_base_url=DEFAULT_API_BASE_URL):
    url = AVAILABLE_TIME_RANGES_TEMPLATE.format(api_base_url=api_base_url,
                                                model=model, parameters=",".join(parameters))

    headers = {'Accept': 'text/csv'}
    response = query_api(url, username, password, request_type='GET', headers=headers)

    try:
        df = pd.read_csv(
            StringIO(response.text),
            sep=";",
            header=0,
            encoding="utf-8",
            index_col=0,
            na_values=["0000-00-00T00:00:00Z"],
            parse_dates=['min_date', 'max_date']
        )
    except Exception:
        raise WeatherApiException(response.text)

    return df


def query_grid_png(filename, startdate, parameter_grid, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon, username,
                   password, model=None, ens_select=None, interp_select=None, api_base_url=DEFAULT_API_BASE_URL,
                   request_type='GET'):
    """Gets a png image generated by the Meteomatics API from grid data (see method query_grid)
    and saves it to the specified filename.
    request_type is one of 'GET'/'POST'
    """

    # interpret time as UTC
    startdate = sanitize_datetime(startdate)

    # build URL

    url_params = dict()
    url_params['connector'] = VERSION
    if model is not None:
        url_params['model'] = model

    if ens_select is not None:
        url_params['ens_select'] = ens_select

    if interp_select is not None:
        url_params['interp_select'] = interp_select

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
        urlParams="&".join(["{}={}".format(k, v) for k, v in url_params.items()])
    )

    headers = {'Accept': 'image/png'}
    response = query_api(url, username, password, request_type=request_type, headers=headers)

    # save to the specified filename
    with open(filename, 'wb') as f:
        _logger.debug('Create File {}'.format(filename))
        for chunk in response.iter_content(chunk_size=1024):
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


def arange(start, stop, step):
    data = []
    if start >= stop:
        return data
    while start <= stop:
        data.append(start)
        start = round(start + step, 10)
    return data
