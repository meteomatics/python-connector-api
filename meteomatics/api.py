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
from typing import Any, Dict, List, Optional, Tuple, Union

import datetime as dt
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
    """Connector configuration.

    Attributes:
        VERIFY_SSL (bool): If False, SSL verification is disabled for requests.
            This can be useful for corporate environments where "secure" proxies are deployed.
        PROXIES (dict): Optional dictionary mapping protocol to the proxy URL, e.g. {'http': 'http://proxy:8080'}.
            If empty, no proxies are used.
    """
    _config = {
        "VERIFY_SSL": True,
        "PROXIES": {}
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
    def wrapper(*args: Any, **kwargs: Any) -> Any:
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
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not len(Config.get("PROXIES")) == 0:
            return func(*args, proxies=Config.get("PROXIES"), **kwargs)
        return func(*args, **kwargs)

    return wrapper


@handle_ssl
@handle_proxy
def get_request(*args: Any, **kwargs: Any) -> requests.Response:
    return requests.get(*args, **kwargs)


@handle_ssl
@handle_proxy
def post_request(*args: Any, **kwargs: Any) -> requests.Response:
    return requests.post(*args, **kwargs)


def create_path(_file: str) -> None:
    _path = os.path.dirname(_file)
    if not os.path.exists(_path) and len(_path) > 0:
        _logger.info("Create Path: {}".format(_path))
        os.makedirs(_path)


def query_api(
    url: str,
    username: str,
    password: str,
    request_type: str = "GET",
    timeout_seconds: int = 330,
    headers: Optional[Dict[str, str]] = None,
) -> requests.Response:
    # avoid mutable default for headers
    if headers is None:
        headers = {'Accept': 'application/octet-stream'}

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


def query_user_limits(username: str, password: str) -> Dict[str, Tuple[int, int]]:
    """Get users usage and limits

    returns {limit[name]: (current_count, limit[value]) for limit in defined_limits}
    """
    response = get_request(DEFAULT_API_BASE_URL + '/user_stats_json', auth=(username, password))
    if response.status_code != requests.codes.ok:
        exc = API_EXCEPTIONS[response.status_code]
        raise exc(response.text)
    return extract_user_limits(response)


def convert_time_series_binary_response_to_df(
    bin_input: bytes,
    coordinate_list: List[Union[str, Tuple[float, float]]],
    parameters: List[str],
    station: bool = False,
    na_values: Tuple[Any, ...] = NA_VALUES,
) -> pd.DataFrame:
    df = raw_df_from_bin(bin_input, coordinate_list, parameters, na_values, station)
    # parse parameters which are queried as sql dates but arrive as date_num
    df = df.apply(lambda col: parse_date_num(col) if col.name.endswith(":sql") else col)
    df = set_index_for_ts(df, station, coordinate_list)
    return df


def raw_df_from_bin(
    bin_input: bytes,
    coordinate_list: List[Union[str, Tuple[float, float]]],
    parameters: List[str],
    na_values: Tuple[Any, ...],
    station: bool,
) -> pd.DataFrame:
    binary_parser = BinaryParser(BinaryReader(bin_input), na_values)
    df = binary_parser.parse(parameters, station, coordinate_list)
    return df


def query_station_list(
    username: str,
    password: str,
    source: Optional[str] = None,
    parameters: Optional[List[str]] = None,
    startdate: Optional[dt.datetime] = None,
    enddate: Optional[dt.datetime] = None,
    location: Optional[str] = None,
    api_base_url: str = DEFAULT_API_BASE_URL,
    request_type: str = "GET",
    elevation: Optional[float] = None,
    id: Optional[str] = None,
) -> pd.DataFrame:
    """Query available observation stations.

    Filters can be provided for source, parameters, date range, location (as "lat,lon"),
    elevation and station id. The query can be send to API as "GET" or "POST" request type.

    See also: https://www.meteomatics.com/en/api/request/advanced-requests/api-request-weather-station-mos-data/#find_station

    Returns:
        pandas.DataFrame: Station metadata with 'lat' and 'lon' columns added.
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


def query_station_timeseries(
    startdate: dt.datetime,
    enddate: dt.datetime,
    interval: dt.timedelta,
    parameters: List[str],
    username: str,
    password: str,
    model: Optional[str] = "mix-obs",
    latlon_tuple_list: Optional[List[Tuple[float, float]]] = None,
    wmo_ids: Optional[List[str]] = None,
    mch_ids: Optional[List[str]] = None,
    general_ids: Optional[List[str]] = None,
    hash_ids: Optional[List[str]] = None,
    metar_ids: Optional[List[str]] = None,
    temporal_interpolation: Optional[str] = None,
    spatial_interpolation: Optional[str] = None,
    on_invalid: Optional[str] = None,
    api_base_url: str = DEFAULT_API_BASE_URL,
    request_type: str = "GET",
    na_values: Tuple[Any, ...] = NA_VALUES,
) -> pd.DataFrame:
    """Retrieve station data as time series for given IDs or coordinates.

    Start and end datetimes must be timezone-aware in UTC or will be interpreted as UTC.
    Coordinates or station identifiers (WMO ID, METAR ID) may be used.
    na_values: numeric values in the response which will be converted to NaN. Defaults to [-666, -777, -888, -999].

    See also: https://www.meteomatics.com/en/api/request/advanced-requests/api-request-weather-station-mos-data/#station_obs

    Returns:
        pandas.DataFrame: Time series indexed by timestamp (DateTimeIndex).
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


def query_special_locations_timeseries(
    startdate: dt.datetime,
    enddate: dt.datetime,
    interval: dt.timedelta,
    parameters: List[str],
    username: str,
    password: str,
    model: Optional[str] = "mix",
    postal_codes: Optional[Dict[str, List[int]]] = None,
    temporal_interpolation: Optional[str] = None,
    spatial_interpolation: Optional[str] = None,
    on_invalid: Optional[str] = None,
    api_base_url: str = DEFAULT_API_BASE_URL,
    request_type: str = "GET",
    na_values: Tuple[Any, ...] = NA_VALUES,
) -> pd.DataFrame:
    """Retrieve time series for locations specified by postal codes.

    postal_codes should be a dict mapping country codes to lists of postal codes,
    e.g. {'DE': [71679, 70173]}.

    Start and end datetimes must be timezone-aware in UTC or will be interpreted as UTC.
    na_values: numeric values in the response which will be converted to NaN. Defaults to [-666, -777, -888, -999].

    Returns:
        pandas.DataFrame: Time series indexed by timestamp (DateTimeIndex).
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


def query_time_series(
    coordinate_list: List[Union[str, Tuple[float, float]]],
    startdate: dt.datetime,
    enddate: dt.datetime,
    interval: dt.timedelta,
    parameters: List[str],
    username: str,
    password: str,
    model: Optional[str] = None,
    ens_select: Optional[str] = None,
    interp_select: Optional[str] = None,
    on_invalid: Optional[str] = None,
    api_base_url: str = DEFAULT_API_BASE_URL,
    request_type: str = "GET",
    cluster_select: Optional[str] = None,
    na_values: Tuple[Any, ...] = NA_VALUES,
    **kwargs: Any
) -> pd.DataFrame:
    """Retrieve a time series for one or more coordinates.

    coordinate_list should be either one (lat,lon) tuples or postal-code strings
    (i.e. must contain 'postal_' prefix), mixing the two doesn't work

    If ensemble selection is requested, additional columns for ensemble members may be returned.

    Start and end datetimes must be timezone-aware in UTC or will be interpreted as UTC.
    na_values: numeric values in the response which will be converted to NaN. Defaults to [-666, -777, -888, -999].

    Returns:
        pandas.DataFrame: Time series indexed by timestamp (DateTimeIndex).
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


def query_grid(
    startdate: dt.datetime,
    parameter_grid: str,
    lat_N: float,
    lon_W: float,
    lat_S: float,
    lon_E: float,
    res_lat: float,
    res_lon: float,
    username: str,
    password: str,
    model: Optional[str] = None,
    ens_select: Optional[str] = None,
    interp_select: Optional[str] = None,
    on_invalid: Optional[str] = None,
    api_base_url: str = DEFAULT_API_BASE_URL,
    request_type: str = "GET",
    na_values: Tuple[Any, ...] = NA_VALUES,
    **kwargs: Any
) -> pd.DataFrame:
    """Retrieve a rectangular grid for a single valid date.

    Start datetime must be timezone-aware in UTC or will be interpreted as UTC.

    Parameters:
        lat_N, lon_W, lat_S, lon_E (float): Bounding box coordinates (north, west, south, east).
        res_lat, res_lon (float): Spatial resolution in degrees for latitude and longitude.
        ens_select (Optional[str]): Ensemble selection string (returns ensemble members if used).
        na_values (tuple): Values to be interpreted as missing (converted to NaN).

    Returns:
        pd.DataFrame with values of the parameter, lat as index, lon as columns
    """
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

def query_grid_unpivoted(
    valid_dates: List[dt.datetime],
    parameters: List[str],
    lat_N: float,
    lon_W: float,
    lat_S: float,
    lon_E: float,
    res_lat: float,
    res_lon: float,
    username: str,
    password: str,
    model: Optional[str] = None,
    ens_select: Optional[str] = None,
    interp_select: Optional[str] = None,
    on_invalid: Optional[str] = None,
    request_type: str = "GET",
    na_values: Tuple[Any, ...] = NA_VALUES,
) -> pd.DataFrame:
    """Retrieve a rectangular grid for a more then one valid date.

    Internally calls query_grid for each of `valid_dates`. Then parses the
    responses into a data frame with MultiIndex(valid_date, lat, lon) and a
    column per each of requested `parameters`.

    """
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


def query_grid_timeseries(
    startdate: dt.datetime,
    enddate: dt.datetime,
    interval: dt.timedelta,
    parameters: List[str],
    lat_N: float,
    lon_W: float,
    lat_S: float,
    lon_E: float,
    res_lat: float,
    res_lon: float,
    username: str,
    password: str,
    model: Optional[str] = None,
    ens_select: Optional[str] = None,
    interp_select: Optional[str] = None,
    on_invalid: Optional[str] = None,
    api_base_url: str = DEFAULT_API_BASE_URL,
    request_type: str = "GET",
    na_values: Tuple[Any, ...] = NA_VALUES,
    **kwargs: Any
) -> pd.DataFrame:
    """Retrieve a grid (aka Rectangle) time series as tabular data.

    Start and end datetimes must be timezone-aware in UTC or will be interpreted as UTC.
    If ensemble selection is requested, additional columns for ensemble members may be returned.
    na_values: numeric values in the response which will be converted to NaN. Defaults to [-666, -777, -888, -999].

    See also: https://www.meteomatics.com/en/api/request/required-parameters/coordinate-description/

    Returns:
        pandas.DataFrame: Time-indexed grid time series; each column corresponds to a grid point.
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


def query_polygon(
    latlon_tuple_lists: List[List[Tuple[float, float]]],
    startdate: dt.datetime,
    enddate: dt.datetime,
    interval: dt.timedelta,
    parameters: List[str],
    aggregation: List[str],
    username: str,
    password: str,
    operator: Optional[str] = None,
    model: Optional[str] = None,
    ens_select: Optional[str] = None,
    interp_select: Optional[str] = None,
    on_invalid: Optional[str] = None,
    api_base_url: str = DEFAULT_API_BASE_URL,
    request_type: str = "GET",
    cluster_select: Optional[str] = None,
    **kwargs: Any
) -> pd.DataFrame:
    """Retrieve time series aggregated over polygon geometries.

    Polygons must be lists of (lat,lon) tuples. Aggregation may be specified per polygon (mean, max, min, etc.).
    Start and end datetimes must be timezone-aware in UTC or will be interpreted as UTC.
    If ensemble selection is requested, additional columns for ensemble members may be returned.
    na_values: numeric values in the response which will be converted to NaN. Defaults to [-666, -777, -888, -999].

    See also: https://www.meteomatics.com/en/api/request/required-parameters/coordinate-description/

    Returns:
        pandas.DataFrame: Aggregated time series indexed by timestamp (DateTimeIndex).
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


def query_lightnings(
    startdate: dt.datetime,
    enddate: dt.datetime,
    lat_N: float,
    lon_W: float,
    lat_S: float,
    lon_E: float,
    username: str,
    password: str,
    api_base_url: str = DEFAULT_API_BASE_URL,
    request_type: str = "GET",
    model: str = "mix",
) -> pd.DataFrame:
    """Query lightning strokes in a bounding box and time range.

    Start and end datetimes must be timezone-aware in UTC or will be interpreted as UTC.
    The response is parsed into a DataFrame with one row per lightning stroke
    and relevant metadata columns.

    See also: https://www.meteomatics.com/en/api/available-parameters/weather-parameter/lightnings/#lightningdensity

    Returns:
        pandas.DataFrame: Lightning stroke records for the requested area and period.
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


def query_netcdf(
    filename: str,
    startdate: dt.datetime,
    enddate: dt.datetime,
    interval: dt.timedelta,
    parameter_netcdf: str,
    lat_N: float,
    lon_W: float,
    lat_S: float,
    lon_E: float,
    res_lat: float,
    res_lon: float,
    username: str,
    password: str,
    model: Optional[str] = None,
    ens_select: Optional[str] = None,
    interp_select: Optional[str] = None,
    api_base_url: str = DEFAULT_API_BASE_URL,
    request_type: str = "GET",
    cluster_select: Optional[str] = None,
) -> None:
    """Download a netCDF file from the Meteomatics API and save to `filename`.

    Request returns binary netCDF content which is written to the given path.
    Note: the target directory will be created if it does not exist.

    Start and end datetimes must be timezone-aware in UTC or will be interpreted as UTC.
    If ensemble selection is requested, additional columns for ensemble members may be returned.

    Returns:
        None
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


def query_init_date(
    startdate: dt.datetime,
    enddate: dt.datetime,
    interval: dt.timedelta,
    parameter: str,
    username: str,
    password: str,
    model: str,
    api_base_url: str = DEFAULT_API_BASE_URL,
) -> pd.DataFrame:
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


def query_available_time_ranges(
    parameters: List[str],
    username: str,
    password: str,
    model: str,
    api_base_url: str = DEFAULT_API_BASE_URL,
) -> pd.DataFrame:
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


def query_grid_png(
    filename: str,
    startdate: dt.datetime,
    parameter_grid: str,
    lat_N: float,
    lon_W: float,
    lat_S: float,
    lon_E: float,
    res_lat: float,
    res_lon: float,
    username: str,
    password: str,
    model: Optional[str] = None,
    ens_select: Optional[str] = None,
    interp_select: Optional[str] = None,
    api_base_url: str = DEFAULT_API_BASE_URL,
    request_type: str = "GET",
) -> None:
    """Request a PNG image for a single grid timestamp and save to `filename`.

    The PNG is produced server-side from the specified grid parameter and written to the
    specified file path.

    Start datetime must be timezone-aware in UTC or will be interpreted as UTC.
    If ensemble selection is requested, additional columns for ensemble members may be returned.

    Returns:
        None
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


def query_png_timeseries(
    prefixpath: str,
    startdate: dt.datetime,
    enddate: dt.datetime,
    interval: dt.timedelta,
    parameter: str,
    lat_N: float,
    lon_W: float,
    lat_S: float,
    lon_E: float,
    res_lat: float,
    res_lon: float,
    username: str,
    password: str,
    model: Optional[str] = None,
    ens_select: Optional[str] = None,
    interp_select: Optional[str] = None,
    api_base_url: str = DEFAULT_API_BASE_URL,
    request_type: str = "GET",
) -> None:
    """Download a sequence of PNG images for a time range and save them under `prefixpath`.

    Each image filename is constructed from the parameter and timestamp. The function iterates
    from startdate to enddate with the provided interval and saves each PNG using query_grid_png.

    Start and end datetimes must be timezone-aware in UTC or will be interpreted as UTC.
    If ensemble selection is requested, additional columns for ensemble members may be returned.

    Returns:
        None
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


def arange(start: float, stop: float, step: float) -> List[float]:
    data: List[float] = []
    if start >= stop:
        return data
    while start <= stop:
        data.append(start)
        start = round(start + step, 10)
    return data
