# -*- coding: utf-8 -*-

"""Meteomatics Weather API Connector"""

import datetime as dt
import itertools
import os
import sys
from io import StringIO
from . import __version__

import isodate
import pandas as pd
import pytz
import requests

from . import rounding
from .binary_reader import BinaryReader

logdepth = 0


def log(lvl, msg, depth=-1):
    global logdepth
    if depth == -1:
        depth = logdepth
    else:
        logdepth = depth

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = "   " * depth
    print (now + "| " + lvl + " |" + prefix + msg)
    sys.stdout.flush()


def log_info(msg, depth=-1):
    log("INFO", msg, depth)


def create_path(_file):
    _path = os.path.dirname(_file)
    if (os.path.exists(_path) == False) & (len(_path) > 0):
        log_info("Create Path: {}".format(_path))
        os.makedirs(_path)


DEFAULT_API_BASE_URL = "https://api.meteomatics.com"
VERSION = 'python_{}'.format(__version__)

# Templates
TIME_SERIES_TEMPLATE = "{api_base_url}/{startdate}--{enddate}:{interval}/{parameters}/{coordinates}/bin?{urlParams}"
GRID_TEMPLATE = "{api_base_url}/{startdate}/{parameter_grid}/{lat_N},{lon_W}_{lat_S},{lon_E}:{res_lat},{res_lon}/bin?{urlParams}"
GRID_TIME_SERIES_TEMPLATE = "{api_base_url}/{startdate}--{enddate}:{interval}/{parameters}/{lat_N},{lon_W}_{lat_S},{lon_E}:{res_lat},{res_lon}/bin?{urlParams}"
GRID_PNG_TEMPLATE = "{api_base_url}/{startdate}/{parameter_grid}/{lat_N},{lon_W}_{lat_S},{lon_E}:{res_lat},{res_lon}/png?{urlParams}"
LIGHTNING_TEMPLATE = "{api_base_url}/get_lightning_list?time_range={startdate}--{enddate}&bounding_box={lat_N},{lon_W}_{lat_S},{lon_E}&format=csv"
GRADS_TEMPLATE = "{api_base_url}/{startdate}/{parameters}/{area}/grads?model={model}&{urlParams}"
NETCDF_TEMPLATE = "{api_base_url}/{startdate}--{enddate}:{interval}/{parameter_netcdf}/{lat_N},{lon_W}_{lat_S},{lon_E}:{res_lat},{res_lon}/netcdf?{urlParams}"
STATIONS_LIST_TEMPLATE = "{api_base_url}/find_station?{urlParams}"
INIT_DATE_TEMPLATE = "{api_base_url}/get_init_date?model={model}&valid_date={interval_string}&parameters={parameter}"
AVAILABLE_TIME_RANGES_TEMPLATE = "{api_base_url}/get_time_range?model={model}&parameters={parameters}"

NA_VALUES = [-666, -777, -888, -999]


class WeatherApiException(Exception):
    def __init__(self, message):
        super(WeatherApiException, self).__init__(message)


def datenum2date(date_num):
    if pd.isnull(date_num):
        return pd.NaT
    else:
        total_seconds = round(dt.timedelta(days=date_num - 366).total_seconds())
        return dt.datetime(1, 1, 1) + dt.timedelta(seconds=total_seconds) - dt.timedelta(days=1)


def parse_date_num(s):
    dates = {date: datenum2date(date) for date in s.unique()}
    return s.map(dates)


def parse_ens(ens_str):
    """Build the members strings for the ensemble answer"""
    components = ens_str.split(',')
    out = []
    for c in components:
        if 'member:' in c:
            numbers = c.lstrip('member:')
            if '-' in numbers:
                start, end = numbers.split('-')
                numbers = range(int(start), int(end) + 1)
            else:
                numbers = (int(numbers), )
            for n in numbers:
                out.append('m{}'.format(n))
        else:
            out.append(c)
    return out


def build_response_params(params, ens_params):
    """Combine member strings with the parameter list"""
    out = []
    for param in params:
        for ens in ens_params:
            if ens == 'm0':
                out.append(param)
            else:
                out.append('{}-{}'.format(param, ens))
    return out


def sanitize_datetime(in_date):
    try:
        if in_date.tzinfo is None:
            return in_date.replace(tzinfo=pytz.UTC)
        return in_date
    except AttributeError:
        raise TypeError('Please use datetime.datetime instead of {}'.format(type(in_date)))


def query_api(url, username, password, request_type="GET", timeout_seconds=300,
              headers={'Accept': 'application/octet-stream'}):
    try:
        if request_type.lower() == "get":
            log_info("Calling URL: {} (username = {})".format(url, username))
            response = requests.get(url, timeout=timeout_seconds, auth=(username, password), headers=headers)
        elif request_type.lower() == "post":
            url_splitted = url.split("/", 4)
            if len(url_splitted) > 4:
                url = "/".join(url_splitted[0:4])
                data = url_splitted[4]
            else:
                data = None

            headers['Content-Type'] = "text/plain"

            log_info("Calling URL: {} (username = {})".format(url, username))
            response = requests.post(url, timeout=timeout_seconds, auth=(username, password), headers=headers,
                                     data=data)
        else:
            raise ValueError('Unknown request_type: {}.'.format(request_type))

        if response.status_code != requests.codes.ok:
            raise WeatherApiException(response.text)

        return response
    except requests.ConnectionError as e:
        raise WeatherApiException(e)


def query_user_features(username, password):
    """Get user features"""
    response = requests.get('http://api.meteomatics.com/user_stats_json',
                            auth=(username, password)
                            )
    data = response.json()
    limits_of_interest = ['historic request option', 'model select option', 'area request option']
    try:
        return {key: data['user statistics'][key] for key in limits_of_interest}
    except TypeError:
        user_data = next(d for d in data['user statistics'] if d['username'] == username)
        return {key: user_data[key] for key in limits_of_interest}


def convert_time_series_binary_response_to_df(input, latlon_tuple_list, parameters, station=False):
    binary_reader = BinaryReader(input)

    parameters_ts = parameters[:]

    if station:
        # add station_id in the list of parameters
        parameters_ts.extend(["station_id"])
    else:
        # add lat, lon in the list of parameters
        parameters_ts.extend(["lat", "lon"])
    dfs = []
    # parse response
    num_of_coords = binary_reader.get_int() if len(latlon_tuple_list) > 1 else 1

    for i in range(num_of_coords):
        dict_data = {}
        num_of_dates = binary_reader.get_int()

        for _ in range(num_of_dates):
            num_of_params = binary_reader.get_int()
            date = binary_reader.get_double()
            if station:
                latlon = [latlon_tuple_list[i]]
            else:
                latlon = latlon_tuple_list[i]
            # ensure tuple
            latlon = tuple(latlon)

            value = binary_reader.get_double(num_of_params)
            if type(value) is not tuple:
                value = (value,)
            dict_data[date] = value + latlon

        df = pd.DataFrame.from_items(dict_data.items(), orient="index", columns=parameters_ts)
        df = df.sort_index()
        dfs.append(df)

    df = pd.concat(dfs)
    df = df.replace(NA_VALUES, float('NaN'))
    df.index.name = "validdate"

    df.index = parse_date_num(df.reset_index()["validdate"])

    # mark index as UTC timezone
    df.index = df.index.tz_localize("UTC")

    # parse parameters which are queried as sql dates but arrive as date_num
    for parameter in parameters_ts:
        if parameter.endswith(":sql"):
            df[parameter] = parse_date_num(df[parameter])

    if not station:
        parameters_ts = [c for c in df.columns if c not in ['lat', 'lon']]

        # extract coordinates
        if 'lat' not in df.columns:
            if 'station_id' in df.columns:
                df['lat'] = df['station_id'].apply(lambda x: float(x.split(',')[0]))
                df['lon'] = df['station_id'].apply(lambda x: float(x.split(',')[1]))
                df.drop('station_id', axis=1, inplace=True)
                parameters_ts.remove('station_id')
            else:
                df['lat'] = latlon_tuple_list[0][0]
                df['lon'] = latlon_tuple_list[0][1]

        # replace lat lon with inital coordinates
        split_point = len(df) / len(latlon_tuple_list)
        df.reset_index(inplace=True)
        for i in range(len(latlon_tuple_list)):
            df.loc[i * split_point: (i + 1) * split_point, 'lat'] = latlon_tuple_list[i][0]
            df.loc[i * split_point: (i + 1) * split_point, 'lon'] = latlon_tuple_list[i][1]
        # set multiindex
        df = df.set_index(['lat', 'lon', 'validdate'])
    else:
        parameters_ts = [c for c in df.columns if c not in ['station_id']]
        split_point = len(df) / len(latlon_tuple_list)
        if 'station_id' not in df.columns:
            for i in range(len(latlon_tuple_list)):
                df.loc[int(i * split_point): int((i + 1) * split_point), 'station_id'] = latlon_tuple_list[i]

        # set multiindex
        df = df.reset_index().set_index(['station_id', 'validdate'])
        df = df.sort_index()
    df = rounding.round_df(df)
    return df[parameters_ts]


def query_station_list(username, password, source=None, parameters=None, startdate=None, enddate=None, location=None,
                       api_base_url=DEFAULT_API_BASE_URL, request_type='GET', elevation=None, id=None):
    '''Function to query available stations in API
    source as string
    parameters as list
    enddate as datetime object
    location as string (e.g. "40,10")
    request_type is one of 'GET'/'POST'
    elevation as integer/float (e.g. 1050 ; 0.5)
    '''
    urlParams = {}
    if source is not None:
        urlParams['source'] = source

    if parameters is not None:
        urlParams['parameters'] = ",".join(parameters)

    if startdate is not None:
        urlParams['startdate'] = dt.datetime.strftime(startdate, "%Y-%m-%dT%HZ")

    if enddate is not None:
        urlParams['enddate'] = dt.datetime.strftime(enddate, "%Y-%m-%dT%HZ")

    if location is not None:
        urlParams['location'] = location

    if elevation is not None:
        urlParams['elevation'] = elevation

    if id is not None:
        urlParams['id'] = id

    url = STATIONS_LIST_TEMPLATE.format(
        api_base_url=api_base_url,
        urlParams="&".join(["{}={}".format(k, v) for k, v in urlParams.items()])
    )

    response = query_api(url, username, password, request_type=request_type)

    sl = pd.read_csv(StringIO(response.text), sep=";")
    sl['lat'] = sl['Location Lat,Lon'].apply(lambda x: float(x.split(",")[0]))
    sl['lon'] = sl['Location Lat,Lon'].apply(lambda x: float(x.split(",")[1]))
    sl.drop('Location Lat,Lon', 1, inplace=True)

    return sl


def query_station_timeseries(startdate, enddate, interval, parameters, username, password, model='mix-obs',
                             latlon_tuple_list=None, wmo_ids=None, mch_ids=None, general_ids=None, hash_ids=None,
                             metar_ids=None, temporal_interpolation=None, spatial_interpolation=None, on_invalid=None,
                             api_base_url=DEFAULT_API_BASE_URL, request_type='GET'):
    """Retrieve a time series from the Meteomatics Weather API.
    Requested can be by WMO ID, Metar ID or coordinates.
    Start and End dates have to be in UTC.
    Returns a Pandas `DataFrame` with a `DateTimeIndex`.
    request_type is one of 'GET/POST'
    """

    # set time zone info to UTC if necessary
    startdate = sanitize_datetime(startdate)
    enddate = sanitize_datetime(enddate)

    # build URL

    coordinate_blocks = []
    urlParams = {}
    urlParams['connector'] = VERSION
    if latlon_tuple_list is not None:
        coordinate_blocks += ("+".join(["{},{}".format(*latlon_tuple) for latlon_tuple in latlon_tuple_list]),)

    if wmo_ids is not None:
        coordinate_blocks += ("+".join(['wmo_' + s for s in wmo_ids]),)

    if metar_ids is not None:
        coordinate_blocks += ("+".join(['metar_' + s for s in metar_ids]),)

    if mch_ids is not None:
        coordinate_blocks += ("+".join(['mch_' + s for s in mch_ids]),)

    if general_ids is not None:
        coordinate_blocks += ("+".join(['id_' + s for s in general_ids]),)

    if hash_ids is not None:
        coordinate_blocks += ("+".join([s for s in hash_ids]),)

    coordinates = '+'.join(coordinate_blocks)

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

    headers = {'Accept': 'text/csv'}
    response = query_api(url, username, password, request_type=request_type, headers=headers)

    coordinates_list = coordinates.split("+")
    return convert_time_series_binary_response_to_df(response.content, coordinates_list, parameters, station=True)


def query_special_locations_timeseries(startdate, enddate, interval, parameters, username, password, model='mix',
                                       postal_codes=None, temporal_interpolation=None, spatial_interpolation=None,
                                       on_invalid=None,
                                       api_base_url=DEFAULT_API_BASE_URL, request_type='GET'):
    """Retrieve a time series from the Meteomatics Weather API.
    Requested locations can be soecified by Postal Codes; Input as dictionary, e.g.: postal_codes={'DE': [71679,70173], ...}.
    Start and End dates have to be in UTC.
    Returns a Pandas `DataFrame` with a `DateTimeIndex`.
    request_type is one of 'GET/POST'
    """

    # set time zone info to UTC if necessary
    startdate = sanitize_datetime(startdate)
    enddate = sanitize_datetime(enddate)

    # build URL
    coordinates = ""
    urlParams = {}
    urlParams['connector'] = VERSION
    if postal_codes is not None:
        for country, pcs in postal_codes.items():
            coordinates += "+".join(['postal_' + country.upper() + s for s in pcs])

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

    headers = {'Accept': 'text/csv'}
    response = query_api(url, username, password, request_type=request_type, headers=headers)

    coordinates_list = coordinates.split("+")
    return convert_time_series_binary_response_to_df(response.content, coordinates_list, parameters, station=True)


def query_time_series(latlon_tuple_list, startdate, enddate, interval, parameters, username, password, model=None,
                      ens_select=None, interp_select=None, on_invalid=None,
                      api_base_url=DEFAULT_API_BASE_URL, request_type='GET', cluster_select=None,
                      **kwargs):
    """Retrieve a time series from the Meteomatics Weather API.
    Start and End dates have to be in UTC.
    Returns a Pandas `DataFrame` with a `DateTimeIndex`.
    request_type is one of 'GET'/'POST'
    """

    # set time zone info to UTC if necessary
    startdate = sanitize_datetime(startdate)
    enddate = sanitize_datetime(enddate)

    # build URL

    urlParams = {}
    urlParams['connector'] = VERSION
    if model is not None:
        urlParams['model'] = model

    if ens_select is not None:
        urlParams['ens_select'] = ens_select
        ens_parameters = parse_ens(ens_select)
        extended_params = build_response_params(parameters, ens_parameters)

    if cluster_select is not None:
        urlParams['cluster_select'] = cluster_select

    if interp_select is not None:
        urlParams['interp_select'] = interp_select

    if on_invalid is not None:
        urlParams['on_invalid'] = on_invalid

    for (key, value) in kwargs.items():
        if key not in urlParams:
            urlParams[key] = value

    url = TIME_SERIES_TEMPLATE.format(
        api_base_url=api_base_url,
        coordinates="+".join(["{},{}".format(*latlon_tuple) for latlon_tuple in latlon_tuple_list]),
        startdate=startdate.isoformat(),
        enddate=enddate.isoformat(),
        interval=isodate.duration_isoformat(interval),
        parameters=",".join(parameters),
        urlParams="&".join(["{}={}".format(k, v) for k, v in urlParams.items()])
    )

    response = query_api(url, username, password, request_type=request_type)

    if ens_select is not None:
        df = convert_time_series_binary_response_to_df(response.content, latlon_tuple_list, extended_params)
    else:
        df = convert_time_series_binary_response_to_df(response.content, latlon_tuple_list, parameters)

    return df


def convert_grid_binary_response_to_df(input, parameter_grid):
    binary_reader = BinaryReader(input)

    header = binary_reader.get_string(length=4)

    if header != "MBG_":
        raise WeatherApiException("No MBG received, instead: {}".format(header))

    version = binary_reader.get_int()
    precision = binary_reader.get_int()
    num_payloads_per_forecast = binary_reader.get_int()
    payload_meta = binary_reader.get_int()
    num_forecasts = binary_reader.get_int()
    forecast_dates_ux = [binary_reader.get_unsigned_long() for _ in range(num_forecasts)]

    # precision in bytes
    DOUBLE = 8
    FLOAT = 4

    if version != 2:
        raise WeatherApiException("Only MBG version 2 supported, this is version {}".format(version))

    if precision not in [FLOAT, DOUBLE]:
        raise WeatherApiException("Received wrong precision {}".format(precision))

    if num_payloads_per_forecast > 100000:
        raise WeatherApiException("numForecasts too big (possibly big-endian): {}".format(num_payloads_per_forecast))

    if num_payloads_per_forecast != 1:
        raise WeatherApiException(
            "Wrong number of payloads per forecast date received: {}".format(num_payloads_per_forecast))

    if payload_meta != 0:
        raise WeatherApiException("Wrong payload type received: {}".format(payload_meta))

    lons = []
    lats = []

    value_data_type = "float" if precision == FLOAT else "double"
    num_lat = binary_reader.get_int()

    for _ in range(num_lat):
        lats.append(binary_reader.get_double())

    num_lon = binary_reader.get_int()
    for _ in range(num_lon):
        lons.append(binary_reader.get_double())

    dates_dict = dict()
    for forecast_date_ux in forecast_dates_ux:
        dict_data = {}
        for _ in range(num_payloads_per_forecast):
            for lat in lats:
                values = binary_reader.get(value_data_type, num_lon)
                dict_data[lat] = values

        df = pd.DataFrame.from_items(dict_data.items(), orient="index", columns=lons)
        df = df.replace(NA_VALUES, float('NaN'))
        df = df.sort_index(ascending=False)

        df.index.name = 'lat'
        df.columns.name = 'lon'

        if parameter_grid is not None and parameter_grid.endswith(":sql"):
            df = df.apply(parse_date_num, axis='index')
        else:
            df = df.round(rounding.get_num_decimal_places(parameter_grid))

        if num_forecasts == 1:
            return df
        else:
            dates_dict[dt.datetime.utcfromtimestamp(forecast_date_ux)] = df.copy()

    return dates_dict


def query_grid(startdate, parameter_grid, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon, username, password, model=None,
               ens_select=None, interp_select=None, api_base_url=DEFAULT_API_BASE_URL, request_type='GET',
               **kwargs):
    # interpret time as UTC
    startdate = sanitize_datetime(startdate)

    # build URL

    urlParams = {}
    urlParams['connector'] = VERSION
    if model is not None:
        urlParams['model'] = model

    if ens_select is not None:
        urlParams['ens_select'] = ens_select

    if interp_select is not None:
        urlParams['interp_select'] = interp_select

    for (key, value) in kwargs.items():
        if key not in urlParams:
            urlParams[key] = value

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

    response = query_api(url, username, password, request_type=request_type)

    return convert_grid_binary_response_to_df(response.content, parameter_grid)


def query_grid_unpivoted(valid_dates, parameters, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon, username, password,
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


def query_grid_timeseries(startdate, enddate, interval, parameters, lat_N, lon_W, lat_S, lon_E,
                          res_lat, res_lon, username, password, model=None, ens_select=None, interp_select=None,
                          on_invalid=None, api_base_url=DEFAULT_API_BASE_URL, request_type='GET'):
    """Retrieve a grid time series from the Meteomatics Weather API.
       Start and End dates have to be in UTC.
       Returns a Pandas `DataFrame` with a `DateTimeIndex`.
       request_type is one of 'GET'/'POST'
       """

    # set time zone info to UTC if necessary
    startdate = sanitize_datetime(startdate)
    enddate = sanitize_datetime(enddate)

    # build URL

    urlParams = {}
    urlParams['connector'] = VERSION
    if model is not None:
        urlParams['model'] = model

    if ens_select is not None:
        urlParams['ens_select'] = ens_select

    if interp_select is not None:
        urlParams['interp_select'] = interp_select

    if on_invalid is not None:
        urlParams['on_invalid'] = on_invalid

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
        urlParams="&".join(["{}={}".format(k, v) for k, v in urlParams.items()])
    )

    response = query_api(url, username, password, request_type=request_type)

    lats = arange(lat_S, lat_N, res_lat)
    lons = arange(lon_W, lon_E, res_lon)

    latlon_tuple_list = list(itertools.product(lats, lons))
    df = convert_time_series_binary_response_to_df(response.content, latlon_tuple_list, parameters)

    return df


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

            if df.empty:
                return df

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
        lon_E=lon_E
    )

    headers = {'Accept': 'text/csv'}
    response = query_api(url, username, password, request_type=request_type, headers=headers)

    return convert_lightning_response_to_df(response.text)


def query_netcdf(filename, startdate, enddate, interval, parameter_netcdf, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon,
                 username, password, model=None, ens_select=None, interp_select=None,
                 api_base_url=DEFAULT_API_BASE_URL, request_type='GET'):
    """Queries a netCDF file form the Meteomatics API and stores it in filename.
    request_type is one of 'GET'/'POST'
    """

    # set time zone info to UTC if necessary
    startdate = sanitize_datetime(startdate)
    enddate = sanitize_datetime(enddate)

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

    headers = {'Accept': 'application/netcdf'}
    response = query_api(url, username, password, request_type=request_type, headers=headers)

    # Check if target directory exists
    create_path(filename)

    # save to the specified filename
    with open(filename, 'wb') as f:
        log_info('Create File {}'.format(filename))
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
    except:
        raise WeatherApiException(response.text)

    # mark index as UTC timezone
    df.index = df.index.tz_localize("UTC")

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
    except:
        raise WeatherApiException(response.text)

    return df


def query_grid_png(filename, startdate, parameter_grid, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon, username,
                   password, model=None, ens_select=None, interp_select=None, api_base_url=DEFAULT_API_BASE_URL,
                   request_type='GET'):
    """Gets a png image generated by the Meteomatics API from grid data (see method query_grid) and saves it to the specified filename.
    request_type is one of 'GET'/'POST'
    """

    # interpret time as UTC
    startdate = sanitize_datetime(startdate)

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

    headers = {'Accept': 'image/png'}
    response = query_api(url, username, password, request_type=request_type, headers=headers)

    # save to the specified filename
    with open(filename, 'wb') as f:
        log_info('Create File {}'.format(filename))
        for chunk in response.iter_content(chunk_size=1024):
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
    startdate = sanitize_datetime(startdate)

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

    headers = {'Accept': 'image/png'}
    response = query_api(url, username, password, request_type=request_type, headers=headers)

    # save to the specified filename
    create_path(filename)
    with open(filename, 'wb') as f:
        log_info('Create File {}'.format(filename))
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


def arange(start, stop, step):
    data = []
    if start >= stop:
        return data
    while start <= stop:
        data.append(start)
        start = round(start + step, 10)
    return data
