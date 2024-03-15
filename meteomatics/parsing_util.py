# -*- coding: utf-8 -*-

import datetime as dt
from io import StringIO

import pandas as pd
import pytz
from meteomatics.deprecated import deprecated

from . import rounding
from ._constants_ import VERSION, NA_VALUES
from .binary_reader import BinaryReader
from .exceptions import WeatherApiException


def all_entries_postal(coordinate_list):
    return all([isinstance(coord, str) and coord.startswith('postal_') for coord in coordinate_list])


def build_coordinates_str_for_polygon(latlon_tuple_lists, aggregation, operator=None):
    coordinates_polygon_list = ["_".join(["{},{}".format(*latlon_tuple) for latlon_tuple in latlon_tuple_list]) for
                                latlon_tuple_list in latlon_tuple_lists]

    if len(coordinates_polygon_list) > 1:
        if operator is not None:
            coordinates = operator.join(coordinates_polygon_list)
            coordinates = coordinates + ':' + aggregation[0]
        else:
            coordinates_polygon_list_aggregator_included = []
            for i, coordinates_polygon in enumerate(coordinates_polygon_list):
                coordinates_polygon = coordinates_polygon + ':' + aggregation[i]
                coordinates_polygon_list_aggregator_included.append(coordinates_polygon)
            coordinates = '+'.join(coordinates_polygon_list_aggregator_included)

    else:
        coordinates = coordinates_polygon_list[0] + ':' + aggregation[0]

    return coordinates


def build_coordinates_str(latlon_tuple_list, wmo_ids, metar_ids, mch_ids, general_ids, hash_ids):
    coordinate_blocks = []
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

    return '+'.join(coordinate_blocks)


def build_coordinates_str_from_postal_codes(postal_codes=None):
    if postal_codes is None:
        return ""
    return "+".join(['postal_' + country.upper() + str(s) for (country, pcs) in postal_codes.items() for s in pcs])


def build_response_params(params, ens_params):
    """Combine member strings with the parameter list"""
    out = [param if ens == 'm0' else '{}-{}'.format(param, ens) for param in params for ens in ens_params]
    return out


def convert_grid_binary_response_to_df(bin_input, parameter_grid, na_values=NA_VALUES):
    binary_reader = BinaryReader(bin_input)

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
    double_precision = 8
    float_precision = 4

    if version != 2:
        raise WeatherApiException("Only MBG version 2 supported, this is version {}".format(version))

    if precision not in [float_precision, double_precision]:
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

    value_data_type = "float" if precision == float_precision else "double"
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

        df = pd.DataFrame.from_dict(dict_data, orient="index", columns=lons)
        df = df.replace(na_values, float('NaN'))
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


def convert_lightning_response_to_df(data):
    """converts the response of the query of query_lightnings to a pandas DataFrame."""
    is_str = False
    try:
        is_str = isinstance(data, basestring)  # python 2
    except NameError:
        is_str = isinstance(data, str)  # python 3
    finally:
        if is_str:
            data = StringIO(data)

        # parse response
        try:
            df = pd.read_csv(
                data,
                sep=";",
                header=0,
                encoding="utf-8",
                parse_dates=['stroke_time:sql'],
                index_col='stroke_time:sql'
            )

            if df.empty:
                return df

            # mark index as UTC timezone
            df = localize_datenum(df)

        except:
            raise WeatherApiException(input.getvalue())

        # rename columns to make consistent with other csv file headers
        df = df.reset_index().rename(
            columns={'stroke_time:sql': 'validdate', 'stroke_lat:d': 'lat', 'stroke_lon:d': 'lon'})
        df.set_index(['validdate', 'lat', 'lon'], inplace=True)

        return df


def convert_polygon_response_to_df(csv):
    # Example for header of CSV (single polygon, polygon united or polygon difference): 'validdate;t_2m:C,precip_1h:mm'
    # Example for header of CSV (multiple polygon): 'station_id, validdate;t_2m:C,precip_1h:mm'
    df = pd.read_csv(StringIO(csv), sep=";", header=0, encoding="utf-8", parse_dates=['validdate'])
    if 'station_id' not in df.columns:
        df['station_id'] = 'polygon1'
    df.set_index(['station_id', 'validdate'], inplace=True)
    return df


def datenum_to_date(date_num):
    """Transform date_num to datetime object.

    Returns pd.NaT on invalid input"""
    try:
        total_seconds = round(dt.timedelta(days=date_num - 366).total_seconds())
        return dt.datetime(1, 1, 1) + dt.timedelta(seconds=total_seconds) - dt.timedelta(days=1)
    except (OverflowError, ValueError):
        return pd.NaT


@deprecated("This function will be removed/renamed because it only provides info about the licensing options "
            "and not real user statistics. "
            "In addition, do not programmatically rely on user features since the returned keys can change "
            "over time due to internal changes.")
def extract_user_statistics(response):
    """Extract user statistics from HTTP response"""
    data = response.json()
    limits_of_interest = ['historic request option', 'model select option', 'area request option']
    try:
        return {key: data['user statistics'][key] for key in limits_of_interest}
    except TypeError:
        user_data = next(d for d in data['user statistics'] if d['username'] == username)
        return {key: user_data[key] for key in limits_of_interest}


def extract_user_limits(response):
    """Extract user limits from HTTP response

    returns {limit[name]: (current_count, limit[value]) for limit in defined_limits}
    """
    data = response.json()
    limits_of_interest = ['requests total', 'requests since last UTC midnight', 'requests since HH:00:00',
                          'requests in the last 60 seconds', 'requests in parallel']
    try:
        return {key: (data['user statistics'][key]['used'], data['user statistics'][key]['hard limit'])
                for key in limits_of_interest if data['user statistics'][key]['hard limit'] != 0}
    except TypeError:
        user_data = next(d for d in data['user statistics'] if d['username'] == username)
        return {key: (user_data[key]['used'], user_data[key]['hard limit'])
                for key in limits_of_interest if user_data['user statistics'][key]['hard limit'] != 0}


def filter_none_from_dict(d):
    return {k: v for k, v in d.items() if v is not None}


def localize_datenum(df):
    """Localize a dataframe index to UTC if needed"""
    try:
        df.index = df.index.tz_localize("UTC")
    except TypeError as e:
        df.index = df.index.tz_convert("UTC")
    return df


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
                numbers = (int(numbers),)
            for n in numbers:
                out.append('m{}'.format(n))
        else:
            out.append(c)
    return out


def parse_date_num(s):
    dates = {date: datenum_to_date(date) for date in s.unique()}
    return s.map(dates)


def parse_query_station_params(source=None, parameters=None, startdate=None, enddate=None, location=None,
                               elevation=None, id=None):
    if parameters is None:
        parameters_string = None
    elif isinstance(parameters, list):
        parameters_string = ','.join((p for p in parameters))
    elif isinstance(parameters, str):
        parameters_string = parameters
    else:
        raise TypeError("Please use a string or a list of strings for parameters.")
    url_params_dict = {
        'source': source,
        'parameters': parameters_string,
        'startdate': startdate.strftime("%Y-%m-%dT%HZ") if startdate is not None else None,
        'enddate': enddate.strftime("%Y-%m-%dT%HZ") if enddate is not None else None,
        'location': location,
        'elevation': elevation,
        'id': id
    }
    # Filter out keys that do not have any value
    return filter_none_from_dict(url_params_dict)


def parse_query_station_timeseries_params(model=None, on_invalid=None, temporal_interpolation=None,
                                          spatial_interpolation=None):
    url_params_dict = {
        'connector': VERSION,
        'model': model,
        'on_invalid': on_invalid,
        'temporal_interpolation': temporal_interpolation,
        'spatial_interpolation': spatial_interpolation
    }
    return filter_none_from_dict(url_params_dict)


def parse_time_series_params(model=None, ens_select=None, cluster_select=None, interp_select=None, on_invalid=None,
                             kwargs=None):
    url_params_dict = {
        'connector': VERSION,
        'model': model,
        'ens_select': ens_select,
        'cluster_select': cluster_select,
        'interp_select': interp_select,
        'on_invalid': on_invalid,
    }
    if kwargs is not None:
        for (key, value) in kwargs.items():
            if key not in url_params_dict:
                url_params_dict[key] = value
    return filter_none_from_dict(url_params_dict)


def parse_url_for_post_data(url):
    """Split the url between url and data if needed"""
    url_splitted = url.split("/", 4)
    data = None
    max_length_url = 2000
    if len(url_splitted) > 4:
        url = "/".join(url_splitted[0:4])
        data = url_splitted[4]
        if len(url) > max_length_url:
            url = "/".join(url_splitted[0:3])
            data = "/".join(url_splitted[3:])
    return url, data


def sanitize_datetime(in_date):
    try:
        if in_date.tzinfo is None:
            return in_date.replace(tzinfo=pytz.UTC)
        return in_date
    except AttributeError:
        raise TypeError('Please use datetime.datetime instead of {}'.format(type(in_date)))


def set_index_for_ts(df, is_station, coordinate_list):
    is_postal = all_entries_postal(coordinate_list)
    if not is_station and not is_postal:
        parameters = list(set(df.columns) - set(['lat', 'lon']))
        # extract coordinates
        if 'lat' not in df.columns:
            if 'station_id' in df.columns:
                df['lat'] = df['station_id'].apply(lambda x: float(x.split(',')[0]))
                df['lon'] = df['station_id'].apply(lambda x: float(x.split(',')[1]))
                df.drop('station_id', axis=1, inplace=True)
                parameters.remove('station_id')
            else:
                df['lat'] = coordinate_list[0][0]
                df['lon'] = coordinate_list[0][1]
        # replace lat lon with inital coordinates
        split_point = len(df) / len(coordinate_list)
        df.reset_index(inplace=True)
        for i in range(len(coordinate_list)):
            df.loc[i * split_point: (i + 1) * split_point, 'lat'] = coordinate_list[i][0]
            df.loc[i * split_point: (i + 1) * split_point, 'lon'] = coordinate_list[i][1]
        # set multiindex
        df = df.set_index(['lat', 'lon', 'validdate'])
    else:
        col_name = 'postal_code' if is_postal else 'station_id'
        split_point = len(df) / len(coordinate_list)
        if col_name not in df.columns:
            df[col_name] = ""  # create new column
            new_col_index = len(df.columns)-1  # use iloc for indexed based slicing
            for i in range(len(coordinate_list)):
                df.iloc[int(i * split_point): int((i + 1) * split_point), new_col_index] = coordinate_list[i]
        # set multiindex
        df = df.reset_index().set_index([col_name, 'validdate'])
        df = df.sort_index()
    df = rounding.round_df(df)
    return df
