from examples.example import run_example
import meteomatics.api as api
from meteomatics.exceptions import Forbidden
import datetime as dt


def is_forecast_possible(username: str, password: str):
    startdate_ts = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    enddate_ts = startdate_ts + dt.timedelta(days=1)
    interval_ts = dt.timedelta(hours=8)
    coordinates_ts = [(47.249297, 9.342854)]
    parameters_ts = ['t_2m:C']

    try:
        api.query_time_series(coordinates_ts, startdate_ts, enddate_ts, interval_ts, parameters_ts, username, password)
    except Forbidden:
        return False
    return True


def is_historic_possible(username: str, password: str):
    startdate_ts = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - dt.timedelta(days=10)
    enddate_ts = startdate_ts + dt.timedelta(days=1)
    interval_ts = dt.timedelta(hours=8)
    coordinates_ts = [(47.249297, 9.342854)]
    parameters_ts = ['t_2m:C']

    try:
        api.query_time_series(coordinates_ts, startdate_ts, enddate_ts, interval_ts, parameters_ts, username, password)
    except Forbidden:
        return False
    return True


def is_grid_query_possible(username: str, password: str):
    lat_N = 50
    lon_W = -15
    lat_S = 20
    lon_E = 10
    res_lat = 3
    res_lon = 3
    startdate_grid = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    parameter_grid = 'evapotranspiration_1h:mm'  # 't_2m:C'

    try:
        api.query_grid(startdate_grid, parameter_grid, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon,
                       username, password)
    except Forbidden:
        return False
    return True


def is_model_selection_possible(username: str, password: str):
    try:
        api.query_available_time_ranges(['t_2m:C', 'precip_6h:mm'], username, password, 'dwd-icon-eu')
    except Forbidden:
        return False
    return True


def account_options_example(username: str, password: str, _logger):
    _logger.info("time series in the future: {}".format(is_forecast_possible(username, password)))
    _logger.info("grid queries: {}".format(is_grid_query_possible(username, password)))
    _logger.info("historic queries: {}".format(is_historic_possible(username, password)))
    _logger.info("model selection: {}".format(is_model_selection_possible(username, password)))


if __name__ == "__main__":
    run_example(account_options_example)
