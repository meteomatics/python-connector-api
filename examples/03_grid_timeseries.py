from example import run_example
import datetime as dt
import meteomatics.api as api


def grid_time_series_example(username: str, password: str, _logger):
    lat_N = 50
    lon_W = -15
    lat_S = 20
    lon_E = 10
    res_lat = 3
    res_lon = 3
    startdate_ts = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    enddate_ts = startdate_ts + dt.timedelta(days=1)
    interval_ts = dt.timedelta(hours=1)
    parameters_ts = ['t_2m:C', 'precip_1h:mm']

    _logger.info("\ngrid timeseries:")
    try:
        df_grid_timeseries = api.query_grid_timeseries(startdate_ts, enddate_ts, interval_ts, parameters_ts, lat_N,
                                                       lon_W, lat_S, lon_E, res_lat, res_lon, username, password)
        _logger.info("Dataframe head \n" + df_grid_timeseries.head().to_string())
    except Exception as e:
        _logger.error("Failed, the exception is {}".format(e))
        return False
    return True


if __name__ == "__main__":
    run_example(grid_time_series_example)
