from example import run_example
import datetime as dt
import meteomatics.api as api


def stations_coordinates_example(username: str, password: str, _logger):
    now = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    startdate_station_ts = now - dt.timedelta(days=2)
    enddate_station_ts = now - dt.timedelta(hours=3)
    parameters_station_ts = ['t_2m:C', 'wind_speed_10m:ms', 'precip_1h:mm']
    interval_station_ts = dt.timedelta(hours=1)
    model_station_ts = 'mix-obs'
    coordinates_station_ts = [(47.43, 9.4), (50.03, 8.52)]  # St. Gallen / Frankfurt/Main

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
        return False
    return True


if __name__ == "__main__":
    run_example(stations_coordinates_example)
