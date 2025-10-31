from example import run_example
import datetime as dt
import meteomatics.api as api


def find_stations_example(username: str, password: str, _logger):
    now = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    startdate_station_ts = now - dt.timedelta(days=2)
    enddate_station_ts = now - dt.timedelta(hours=3)
    parameters_station_ts = ['t_2m:C', 'wind_speed_10m:ms', 'precip_1h:mm']

    _logger.info("\nfind stations:")
    try:
        met = api.query_station_list(username, password, startdate=startdate_station_ts, enddate=enddate_station_ts,
                                     parameters=parameters_station_ts)
        _logger.info("Dataframe head \n" + met.head().to_string())
    except Exception as e:
        _logger.error("Failed, the exception is {}".format(e))
        return False
    return True


if __name__ == "__main__":
    run_example(find_stations_example)
