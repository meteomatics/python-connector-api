from examples.example import run_example
import datetime as dt
import meteomatics.api as api


def unpivoted_grid_example(username: str, password: str, _logger):
    lat_N = 50
    lon_W = -15
    lat_S = 20
    lon_E = 10
    res_lat = 1
    res_lon = 1
    parameters_grid_unpiv = ['t_2m:C', 'precip_1h:mm']
    valid_dates_unpiv = [dt.datetime.utcnow(), dt.datetime.utcnow() + dt.timedelta(days=1)]

    _logger.info("\nunpivoted grid:")
    try:
        df_grid_unpivoted = api.query_grid_unpivoted(valid_dates_unpiv, parameters_grid_unpiv, lat_N, lon_W, lat_S,
                                                     lon_E, res_lat, res_lon, username, password)
        _logger.info("Dataframe head \n" + df_grid_unpivoted.head().to_string())
    except Exception as e:
        _logger.error("Failed, the exception is {}".format(e))
        return False
    return True


if __name__ == "__main__":
    run_example(unpivoted_grid_example)
