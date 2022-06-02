from examples.example import run_example
import datetime as dt
import meteomatics.api as api


def grid_example(username: str, password: str, _logger):
    lat_N = 50
    lon_W = -15
    lat_S = 20
    lon_E = 10
    res_lat = 3
    res_lon = 3
    startdate_grid = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    parameter_grid = 'evapotranspiration_1h:mm'  # 't_2m:C'

    _logger.info("\ngrid:")
    try:
        df_grid = api.query_grid(startdate_grid, parameter_grid, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon,
                                 username, password)
        _logger.info("Dataframe head \n" + df_grid.head().to_string())
    except Exception as e:
        _logger.error("Failed, the exception is {}".format(e))
        return False
    return True


if __name__ == "__main__":
    run_example(grid_example)
