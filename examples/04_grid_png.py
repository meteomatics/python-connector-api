from example import run_example
import datetime as dt
import meteomatics.api as api


def grid_png_example(username: str, password: str, _logger):
    lat_N = 50
    lon_W = -15
    lat_S = 20
    lon_E = 10
    res_lat = 0.1
    res_lon = 0.1
    filename_png = "grid_target.png"
    startdate_png = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    parameter_png = 't_2m:C'

    _logger.info("\ngrid as a png:")
    try:
        api.query_grid_png(filename_png, startdate_png, parameter_png, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon,
                           username, password)
        _logger.info("filename = {}".format(filename_png))
    except Exception as e:
        _logger.error("Failed, the exception is {}".format(e))
        return False
    return True


if __name__ == "__main__":
    run_example(grid_png_example)
