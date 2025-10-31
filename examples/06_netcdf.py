from example import run_example
import datetime as dt
import meteomatics.api as api


def netcdf_example(username: str, password: str, _logger):
    filename_nc = "path_netcdf/netcdf_target.nc"
    startdate_nc = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    enddate_nc = startdate_nc + dt.timedelta(days=1)
    interval_nc = dt.timedelta(days=1)
    parameter_nc = 't_2m:C'
    lat_N = 50
    lon_W = -15
    lat_S = 20
    lon_E = 10
    res_lat = 3
    res_lon = 3

    _logger.info("\nnetCDF file:")
    try:
        api.query_netcdf(filename_nc, startdate_nc, enddate_nc, interval_nc, parameter_nc, lat_N, lon_W, lat_S,
                         lon_E,
                         res_lat, res_lon, username, password)
        _logger.info("filename = {}".format(filename_nc))
    except Exception as e:
        _logger.error("Failed, the exception is {}".format(e))
        return False
    return True


if __name__ == "__main__":
    run_example(netcdf_example)
