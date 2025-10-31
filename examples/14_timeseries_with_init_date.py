from example import run_example
import datetime as dt
import meteomatics.api as api


def time_series_example(username: str, password: str, _logger):
    startdate_ts = dt.datetime(year=2022, month=5, day=19, hour=14)
    enddate_ts = dt.datetime(year=2022, month=5, day=22, hour=17)
    interval_ts = dt.timedelta(hours=1)
    coordinates_ts = [(41.5685, 2.2573)]
    parameters_ts = ['t_2m:C']
    model = 'ecmwf-ifs'
    ens_select = None  # e.g. 'median'
    cluster_select = None  # e.g. "cluster:1", see http://api.meteomatics.com/API-Request.html#cluster-selection

    _logger.info("\ntime series:")
    try:
        # init_date goes as part of kwargs of the function and is added to the final URL
        df_ts = api.query_time_series(coordinates_ts, startdate_ts, enddate_ts, interval_ts, parameters_ts,
                                      username, password, model, ens_select,
                                      cluster_select=cluster_select, init_date="2022-05-19T00Z")
        _logger.info("Dataframe head \n" + df_ts.head().to_string())
    except Exception as e:
        _logger.info("Failed, the exception is {}".format(e))
        return False
    return True


if __name__ == "__main__":
    run_example(time_series_example)
