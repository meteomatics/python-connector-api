from examples.example import run_example
import datetime as dt
import meteomatics.api as api


def time_series_example_using_proxy(username: str, password: str, _logger):
    startdate_ts = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    enddate_ts = startdate_ts + dt.timedelta(days=1)
    interval_ts = dt.timedelta(hours=1)
    coordinates_ts = [(47.249297, 9.342854), (50., 10.)]
    parameters_ts = ['t_2m:C', 'rr_1h:mm']
    model = 'mix'
    ens_select = None  # e.g. 'median'
    cluster_select = None  # e.g. "cluster:1", see http://api.meteomatics.com/API-Request.html#cluster-selection
    interp_select = 'gradient_interpolation'
    proxies = {
        'http': 'http://localhost:8080',
        'https': 'https://localhost:8080',
    }
    config = api.Config()
    config.set("PROXIES", proxies)

    _logger.info("\ntime series:")
    try:
        df_ts = api.query_time_series(coordinates_ts, startdate_ts, enddate_ts, interval_ts, parameters_ts,
                                      username, password, model, ens_select, interp_select,
                                      cluster_select=cluster_select)
        _logger.info("Dataframe head \n" + df_ts.head().to_string())
    except Exception as e:
        _logger.info("Failed, the exception is {}".format(e))
        return False
    return True


if __name__ == "__main__":
    run_example(time_series_example_using_proxy)
