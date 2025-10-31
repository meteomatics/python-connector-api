from example import run_example
import datetime as dt
import meteomatics.api as api


def get_init_dates_example(username: str, password: str, _logger):
    now = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    _logger.info("\nget init dates:")
    try:
        df_init_dates = api.query_init_date(now, now + dt.timedelta(days=2), dt.timedelta(hours=3), 't_2m:C',
                                            username,
                                            password, 'ecmwf-ens')
        _logger.info("Dataframe head \n" + df_init_dates.head().to_string())
    except Exception as e:
        _logger.error("Failed, the exception is {}".format(e))
        return False
    return True


if __name__ == "__main__":
    run_example(get_init_dates_example)
