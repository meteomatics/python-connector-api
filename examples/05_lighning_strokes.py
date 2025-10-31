from example import run_example
import datetime as dt
import meteomatics.api as api


def lightning_strokes_example(username: str, password: str, _logger):
    startdate_l = dt.datetime.utcnow() - dt.timedelta(days=1)
    enddate_l = dt.datetime.utcnow() - dt.timedelta(minutes=5)
    lat_N_l = 90
    lon_W_l = -180
    lat_S_l = -90
    lon_E_l = 180

    _logger.info("\nlighning strokes as csv:")
    try:
        df_lightning = api.query_lightnings(startdate_l, enddate_l, lat_N_l, lon_W_l, lat_S_l, lon_E_l, username,
                                            password)
        _logger.info("Dataframe head \n" + df_lightning.head().to_string())
    except Exception as e:
        _logger.error("Failed, the exception is {}".format(e))
        return False
    return True


if __name__ == "__main__":
    run_example(lightning_strokes_example)
