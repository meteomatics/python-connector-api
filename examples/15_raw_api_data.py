import datetime
import datetime as dt

import pytz

from examples.example import run_example
from meteomatics._constants_ import DEFAULT_API_BASE_URL, NA_VALUES
from meteomatics.api import query_api, raw_df_from_bin


def data_changer(df):
    value = 10
    startdate = dt.datetime(2025, 4, 28, 1, 0, tzinfo=pytz.utc)
    enddate = dt.datetime(2025, 4, 28, 5, 0, tzinfo=pytz.utc)
    df.loc[:, :] = 0
    df = df.reset_index()
    idx_start = list(df[df['validdate'] == startdate].index)
    idx_end = list(df[df['validdate'] == enddate].index)
    for i in range(len(idx_start)):
        df.loc[idx_start[i]: idx_end[i], 'wind_speed_10m:ms'] = value
    df = df.set_index(['lat', 'lon', 'validdate'], drop=True)
    return df


def dataframe_directly_from_binary_data(username: str, password: str, _logger):
    startdate_ts = dt.datetime.now(datetime.UTC).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    enddate_ts = (startdate_ts + dt.timedelta(days=1))
    url = f'{DEFAULT_API_BASE_URL}/{startdate_ts.isoformat()}Z--{enddate_ts.isoformat()}Z:PT1H/t_2m:C,wind_speed_10m:ms/47.249297,9.342854+50.0,10.0/bin?connector=python_v2.11.4&model=mix'

    # The following data are needed to decode the api response
    coordinates_ts = [(47.249297, 9.342854), (50., 10.)]
    parameters_ts = ['t_2m:C', 'wind_speed_10m:ms']

    try:
        response = query_api(url, username, password, request_type='GET')
        df = raw_df_from_bin(response.content, coordinates_ts, parameters_ts, na_values=NA_VALUES, station=False)

        _logger.info("\nData frame shape: {}".format(df.shape))
        _logger.info("Dataframe head \n" + df.head().to_string())
        _logger.info("Dataframe head complete indexes \n" + df.reset_index().head().to_string())
    except Exception as e:
        _logger.info("Failed, the exception is {}".format(e))
        _logger.info("URL: {}".format(url))
        return False
    return True


if __name__ == "__main__":
    run_example(dataframe_directly_from_binary_data)
