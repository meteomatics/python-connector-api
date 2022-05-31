from examples.example import run_example
import meteomatics.api as api


def get_available_time_ranges_example(username: str, password: str, _logger):
    _logger.info("\nget available time ranges:")
    try:
        df_time_ranges = api.query_available_time_ranges(['t_2m:C', 'precip_6h:mm'], username, password,
                                                         'ukmo-euro4')
        _logger.info("Dataframe head \n" + df_time_ranges.head().to_string())
    except Exception as e:
        _logger.error("Failed, the exception is {}".format(e))
        return False
    return True


if __name__ == "__main__":
    run_example(get_available_time_ranges_example)
