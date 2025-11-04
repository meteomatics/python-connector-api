# Logging
The connector uses the Python built-in logging library internally (https://docs.python.org/3/library/logging.html). You 
can change the log level by using the function `set_log_level()` from the [logger module](logger.py).

**Example**:
Use the code snippet shown below to get the most verbose log output of the connector (this will also print the HTTP
URLs).
```python
# your other imports
from meteomatics.logger import set_log_level
import logging

set_log_level(logging.DEBUG)
# your other code
```