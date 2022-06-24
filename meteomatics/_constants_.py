# -*- coding: utf-8 -*-
# coding:=utf-8
"""
Collect all constant definitions here
"""
from . import __version__

DEFAULT_API_BASE_URL = "https://api.meteomatics.com"

VERSION = 'python_v{}'.format(__version__)

NA_VALUES = (-666, -777, -888, -999)

LOGGERNAME="meteomatics-api"

# templates
TIME_SERIES_TEMPLATE = "{api_base_url}/{startdate}--{enddate}:{interval}/{parameters}/{coordinates}/bin?{urlParams}"
GRID_TEMPLATE = "{api_base_url}/{startdate}/{parameter_grid}/{lat_N},{lon_W}_{lat_S},{lon_E}:{res_lat},{res_lon}/bin?{urlParams}"
GRID_TIME_SERIES_TEMPLATE = "{api_base_url}/{startdate}--{enddate}:{interval}/{parameters}/{lat_N},{lon_W}_{lat_S},{lon_E}:{res_lat},{res_lon}/bin?{urlParams}"
POLYGON_TEMPLATE = "{api_base_url}/{startdate}--{enddate}:{interval}/{parameters}/{coordinates_aggregation}/csv?{urlParams}"
GRID_PNG_TEMPLATE = "{api_base_url}/{startdate}/{parameter_grid}/{lat_N},{lon_W}_{lat_S},{lon_E}:{res_lat},{res_lon}/png?{urlParams}"
LIGHTNING_TEMPLATE = "{api_base_url}/get_lightning_list?time_range={startdate}--{enddate}&bounding_box={lat_N},{lon_W}_{lat_S},{lon_E}&source={source}&format=csv"
NETCDF_TEMPLATE = "{api_base_url}/{startdate}--{enddate}:{interval}/{parameter_netcdf}/{lat_N},{lon_W}_{lat_S},{lon_E}:{res_lat},{res_lon}/netcdf?{urlParams}"
STATIONS_LIST_TEMPLATE = "{api_base_url}/find_station?{urlParams}"
INIT_DATE_TEMPLATE = "{api_base_url}/get_init_date?model={model}&valid_date={interval_string}&parameters={parameter}"
AVAILABLE_TIME_RANGES_TEMPLATE = "{api_base_url}/get_time_range?model={model}&parameters={parameters}"
logdepth = 0
