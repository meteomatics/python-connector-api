"""
In this script a user can specify several regions for which weather data will then be plotted. It was conceived as a
regular energy market update with temperature anomaly (indicative of heating demand) and 90m wind speed (indicative of
onshore wind generation) as the time-series variables. Two sets of ouput plots are produced:
- time-series data for each region and each variable
- an animation of temperture and pressure in a given region and map projection (frames also produced)
The variables used in the time-series may be straightforwardly changed; the variables shown in the animation are
not configurable (but may be changed in the function) since, in general, a different choice of variable will imply
a different optimal plot type.

I've experimented with a few different configurations but the code has not been robustly tested: let me know if you
try something that doesn't work and I'll think about an update.
"""

import shapely.geometry as sgeometry
import matplotlib.pyplot as plt
import mm_python_module.api_connector.meteomatics.api as api
import cartopy.crs as ccrs
import geopandas as gpd
from PIL import Image
import datetime as dt
import pandas as pd
import xarray as xr
import numpy as np
import functools
import glob
import time
import os

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CONFIGURATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 1. What countries/regions do you want in your time-series?
# These are provided in two arguments. 'standard_countries' is a list of strings which are the names of polygons in
# geopandas' "naturalearth_lowres" built-in dataset. 'manual_geometries' is a dictionary whose keys are the names you
# want to give to your user-created polygons, and whose keys are the paths to geojson files containing those polygons.
# I advise that, if you make changes to this, you check that they have been interpreted correctly by switching
# 'show_polygons' to True. This will show them in your chosen map projection. Speaking of which...
standard_countries = ['United Kingdom', 'Germany', 'Netherlands', 'Italy']
manual_geometries = {'Northern France': 'polygons/northern_france.geojson'}
show_polygons = False

# 2. What map projection do you want to show your animated data in?
# Change entries in the dictionaries to change the bounding box for your animations. This will make a rectangular plot
# regardless of the projection. The dictionary values are then used to define the map projection. You can choose from
# any cartopy projection, but be aware that a) some projections take different arguments to define the transformations
# and b) some projections will not fill the rectangle defined by the dictionaries in certain parts of the globe (for
# instance, if you use AzimuthalEquidistant in Europe you'll see the fetched data get squeezed at the top of the plot).
top_left = {
    'lat': 66.211199,
    'lon': -32.364446
}
bottom_right = {
    'lat': 35.526388,
    'lon': 32
}
# crs = ccrs.AzimuthalEquidistant(central_longitude=(top_left['lon']+bottom_right['lon'])/2,
#                                 central_latitude=(top_left['lat']+bottom_right['lat'])/2)
crs = ccrs.Mercator(central_longitude=(top_left['lon']+bottom_right['lon'])/2,
                    min_latitude=bottom_right['lat'],
                    max_latitude=top_left['lat'])

# 3. What variables are you interested in?
# Currently this choice only affects time-series plots: the variables used in the animation are static (temperature and
# pressure, see animate <== make_nc). Keys should be the names you want on your axis labels, and should feature exactly
# one set of parentheses enclosing the units; values are the names of the corresponding meteomatics strings.
timeseries_vars = {
    'Temperature (C)': 't_2m:C',
    '90m Wind Speed (m/s)': 'wind_speed_90m:ms',
    'Global Radiation (W/m2)': 'global_rad:W'
}

# 4. How do you want to define the baseline (climatology) for your time-series?
# 15 years is apparently standard for energy industry; 30 years is standard in academic texts. Climatologies are all
# interpolated to 1hrly resolution, but I recommend sub-6hrly for data acquisition, since 0600 and 1800 vary between
# being daytime- and nighttime values throughout the year. Also note that the API assumes all times are UTC, so a
# smaller time-step is better for translating to other time-zones). Don't let your climatology get too out of date!
climatology_start_year = 2005
climatology_stop_year = 2020
climatology_step = dt.timedelta(hours=3)

# IMPORTANT! Climatologies are obtained once per new run environment and then imported from .csv in order to reduce
# runtime. The script is not clever enough to know whether you have changed timeseries_vars or the period to be covered
# by climatology, so make sure you force the recalculation of climatology whenever changes are made.
rebuild_climatology = False

# 5. Plot formatting.
#   What period do you want your output (time-series and animations) to cover?
#   What temporal resolution do you want for them both (set separately)?
#   What spatial resolution do you want in your
# Note that the animation may fail if you set the time-step very high
lead_time_days = 28
start_time = dt.datetime.combine(dt.datetime.now().date(), dt.time(12))  # starts at 12pm on the day called
t_series_lead = dt.timedelta(days=lead_time_days)
t_series_step = dt.timedelta(hours=1)
animation_step = dt.timedelta(hours=6)

# API access
username = 'XXXXXX'
password = 'XXXXXX'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ END CONFIGURATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        t_start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - t_start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


def get_geometries(standard_countries, manual_geometries, show=False):
    """
    Uses geopandas' built-in countries for those polygons which correspond to a whole country; otherwise can use custom
    geometries provided in GeoJSON format (with corresponding user-defined polygon names: see parameter docstrings).
    :param standard_countries: list of country names as they appear in naturalearth_lowres
    :param manual_geometries: dictionary of:
                polygon names : GeoJSON paths
    :param show: Bool, if True will illustrate location and shape of polygons on given map projection
    :return: dictionary of:
                polygon names (standard_countries + manual_geometries.keys) : shapely.[Multi]Polygon
    """
    countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    geometries = {}
    geometries_projected = {}  # will be filled with geometries in chosen projection if show is True
    for country in standard_countries:
        geometries[country] = countries[countries.name == country].geometry.values[0]
        if show:
            geometries_projected[country] = countries[countries.name == country].to_crs(crs.proj4_init).geometry
    for country in manual_geometries:
        geometries[country] = gpd.read_file(manual_geometries[country]).geometry.values[0]
        if show:
            geometries_projected[country] = gpd.read_file(manual_geometries[country]).to_crs(crs.proj4_init).geometry
    if show:
        show_geometries(geometries_projected)
    return geometries


def show_geometries(geometries):  # TODO could add an option to show overlap
    """
    Plots the projected polygons
    :param geometries: shapely.[Multi]Polygon objects corresponding to polygons projected into crs (defined in header)
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection=crs)
    ax.stock_img()
    ax.set_extent([top_left['lon'], bottom_right['lon'], top_left['lat'], bottom_right['lat']])
    for geom in geometries:
        geometries[geom].plot(ax=ax)
    plt.show()


@timer
def get_polygon_climatology(geometries, location):  # TODO could save climatology locally and only fetch if required
    if os.path.exists(os.path.join('climatologies', '{}.csv'.format(location))) and not rebuild_climatology:
        print('Climatology for {} exists: reading climatology'.format(location))
    else:
        print('New climatology required for {}.'.format(location))
        os.makedirs('climatologies', exist_ok=True)
        write_climatology(geometries, location)
    df = pd.read_csv(os.path.join('climatologies', '{}.csv'.format(location)), index_col=[0, 1])
    return df


def write_climatology(geometries, location):
    """
    This only looks hefty because of the if/else block, plus the fact that I've extended API calls over multiple lines.
    Makes a climatology dataframe by joining years together sequentially.
    :param geometries: dictionary of:
                locations : shapely.[Multi]Polygons
    :param location: the name of the location. passed because desired for name of written .csv
    """
    geometry = geometries[location]
    tuple_lists = make_tuple_lists(geometry)
    print('Fetching climatology for {}.'.format(location))
    # had to break this down because of 300s timeout. 3hrly still seems reasonable and a bit faster; could go 1hrly now
    for year in range(climatology_start_year, climatology_stop_year):
        if year == climatology_start_year:
            df = api.query_polygon(
                latlon_tuple_lists=tuple_lists,
                startdate=dt.datetime(year, 1, 1),
                enddate=dt.datetime(year, 12, 31, int(24 - climatology_step.seconds/3600.)),
                interval=climatology_step,
                parameters=timeseries_vars.values(),
                aggregation=['mean'],
                username=username,
                password=password,
                operator='U',  # in case of MultiPolygons
                model='ecmwf-era5',
                polygon_sampling='adaptive_grid'
            )
        else:
            df = df.append(
                api.query_polygon(
                    latlon_tuple_lists=tuple_lists,
                    startdate=dt.datetime(year, 1, 1),
                    enddate=dt.datetime(year, 12, 31, int(24 - climatology_step.seconds/3600.)),
                    interval=climatology_step,
                    parameters=timeseries_vars.values(),
                    aggregation=['mean'],
                    username=username,
                    password=password,
                    operator='U',  # in case of MultiPolygons
                    model='ecmwf-era5',
                    polygon_sampling='adaptive_grid'
                )
            )
    # Query returns MultiIndexed DataFrame, but outer index is polygon1 even if geometry is a MultiPolygon (because
    # of the union operation). To make this easier to work with, I cross-section to get rid of the outer index level.
    # To future-proof, make sure that there is only one outer index - if so, remove it.
    assert len(df.index.get_level_values(0).unique()) == 1
    df = df.xs(key='polygon1')
    df = df.resample('H').interpolate(method='cubic')
    climatology = df.groupby([df.index.dayofyear, df.index.time]).mean()
    climatology.index.names = ['doy', 'time']
    climatology.to_csv(os.path.join('climatologies', '{}.csv'.format(location)))


@timer
def get_polygon_data(geometry):  # TODO could make parameters variable
    """
    Gets mean weather data for 2m temperature and 90m wind-speed (currently static) within polygon defined by geometry
    :param geometry: shapely.[Multi]Polygon
    :return: pandas.DataFrame of spatial mean time-series data for the polygon; time-series parameters defined in header
    """
    tuple_lists = make_tuple_lists(geometry)
    print('Fetching time-series data for the next {} days.'.format(lead_time_days))
    df = api.query_polygon(
        latlon_tuple_lists=tuple_lists,
        startdate=start_time,
        enddate=start_time + t_series_lead,
        interval=t_series_step,
        parameters=timeseries_vars.values(),
        aggregation=['mean'],
        username=username,
        password=password,
        operator='U',  # not necessary for the single polygon, but doesn't break and is necessary for multi-polygons
        model='ecmwf-vareps',
        polygon_sampling='adaptive_grid'
    )
    # Query returns MultiIndexed DataFrame, but outer index is polygon1 even if geometry is a MultiPolygon (because
    # of the union operation). To make this easier to work with, I cross-section to get rid of the outer index level.
    # To future-proof, make sure that there is only one outer index - if so, remove it.
    assert len(df.index.get_level_values(0).unique()) == 1
    df = df.xs(key='polygon1')
    return df


def make_tuple_lists(geometry):
    """
    Prepare the tuple list argument for API query from polygon data
    :param geometry: shapely.[Multi]Polygon
    :return: tuple lists (tuple list for each Polygon)
    """
    retval = []
    if type(geometry) is sgeometry.multipolygon.MultiPolygon:
        for polygon in geometry:  # this is the correct way to access Polygons within a MultiPolygon...
            retval.append(make_tuple_list(polygon))
    elif type(geometry) is sgeometry.polygon.Polygon:
        retval.append(make_tuple_list(geometry))  # ...but you can't loop over a single Polygon (which is dumb)
    else:
        # code should be written such that only [Multi]Polygons make it here; if imported modules change, will raise
        raise TypeError('geometries should contain only polygons or multipolygons')
    return retval


def make_tuple_list(polygon):
    """
    Subroutine of make_tuple_lists. Argument will always be a Polygon, since parent function loops over MultiPolygon
    :param polygon: shapely.Polygon
    :return: tuple list
    """
    tuple_list = []
    lons, lats = polygon.exterior.coords.xy
    for i in range(len(lons)):
        tuple_list.append((lats[i], lons[i]))
    return tuple_list


def get_combined_data(geometries, key, mins, maxs):
    """
    Reads the climatology and forecast data, combines them, writes the result to file and updates the min/max dicts.
    :param geometries: dict of shapely.[Multi]Polygons
    :param key: the country/region of interest
    :param mins: dict of minimum values for variables across all regions
    :param maxs: dict of maximum values for variables across all regions
    :return: (updated) dicts of minimum/maximum values for variables across all regions
    """
    clim = get_polygon_climatology(geometries, key)  # pass the key in order to name/lookup the file
    fcst = get_polygon_data(geometries[key])
    combined = combine(fcst, clim)
    for var in fcst.columns:
        mins[var] = min(mins[var], min(fcst[var]))
        maxs[var] = max(maxs[var], max(fcst[var]))
    os.makedirs('time-series', exist_ok=True)
    combined.to_csv(os.path.join('time-series', 'time-series data {}.csv'.format(key)))
    return mins, maxs


def combine(fcst, clim):
    """
    Joins the forecast and the corresponding climatology data together into one DataFrame.
    :param fcst:
    :param clim:
    :return:
    """
    relevant_clim = pd.DataFrame(index=fcst.index, columns=fcst.columns)
    for i in range(len(fcst)):
        tstamp = fcst.index[i]
        relevant_clim.loc[tstamp] = clim.loc[(tstamp.dayofyear, tstamp.time().strftime('%H:%M:%S'))]
    return pd.merge(fcst, relevant_clim, left_index=True, right_index=True, suffixes=('_forecast', '_climatology'))


def plot_timeseries(title, mins, maxs):
    """
    Generic function for plotting time-series data and anomaly from climatology of any variable (specified in header)
    :param title: the name of the region being plotted: will be the title of the plot and the name of the file
    :param mins: dictionary of global minimum values corresponding to each variable
    :param maxs: dictionary of global maximum values corresponding to each variable
    """
    df = pd.read_csv(os.path.join('time-series', 'time-series data {}.csv'.format(title)), index_col=0, parse_dates=True)
    for var in timeseries_vars:
        try:
            assert '(' in var
        except AssertionError:
            raise ValueError('Make sure the timeseries_vars dict is formatted correctly')
        mm_string = timeseries_vars[var]
        unit_index = len(var.split('(')[0])
        vmin = mins[mm_string] - (maxs[mm_string] - mins[mm_string]) * 0.1
        vmax = maxs[mm_string] + (maxs[mm_string] - mins[mm_string]) * 0.1
        f, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(12, 7))
        climatology_years = climatology_stop_year - climatology_start_year
        df['{}_climatology'.format(mm_string)].plot(label='{}-year mean'.format(climatology_years), ax=ax1, c='b')
        df['{}_forecast'.format(mm_string)].plot(label='forecast', ax=ax1, ylim=[vmin, vmax], c='k')
        pd.Series(np.zeros(len(df.index)), index=df.index).plot(ax=ax2, linestyle='dashed', c='b')
        (df['{}_forecast'.format(mm_string)] - df['{}_climatology'.format(mm_string)]).plot(ax=ax2, c='k')
        # plot formatting options
        f.suptitle('{}'.format(title))
        ax1.set_ylabel(var)
        ax1.legend()
        ax2.set_ylabel(var[:unit_index]+'Anomaly '+var[unit_index:])
        ax1.grid(True)
        ax2.grid(True)
        f.tight_layout()
        os.makedirs(os.path.join('time-series', '{}'.format(var.replace('/', ''))), exist_ok=True)
        plt.savefig(os.path.join('time-series', '{}'.format(var.replace('/', '')), '{}.png'.format(title)))
        plt.close(f)


@ timer
def make_nc():
    """
    A single API call to retrieve all this data is too big, so I loop over days in our forecast range and stack the
    grids together. Using query_grid_timeseries is still desirable for this, since it can be manipulated into an
    xarray.Dataset very easily.
    :return: xarray.Dataset containing 2m temperature and MSLP
    """
    for day in range(lead_time_days):
        query_start = start_time + dt.timedelta(days=day)
        df = api.query_grid_timeseries(
            startdate=query_start,
            enddate=query_start + dt.timedelta(hours=23), # ensures we don't double-count days
            interval=animation_step,
            parameters=['t_2m:C', 'msl_pressure:hPa'],
            lat_N=np.ceil(top_left['lat'] + 1),
            lon_W=np.floor(top_left['lon'] - 1),
            lat_S=np.floor(bottom_right['lat'] - 1),
            lon_E=np.ceil(bottom_right['lon'] + 1),
            res_lon=0.5,
            res_lat=0.5,
            username=username,
            password=password,
            model='ecmwf-vareps'
        )
        if day == 0:
            nc = df.to_xarray()
        else:
            tmp = df.to_xarray()
            nc = xr.concat([nc, tmp], dim='validdate')
    try:
        # TODO this currently fails because Meteomatics data is float64 and to_netcdf wants float32
        nc.to_netcdf('run_{}'.format(dt.datetime.now().strftime('%Y-%m-%d_%Hh00.nc')))
    except ValueError:
        pass
    return nc


def transform_coords(nc):
    """
    I found the suggested method of adding a transorm=crs keyword argument to plotting functions not to work with
    e.g. Azimuthal equidistant. Perhaps this is something to do with pcolor. Anyway, here's my solution: define a
    2D latitude and longitude variable which corresponds to the data variables; transform those and return them.
    :param nc: xarray dataset containing latitude and longitude coordinates
    :return: longitude and latitudes transformed to your chosen crs
    """
    # We first need latitude- and longitude arrays of equal size. Since we may be working with a map which has different
    # x- and y-dimensions, and also since we may not be mapping to a rectilinear coordinate system, we first have to
    # make 2D arrays of each
    meshed_lon, meshed_lat = np.meshgrid(nc.lon.values, nc.lat.values)
    # We can then transform these to our target crs. The source crs is PlateCarree() i.e. PlateCarree with default args
    # i.e. quadratic grid, as this is the coordinate system returned from the Meteomatics API.
    transformed_output = crs.transform_points(ccrs.PlateCarree(), meshed_lon, meshed_lat)
    # This gives us a 3D array of x, y, z; the latter will, for our purposes, be identically 0. We can subset this as
    transformed_lons = transformed_output[:, :, 0]
    transformed_lats = transformed_output[:, :, 1]
    # Note that these are again 2D (because the same latitudes are not used for all longitudes and vice versa in a
    # non-rectangular projection) and hence cannot be used as coordinates in an xarray.Dataset, but can be added as
    # additional data variables if you like.
    # This produces lat/lon fields for each element of the weather variables which allow them to be plotted in this
    # projection. The process needn't be repeated for each time-step since we assume that the spatial domain is static.
    return transformed_lons, transformed_lats


def make_frames(nc, lons, lats, animation_path):
    """
    Make all the individual images which will comprise the final gif.
    :param nc: the netCDF containing the data
    :param lons: the 2D grid of longitudes transformed to the crs != nc.lon.values
    :param lats: the 2D grid of latitudes transformed to the crs != nc.lat.values
    :param animation_path: location in which to save the frames
    :return:
    """
    # one advantage of having built a Dataset of all the values we're going to animate is that I can now access
    # the global min and max of all the data for our colorbar/contour levels
    mslp_min = nc['msl_pressure:hPa'].min()
    mslp_max = nc['msl_pressure:hPa'].max()
    contour_levels = np.arange(np.floor(mslp_min), np.ceil(mslp_max), 2)
    t2m_min = nc['t_2m:C'].min()
    t2m_max = nc['t_2m:C'].max()

    os.makedirs(animation_path, exist_ok=True)
    for fle in glob.glob(os.path.join(animation_path, '*')):
        os.remove(fle)  # remove all the previous animation bits and pieces

    # TODO various bits of formatting can be done
    # this will save a bunch of static images, which can of course be giffed
    for t_step in range(len(nc.validdate)):
        fig = plt.figure()
        ax = fig.add_subplot(projection=crs)
        ax.set_extent([top_left['lon'], bottom_right['lon'], top_left['lat'], bottom_right['lat']])
        nc_step = nc.isel(validdate=t_step)
        cbar_map = ax.pcolor(lons, lats, nc_step['t_2m:C'], vmin=t2m_min, vmax=t2m_max)
        contours = ax.contour(lons, lats, nc_step['msl_pressure:hPa'], levels=contour_levels, colors='black')
        ax.clabel(contours, contours.levels[::5])
        ax.coastlines()
        plt.colorbar(cbar_map)
        plt.title(pd.to_datetime(nc.validdate[t_step].values).strftime('%Y-%m-%d %Hh00'))
        plt.savefig(os.path.join(animation_path, '{}_{}'.format(
            str(t_step).zfill(4),  # include this so that order is preserved even if we cross into January
            pd.to_datetime(nc.validdate[t_step].values).strftime('%Y-%m-%d %Hh00.png')
        )))
        plt.close(fig)


def animate(animation_path='animation'):
    """
    Makes an animation of the temperature and pressure situation over the forecast time- and region specified.
    All arguments other than the name of the animation directory are defined in the preamble.
    :param animation_path:
    :return:
    """
    nc = make_nc()
    lons, lats = transform_coords(nc)
    make_frames(nc, lons, lats, animation_path)

    # I'd have liked this to be a FuncAnimation with a slider for time control, but that seems complicated when
    # including multiple functions on a single frame (as I do with pcolor and contour) so I cheat by making a GIF
    # out of multiple images per https://bit.ly/3kRZOmB, and please check out my question https://bit.ly/3nwhNAT
    # if you have suggestions on how to improve the readability of the GIF"
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(os.path.join(animation_path, '*.png')))]
    # 'duration' (below) means duration of each frame in milliseconds
    img.save(fp=os.path.join(animation_path, 'animation.gif'), format='GIF', append_images=imgs, save_all=True,
             duration=200, loop=0)


if __name__ == '__main__':
    # get the geometries of the polygons for which we want time-series plots
    geometries = get_geometries(
        standard_countries=standard_countries,
        manual_geometries=manual_geometries,
        show=show_polygons
    )

    # for each of those polygons, write the time-series data and get a global min/max for each variable
    mins = {var: np.inf for var in timeseries_vars.values()}
    maxs = {var: -np.inf for var in timeseries_vars.values()}
    for key in geometries:
        mins, maxs = get_combined_data(geometries, key, mins, maxs)

    # plot the time-series using these global values
    for key in geometries:
        plot_timeseries(key, mins, maxs)

    # now make the GIF
    animate('animation')  # all the other variables for this are controlled in the preamble
