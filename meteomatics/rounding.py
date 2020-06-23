# -*- coding: utf-8 -*-

decimal_places = {
    "bft":      0,  # Beauforts (wind)
    "cm":      -1,  # centimeters (snow height, hail)
    "C":        1,  # temperature
    "d":       -1,  # days AND degrees...
    "dn":       5,  # datenums
    "eur":      1,  # euro
    "F":        1,  # temperature
    "ft":       1,  # feet (ceiling height)
    "gcm2":     4,  # total column air pollution
    "gm3":      1,  # absolute humidity
    "gdd":      1,  # growind degree days
    "grainsm3": 1,  # grains per m3 (pollen)
    "hdd":      1,  # heating degree days
    "h":        1,  # hours
    "hft":      1,  # hectofeet
    "hPa":      1,  # pressures
    "Hz":       6,  # vorticity
    "mHz":      3,  # vorticity
    "uHz":      1,  # vortictiy
    "Pa":       0,  # pressures
    "idx":     -1,  # index
    "J":        1,  # energy
    "K":        1,  # temperature
    "kA":       1,  # kilo Amperes for lightning strokes
    "kgkg":     5,  # kg/kg for mixing ratio
    "kgm2":    -1,  # kg/m2 (snow melt & super cooled liquid water, total column air pollution)
    "kgm3":    -1,  # density
    "km":       1,  # km (e.g. visibility)
    "kmh":      1,  # wind speed
    "kn":       1,  # wind speed
    "kW":       1,  # power
    "Jkg":      3,  # cape
    "l":        1,  # liters (discharge)
    "m":       -1,  # meters (elevation, roughness length)
    "m3":       2,  # cubic meters (discharge)
    "m3m3":     2,  # fraction (volumetric soil moisture)
    "m3s":      9,  # discharge
    "min":      1,  # minutes
    "mm":      -1,  # millimeters (precipitation)
    "mph":      1,  # wind speed
    "MW":       3,  # power
    "ms":       1,  # wind speed
    "nms":      3,  # runoff rate
    "octas":    0,  # clouds
    "p":        1,  # percent
    "Pas":      2,  # vertical wind speed Pas / s
    "psu":      1,  # psu (for water salinity)
    "s":        1,  # seconds
    "ugm2":     3,  # air pollution
    "ugm3":     3,  # air pollution
    "ux":       1,  # unix time stamps
    "W":        1,  # power
    "Ws":       1,  # energy
    "Wh":       3,  # energy
    "x":        0,  # number
    "y":        0,  # year
}


def get_num_decimal_places(parameter):
    try:
        _, unit = parameter.split(":")
        if "-" in unit:
            unit = unit.split("-")[0]
    except ValueError:
        return 3

    try:
        number_of_decimals = decimal_places[unit]
    except KeyError:
        return 3

    if number_of_decimals >= 0:
        return number_of_decimals

    if unit == "cm":
        if parameter.find("hail") == 0:
            return 0
        return 1

    elif unit == "mm":
        if "surface_runoff_1h" in parameter:
            return 3
        else:
            return 2

    elif unit == "idx":
        if parameter in ["phytophthora_negative:idx", "phytophthora_negative_prognose:idx",
                         "fosberg_fire_weather_index:idx", "santa_ana_wind:idx", "north_atlantic_oscillation:idx",
                         "forest_fire_warning:idx", "pollen_warning:idx"]:
            return 1

        elif parameter.find("icing_potential") == 0 or parameter.find("sld_potential") == 0 or \
                parameter.find("uv_bio_effective_dose") == 0 or parameter.find("leisure_") == 0:
            return 2

        elif parameter == "moon_phase:idx":
            return 3

        elif parameter.find("soil_moisture_index") == 0 or parameter.find("soil_moisture_deficit:idx") == 0:
            return 3

        elif "_aod" in parameter:
            return 3

        elif parameter.find("cloud_layer") == 0:
            return 5

        elif parameter.find("precip_layer") == 0:
            return 2

        elif parameter.find("sat_") != -1:
            if parameter.find("cloud_type", 4) != -1:
                return 0
            elif parameter.find("ndvi", 4) != -1:
                return 2
            else:
                return 5

        return 0
    elif unit == "d":
        if "lats" in parameter or "lons" in parameter:
            return 5
        for marker in ["dir", "days", "nights", "moon_age"]:
            if marker in parameter:
                return 1

    elif unit == "kgm2":
        if "tc_" in parameter:
            return 3
        else:
            return 1

    elif unit == "kgm3":
        if "air_density" in parameter:
            return 3
        else:
            return 1

    elif unit == "m":
        if parameter == "roughness_length:m":
            return 3
        elif "depth:m" in parameter:
            return 2
        else:
            return 1

    return 3


def round_df(df):
    for column in df.columns:
        if not column.endswith(":sql"):
            sig_num = get_num_decimal_places(column)
            df[column] = df[column].round(sig_num)
    return df
