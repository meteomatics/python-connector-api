# -*- coding: utf-8 -*-

import pandas as pd

from .binary_reader import BinaryReader
from .parsing_util import all_entries_postal, parse_date_num, localize_datenum

class BinaryParser(object):
    
    def __init__(self, binary_reader, na_values):
        self.binary_reader = binary_reader
        self.na_values = na_values

    def parse(self, parameters, is_station=False, coordinate_list=None):
        is_postal = all_entries_postal(coordinate_list)
        if is_station:
            return self._parse_station(parameters[:], coordinate_list)
        elif is_postal:
            return self._parse_postal(parameters[:], coordinate_list)
        else:
            return self._parse_latlon(parameters[:], coordinate_list)
    
    def _parse_station(self, parameters, coordinate_list):
        parameters.extend(["station_id"])
        return self._parse_internal(parameters, 'station', coordinate_list)

    def _parse_postal(self, parameters, coordinate_list):
        parameters.extend(['postal_code'])
        return self._parse_internal(parameters, 'postal', coordinate_list)

    def _parse_latlon(self, parameters, coordinate_list):
        # add lat, lon in the list of parameters
        parameters.extend(["lat", "lon"])
        return self._parse_internal(parameters, 'latlon', coordinate_list)
        
    
    def _parse_internal(self, parameters, parse_type, coordinate_list):
        dfs = []
        # parse response
        num_of_coords = self.binary_reader.get_int() if len(coordinate_list) > 1 else 1

        for i in range(num_of_coords):
            dict_data = {}
            num_of_dates = self.binary_reader.get_int()

            for _ in range(num_of_dates):
                num_of_params = self.binary_reader.get_int()
                date = self.binary_reader.get_double()
                if parse_type == 'station':
                    latlon = [coordinate_list[i]]
                elif parse_type == 'postal':
                    latlon = [coordinate_list[i]]
                else:
                    latlon = coordinate_list[i]
                # ensure tuple
                latlon = tuple(latlon)

                value = self.binary_reader.get_double(num_of_params)
                if type(value) is not tuple:
                    value = (value,)
                dict_data[date] = value + latlon

            df = pd.DataFrame.from_dict(dict_data, orient="index", columns=parameters)
            df = df.sort_index()
            dfs.append(df)

        df = pd.concat(dfs)
        df = df.replace(self.na_values, float('NaN'))
        df.index.name = "validdate"

        df.index = parse_date_num(df.reset_index()["validdate"])

        # mark index as UTC timezone
        df = localize_datenum(df)
        return df