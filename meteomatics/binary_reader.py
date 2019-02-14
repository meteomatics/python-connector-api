# -*- coding: utf-8 -*-

import struct


class BinaryReaderException(Exception):
    def __init__(self, message):
        super(BinaryReaderException, self).__init__(message)


class BinaryReader(object):
    DATA_TYPES = {
        "double": {"format_char": "d", "num_of_bytes": 8},
        "float": {"format_char": "f", "num_of_bytes": 4},
        "int": {"format_char": "i", "num_of_bytes": 4},
        "char": {"format_char": "c", "num_of_bytes": 1},
        "unsigned_long": {"format_char": "Q", "num_of_bytes": 8}
    }

    def __init__(self, binary_data):
        self._binary_data = bytearray(bytes(binary_data))
        self._pointer = 0

    def __str__(self):
        return str(self._binary_data)

    def __len__(self):
        return len(self._binary_data[self._pointer:])

    def _read_data(self, data_type, n):
        try:
            num_of_bytes = self.DATA_TYPES[data_type]["num_of_bytes"]
        except KeyError as e:
            raise BinaryReaderException("Data type: '{}' is not valid. Valid types: [{}]".format(data_type,
                                                                                                 ", ".join(
                                                                                                     self.DATA_TYPES.keys())))

        substring = self._binary_data[self._pointer:self._pointer + num_of_bytes * n]

        if len(substring) != num_of_bytes * n:
            raise BinaryReaderException(
                "Not possible to read {} bytes, length of binary data is {} bytes".format(num_of_bytes * n,
                                                                                          len(self)))
        self._pointer += num_of_bytes * n
        return struct.unpack("<{}".format(self.DATA_TYPES[data_type]["format_char"] * n), substring)

    def get(self, data_type, n=1):
        try:
            data_type = data_type.lower()
        except AttributeError:
            raise BinaryReaderException("Argument data_type has to be string")

        data = self._read_data(data_type, n)
        if n == 1:
            return data[0]
        else:
            return data

    def get_int(self, n=1):
        return self.get("int", n)

    def get_double(self, n=1):
        return self.get("double", n)

    def get_char(self, n=1):
        return self.get("char", n)

    def get_float(self, n=1):
        return self.get("float", n)

    def get_unsigned_long(self, n=1):
        return self.get("unsigned_long", n)

    def get_string(self, length):
        if length > len(self):
            raise BinaryReaderException(
                "Not possible to read string with length {}, length of binary data is {}".format(length,
                                                                                                 len(self)))
        data = self.get("char", length)
        return "".join([c.decode("utf-8") for c in data])
