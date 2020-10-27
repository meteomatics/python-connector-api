# -*- coding: utf-8 -*-

from collections import defaultdict


class WeatherApiException(Exception):
    pass


class BadRequest(WeatherApiException):
    pass


class Unauthorized(WeatherApiException):
    pass


class PaymentRequired(WeatherApiException):
    pass


class Forbidden(WeatherApiException):
    pass


class NotFound(WeatherApiException):
    pass


class RequestTimeout(WeatherApiException):
    pass


class PayloadTooLarge(WeatherApiException):
    pass


class UriTooLong(WeatherApiException):
    pass


class TooManyRequests(WeatherApiException):
    pass


class InternalServerError(WeatherApiException):
    pass


_exceptions = {
    400: BadRequest,
    401: Unauthorized,
    403: Forbidden,
    404: NotFound,
    408: RequestTimeout,
    413: PayloadTooLarge,
    414: UriTooLong,
    429: TooManyRequests,
    500: InternalServerError
}

API_EXCEPTIONS = defaultdict(lambda: WeatherApiException, _exceptions)
