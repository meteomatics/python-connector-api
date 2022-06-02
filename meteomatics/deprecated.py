import warnings


def deprecated(message):
  """https://stackoverflow.com/a/48632082
      This is a decorator which can be used to mark functions
      as deprecated. It will result in a warning being emitted
      when the function is used."""
  def deprecated_decorator(func):
      def deprecated_func(*args, **kwargs):
          warnings.warn("{} is a deprecated function. {}".format(func.__name__, message),
                        category=DeprecationWarning,
                        stacklevel=2)
          warnings.simplefilter('default', DeprecationWarning)
          return func(*args, **kwargs)
      return deprecated_func
  return deprecated_decorator
