"""The LibFM factorization library."""
try:
    from bin import pyfm
except ImportError as error:
    raise "Could not find pyfm package. You probably need to run make in the libfm directory."
