from .transformer import transform_tests


def load_ipython_extension(ipython):
    ipython.register_magics(transform_tests)