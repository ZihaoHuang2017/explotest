import IPython
from IPython.core.magic import register_line_magic

from explotest.explotest import transform_tests_wrapper


def load_ipython_extension(ipython: IPython.InteractiveShell):
    register_line_magic(transform_tests_wrapper(ipython))
