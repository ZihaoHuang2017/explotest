import ast
import builtins
import logging
import sys
from io import open
from pathlib import Path

import IPython
from IPython.core.error import StdinNotImplementedError
from IPython.core.magic_arguments import magic_arguments, argument, parse_argstring
from IPython.utils import io

from .carver import Carver, add_call_string
from .constants import INDENT_SIZE
from .generate_tests import generate_tests
from .incrementor import Incrementor
from .utils import revise_line_input, has_bad_repr, PredicateType


logger = logging.getLogger(__name__)


def transform_tests_wrapper(ipython: IPython.InteractiveShell):
    @magic_arguments()
    @argument(
        "-f",
        dest="filename",
        help="""
        FILENAME: instead of printing the output to the screen, redirect
        it to the given file.  The file is always overwritten, though *when
        it can*, IPython asks for confirmation first. In particular, running
        the command 'history -f FILENAME' from the IPython Notebook
        interface will replace FILENAME even if it already exists *without*
        confirmation.
        """,
    )
    @argument(
        "-v",
        dest="verbose",
        action="store_true",
        help="""
        VERBOSE: If set to True, then the program will try to expand the test case into 
        individual assertions; if False, then the whole list/dict/tuple will be asserted at once.
        """,
    )
    @argument(
        "-d",
        dest="dest",
        default=f"./test_resources",
        help="""
        The location that the pickled arguments will go.
        """,
    )
    def transform_tests(parameter_s=""):
        args = parse_argstring(transform_tests, parameter_s)
        transform_tests_outer(ipython, args.filename, args.verbose, args.dest)

    return transform_tests


def transform_tests_outer(ipython: IPython.InteractiveShell, filename, verbose, dest):
    logging.basicConfig(filename="explotest.log", level=logging.INFO)
    logger.info(f"{filename}, {verbose}, {dest}")
    if not filename:
        outfile = sys.stdout  # default
        # We don't want to close stdout at the end!
        close_at_end = False
    else:
        if Path(filename).exists():
            try:
                ans = io.ask_yes_no(f"File {filename} exists. Overwrite?")
            except StdinNotImplementedError:
                ans = True
            if not ans:
                print("Aborting.")
                return
            print("Overwriting file.")
        outfile = open(Path(filename), "w", encoding="utf-8")
        close_at_end = True
    if Path(dest).exists():
        try:
            ans = io.ask_yes_no(
                f"Dest folder {dest} exists. Proceed? (Will potentially override content)"
            )
        except StdinNotImplementedError:
            ans = True
        if not ans:
            print("Aborting. Specify a folder for test resource using -d")
            return
        print("Overwriting directory.")
    import_statements = set()
    normal_statements = []
    output_lines = [0, 0, 0]
    original_print = builtins.print
    histories = ipython.history_manager.get_range(output=True, raw=False)
    nondeterministic_counter = 0
    ipython.builtin_trap.remove_builtin("print", original_print)
    ipython.builtin_trap.add_builtin("print", return_hijack_print(original_print))
    for session, line_number, (lin, lout) in histories:
        logger.info(f"Processing line {line_number}, line {lin}, out {lout}")
        try:
            if lin.startswith("get_ipython()"):  # magic methods
                continue
            if lin.startswith("from ") or lin.startswith("import "):
                import_statements.add(lin)
                continue
            revised_statements = revise_line_input(lin, output_lines)
            if lout is None:
                for statement in revised_statements:
                    normal_result, import_result = execute_parsed_statement(
                        ipython, statement, verbose, dest, line_number
                    )
                    normal_statements.extend(normal_result)
                    import_statements |= import_result
                continue
            output_lines.append(line_number)
            var_name = f"_{line_number}"
            for index in range(len(revised_statements) - 1):
                normal_result, import_result = execute_parsed_statement(
                    ipython, revised_statements[index], verbose, dest, line_number
                )
                normal_statements.extend(normal_result)
                import_statements |= import_result
            normal_statements.extend(revised_statements[:-1])
            obj_result = ipython.ev(revised_statements[-1])
            normal_statements.append(f"{var_name} = {revised_statements[-1]}")
            if not has_bad_repr(lout) and repr(obj_result) != lout:
                nondeterministic_counter += 1
                print(f"nondeterministic {repr(obj_result)} encountered {repr(lout)}")
            else:
                logger.info(f"obj generated for line {line_number}")
                normal_statements.extend(
                    generate_tests(obj_result, var_name, ipython, verbose)
                )
        except (SyntaxError, NameError) as e:
            print(e)
            # raise e
            continue
        except SystemExit:
            logger.info(f"System exited")
            pass
        except Exception as e:
            logger.info(f"Exception encountered {e}")
            import_statements.add("import pytest")
            normal_statements.append(f"with pytest.raises({type(e).__name__}):")
            normal_statements.append(" " * INDENT_SIZE + lin)
            continue
    print(f"{nondeterministic_counter} nondeterministic objects encountered in total")
    for statement in import_statements:
        lines = statement.split("\n")
        for line_number in lines:
            print(line_number, file=outfile)
    print("\n", file=outfile)
    print("def test_func():", file=outfile)
    for statement in normal_statements:
        lines = statement.split("\n")
        for line_number in lines:
            print(" " * INDENT_SIZE + line_number, file=outfile)
    if close_at_end:
        outfile.close()


def execute_parsed_statement(
    ipython, statement, verbose, dest, line_number
) -> tuple[list, set]:
    parsed_in = ast.parse(statement).body[0]
    incrementor = Incrementor()
    with Carver(parsed_in, ipython, verbose) as carver:
        exec_result = sys_exec(ipython, statement)
    if carver.desired_function is None:
        return exec_result
    normal_result, import_result = exec_result
    for call_stat in carver.call_statistics(carver.desired_function.__qualname__):
        if (
            carver.desired_function == carver.called_function
            or call_stat.appendage == []
        ):
            normal_result.extend(call_stat.appendage)
            continue
        import_str = get_import_str(call_stat.predicate, carver.desired_function)
        import_result.add(
            f"from {carver.desired_function.__module__} import {import_str}"
        )

        call_string, pickle_setup = add_call_string(
            carver.desired_function.__name__,
            call_stat,
            ipython,
            dest,
            line_number,
            incrementor,
        )
        if pickle_setup:
            import_result.add("import dill as pickle")
            normal_result.extend(pickle_setup)
        normal_result.append("ret = " + call_string)
        normal_result.extend(call_stat.appendage)
    # not the most ideal way if we have some weird crap going on (remote apis???)
    return normal_result, import_result


def get_import_str(predicate, desired_function):
    if predicate == PredicateType.NONE:
        return desired_function.__name__
    elif predicate == PredicateType.OBJECT:
        return type(desired_function.__self__).__name__
    else:
        return desired_function.__qualname__.split(".")[0]


def return_hijack_print(original_print):
    def hijack_print(
        *values: object,
        sep: str | None = " ",
        end: str | None = "\n",
        file=None,
        flush=False,
    ):
        original_print(*values, sep=sep, end=end, file=file, flush=flush)

    return hijack_print


def sys_exec(ipython, statement):
    try:
        ipython.ex(statement)
        return [statement], set()  # normal
    except SystemExit:
        return [], set()  # system run
