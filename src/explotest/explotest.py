import ast
import builtins
import os
import sys
from io import open

import IPython
from IPython.core.error import StdinNotImplementedError
from IPython.core.magic_arguments import magic_arguments, argument, parse_argstring
from IPython.utils import io

from .carver import Carver
from .constants import INDENT_SIZE
from .generate_tests import generate_tests, add_call_string
from .utils import revise_line_input


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
        """
    )
    def transform_tests(parameter_s=""):
        args = parse_argstring(transform_tests, parameter_s)
        outfname = args.filename
        if not outfname:
            outfile = sys.stdout  # default
            # We don't want to close stdout at the end!
            close_at_end = False
        else:
            outfname = os.path.expanduser(outfname)
            if os.path.exists(outfname):
                try:
                    ans = io.ask_yes_no("File %r exists. Overwrite?" % outfname)
                except StdinNotImplementedError:
                    ans = True
                if not ans:
                    print("Aborting.")
                    return
                print("Overwriting file.")
            outfile = open(outfname, "w", encoding="utf-8")
            close_at_end = True

        import_statements = set()
        normal_statements = []
        output_lines = [0, 0, 0]
        original_print = builtins.print
        histories = ipython.history_manager.get_range(output=True)
        for session, line, (lin, lout) in histories:
            ipython.builtin_trap.remove_builtin("print", original_print)
            ipython.builtin_trap.add_builtin(
                "print", return_hijack_print(original_print)
            )
            try:
                if (
                    lin.startswith("%")
                    or lin.endswith("?")
                    or lin.startswith("get_ipython()")
                ):  # magic methods
                    continue
                if lin.startswith("from ") or lin.startswith("import "):
                    import_statements.add(lin)
                    continue
                revised_statement = revise_line_input(lin, output_lines)
                if lout is None:
                    parsed_in = ast.parse(revised_statement[-1]).body[0]
                    with Carver(parsed_in, ipython, args.verbose) as carver:
                        for stmt in revised_statement:
                            ipython.ex(stmt)
                    normal_statements.extend(revised_statement)

                    for call_stat in carver.call_statistics(
                        carver.desired_function_name
                    ):
                        if (
                            carver.desired_function_name
                            not in carver.called_function_name.split(".")
                            and call_stat.appendage != []
                        ):
                            import_statements.add(
                                f"from {carver.module.__name__} import {carver.desired_function_name}"
                            )
                            exec("import dill as pickle", ipython.user_global_ns, ipython.user_ns)
                            call_string, setup = add_call_string(
                                carver.desired_function_name, call_stat, ipython, args.dest, line
                            )
                            if setup:
                                import_statements.add("import dill as pickle")
                                normal_statements.extend(setup)
                            normal_statements.append("ret = " + call_string)
                        normal_statements.extend(call_stat.appendage)
                    # not the most ideal way if we have some weird crap going on (remote apis???)
                    continue
                output_lines.append(line)
                var_name = f"_{line}"
                for index in range(len(revised_statement) - 1):
                    ipython.ex(revised_statement[index])
                normal_statements.extend(revised_statement[:-1])
                obj_result = ipython.ev(revised_statement[-1])
                normal_statements.append(f"{var_name} = {revised_statement[-1]}")
                normal_statements.extend(
                    generate_tests(obj_result, var_name, ipython, args.verbose)
                )
            except (SyntaxError, NameError) as e:
                raise e
                continue
            # except Exception as e:
            #     import_statements.add("import pytest")
            #     normal_statements.append(f"with pytest.raises({type(e).__name__}):")
            #     normal_statements.append(" " * INDENT_SIZE + lin)
            #     continue
        for statement in import_statements:
            lines = statement.split("\n")
            for line in lines:
                print(line, file=outfile)
        print("\n", file=outfile)
        print("def test_func():", file=outfile)
        for statement in normal_statements:
            lines = statement.split("\n")
            for line in lines:
                print(" " * INDENT_SIZE + line, file=outfile)
        if close_at_end:
            outfile.close()

    return transform_tests


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


