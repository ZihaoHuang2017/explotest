import argparse
import ast
import sys
from pathlib import Path

from IPython.terminal.interactiveshell import TerminalInteractiveShell

if __name__ == "__main__":
    ipython = TerminalInteractiveShell()
    parser = argparse.ArgumentParser()
    parser.add_argument("input_filename")
    parser.add_argument(
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
    parser.add_argument(
        "-v",
        dest="verbose",
        action="store_true",
        help="""
        VERBOSE: If set, then the program will try to expand the test case into
        individual assertions; if False, then the whole list/dict/tuple will be asserted at once.
        """,
    )
    parser.add_argument(
        "-d",
        dest="dest",
        default=f"./test_resources",
        help="""
        The location that the pickled arguments will go.
        """,
    )

    args, other_args = parser.parse_known_args()
    if not Path(args.input_filename).exists():
        print("This file does not exist!")
        sys.exit(1)
    infile = open(Path(args.input_filename), "r", encoding="utf-8")
    end_index = len(sys.argv)
    for index, arg in enumerate(sys.argv):
        if arg == "--":
            end_index = index
    extracted_args = sys.argv[2:end_index]
    if len(other_args) > 0:
        other_args[0] = args.filename  # Simulate the passed in argv
        ipython.run_cell(
            f"import sys\nsys.argv = {other_args}", store_history=True
        )  # We no longer need argv
    statements: list[ast.stmt] = ast.parse(infile.read()).body
    for statement in statements:
        match statement:
            case ast.If(
                test=ast.Compare(
                    left=ast.Name(id="__name__"),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(value="__main__")],
                )
            ) as if_statement:
                for subStatement in if_statement.body:
                    ipython.run_cell(ast.unparse(subStatement), store_history=True)
            case _:
                ipython.run_cell(ast.unparse(statement), store_history=True)

    ipython.run_line_magic("reload_ext", "explotest")
    ipython.run_line_magic("transform_tests", " ".join(extracted_args))
