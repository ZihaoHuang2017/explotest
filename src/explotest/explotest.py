import ast
import builtins
import dataclasses
import enum
import os
import sys
import typing
from io import open

import IPython
from IPython.core.error import StdinNotImplementedError
from IPython.core.magic_arguments import magic_arguments, argument, parse_argstring
from IPython.utils import io

from .carver import call_string, Carver

INDENT_SIZE = 4
EXPLORATORY_PREFIX = "--explore"


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
                            import_statements.add("import pickle")
                            normal_statements.append(
                                "ret = "
                                + call_string(
                                    carver.desired_function_name, call_stat, ipython
                                )
                            )
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
                # raise e
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


def generate_tests(obj: any, var_name: str, ipython, verbose: bool) -> list[str]:
    if verbose:
        result = generate_verbose_tests(obj, var_name, dict(), ipython)
    else:
        representation, assertions = generate_concise_tests(
            obj, var_name, dict(), True, ipython
        )
        result = assertions
    if len(result) <= 20:  # Arbitrary
        return result
    proper_string_representation = str(obj).replace("\n", "\\n")
    return [
        f'assert str({var_name}) == "{proper_string_representation}"'
    ]  # Too lengthy!


def generate_verbose_tests(
    obj: any, var_name: str, visited: dict[int, str], ipython: IPython.InteractiveShell
) -> list[str]:
    """Parses the object and generates verbose tests.

    We are only interested in the top level assertion as well as the objects that can't be parsed directly,
    in which case it is necessary to compare the individual fields.

    Args:
        obj (any): The object to be transformed into tests.
        var_name (str): The name referring to the object.
        visited (dict[int, str]): A dict associating the obj with the var_names. Used for cycle detection.
        ipython (IPython.InteractiveShell):  bruh

    Returns:
        list[str]: A list of assertions to be added.

    """
    if obj is True:
        return [f"assert {var_name}"]
    if obj is False:
        return [f"assert not {var_name}"]
    if obj is None:
        return [f"assert {var_name} is None"]
    if type(type(obj)) is enum.EnumMeta and is_legal_python_obj(
        type(obj).__name__, type(obj), ipython
    ):
        return [f"assert {var_name} == {str(obj)}"]
    if type(obj) is type:
        class_name = obj.__name__
        if is_legal_python_obj(class_name, obj, ipython):
            return [f"assert {var_name} is {class_name}"]
        else:
            return [f'assert {var_name}.__name__ == "{class_name}"']
    if is_legal_python_obj(repr(obj), obj, ipython):
        return [f"assert {var_name} == {repr(obj)}"]
    if id(obj) in visited:
        return [f"assert {var_name} == {visited[id(obj)]}"]
    visited[id(obj)] = var_name
    result = [get_type_assertion(obj, var_name, ipython)]
    if isinstance(obj, typing.Sequence):
        for idx, val in enumerate(obj):
            result.extend(
                generate_verbose_tests(val, f"{var_name}[{idx}]", visited, ipython)
            )
    elif type(obj) is dict:
        for key, value in obj.items():
            result.extend(
                generate_verbose_tests(value, f'{var_name}["{key}"]', visited, ipython)
            )
    else:
        attrs = dir(obj)
        for attr in attrs:
            if not attr.startswith("_"):
                value = getattr(obj, attr)
                if not callable(value):
                    result.extend(
                        generate_verbose_tests(
                            value, f"{var_name}.{attr}", visited, ipython
                        )
                    )
    return result


def generate_concise_tests(
    obj: any,
    var_name: str,
    visited: dict[int, str],
    propagation: bool,
    ipython: IPython.InteractiveShell,
) -> tuple[str, list[str]]:
    """Parses the object and generates concise tests.

    We are only interested in the top level assertion as well as the objects that can't be parsed directly,
    in which case it is necessary to compare the individual fields.

    Args:
        obj (any): The object to be transformed into tests.
        var_name (str): The name referring to the object.
        visited (dict[int, str]): A dict associating the obj with the var_names. Used for cycle detection.
        propagation (bool): Whether the result should be propagated.
        ipython (IPython.InteractiveShell):  bruh

    Returns:
        tuple[str, list[str]]: The repr of the obj if it can be parsed easily, var_name if it can't, and a list of
    """
    # readable-repr, assertions
    if type(type(obj)) is enum.EnumMeta and is_legal_python_obj(
        type(obj).__name__, type(obj), ipython
    ):
        if propagation:
            return str(obj), [f"assert {var_name} == {str(obj)}"]
        return str(obj), []
    if is_legal_python_obj(repr(obj), obj, ipython):
        if propagation:
            return repr(obj), generate_verbose_tests(
                obj, var_name, visited, ipython
            )  # to be expanded
        return repr(obj), []
    if id(obj) in visited:
        return var_name, [f"assert {var_name} == {visited[id(obj)]}"]
    visited[id(obj)] = var_name
    if isinstance(obj, typing.Sequence):
        reprs, overall_assertions = [], []
        for idx, val in enumerate(obj):
            representation, assertions = generate_concise_tests(
                val, f"{var_name}[{idx}]", visited, False, ipython
            )
            reprs.append(representation)
            overall_assertions.extend(assertions)
        if type(obj) is tuple:
            repr_str = f'({", ".join(reprs)})'
        else:
            repr_str = f'[{", ".join(reprs)}]'
        if propagation:
            overall_assertions.insert(0, f"assert {var_name} == {repr_str}")
        return repr_str, overall_assertions
    elif type(obj) is dict:
        reprs, overall_assertions = [], []
        for field, value in obj.items():
            representation, assertions = generate_concise_tests(
                value, f'{var_name}["{field}"]', visited, False, ipython
            )
            reprs.append(f'"{field}": {representation}')
            overall_assertions.extend(assertions)
        repr_str = "{" + ", ".join(reprs) + "}"
        if propagation:
            overall_assertions.insert(0, f"assert {var_name} == {repr_str}")
        return repr_str, overall_assertions
    elif dataclasses.is_dataclass(obj):
        reprs, overall_assertions = [], []
        for field in dataclasses.fields(obj):
            representation, assertions = generate_concise_tests(
                getattr(obj, field.name),
                f"{var_name}.{field.name}",
                visited,
                False,
                ipython,
            )
            reprs.append(f'"{field.name}": {representation}')
            overall_assertions.extend(assertions)
        repr_str = "{" + ", ".join(reprs) + "}"
        if propagation:
            overall_assertions.insert(0, f"assert {var_name} == {repr_str}")
        return repr_str, overall_assertions
    else:
        overall_assertions = [get_type_assertion(obj, var_name, ipython)]
        attrs = dir(obj)
        for attr in attrs:
            if not attr.startswith("_"):
                value = getattr(obj, attr)
                if not callable(value):
                    _, assertions = generate_concise_tests(
                        value, f"{var_name}.{attr}", visited, True, ipython
                    )
                    overall_assertions.extend(assertions)
        return var_name, overall_assertions


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


def get_type_assertion(
    obj: any, var_name: str, ipython: IPython.InteractiveShell
) -> str:
    class_name = type(obj).__name__
    if is_legal_python_obj(class_name, type(obj), ipython):
        return f"assert type({var_name}) is {class_name}"
    else:
        return f'assert type({var_name}).__name__ == "{class_name}"'


def is_legal_python_obj(
    statement: str, obj: any, ipython: IPython.InteractiveShell
) -> bool:
    try:
        return obj == ipython.ev(statement)
    except (SyntaxError, NameError):
        return False


def is_builtin_obj(obj: any) -> bool:
    if type(obj) in [int, str, bool, float, complex]:
        return True
    if type(obj) in [dict, tuple, set, frozenset]:
        return all(is_builtin_obj(item) for item in obj)
    return False


class RewriteUnderscores(ast.NodeTransformer):
    def __init__(self, one_underscore, two_underscores, three_underscores):
        self.one_underscore = one_underscore
        self.two_underscores = two_underscores
        self.three_underscores = three_underscores

    def visit_Name(self, node):
        if node.id == "_":
            return ast.Name(id=f"_{self.one_underscore}", ctx=ast.Load())
        elif node.id == "__":
            return ast.Name(id=f"_{self.two_underscores}", ctx=ast.Load())
        elif node.id == "___":
            return ast.Name(id=f"_{self.three_underscores}", ctx=ast.Load())
        else:
            return node


def revise_line_input(lin: str, output_lines: list[str]) -> list[str]:
    # Undefined Behaviour if the user tries to invoke _ with len < 3. Why would you want to do that?
    one_underscore, two_underscores, three_underscores = (
        output_lines[-1],
        output_lines[-2],
        output_lines[-3],
    )
    node = ast.parse(lin)
    revised_node = RewriteUnderscores(
        one_underscore, two_underscores, three_underscores
    ).visit(node)
    return [ast.unparse(stmt) for stmt in revised_node.body]


def assert_recursive_depth(
    obj: any, ipython: IPython.InteractiveShell, visited: list[any]
) -> bool:
    if is_legal_python_obj(repr(obj), obj, ipython):
        return True
    if type(type(obj)) is enum.EnumMeta:
        return True
    if obj in visited:
        return False
    visited.append(obj)
    if type(obj) in [list, tuple, set]:
        for item in obj:
            if not assert_recursive_depth(item, ipython, visited):
                return False
        return True
    if type(obj) is dict:
        for k, v in obj.items():
            if not assert_recursive_depth(v, ipython, visited):
                return False
        return True
    attrs = dir(obj)
    for attr in attrs:
        if not attr.startswith("_") and not callable(attr):
            if not assert_recursive_depth(getattr(obj, attr), ipython, visited):
                return False
    return True
