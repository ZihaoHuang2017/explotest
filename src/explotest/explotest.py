import ast
import builtins
import dataclasses
import enum
import inspect
import os
import pickle
import sys
import textwrap
import types
import typing
from io import open

import IPython
from IPython.core.error import StdinNotImplementedError
from IPython.core.magic import register_line_magic
from IPython.core.magic_arguments import magic_arguments, argument, parse_argstring
from IPython.utils import io

INDENT_SIZE = 4
EXPLORATORY_PREFIX = "--explore"

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


def load_ipython_extension(ipython: IPython.InteractiveShell):
    @register_line_magic
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
                if lin.startswith("%") or lin.endswith("?") or lin.startswith("get_ipython()"):  # magic methods
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
                                + call_string(carver.desired_function_name, call_stat, ipython)
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
    return [f'assert str({var_name}) == "{proper_string_representation}"']  # Too lengthy!


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


class DetermineReturnType(ast.NodeVisitor):
    def __init__(self):
        self.ret = None

    def visit_Return(self, node):
        self.ret = node.value


class ExpressionParser(ast.NodeVisitor):
    def __init__(self, caller_frame: types.FrameType, global_index_start: int):
        self.expression: str = ""
        self.caller_frame: types.FrameType = caller_frame
        self.lineno: int = caller_frame.f_lineno - global_index_start + 1
        self.stack: dict[str, tuple[str, str]] = dict()

    def visit_For(
        self, node
    ):  # method is quite scuffed. There's quite a load of ways ppl can write scuffed
        if not (
            node.lineno <= self.lineno <= node.end_lineno
        ):  # The loop actually contains the desired print statement
            self.generic_visit(node)
            return
        self.stack.update(
            extract_loop_params(node.target, node.iter, self.caller_frame, -1)
        )
        self.generic_visit(node)

    def visit_Call(self, node):
        if node.lineno == self.lineno and getattr(node.func, "id", "") == "print":
            name_replacer = ReplaceNamesWithSuffix(self.stack)
            parsed_obj_name = name_replacer.visit(node.args[1])
            self.expression = ast.unparse(parsed_obj_name)


def extract_loop_params(
    target_node: ast.expr,
    iterator_node: ast.expr,
    caller_frame: types.FrameType,
    override_index: int,
) -> dict[str, tuple[str, str]]:
    match target_node, iterator_node:
        case (ast.Name(), ast.Call(func=ast.Name(id="range"))):
            return {
                target_node.id: (
                    str(
                        eval(
                            target_node.id,
                            caller_frame.f_globals,
                            caller_frame.f_locals,
                        )
                    ),
                    "",
                )
            }
        case (ast.Name(), _):
            unparsed_iterator = ast.unparse(iterator_node)
            evaluated_iterator = eval(
                unparsed_iterator, caller_frame.f_globals, caller_frame.f_locals
            )
            obj = eval(
                target_node.id,
                caller_frame.f_globals,
                caller_frame.f_locals,
            )
            if isinstance(evaluated_iterator, dict):
                return {target_node.id: (f'"{obj}"', "")}
            if isinstance(evaluated_iterator, typing.Sequence):
                if override_index == -1:
                    index = evaluated_iterator.index(obj)
                else:
                    index = override_index
                return {
                    target_node.id: (
                        f"{unparsed_iterator}",
                        f"[{index}]",  # TODO: support nonunique lists
                    )
                }

            try:  # handles sets, hopefully
                iterator_node_list = list(evaluated_iterator)
                if override_index == -1:
                    index = iterator_node_list.index(obj)
                else:
                    index = override_index
                return {
                    target_node.id: (
                        f"list({unparsed_iterator})",
                        f"[{index}]",  # TODO: support nonunique lists
                    )
                }
            except Exception as e:
                pass
            return dict()
        case (ast.Tuple() | ast.List(), ast.Call(func=ast.Attribute(attr="items"))):
            key, value_node = target_node.elts
            key_str = eval(
                ast.unparse(key),
                caller_frame.f_globals,
                caller_frame.f_locals,
            )
            return {
                key.id: (f"'{key_str}'", ""),
                value_node.id: (
                    f"{ast.unparse(iterator_node.func.value)}",
                    f'["{key_str}"]',
                ),
            }
        case (ast.Tuple() | ast.List(), ast.Call(func=ast.Name(id="enumerate"))):
            index_node, value_node = target_node.elts
            index_num = eval(
                ast.unparse(index_node),
                caller_frame.f_globals,
                caller_frame.f_locals,
            )
            result = {index_node.id: (f"{index_num}", "")}
            result.update(
                extract_loop_params(
                    value_node, iterator_node.args[0], caller_frame, index_num
                )
            )
            return result
        case (ast.Tuple() | ast.List(), ast.Call(func=ast.Name(id="zip"))):
            assert isinstance(target_node, ast.Tuple)
            assert len(target_node.elts) == len(iterator_node.args)
            result = dict()
            for item, item_list in zip(target_node.elts, iterator_node.args):
                result.update(
                    extract_loop_params(item, item_list, caller_frame, override_index)
                )
            return result
        case _:
            raise Exception("unhandled", ast.dump(target_node), ast.dump(iterator_node))


class ReplaceNames(ast.NodeTransformer):
    def __init__(self, names: dict[str, str]):
        self.names = names

    def visit_Name(self, node):
        temp_id = node.id
        if temp_id in self.names:
            temp_id = self.names[temp_id]
        node.id = temp_id
        return node


class ReplaceNamesWithSuffix(ast.NodeTransformer):
    def __init__(self, names: dict[str, tuple[str, str]]):
        self.names = names

    def visit_Name(self, node):
        temp_id = node.id
        suffixes = []
        while temp_id in self.names:
            temp_id, suffix = self.names.get(temp_id)
            suffixes.append(suffix)
        suffixes.reverse()
        for suf in suffixes:
            temp_id += suf
        node.id = temp_id
        return node


class RewriteToName(ast.NodeTransformer):
    def visit_Name(self, node):
        return ast.Constant(node.id)


class ContainCorrectCall(ast.NodeVisitor):
    def __init__(self):
        self.is_correct_call = False

    def visit_Call(self, node):
        match node:
            case ast.Call(func=ast.Name(id="print"), args=[ast.Constant(value="--explore"), x]):
                self.is_correct_call = True


def extract_tests_from_frame(obj, frame, assignment_target_names, ipython, verbose):
    caller_frame = frame.f_back
    code_list, global_index_start = inspect.getsourcelines(caller_frame)
    parsed_ast = ast.parse(inspect.getsource(caller_frame))
    expression_parser = ExpressionParser(caller_frame, global_index_start)
    expression_parser.visit(parsed_ast)
    explore_expression = expression_parser.expression
    return_type_determiner = DetermineReturnType()
    return_type_determiner.visit(
        ast.parse(code_list[-1].strip())
    )  # Assuming that this is the correct deal
    name_rewriter = RewriteToName()
    ret = ipython.ev(ast.unparse(name_rewriter.visit(return_type_determiner.ret)))
    name_replacements = match_return_with_assignment(assignment_target_names, ret)
    reparsed_var_expression = ast.parse(explore_expression)
    name_replacer = ReplaceNames(name_replacements)
    var_name = ast.unparse(name_replacer.visit(reparsed_var_expression))
    return generate_tests(obj, var_name, ipython, verbose)


def match_return_with_assignment(
    assign_to: str or tuple[any] or list[any],
    return_from: str or tuple[any] or list[any],
) -> dict[str, str]:
    match assign_to, return_from:
        case str(), str():
            return {return_from: assign_to}
        case str(), _:
            result = dict()
            for i, sub_ret in enumerate(return_from):
                result[sub_ret] = f"{assign_to}[{i}]"
            return result
        case _, str():
            return {return_from: f"({', '.join(assign_to)})"}
        case _, _:
            result = dict()
            for sub_assign, sub_ret in zip(assign_to, return_from):
                result.update(match_return_with_assignment(sub_assign, sub_ret))
            return result


class CallStatistics:
    def __init__(self, args, varargs, keywords, fn_locals):
        self.args = args
        self.varargs = varargs
        self.keywords = keywords
        self.locals: dict = fn_locals
        self.appendage = []


def get_arguments(frame: types.FrameType) -> CallStatistics:
    """Return call arguments in the given frame"""
    # When called, all arguments are local variables
    arg_info = inspect.getargvalues(frame)
    local_variables = frame.f_locals.copy()
    function_locals = dict()
    for key in local_variables:
        value = frame.f_locals[key]
        try:
            if is_builtin_obj(value):
                function_locals[key] = ("DIRECT", value)
            else:
                function_locals[key] = ("PICKLE", pickle.dumps(value))
        except Exception:
            function_locals[key] = ("NO-GO", value)
    return CallStatistics(arg_info.args, arg_info.varargs, arg_info.keywords, function_locals)


def call_value(argument, ipython: IPython.InteractiveShell) -> str:
    mode, representation = argument
    if mode == "DIRECT":
        return repr(representation)
    if mode == "PICKLE":
        unpickled = pickle.loads(representation)
        for key in ipython.user_ns:
            if ipython.user_ns[key] == unpickled:
                return key
        return f"pickle.loads({representation})"
    for key in ipython.user_ns:
        if ipython.user_ns[key] == representation:
            return key
    return repr(representation)


def call_string(function_name, stat: CallStatistics, ipython) -> str:
    """Return function_name(arg[0], arg[1], ...) as a string, pickling complex objects"""
    start_index = 0
    if len(stat.args) > 0:
        first_var = stat.args[0]
        if first_var == "self":
            start_index = 1
            function_name = call_value(stat.locals["self"], ipython) + "." + function_name
    arglist = []
    if stat.varargs is not None:
        arglist.extend(call_value(stat.locals[x], ipython) for x in stat.varargs)

    arglist.extend(f"{arg}={call_value(stat.locals[arg], ipython)}" for arg in stat.args[start_index:])

    if stat.keywords is not None:
        arglist.extend(f"{arg}={call_value(stat.locals[arg], ipython)}" for arg in stat.keywords)

    return f"{function_name}({', '.join(arglist)})"

class Carver:
    def __init__(self, parsed_in, ipython, verbose):
        self.reset()
        self.desired_function_name = None
        self.module = None
        self.parsed_in = parsed_in
        self.ipython = ipython
        self.verbose = verbose
        self.assignment_targets = None
        self.called_function_name = None
        self.visited_functions = set()
        self.called_function_name = ""
        match parsed_in:
            case ast.Assign(targets=[x], value=ast.Call(func=y)):
                self.assignment_targets = ipython.ev(
                    ast.unparse(RewriteToName().visit(x))
                )
                self.called_function_name = ast.unparse(y)


    def reset(self):
        self._calls = {}

    # Start of `with` block
    def __enter__(self):
        self.original_trace_function = sys.gettrace()
        sys.settrace(self.traceit)
        return self

    # End of `with` block
    def __exit__(self, exc_type, exc_value, tb):
        sys.settrace(self.original_trace_function)

    def add_call(self, function_name, call_stats):
        """Add given call to list of calls"""
        if function_name not in self._calls:
            self._calls[function_name] = []
        self._calls[function_name].append(call_stats)

    # Tracking function: Record all calls and all args
    def traceit(self, frame: types.FrameType, event, arg):
        if event != "call":
            return None
        code = frame.f_code
        function_name = code.co_name

        if self.desired_function_name is None and (code.co_filename, function_name) not in self.visited_functions:
            self.visited_functions.add((code.co_filename, function_name))
            try:
                parsed_ast = ast.parse(textwrap.dedent(inspect.getsource(frame)))
                correct_call_checker = ContainCorrectCall()
                correct_call_checker.visit(parsed_ast)
                if correct_call_checker.is_correct_call:
                    self.desired_function_name = function_name
                    self.module = inspect.getmodule(code)
            except (OSError, SyntaxError):  # Cython Exec
                pass

        if function_name == self.desired_function_name:
            call_stats = get_arguments(frame)
            self.add_call(function_name, call_stats)
        if function_name == "hijack_print":
            value = frame.f_locals["values"]
            if (
                isinstance(value, tuple)
                and len(value) == 2
                and value[0] == EXPLORATORY_PREFIX
            ):
                if self.desired_function_name == self.called_function_name:
                    self._calls[self.desired_function_name][-1].appendage.extend(
                        extract_tests_from_frame(
                            value[1],
                            frame,
                            self.assignment_targets,
                            self.ipython,
                            self.verbose,
                        )
                    )
                else:
                    self._calls[self.desired_function_name][-1].appendage.extend(
                        extract_tests_from_frame(
                            value[1], frame, "ret", self.ipython, self.verbose
                        )
                    )

        return None

    def calls(self):
        """Return a dictionary of all calls traced."""
        return self._calls

    def call_statistics(self, function_name):
        """Return a list of all arguments of the given function
        as (VAR, VALUE) pairs."""
        return self._calls.get(function_name, [])

    def called_functions(self):
        """Return all functions called."""
        return [function_name for function_name in self._calls.keys()]