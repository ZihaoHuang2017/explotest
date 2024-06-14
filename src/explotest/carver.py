import ast
import copy
import inspect
import dill as pickle
import sys
import textwrap
import types
import typing

from .generate_tests import generate_tests
from .utils import is_builtin_obj, CallStatistics
from .constants import EXPLORATORY_PREFIX


class RewriteToName(ast.NodeTransformer):
    def visit_Name(self, node):
        return ast.Constant(node.id)


class ContainCorrectCall(ast.NodeVisitor):
    def __init__(self):
        self.is_correct_call = False

    def visit_Call(self, node):
        match node:
            case ast.Call(
                func=ast.Name(id="print"), args=[ast.Constant(value="--explore"), _]
            ):
                self.is_correct_call = True


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
                function_locals[key] = ("DIRECT", str(value))
            else:
                function_locals[key] = ("PICKLE", pickle.dumps(value))
        except Exception:
            function_locals[key] = ("NO-GO", value)
    return CallStatistics(
        arg_info.args, arg_info.varargs, arg_info.keywords, function_locals
    )


class Carver:
    def __init__(self, parsed_in, ipython, verbose):
        self.calls = dict()
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
        if function_name not in self.calls:
            self.calls[function_name] = []
        self.calls[function_name].append(call_stats)

    # Tracking function: Record all calls and all args
    def traceit(self, frame: types.FrameType, event, _):
        if event != "call":
            return None
        code = frame.f_code
        function_name = code.co_name

        if (
            self.desired_function_name is None
            and (code.co_filename, function_name) not in self.visited_functions
        ):
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
                if self.desired_function_name in self.called_function_name.split("."):
                    self.calls[self.desired_function_name][-1].appendage.extend(
                        extract_tests_from_frame(
                            value[1],
                            frame,
                            self.assignment_targets,
                            self.ipython,
                            self.verbose,
                        )
                    )
                else:
                    self.calls[self.desired_function_name][-1].appendage.extend(
                        extract_tests_from_frame(
                            value[1], frame, "ret", self.ipython, self.verbose
                        )
                    )

        return None

    def calls(self):
        """Return a dictionary of all calls traced."""
        return self.calls

    def call_statistics(self, function_name):
        """Return a list of all arguments of the given function
        as (VAR, VALUE) pairs."""
        return self.calls.get(function_name, [])

    def called_functions(self):
        """Return all functions called."""
        return [function_name for function_name in self.calls.keys()]


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
            except Exception:
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
        if len(temp_id) >= 2 and temp_id[0] == temp_id[-1] == "'":
            temp_id = temp_id[0] + temp_id[1:-1].replace("'", "\\'") + temp_id[-1]  # sanitize string
        for suf in suffixes:
            temp_id += suf
        node.id = temp_id
        return node


class DetermineReturnType(ast.NodeVisitor):
    def __init__(self):
        self.ret = None

    def visit_Return(self, node):
        self.ret = node.value
