import ast
import enum

import IPython


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
    if type(obj) in [list, dict, tuple, set, frozenset]:
        return all(is_builtin_obj(item) for item in obj)
    if type(obj) is dict:
        return all(is_builtin_obj(key) and is_builtin_obj(value) for key, value in obj.items())
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


class CallStatistics:
    def __init__(self, args, varargs, keywords, fn_locals):
        self.args = args
        self.varargs = varargs
        self.keywords = keywords
        self.locals: dict = fn_locals
        self.appendage = []


