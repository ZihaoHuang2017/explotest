import dataclasses
import enum
from inspect import Parameter

import dill
import typing

import IPython
import pathlib
from .utils import is_legal_python_obj, get_type_assertion, CallStatistics


def generate_tests(obj: any, var_name: str, ipython, verbose: bool) -> list[str]:
    try:
        if verbose:
            result = generate_verbose_tests(obj, var_name, dict(), ipython)
        else:
            representation, assertions = generate_concise_tests(
                obj, var_name, dict(), True, ipython
            )
            result = assertions
        if len(result) <= 20:  # Arbitrary
            return result
        if "object at 0x" in str(obj):
            return result[0:10]
    except Exception as e:
        print(f"Exception encountered when generating tests for {var_name}", e)
    if "object at 0x" in str(obj):  # Can't do crap
        return []
    proper_string_representation = str(obj).replace('"', '\\"').replace("\n", "\\n")
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


class Incrementor:
    def __init__(self):
        self.arg_counter = 0


def add_call_string(
    function_name, stat: CallStatistics, ipython, dest, line
) -> tuple[str, list[str]]:
    """Return function_name(arg[0], arg[1], ...) as a string, pickling complex objects
    function call, setup
    """
    incrementor = Incrementor()
    setup_code = []
    arglist = []
    if stat.is_method_call:
        value, setup = call_value_wrapper(
            stat.locals["self"], ipython, line, incrementor, dest
        )
        setup_code.extend(setup)
        function_name = value + "." + function_name

    for (
        param_name,
        param_obj,
    ) in stat.parameters.items():  # note: well-order is guaranteed
        if param_name in ["/", "*"]:
            continue
        match param_obj.kind:
            case Parameter.POSITIONAL_ONLY:
                value, setup = call_value_wrapper(
                    stat.locals[param_name], ipython, line, incrementor, dest
                )
                setup_code.extend(setup)
                arglist.append(value)
            case Parameter.KEYWORD_ONLY | Parameter.POSITIONAL_OR_KEYWORD:
                value, setup = call_value_wrapper(
                    stat.locals[param_name], ipython, line, incrementor, dest
                )
                setup_code.extend(setup)
                arglist.append(f"{param_name}={value}")
            case Parameter.VAR_POSITIONAL:
                for arg_name in stat.locals[param_name]:
                    value, setup = call_value_wrapper(
                        stat.locals[arg_name], ipython, line, incrementor, dest
                    )
                    setup_code.extend(setup)
                    arglist.append(value)
            case Parameter.VAR_KEYWORD:
                for arg_name in stat.locals[param_name]:
                    value, setup = call_value_wrapper(
                        stat.locals[arg_name], ipython, line, incrementor, dest
                    )
                    setup_code.extend(setup)
                    arglist.append(f"{arg_name}={value}")
    return f"{function_name}({', '.join(arglist)})", setup_code


def call_value_wrapper(
    argument,
    ipython: IPython.InteractiveShell,
    line: int,
    incrementor: Incrementor,
    dest,
) -> tuple[str, list[str]]:
    mode, representation = argument
    varname = f"line{line}_arg{incrementor.arg_counter}"
    if mode == "DIRECT":
        return representation, []
    if mode == "PICKLE":
        unpickled = dill.loads(representation)
        for key in ipython.user_ns:
            if ipython.user_ns[key] == unpickled:
                return key, []
        pathlib.Path(dest).mkdir(parents=True, exist_ok=True)
        full_path = f"{dest}/{varname}"
        with open(full_path, "wb") as f:
            f.write(representation)
        setup_code = [
            f"with open('{full_path}', 'rb') as f: \n    {varname} = pickle.load(f)"
        ]
        setup_code.extend(generate_tests(unpickled, varname, ipython, False))
        return varname, setup_code
    for key in ipython.user_ns:
        if ipython.user_ns[key] == representation:
            return key, []
    return repr(representation), []
