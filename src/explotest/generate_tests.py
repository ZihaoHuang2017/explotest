import dataclasses
import enum
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


def add_call_string(function_name, stat: CallStatistics, ipython, dest, line) -> tuple[str, list[str]]:
    """Return function_name(arg[0], arg[1], ...) as a string, pickling complex objects
    function call, setup
    """
    start_index = 0
    arg_counter = 0
    setup_code = []
    if len(stat.args) > 0:
        first_var = stat.args[0]
        if first_var == "self":
            start_index = 1
            value, setup = call_value_wrapper(stat.locals["self"], ipython, f"line{line}_arg{arg_counter}", dest)
            function_name = value + "." + function_name
            arg_counter += 1
            setup_code.extend(setup)

    # TODO: more correct parsing of fn signature
    arglist = []
    if stat.varargs is not None:
        for arg in stat.varargs:
            value, setup = call_value_wrapper(stat.locals[arg], ipython, f"line{line}_arg{arg_counter}", dest)
            arglist.append(value)
            arg_counter += 1
            setup_code.extend(setup)

    for arg in stat.args[start_index:]:
        value, setup = call_value_wrapper(stat.locals[arg], ipython, f"line{line}_arg{arg_counter}", dest)
        arglist.append(f"{arg}={value}")
        arg_counter += 1
        setup_code.extend(setup)

    if stat.keywords is not None:
        for arg in stat.keywords:
            value, setup = call_value_wrapper(stat.locals[arg], ipython, f"line{line}_arg{arg_counter}", dest)
            arglist.append(f"{arg}={value}")
            arg_counter += 1
            setup_code.extend(setup)

    return f"{function_name}({', '.join(arglist)})", setup_code


def call_value_wrapper(argument, ipython: IPython.InteractiveShell, varname, dest) -> tuple[str, list[str]]:
    mode, representation = argument
    if mode == "DIRECT":
        return representation, []
    if mode == "PICKLE":
        unpickled = dill.loads(representation)
        for key in ipython.user_ns:
            if ipython.user_ns[key] == unpickled:
                return key, []
        pathlib.Path(dest).mkdir(parents=True, exist_ok=True)
        with open(f"{dest}/{varname}", "wb") as f:
            f.write(representation)
        setup_code = [f"with open('{dest}/{varname}', 'rb') as f: \n    {varname} = pickle.load(f)"]
        setup_code.extend(generate_tests(unpickled, varname, ipython, False))
        return varname, setup_code
    for key in ipython.user_ns:
        if ipython.user_ns[key] == representation:
            return key, []
    return repr(representation), []
