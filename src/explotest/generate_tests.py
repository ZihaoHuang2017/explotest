import dataclasses
import enum
import logging
import typing
from collections import deque

import IPython

from .utils import is_legal_python_obj, get_type_assertion, has_bad_repr

logger = logging.getLogger(__name__)


def generate_tests(obj: typing.Any, var_name: str, ipython, verbose: bool) -> list[str]:
    try:
        if verbose:
            result = generate_verbose_tests(obj, var_name, dict(), ipython)
        else:
            representation, assertions = generate_concise_tests(
                obj, var_name, dict(), True, ipython, 0
            )
            assertions.sort(key=lambda x: x[0])
            result = [assertion for priority, assertion in assertions]
        if len(result) <= 20:  # Arbitrary
            return result
    except Exception as e:
        print(f"Exception encountered when generating tests for {var_name}", e)
        return []
    return result[0:10]


def generate_verbose_tests(
    obj: typing.Any,
    var_name: str,
    visited: dict[int, str],
    ipython: IPython.InteractiveShell,
) -> list[str]:
    """Parses the object and generates verbose tests.

    We are only interested in the top level assertion as well as the objects that can't be parsed directly,
    in which case it is necessary to compare the individual fields.

    Args:
        obj (typing.Any): The object to be transformed into tests.
        var_name (str): The name referring to the object.
        visited (dict[int, str]): A dict associating the obj with the var_names. Used for cycle detection.
        ipython (IPython.InteractiveShell):  bruh

    Returns:
        list[str]: A list of assertions to be added.

    """
    queue: typing.Deque[tuple[str, typing.Any]] = deque([(var_name, obj)])
    result: list[str] = []
    while queue:
        var_name, obj = queue.popleft()
        if obj is True:
            result.append(f"assert {var_name}")
        elif obj is False:
            result.append(f"assert not {var_name}")
        elif obj is None:
            result.append(f"assert {var_name} is None")
        elif type(type(obj)) is enum.EnumMeta and is_legal_python_obj(
            type(obj).__name__, type(obj), ipython
        ):
            result.append(f"assert {var_name} == {str(obj)}")
        elif type(obj) is type:
            class_name = obj.__name__
            if is_legal_python_obj(class_name, obj, ipython):
                result.append(f"assert {var_name} is {class_name}")
            else:
                result.append(f'assert {var_name}.__name__ == "{class_name}"')
        elif type(obj).__module__.split(".")[0] in ["pandas", "scipy", "numpy"]:
            result.append(f"assert str({var_name}) == {repr(str(obj))}")
        elif is_legal_python_obj(repr(obj), obj, ipython):
            result.append(f"assert {var_name} == {repr(obj)}")
        elif id(obj) in visited:
            result.append(f"assert {var_name} == {visited[id(obj)]}")
        else:
            visited[id(obj)] = var_name
            result.append(get_type_assertion(obj, var_name, ipython))
            if type(obj).__name__ == "ndarray":
                queue.append((f"{var_name}.tolist()", obj.tolist()))
            elif isinstance(obj, typing.Sequence):
                queue.extend(
                    [(f"{var_name}[{idx}]", val) for idx, val in enumerate(obj)]
                )
            elif type(obj) is dict:
                queue.extend(
                    [(f"{var_name}[{repr(key)}]", value) for key, value in obj.items()]
                )
            else:
                attrs = dir(obj)
                for attr in attrs:
                    if not attr.startswith("_"):
                        value = getattr(obj, attr)
                        if not callable(value):
                            queue.append((f"{var_name}.{attr}", value))
    return result


def generate_concise_tests(
    obj: typing.Any,
    var_name: str,
    visited: dict[int, str],
    propagation: bool,
    ipython: IPython.InteractiveShell,
    level: int,
) -> tuple[str, list[tuple[int, str]]]:
    """Parses the object and generates concise tests.

    We are only interested in the top level assertion as well as the objects that can't be parsed directly,
    in which case it is necessary to compare the individual fields.

    Args:
        obj (typing.Any): The object to be transformed into tests.
        var_name (str): The name referring to the object.
        visited (dict[int, str]): A dict associating the obj with the var_names. Used for cycle detection.
        propagation (bool): Whether the result should be propagated.
        ipython (IPython.InteractiveShell):  bruh
        level (int): level of the traversal. Used for ensure more useful tests.
    Returns:
        tuple[str, list[str]]: The repr of the obj if it can be parsed easily, var_name if it can't, and a list of

    """

    logger.info(f"{obj, type(obj), var_name}")
    if type(type(obj)) is enum.EnumMeta and is_legal_python_obj(
        type(obj).__name__, type(obj), ipython
    ):
        if propagation:
            return str(obj), [(level, f"assert {var_name} == {str(obj)}")]
        return str(obj), []
    if is_legal_python_obj(repr(obj), obj, ipython):
        if propagation:
            return repr(obj), [
                (level, test)
                for test in generate_verbose_tests(obj, var_name, visited, ipython)
            ]  # to be expanded
        return repr(obj), []

    module_name: str = type(obj).__module__
    if module_name.split(".")[0] in ["pandas", "scipy", "numpy"]:
        return var_name, [(level, f"assert str({var_name}) == {repr(str(obj))}")]
    if id(obj) in visited:
        return var_name, [(level, f"assert {var_name} == {visited[id(obj)]}")]
    visited[id(obj)] = var_name

    if isinstance(obj, typing.Sequence):
        reprs: list[str] = []
        overall_assertions: list[tuple[int, str]] = []
        repr_count, useless_repr_count = 0, 0
        for idx, val in enumerate(obj):
            repr_count += 1
            new_var_name = f"{var_name}[{idx}]"
            representation, assertions = generate_concise_tests(
                val, new_var_name, visited, False, ipython, level + 1
            )
            if representation == new_var_name:
                useless_repr_count += 1
            reprs.append(representation)
            overall_assertions.extend(assertions)
        if isinstance(obj, tuple):
            repr_str = f'({", ".join(reprs)})'
        else:
            repr_str = f'[{", ".join(reprs)}]'
        if (
            propagation and useless_repr_count < repr_count
        ):  # less than 50% useless junk
            overall_assertions.insert(0, (level, f"assert {var_name} == {repr_str}"))
        return repr_str, overall_assertions
    if type(obj) is dict:
        reprs, overall_assertions = [], []
        repr_count, useless_repr_count = 0, 0
        for field, value in obj.items():
            new_var_name = f"{var_name}[{repr(field)}]"
            repr_count += 1
            representation, assertions = generate_concise_tests(
                value, new_var_name, visited, False, ipython, level + 1
            )
            if representation == new_var_name:
                useless_repr_count += 1
            reprs.append(f"{repr(field)}: {representation}")
            overall_assertions.extend(assertions)
        repr_str = "{" + ", ".join(reprs) + "}"
        if propagation and useless_repr_count < repr_count:
            overall_assertions.insert(0, (level, f"assert {var_name} == {repr_str}"))
        return repr_str, overall_assertions
    if dataclasses.is_dataclass(obj):
        reprs, overall_assertions = [], []
        for field in dataclasses.fields(obj):
            representation, assertions = generate_concise_tests(
                getattr(obj, field.name),
                f"{var_name}.{field.name}",
                visited,
                False,
                ipython,
                level + 1,
            )
            reprs.append(f"{repr(field.name)}: {representation}")
            overall_assertions.extend(assertions)
        repr_str = "{" + ", ".join(reprs) + "}"
        if propagation:
            overall_assertions.insert(0, (level, f"assert {var_name} == {repr_str}"))
        return repr_str, overall_assertions
    overall_assertions = [(level + 1, get_type_assertion(obj, var_name, ipython))]
    attrs = dir(obj)
    for attr in attrs:
        if not attr.startswith("_"):
            value = getattr(obj, attr)
            if not callable(value):
                _, assertions = generate_concise_tests(
                    value, f"{var_name}.{attr}", visited, True, ipython, level + 1
                )
                overall_assertions.extend(assertions)
    return var_name, overall_assertions
