class Bar:
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name


class Foo:
    def __init__(self, name: str, bar: Bar):
        self.name = name
        self.bar = bar


# Some triggers:
"""

print_var_name, ret_var_name, shell_var_name


We limit ourselves to print("--explore", obj[attr].attr[0]).
Note that it can't be f"bruh:{bruh}" as it's not helpful
GOOD: we actually receives the entire object
BAD: will need to infer the name somehow
When it is the object itself, can just loop through locals() to find the obj; otherwise need to look at the code itself



NEED: extracting print_var_name, given the line of code. (Shouldn't be that bad?)
NEED: deparse the name passed in to (someobj)[attr].attr[0] or sth
NEED: assert that print_var_name is in ret_var_name
NEED: given that ret_var_name is returned from the same function that shell_var_name calls, associate the two


# Does the return name match with the outer scope's name? (True, False)

 
# Does the return name match the name being feed into result? If not, then is that name
a subset of result? (Happens to share the same name, subset, same)
- Can go see whether the return parts includes a subset of print_var_name.
- Can't do much when they only happen to share the same name.
- If no, then attempts to go up and find the closest place that defined assigned the print_var_name to a subset of 
ret_var_name 

# Is a loop involved? If so, do we know the index somehow? (direct index, enumerate, foreach loop)

- For direct index and enumerate, can just find the current index object (go up and find the nearest loop that defines n)
 in the locals()
- For foreach, can probably infer using list.find(x), and as we are generating tests it shouldn't matter 
the exact position that we are checking.

# Is a conditional statement involved, i.e. does the print only work in certain conds? (True/False)
- This shouldn't really matter, as we hijack all of the prints. This will only run when it's needed
"""


def func0():
    result = 1
    print("--explore", result)
    return result


def func1() -> Bar:
    # ...
    result = Bar("bruh")
    print("--explore", result.name)
    return result


def func2(cond: bool) -> Bar:
    result = Bar("bruh")
    if cond:
        print("--explore", result.name)
    return result


def func3():  # not this!
    result = Foo("oof", Bar("bruh"))
    a = result.bar
    print("--explore", a.name)  # result.bar.name
    return result


def func4() -> list[Bar]:
    results = [Bar("one"), Bar("two"), Bar("three")]
    for i in range(len(results)):
        print("--explore", results[i].name)
    return results


def func5() -> list[Bar]:
    results = [Bar("one"), Bar("two"), Bar("three")]
    for i, bar in enumerate(results):
        print("--explore", bar.name)
    return results


def func6() -> list[Bar]:
    results = [Bar("one"), Bar("two"), Bar("three")]
    for bar in results:
        print("--explore", bar.name)
    return results


def func7() -> list[list[Bar]]:
    results = [
        [Bar("one"), Bar("two"), Bar("three")],
        [Bar("four"), Bar("five"), Bar("six")],
        [Bar("seven"), Bar("eight"), Bar("nine")],
    ]
    for bars in results:
        for bar in bars:
            print("--explore", bar.name)
    return results


def func8() -> list[list[list[int]]]:
    results = [[[1], [2], [3]], [[4], [5], [6]]]
    for i in range(len(results)):
        for j in range(len(results[i])):
            for k in range(len(results[i][j])):
                print("--explore", results[i][j][k])
    return results


def func9():
    result1 = ["a", "b", "c"]
    result2 = ["d", "e", "f"]
    for i, j in zip(result1, result2):
        print("--explore", i)
    return result1, result2


def func10():
    result = {"a": "b", "c": "d", "e": "f"}
    for i in result:
        print("--explore", result[i])
    return result


def func11():
    result = {"a": "b", "c": "d", "e": "f"}
    for i, j in result.items():
        print("--explore", i)
        print("--explore", result[i])
        print("--explore", j)
    return result


def func11a():
    result = {"'a'b": {"'b'a": "a"}, "'c'd": {"'b'a": "a"}, "'e'f": {"'b'a": "a"}}
    for i, j in result.items():
        print("--explore", result[i])
        print("--explore", j)
        for k, l in j.items():
            print("--explore", k)
            print("--explore", l)

    return result


def func12():
    result1 = ["a", "b", "c"]
    result2 = ["d", "e", "f"]
    for i in zip(result1, result2):
        print("--explore", i[0])
    return result1, result2


def func13a():
    result = [1, 2, 3, 3, 3, 2, 1]
    for i, j in enumerate(result):
        print("--explore", j + 1)
    return result


def func14():
    result = [[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2]]
    for a, b in enumerate(result):
        for c, d in enumerate(b):
            print("--explore", a)
            print("--explore", b)
            print("--explore", c)
            print("--explore", d)
    return result


def func15():
    result = {
        # '(\'shell-sort\', \'shell_sort\')': "a"
        "('shell-sort', 'shell_sort')": "a"
    }
    for i, j in result.items():
        print("--explore", i)
        print("--explore", j)
    return result


def bruh(bar: Bar, bar2):
    bar.name = "modified\n bruh"
    print("--explore", bar.name)
    return bar


def bruh2():
    return bruh(Bar("oof"), Bar("crap"))


def func16(arr: list[int]) -> list[int]:
    arr[0] = 2
    print("--explore", arr)
    return arr


def func17() -> list[int]:
    arr = [1, 3, 5, 7]
    return func16(arr)


class Bruh:
    def __init__(self, bruh):
        self.bruh = bruh

    def foo(self):
        temp = self.bruh
        print("--explore", temp)
        return temp


def func18():
    return Bruh(15).foo()


def mult(a: int, b: int):
    return a*b


def mult_alt(a: int, b: int):
    result = a*b
    print("result", result)
    return result



