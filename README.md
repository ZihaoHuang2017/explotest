## Some assumptions on the transformer and how to fix your code:

1. Generally, no global states. As the transformer essentially runs your input again, for any sequence of inputs, it is assumed
that the result running the sequence twice will be the same. This means that global states that may change during  
testing is not testable. This also means singleton patterns are more or less untestable. Pass the mechanism in as a
class instead.

To fix:

Before:
```python
node_id = 0

def get_new_id() -> int:
    global node_id
    node_id += 1
    return node_id


class Foo:
    def __init__(self):
        self.node_id = get_new_id()


if __name__ == "__main__":   # When testing
    assert Foo().node_id == 1
```

After:
```python
class IDGenerator:
    def __init__(self):
        self.node_id = 0
        
        
    def get_next_id(self) -> int:
        self.node_id += 1
        return self.node_id


class Foo:
    def __init__(self, id_gen: IDGenerator):
        self.node_id = id_gen.get_next_id()

if __name__ == "__main__":  # When testing
    id_generator = IDGenerator()
    assert Foo(id_generator).node_id == 1
```

or

```python
class IDGenerator:
    def __init__(self):
        self.node_id = 0


    def get_new_id(self) -> int:
        self.node_id += 1
        return self.node_id

id_generator: IDGenerator = None

def get_next_id():
    if id_generator is None:
        raise Exception()  # Can also return some default value
    return id_generator.get_new_id()

def set_id_generator(id_gen: IDGenerator):
    global id_generator
    id_generator = id_gen
    
class Foo:
    def __init__(self):
        self.node_id = get_next_id()
        
if __name__ == "__main__":  # When testing
    set_id_generator(IDGenerator())
    assert Foo().node_id == 1
```

2. (If using the print exploration function):
   - A single point of return, situated at the bottom of the file
   - The state is not changed after the print
   - A returned value (if tuples) is a subset of exploratory print expression
   - The test generated is not super robust; specifically the test generated might not cover the entire list if the iterator loops through all the items (the index isn't retrievable) 