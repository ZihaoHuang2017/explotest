class Incrementor:
    def __init__(self):
        self.arg_counter = 0

    def get_next_counter(self):
        self.arg_counter += 1
        return self.arg_counter
