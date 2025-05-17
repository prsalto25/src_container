from collections import OrderedDict

class LimitedSizeDict(OrderedDict):
    def __init__(self, max_size):
        self.max_size = max_size
        super().__init__()

    def __setitem__(self, key, value):
        if len(self) >= self.max_size:
            self.popitem(last=False)  # Remove the oldest (first inserted) item
        super().__setitem__(key, value)