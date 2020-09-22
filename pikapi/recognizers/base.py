from collections import defaultdict
import contextlib
import time


class PerformanceMeasurable:
    def __init__(self):
        self.time_measure_result = defaultdict(list)
        
    @contextlib.contextmanager
    def time_measure(self, log_name):
        start_time = time.time()
        yield
        end_time = time.time()
        self.time_measure_result[log_name].append((end_time - start_time) * 1000)
