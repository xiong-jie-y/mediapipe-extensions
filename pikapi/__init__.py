import os

def get_data_path(data_path):
    return os.path.join(os.path.dirname(__file__), data_path)