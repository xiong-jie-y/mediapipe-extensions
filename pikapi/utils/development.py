import numpy as np
import json

import argparse
import streamlit as st

import os
import glob

import time
from collections import defaultdict
import plotly.express as px

import plotly.graph_objects as go

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def run_and_measure(func, arg, n_times=1):
    times = []
    outputs = []
    for i in range(0, n_times):
        start = time.time()
        result = func(*arg)
        end = time.time()
        times.append(end - start)
        outputs.append(result)
    return outputs, times

def compare_function_performances(funcs, args):
    st.markdown("## Performance")
    process_times = defaultdict(list)
    outputs = []

    for func in funcs:
        func_name_base = func.__name__
        func_name = func_name_base
        i = 1
        while func_name in process_times:
            func_name = func_name_base + "_" + str(i)
        for arg in args:
            start_time = time.time()
            output = func(*arg)
            process_times[func_name].append(time.time() - start_time)
            outputs.append(output)

    fig = go.Figure()
    for func_name, times in process_times.items():
        fig.add_trace(go.Box(y=times, name=func_name))
    fig.update_layout(
        yaxis_title='Running Time[ms]',
    )
    st.write(fig)

def write_as_json(obj):
    json_str = json.dumps(obj, indent=4, cls = MyEncoder)
    st.markdown(f"```json\n{json_str}\n```")

def inspect_function(func, arg, n_times=1):
    st.markdown(f"## **Inspect Function**/{func.__name__}")
    outputs, times = run_and_measure(func, arg, n_times=n_times)
    st.write(f"**Input: ** {arg}")
    st.write(f"**Running Time: **{np.mean(times)}")
    st.write(f"**Output Type: ** {type(outputs[0])}")
    st.write("**Output**")
    write_as_json(outputs[0])
    return outputs[0]