# PIKAPI
This repository contains perception algorithms for interaction with agetns.
The name comes from capitals of "Perception toward Interaction with [Kawaii](https://en.wikipedia.org/wiki/Kawaii) Agents and Pretty Individuals". My focus is making the world full of kawaii characters.

## Requirements
### For use perception algorithms
For python, please install following [modules](requirements.txt).

```sh
pip install pikapi
```

### To build library from source.
This repository is currently tested on Ubuntu 18.04 and python3.7.7.
So it might work around these OS and python versions.

* [requirements to run mediapipe](https://google.github.io/mediapipe/getting_started/install)
  * (Running [facemesh](https://google.github.io/mediapipe/solutions/face_mesh.html) example for testing successful installation is recommended.)
* protobuf-compiler

For python, please install following modules.
* numpy
* pyqt5
* scipy
* opencv > 3

## Perception Algorithms
### Head Gesture Recognition
This app is for recognizing head gesture.
Currently supported gesture is nodding and shaking head. ([Video](https://www.youtube.com/watch?v=PshPSOAfv0E))

#### If you install the module with pip
```sh
GLOG_minloglevel=2 python apps/recognize_head_gesture.py
```

#### If want to compile from source.
```sh
# To run on GPU.
python setup.py build_ext
GLOG_minloglevel=2 python apps/recognize_head_gesture.py

# To run on CPU.
cd modules/face
wget -O - https://raw.githubusercontent.com/PINTO0309/PINTO_model_zoo/master/030_BlazeFace/05_float16_quantization/download_new.sh | bash
wget -O - https://raw.githubusercontent.com/PINTO0309/PINTO_model_zoo/master/032_FaceMesh/05_float16_quantization/download.sh | bash
cd ../../

python3 tools/generate_proto.py

# If you use non-system python, it is necessary to point to the path of python.
# to make a python binding appropriately.
# export PYTHON_BIN_PATH=$YOUR_PYTHON_PATH
bash development/prepare_framework_bindings.sh

GLOG_minloglevel=2 python apps/recognize_head_gesture.py
```

## Examples for mediapipe use in python
These are the example of getting the face landmark results from graph.

### Python Graph Runner for GPU
Graph Runner API is an API with higher level than calculator framework API in mediapipe.
This API is provided with python binding using pybind11 and it support GPU.

To run simple app using this graph runner API, please follow these steps.

```sh
python setup.py build_ext
python apps/run_face_mesh_live.py
```

### Python Graph Runner for CPU
To run graph runner for cpu (pure python).
Run the following command.

```sh
cd modules/face
wget -O - https://raw.githubusercontent.com/PINTO0309/PINTO_model_zoo/master/030_BlazeFace/05_float16_quantization/download_new.sh | bash
wget -O - https://raw.githubusercontent.com/PINTO0309/PINTO_model_zoo/master/032_FaceMesh/05_float16_quantization/download.sh | bash
cd ../../

python3 tools/generate_proto.py

# If you use non-system python, it is necessary to point to the path of python.
# to make a python binding appropriately.
# export PYTHON_BIN_PATH=$YOUR_PYTHON_PATH
bash development/prepare_framework_bindings.sh
python apps/run_face_mesh_live_cpu.py
```

## Development Code
Code that is used to develop algorithms are included under `/development`.

Directory structure

```
- apps: Sample app to use perception algorithms and lower apis.
- cpp: C++ code that is eventually built as .so
- development: Code for developing and analyzing algorithms.
- pikapi: All the python code and data that should be included in python package.
- tools: Tools that supports other than developing algorithms.
```

## License
* [Apache License 2](https://www.apache.org/licenses/LICENSE-2.0)