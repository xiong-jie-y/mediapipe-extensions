# PIKA
This repository contains perception algorithms for interaction with agetns.
The name comes from capitals of "Perception toward Interaction with [Kawaii](https://en.wikipedia.org/wiki/Kawaii) Agents". My focus is making the world full of kawaii characters.

## Requirements
This repository is currently tested on Ubuntu 18.04 and python3.7.7.
So it might work around these OS and python versions.

* [requirements to run mediapipe](https://google.github.io/mediapipe/getting_started/install)
* [pybind11](https://pybind11.readthedocs.io/en/stable/basics.html)

For python
```
conda env create -f environment.yml
```

## Graph Runner API
### Graph Runner API for GPU
Graph Runner API is an API with higher level than calculator framework API in mediapipe.
This API is provided with python binding using pybind11 and it support GPU.

To run simple app using this graph runner API, please follow these steps.

```sh
python setup.py build_ext
python apps/run_face_mesh_live.py
```

### Graph Runner API for CPU
To run graph runner for cpu (pure python).
Run the following command.

```sh
cd modules/face
wget -O - https://raw.githubusercontent.com/PINTO0309/PINTO_model_zoo/master/032_FaceMesh/02_weight_quantization/download.sh | bash
cd ../../

bash development/prepare_amework_bindings.sh
python apps/run_face_mesh_live_cpu.py
```

## Apps
### Head Gesture Recognition
This app is for recognizing head gesture.
Currently supported gesture is nodding and shaking head. ([Video](https://www.youtube.com/watch?v=PshPSOAfv0E))

```sh                                
python setup.py build_ext
GLOG_minloglevel=2 python apps/recognize_head_gesture.py
```

## Development Code
Code that is used to develop algorithms are included under `/development`.

## License
* [Apache License 2](https://www.apache.org/licenses/LICENSE-2.0)