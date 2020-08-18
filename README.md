# PIKA
This repository contains perception algorithms for interaction with agetns.
The name comes from capitals of "Perception toward Interaction with [Kawaii](https://en.wikipedia.org/wiki/Kawaii) Agents". My focus is making the world full of kawaii characters.

## Requirements
This repository is currently tested on Ubuntu 18.04 and python3.7.7.
So it might work around these OS and python versions.

(TBD)

## Graph Runner API
Graph Runner API is an API with higher level than calculator framework API in mediapipe.
This API is provided with python binding using pybind11 and it support GPU.

To run simple app using this graph runner API, please follow these steps.

(WIP)
```sh
python setup.py build_graph_runner
python apps/run_face_mesh_live.py
```

## Apps
### Head Gesture Recognition
This app is for recognizing head gesture.
Currently supported gesture is nodding and shaking head.

```sh
python setup.py build_graph_runner
python apps/recognize_head_gesture.py
```

## Development Code
Code that is used to develop algorithms are included under `/development`.

## License
* [Apache License 2](https://www.apache.org/licenses/LICENSE-2.0)