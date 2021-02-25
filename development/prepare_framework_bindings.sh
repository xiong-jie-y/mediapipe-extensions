set -e
# bazel build -c opt --action_env tflite_with_xnnpack=true --define=MEDIAPIPE_DISABLE_GPU=1 --action_env=PYTHON_BIN_PATH=/home/yusuke/miniconda3/envs/py37_ubuntu_vtuber/bin/python //pikapi:_framework_bindings.so

# bazel build -c opt --define=MEDIAPIPE_DISABLE_GPU=1 --action_env=PYTHON_BIN_PATH=$PYTHON_BIN_PATH //pikapi:_framework_bindings.so
bazel build -c opt --define=MEDIAPIPE_DISABLE_GPU=1 //pikapi:_framework_bindings.so
mv bazel-bin/pikapi/_framework_bindings.so mediapipe/python
# pip install -e .
