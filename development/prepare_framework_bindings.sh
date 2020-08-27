bazel build --define=MEDIAPIPE_DISABLE_GPU=1 --action_env=PYTHON_BIN_PATH=/home/yusuke/miniconda3/envs/py37_ubuntu_vtuber/bin/python //pika:_framework_bindings.so
mv bazel-bin/pika/_framework_bindings.so mediapipe/python
pip install -e .
