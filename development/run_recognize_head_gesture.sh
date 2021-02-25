set -e
bazel build  -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 //pikapi:graph_runner.so
mv bazel-bin/pikapi/graph_runner.so pikapi/
PYTHONPATH=.:$PYTHONPATH python apps/recognize_head_gesture.py --running_mode gpu --camera-id 4