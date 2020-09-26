set -e
bazel build  -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 //pikapi:orientation_estimator.so 
cp -f bazel-bin/pikapi/orientation_estimator.so ./pikapi
bazel build  -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 //pikapi:_framework_bindings.so
bazel build  -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 //pikapi:landmark_utils.so
bazel build  -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 //pikapi:graph_runner.so
cp -f bazel-bin/pikapi/graph_runner.so pikapi/
cp -f bazel-bin/pikapi/landmark_utils.so ./pikapi/
