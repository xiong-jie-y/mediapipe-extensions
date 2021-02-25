# cc_library(
#     name = "tensor_rt_lib",
#     srcs = ["trt_pose3d.cpp", "util_trt.cpp"],
#     hdrs = ["trt_pose3d.h", "util_trt.h", "util_debug.h"],
#     visibility = ["//visibility:public"],
# )

# cc_library(
#     name = "pose_landmark_3d_estimator_by_roi_calculator",
#     srcs = ["pose_landmark_3d_estimator_by_roi_calculator.cc"],
#     # hdrs = ["onnx_inference_calculator.h"],
#     visibility = ["//visibility:public"],
#     deps = [
#         "//mediapipe/framework:calculator_framework",
#         ":tensor_rt_lib"
#     ],
#     alwayslink = 1,
# )

cc_library(
    name = "pose_landmark_3d_estimator_by_roi_calculator",
    srcs = ["pose_landmark_3d_estimator_by_roi_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//third_party:onnx_runtime",
    ],
    alwayslink = 1,
)