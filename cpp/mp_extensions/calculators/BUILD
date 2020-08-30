cc_library(
    name = "tensor_remap_calculator",
    srcs = ["tensor_remap_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/port:ret_check",
        "@mediapipe//mediapipe/framework/port:status",
        "@org_tensorflow//tensorflow/lite:framework"
    ],
    alwayslink = 1,
)

cc_test(
    name = "tensor_remap_calculator_test",
    srcs = ["tensor_remap_calculator_test.cc"],
    deps = [
        ":tensor_remap_calculator",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework:calculator_runner",
        "@mediapipe//mediapipe/framework/deps:file_path",
        "@mediapipe//mediapipe/framework/formats/object_detection:anchor_cc_proto",
        "@mediapipe//mediapipe/framework/port:file_helpers",
        "@mediapipe//mediapipe/framework/port:gtest_main",
        "@mediapipe//mediapipe/framework/port:integral_types",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",
        "@mediapipe//mediapipe/framework/tool:validate_type",
    ],
)