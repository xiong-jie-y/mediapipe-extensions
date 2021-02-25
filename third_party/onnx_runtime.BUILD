cc_library(
    name = "onnx_runtime",
    srcs = glob(
        [
            "lib/libonnxruntime.so",
            "lib/libonnxruntime.so.1.4.0",
        ],
    ),
    hdrs = glob([
        "include/**/*.h*",
    ]),
    includes = [
        "include/",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
