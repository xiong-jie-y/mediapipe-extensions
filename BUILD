load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

pybind_extension(
    name = "graph_runner",
    srcs = [
        "cameravtuber_pybind.cc"
    ],
    deps = [
        "@mediapipe//mediapipe/examples/desktop:demo_run_graph_main_gpu",
        "@mediapipe//mediapipe/graphs/hand_tracking:multi_hand_mobile_calculators",
        "@mediapipe//mediapipe/graphs/face_mesh:desktop_live_gpu_calculators",
        # "@mediapipe//mediapipe/graphs/face_mesh:mobile_calculators",
        "@mediapipe//mediapipe/graphs/pose_tracking:upper_body_pose_tracking_gpu_deps"
    ],
)
