load(
    "@mediapipe//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_simple_subgraph",
)

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

mediapipe_simple_subgraph(
    name = "face_detection_front_cpu",
    graph = "face_detection_front_cpu.pbtxt",
    register_as = "FaceDetectionFrontCpuAnyModel",
    deps = [
        "@mediapipe//mediapipe/calculators/image:image_transformation_calculator",
        "@mediapipe//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "@mediapipe//mediapipe/calculators/tflite:tflite_converter_calculator",
        "@mediapipe//mediapipe/calculators/tflite:tflite_inference_calculator",
        "@mediapipe//mediapipe/calculators/tflite:tflite_tensors_to_detections_calculator",
        "@mediapipe//mediapipe/calculators/util:detection_letterbox_removal_calculator",
        "@mediapipe//mediapipe/calculators/util:non_max_suppression_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "face_landmark_cpu",
    graph = "face_landmark_cpu.pbtxt",
    register_as = "FaceLandmarkCpuAnyModel",
    deps = [
        "@mediapipe//mediapipe/calculators/core:gate_calculator",
        "@mediapipe//mediapipe/calculators/core:split_vector_calculator",
        "@mediapipe//mediapipe/calculators/image:image_cropping_calculator",
        "@mediapipe//mediapipe/calculators/image:image_transformation_calculator",
        "@mediapipe//mediapipe/calculators/tflite:tflite_converter_calculator",
        "@mediapipe//mediapipe/calculators/tflite:tflite_inference_calculator",
        "@mediapipe//mediapipe/calculators/tflite:tflite_tensors_to_floats_calculator",
        "@mediapipe//mediapipe/calculators/tflite:tflite_tensors_to_landmarks_calculator",
        "@mediapipe//mediapipe/calculators/util:landmark_projection_calculator",
        "@mediapipe//mediapipe/calculators/util:thresholding_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "face_landmark_front_cpu",
    graph = "face_landmark_front_cpu.pbtxt",
    register_as = "FaceLandmarkFrontCpuAnyModel",
    deps = [
        "@mediapipe//mediapipe/modules/face_landmark:face_detection_front_detection_to_roi",
        ":face_landmark_cpu",
        "@mediapipe//mediapipe/modules/face_landmark:face_landmark_landmarks_to_roi",
        "@mediapipe//mediapipe/calculators/core:begin_loop_calculator",
        "@mediapipe//mediapipe/calculators/core:clip_vector_size_calculator",
        "@mediapipe//mediapipe/calculators/core:end_loop_calculator",
        "@mediapipe//mediapipe/calculators/core:gate_calculator",
        "@mediapipe//mediapipe/calculators/core:merge_calculator",
        "@mediapipe//mediapipe/calculators/core:previous_loopback_calculator",
        "@mediapipe//mediapipe/calculators/image:image_properties_calculator",
        "@mediapipe//mediapipe/calculators/util:association_norm_rect_calculator",
        "@mediapipe//mediapipe/calculators/util:collection_has_min_size_calculator",
        ":face_detection_front_cpu",
        "@mediapipe//mediapipe/modules/face_detection:face_detection_front_cpu",
    ],
)

mediapipe_simple_subgraph(
    name = "face_landmark_gpu",
    graph = "face_landmark_gpu.pbtxt",
    register_as = "FaceLandmarkGpuAnyModel",
    deps = [
        "@mediapipe//mediapipe/calculators/core:gate_calculator",
        "@mediapipe//mediapipe/calculators/core:split_vector_calculator",
        "@mediapipe//mediapipe/calculators/image:image_cropping_calculator",
        "@mediapipe//mediapipe/calculators/image:image_transformation_calculator",
        "@mediapipe//mediapipe/calculators/tflite:tflite_converter_calculator",
        "@mediapipe//mediapipe/calculators/tflite:tflite_inference_calculator",
        "@mediapipe//mediapipe/calculators/tflite:tflite_tensors_to_floats_calculator",
        "@mediapipe//mediapipe/calculators/tflite:tflite_tensors_to_landmarks_calculator",
        "@mediapipe//mediapipe/calculators/util:landmark_projection_calculator",
        "@mediapipe//mediapipe/calculators/util:thresholding_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "face_landmark_front_gpu",
    graph = "face_landmark_front_gpu.pbtxt",
    register_as = "FaceLandmarkFrontGpuAnyModel",
    deps = [
        "@mediapipe//mediapipe/modules/face_landmark:face_detection_front_detection_to_roi",
        ":face_landmark_gpu",
        "@mediapipe//mediapipe/modules/face_landmark:face_landmark_landmarks_to_roi",
        "@mediapipe//mediapipe/calculators/core:begin_loop_calculator",
        "@mediapipe//mediapipe/calculators/core:clip_vector_size_calculator",
        "@mediapipe//mediapipe/calculators/core:end_loop_calculator",
        "@mediapipe//mediapipe/calculators/core:gate_calculator",
        "@mediapipe//mediapipe/calculators/core:merge_calculator",
        "@mediapipe//mediapipe/calculators/core:previous_loopback_calculator",
        "@mediapipe//mediapipe/calculators/image:image_properties_calculator",
        "@mediapipe//mediapipe/calculators/util:association_norm_rect_calculator",
        "@mediapipe//mediapipe/calculators/util:collection_has_min_size_calculator",
        "@mediapipe//mediapipe/modules/face_detection:face_detection_front_gpu",
    ],
)

# exports_files(
#     srcs = [
#         "face_landmark_192_weight_quant.tflite",
#         "face_landmark_192_full_integer_quant.tflite"
#     ],
# )