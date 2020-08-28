import faulthandler
faulthandler.enable()

from pika.graph_runner_cpu import GraphRunnerCpu

pose_tracker = GraphRunnerCpu(
    "graphs/face_mesh_desktop_live_any_model_cpu.pbtxt",
    [], {
        "detection_model_file_path": "modules/face/face_detection_front_128_float16_quant.tflite",
        # "detection_model_file_path": "mediapipe/models/face_detection_front.tflite",
        # "landmark_model_file_path": "mediapipe/modules/face_landmark/face_landmark.tflite",
        "landmark_model_file_path": "modules/face/face_landmark_192_float16_quant.tflite"
    }
)

pose_tracker.run_live()