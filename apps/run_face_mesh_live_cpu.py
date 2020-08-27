import faulthandler
faulthandler.enable()

from pika.graph_runner_cpu import GraphRunnerCpu

pose_tracker = GraphRunnerCpu("graphs/face_mesh_desktop_live_any_model_cpu.pbtxt")
pose_tracker.run_live()