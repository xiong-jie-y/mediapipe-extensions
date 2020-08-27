import faulthandler
faulthandler.enable()

from pika.graph_runner_cpu import UpperBodyPoseTracker

pose_tracker = UpperBodyPoseTracker()
pose_tracker.run_live()