from collections import defaultdict, namedtuple
import glob
import json
import pickle
import os
from typing import Any, Dict, List, NamedTuple
import cv2

import numpy as np
from PIL import Image

class TimestampedData(NamedTuple):
    timestamp: float
    data: Any

class HumanReadableLog:
    """Log that is saved as a human readable format.

    It's not so efficient, but saved in a filesystem for easy data handling.
    Supposed usecases are like data logging for analysis in a small scale.
    """

    def __init__(self, plain_data=None, image_data=None) -> None:
        if plain_data is None:
            self.plain_data: defaultdict[int, List[TimestampedData]] = defaultdict(list)
        else:
            self.plain_data: defaultdict[int, List[TimestampedData]] = defaultdict(list, plain_data)

        if image_data is None:
            self.image_data: defaultdict[int, List[TimestampedData]] = defaultdict(list)
        else:
            self.image_data: defaultdict[int, List[TimestampedData]] = defaultdict(list, image_data)

    @classmethod
    def load_from_path(cls, output_path):
        image_data = {}
        for video_path in glob.glob(os.path.join(output_path, "*.mp4")):
            name = os.path.splitext(os.path.relpath(video_path, output_path))[0]

            # Load video from file_path.
            frames = []
            cap = cv2.VideoCapture(video_path)
            while True:
                _, frame = cap.read()
                if frame is None:
                    break
                frames.append(np.array(frame))

            # Get timestamp data.
            timestamps = json.load(open(os.path.join(output_path, f"{name}_frame_to_timestamp.json")))
            image_datum = []
            for frame, timestamp in zip(frames, timestamps):
                image_datum.append(TimestampedData(timestamp, frame))


            image_data[name] = image_datum
            print(name)

        # for 
        # name_timestamps = json.load(
        #     open(os.path.join(output_path, "name_timestamps.json"), "r"))
        # image_data = defaultdict(list)
        # for name, timestamp in name_timestamps:
        #     image = np.array(Image.open(os.path.join(
        #         output_path, f"{name}_{timestamp}.png")))
        #     image_data[name].append((timestamp, image))

        # plain_data = json.load(
        #     open(os.path.join(output_path, "plain_data.json")))

        plain_data = pickle.load(
            open(os.path.join(output_path, "plain_data.pkl"), "rb"))

        return HumanReadableLog(plain_data, image_data)

    def add_data(self, name, timestamp, data):
        self.plain_data[name].append(TimestampedData(timestamp, data))

    def add_image(self, name, timestamp, image):
        self.image_data[name].append(TimestampedData(timestamp, image))

    def add_opencv_image(self, name, timestamp, image):
        self.image_data[name].append(TimestampedData(timestamp, image[:, :, ::-1]))

    def get_images_with_timestamp(self, name) -> List[TimestampedData]:
        return self.image_data[name]

    def get_data(self, name) -> List[TimestampedData]:
        return self.plain_data[name]

    def save(self, output_path, video_option: Dict[str, Any]):
        os.makedirs(output_path, exist_ok=True)
        # json.dump(dict(self.plain_data), open(
        #     os.path.join(output_path, "plain_data.json"), "w"))
        pickle.dump(dict(self.plain_data), 
            open(os.path.join(output_path, "plain_data.pkl"), "wb"))

        # Save this as a mp4 file.
        name_timestamps = []

        for name, image_tuples in self.image_data.items():
            fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            size = (640, 480)
            writer = cv2.VideoWriter(os.path.join(output_path, f"{name}.mp4"), fmt, video_option["frame_rate"], size)
            for timestamped_data in image_tuples:
                name_timestamps.append(timestamped_data.timestamp)
                writer.write(timestamped_data.data)
                # Image.fromarray(image).save(os.path.join(
                #     output_path, f"{name}_{timestamp}.png"))
            json.dump(name_timestamps, open(os.path.join(
                output_path, f"{name}_frame_to_timestamp.json"), "w"))
            writer.release()
