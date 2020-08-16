from collections import defaultdict
import json
import pickle
import os
from typing import Any, Dict
import cv2

import numpy as np
from PIL import Image


class HumanReadableLog:
    """Log that is saved as a human readable format.

    It's not so efficient, but saved in a filesystem for easy data handling.
    Supposed usecases are like data logging for analysis in a small scale.
    """

    def __init__(self, plain_data=None, image_data=None) -> None:
        if plain_data is None:
            self.plain_data = defaultdict(list)
        else:
            self.plain_data = defaultdict(list, plain_data)

        if image_data is None:
            self.image_data = defaultdict(list)
        else:
            self.image_data = defaultdict(list, image_data)

    @classmethod
    def load_from_path(cls, output_path):
        name_timestamps = json.load(
            open(os.path.join(output_path, "name_timestamps.json"), "r"))
        image_data = defaultdict(list)
        for name, timestamp in name_timestamps:
            image = np.array(Image.open(os.path.join(
                output_path, f"{name}_{timestamp}.png")))
            image_data[name].append((timestamp, image))

        plain_data = json.load(
            open(os.path.join(output_path, "plain_data.json")))

        return HumanReadableLog(plain_data, image_data)

    def add_data(self, name, timestamp, data):
        self.plain_data[name].append((timestamp, data))

    def add_image(self, name, timestamp, image):
        self.image_data[name].append((timestamp, image))

    def add_opencv_image(self, name, timestamp, image):
        self.image_data[name].append((timestamp, image[:, :, ::-1]))

    def get_images_with_timestamp(self, name):
        return self.image_data[name]

    def get_data(self, name):
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
            for timestamp, image in image_tuples:
                name_timestamps.append((name, timestamp))
                writer.write(image)
                # Image.fromarray(image).save(os.path.join(
                #     output_path, f"{name}_{timestamp}.png"))
            writer.release()

        json.dump(name_timestamps, open(os.path.join(
            output_path, "name_timestamps.json"), "w"))
