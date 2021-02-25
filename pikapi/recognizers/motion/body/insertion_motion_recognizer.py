from dataclasses import dataclass
from pikapi.formats.basics import ImageFrame, MotionFrame
from pikapi.utils.logging import HumanReadableLog

class InsertionMotionRecognizerFromWaistMotion():
    @dataclass
    class Input:
        motion: MotionFrame

        # They might be a supplimentary information.
        rgb_image: ImageFrame
        depth_image: ImageFrame

    def __init__(self, do_logging=True):
        self.motions = []
        self.do_logging = do_logging
        if self.do_logging:
            self.log = HumanReadableLog()

    def process_input(self, input: Input):
        self.motions.append(input.motion)

        if self.do_logging:
            self.log.add_input_data(f"{self.__class__.__name__}_Input", input)