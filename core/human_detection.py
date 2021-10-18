import ssl
from os.path import basename
from os.path import join
from numpy import ndarray
from typing import List
from torch.hub import load
from torch import Tensor
import cv2
import numpy as np

from core.deep_sort_pytorch.deep_sort import DeepSort

from core.plot_utils import plot_boxes, xyxy2xywh

class HumanDetection(object):

    """

    Attributes
    ----------
    input_path : str
        Path to the input video
    skip_frame : int
        Frequence to apply detection algorithms
    """


    # Deactivate ssl checking to download pre-trained yolov5 model
    # from torch hub
    ssl._create_default_https_context = ssl._create_unverified_context

    def __init__(self, input_path: str, skip_frame: int = 20):

        self.input_path: str = input_path
        self.persons: List = []

        # Load pre-trained YoloV5 model from pytorchHub
        self.model = load('ultralytics/yolov5', 'yolov5m6')
        
        # Set model to classes to [0] to only detect persons
        self.model.classes = [0]

        # Init deepsort aglorithm
        self.deepsort = DeepSort("core/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7", max_dist=0.1, min_confidence=0.49)
        
        # Keep color of each identified person
        self.color = {}

        self.skip_frame: int= skip_frame

        # Store result of the detection algorithm
        self.predictions = None

        #nth frame of the video
        self.frame_nth = 0

    def detect_image(self, image: ndarray) -> ndarray:
        """
            Detects the people present in the extracted image in the frames

            Attributes
            ----------
            image: opencv image
                Extracted image from the video
            frame_th: int
                The nth frame of the video
            Return
            ----------
            processed_image: opencv image
        """

        # Check if detection algorithm needed

        if (self.predictions is None) or (self.frame_nth % self.skip_frame == 0):
            self.predictions = self.model(image)

        for detected_boxes in self.predictions.pred:
            if detected_boxes is not None and len(detected_boxes):
                xywhs: Tensor = xyxy2xywh(detected_boxes[:, 0:4])
                confs: float = detected_boxes[:, 4]
                clss: int = detected_boxes[:, 5]

                # pass detections to deepsort
                outputs: ndarray = self.deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), image)
                # draw boxes for videoisualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)): 
                        
                        bboxes = output[0:4]
                        id_person = output[4]

                        if id_person not in self.color:
                            self.color[id_person] = tuple(np.random.randint(0, 255, size=(3, )))
                            
                            # convert data types int64 to int
                            self.color[id_person] = ( int (self.color[id_person] [ 0 ]), int (self.color[id_person] [ 1 ]), int (self.color[id_person] [ 2 ])) 


                        plot_boxes(image, bboxes, id_person, conf, self.color[id_person])

            else:
                self.deepsort.increment_ages()

                # Set predictions to None if the model doesn't detect any person
                # It will allow to run another detection in the next frame
                self.predictions = None

        return image


    def detect(self):
        """
            Generate a video with identified boxes around each person

        """

        # Create output path
        filename: str = basename(self.input_path)
        output_path: str = join("data", "output", filename)

        # Set output video writer with codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 25.0, (1920, 1080))

        # Read the video
        video = cv2.VideoCapture(self.input_path)
        frame_read, image = video.read()


        # Iterate over frames and pass each for prediction
        while frame_read:

            # Increment frame pointer
            self.frame_nth += 1

            # Perform human detection
            processed_image = self.detect_image(image)


            # Write frame with predictions to video
            out.write(processed_image)
              
            # Read next frame
            frame_read, image = video.read()

        # Release video file when we're ready
        out.release()

        return True