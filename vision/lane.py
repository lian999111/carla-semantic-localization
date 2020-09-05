# Implementation of lane extraction from semantic segmentation images

import numpy as np
import 

class LaneMarkingExtractor(object):
    """ 
    Class for lane marking extration from semantic segmentation images. 
    It uses 2 approaches to detect lane marking pixels:
    - Sliding window: Used when there was no lane marking detected in previous images.
    - CTRV motion model: Used to predict the current positions of lane markings. It requires lane markings to be detected in previous images.
    """

    def __init__(self, image_size=(800, 600)):
        pass
