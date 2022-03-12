"""
Author: Joshua Martin
Email: jmmartin397@protonmail.com
Created: 3/12/2022
Class: CSE 598 Perception in Robotics
Project: Deadlift Critic
"""

import mediapipe as mp
import numpy as np

ANKLE = 0
KNEE = 1
HIP = 2
SHOULDER = 3
EAR = 4
ELBOW = 5
WRIST = 6
INDEX = 7

class LandmarkExtractor:
    def __init__(self, **kwargs):
        self.pose = mp.solutions.pose.Pose(**kwargs)

    def extract_landmarks(self, image, world_lms=True):
        """ Runs Mediapipe's pose model on the given image and reformats the
        output land marks into a 33x4 numpy array. Row indices are the same
        as the mediapipe landmark indices provided by constants in the 
        mp.solutions.pose.PoseLandmark module. 
        e.g. mp.solutions.pose.PoseLandmark.RIGHT_EAR is the index of the right
        ear's landmark.

        world_lms specifies whether to grab world landmarks (in human coordinate
        frame) or not (pixel coordinate frame).

        Additionally, the raw results.pose_landmarks (pixelspace landmarks) are
        stored in self.pose_landmarks for rendering purposed.

        image is expected to be in RGB format
        """
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        self.pose_landmarks = results.pose_landmarks
        if world_lms:
            landmarks = results.pose_world_landmarks.landmark
        else:
            landmarks = results.pose_landmarks.landmark
        return np.array([
            (lm.x, lm.y, lm.z, lm.visibility) for lm in landmarks
        ])

    def extract_sideview_landmarks(self, image, ground=True, **kwargs):
        """ Calls the extract_landmarks method with **kwargs and then pulls
        out just the right and left side landmarks (ankle, knee, hip, shoulder,
        ear, elbow, wrist, index) and returns two numpy arrays (left, right)
        with the points indexed in that order. The module contains appropriately
        named constant attributes if you need to index a specific landmark.
        Additionally, if ground is set to True, then the sideview landmarks 
        are "grounded" by subtracting the ankle position from all points.
        e.g. 
        ...
        import lm_extractor as lme
        extractor = lme.LandmarkExtractor()
        left, right = extractor.extract_sideview_landmarks(im)
        wrist_lm = right[lme.WRIST]
        """
        lms = self.extract_landmarks(image, **kwargs)
        idxs = mp.solutions.pose.PoseLandmark
        right = lms[[
            idxs.RIGHT_ANKLE, idxs.RIGHT_KNEE, idxs.RIGHT_HIP, idxs.RIGHT_SHOULDER,
            idxs.RIGHT_EAR, idxs.RIGHT_ELBOW, idxs.RIGHT_WRIST, idxs.RIGHT_INDEX
        ]][:,[0,1,3]]
        left = lms[[
            idxs.LEFT_ANKLE, idxs.LEFT_KNEE, idxs.LEFT_HIP, idxs.LEFT_SHOULDER,
            idxs.LEFT_EAR, idxs.LEFT_ELBOW, idxs.LEFT_WRIST, idxs.LEFT_INDEX
        ]][:,[0,1,3]]
        #flip vertically
        right[:, 1] = -right[:, 1]
        left[:, 1] = -left[:, 1]
        if ground:
            right[:, :-1] -= right[ANKLE, :-1]
            left[:, :-1] -= left[ANKLE, :-1]
        return left, right

    def weighted_average(self, landmarks):
        """ Combines multiple landmark numpy arrays using the last column
        (the visibility) as the weight. The idea here is to combine multiple
        side views (both right and left, and possibly additional views from
        multiple cameras) and weight the average by how confident mediapipe
        is that the body part is not occluded.
        """
        weighted_sum = np.sum(
            [(lm[:,:].T * lm[:,-1]).T for lm in landmarks], axis=0
        )
        norm_factors = 1 / np.sum([lm[:,-1] for lm in landmarks], axis=0)
        norm_sum = (weighted_sum.T * norm_factors).T
        return norm_sum
        

if __name__ == '__main__':
    # test landmark extraction
    from PIL import Image
    img = np.array(Image.open('deadlift.jpeg'))
    lme = LandmarkExtractor()
    print(lme.extract_sideview_landmarks(img))

    # test weighted average
    lm1 = np.array([[1,1,.1],[3,2,.1]])
    lm2 = np.array([[1.2,1.2,.1],[3.5,2.5,.3]])
    print(lme.weighted_average([lm1, lm2]))

