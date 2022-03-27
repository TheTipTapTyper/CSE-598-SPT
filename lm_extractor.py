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
        output landmark object into a 33x4 numpy array. Row indices are the same
        as the mediapipe landmark object indices provided by constants in the 
        mp.solutions.pose.PoseLandmark module. 
        e.g. mp.solutions.pose.PoseLandmark.RIGHT_EAR is the index of the right
        ear's landmark.

        world_lms specifies whether to grab world landmarks (in human coordinate
        frame) or not (pixel coordinate frame).

        The return value is a tuple (lm_array, mp_lm_obj) where lm_array is the
        33x4 numpy array and mp_lm_obj is the mediapipe landmark object which
        can be used for rendering purposes.

        If the pose estimator failed to detect a person, this method returns None.

        image is expected to be in RGB format
        """
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        if results.pose_landmarks is None:
            return None
        if world_lms:
            landmarks = results.pose_world_landmarks.landmark
        else:
            landmarks = results.pose_landmarks.landmark
        lm_array = np.array([
            (lm.x, lm.y, lm.z, lm.visibility) for lm in landmarks
        ])
        mp_lm_obj = results.pose_landmarks
        return lm_array, mp_lm_obj

    def extract_sideview_landmarks(self, image, ground=True, **kwargs):
        """Extracts pose landmarks from the image and further breaks them up into
        two 8x4 numpy arrays (for the right and left side of the body). The
        landmarks of the arrays are indexed in the following order:
        (ankle, knee, hip, shoulder, ear, elbow, wrist, index)
        The module contains appropriately named constant attributes if you need
        to index a specific landmark.
        
        The return value is a 3-tuple (left, right, mp_lm_obj) where left and
        right are the left and right side numpy arrays and mp_lm_obj is the
        mediapipe landmark object, from which the side views were extracted,
        which can be used for rendering purposes.

        Additionally, if ground is set to True (default), then the sideview landmarks 
        are "grounded" by subtracting the ankle position from all points.

        If the pose estimator failed to detect a person, this method returns None.

        e.g. 
        ...
        import lm_extractor as lme
        extractor = lme.LandmarkExtractor()
        left, right, mp_lm_obj = extractor.extract_sideview_landmarks(im)
        wrist_lm = right[lme.WRIST]
        """
        results = self.extract_landmarks(image, **kwargs)
        if results is None:
            return None
        lm_array, mp_lm_obj = results
        idxs = mp.solutions.pose.PoseLandmark
        right = lm_array[[
            idxs.RIGHT_ANKLE, idxs.RIGHT_KNEE, idxs.RIGHT_HIP, idxs.RIGHT_SHOULDER,
            idxs.RIGHT_EAR, idxs.RIGHT_ELBOW, idxs.RIGHT_WRIST, idxs.RIGHT_INDEX
        ]][:,[0,1,3]]
        left = lm_array[[
            idxs.LEFT_ANKLE, idxs.LEFT_KNEE, idxs.LEFT_HIP, idxs.LEFT_SHOULDER,
            idxs.LEFT_EAR, idxs.LEFT_ELBOW, idxs.LEFT_WRIST, idxs.LEFT_INDEX
        ]][:,[0,1,3]]
        #flip vertically
        right[:, 1] = -right[:, 1]
        left[:, 1] = -left[:, 1]
        if ground:
            right[:, :-1] -= right[ANKLE, :-1]
            left[:, :-1] -= left[ANKLE, :-1]
        return left, right, mp_lm_obj

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

