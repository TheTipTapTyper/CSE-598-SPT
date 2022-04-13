"""
Author: Joshua Martin
Email: jmmartin397@protonmail.com
Created: 3/12/2022
Class: CSE 598 Perception in Robotics
Project: Deadlift Critic
"""

import mediapipe as mp
import numpy as np
import multiprocessing as mproc

ANKLE = 0
KNEE = 1
HIP = 2
SHOULDER = 3
EAR = 4
ELBOW = 5
WRIST = 6
INDEX = 7

class LandmarkExtractor:
    """ Capable of maintaining multiple subprocesses which each contain an instance
    of medipipe's pose estimator. Contains methods for extracting landmarks from
    one or more images in parallel.
    """
    def __init__(self, **kwargs):
        """ kwargs are saved and passed to mediapipe pose extractor instances when
        an input stream is setup.
        """
        self.pose_kwargs = kwargs
        self.processes = {}
        self.connections = {}

    def __del__(self):
        self.kill_processes()

    def kill_processes(self):
        """ Explicitly terminate all subprocesses. This will happen automatically
        when the LandmarkExtractor instance goes out of scope.
        """
        for proc in self.processes.values():
            proc.terminate()

    def _setup_input_stream(self, input_id):
        """ Create a subprocess to handle landmark extraction for the input stream
        specified by input_id. A pipe is opened between the parent process and the child.
        """
        parent_con, child_con = mproc.Pipe()
        proc = mproc.Process(target=_pose_estimator_proc, args=[child_con], kwargs={**self.pose_kwargs})
        proc.start()
        self.processes[input_id] = proc
        self.connections[input_id] = parent_con

    def extract_landmarks(self, input, id='default'):
        """ Extracts human pose landmarks from the given image(s). input can
        either be a single image or a list of images. If input is a list, then id
        must also be a list of the same length. The value(s) of id specify which subprocess
        will handle the request. This is important to keep straight because individual
        subprocesses track poses through subsequent calls.

        Note: all elements of id (if list) must be UNIQUE

        Returns: 
            two-tuple (mp_lm_obj, mp_lm_world_obj) mediapipe landmark objects. The first gives
                coordinates in the pixel frame, and the coordinates range from 0 to 1. The second
                gives "world coordinates" meaning the coordinate frame is roughly centered on the
                center of the hip, and the coordinates are in meters, roughly.
            If input is not a list, then only a single tuple is returned, else a list of tuples is
            returned where the results are in the same order as the input images.
        """
        if not isinstance(input, list):
            input = [input]
        if not isinstance(id, list):
            id = [id]
        assert len(id) == len(set(id)), 'all input_ids must be unique'
        assert len(input) == len(id), 'must have same number of images as ids'
        # start extractor subprocesses for each id if one doesn't already exist
        for input_id in id:
            if input_id not in self.processes:
                self._setup_input_stream(input_id)
        # start processes
        for image, input_id in zip(input, id):
            self.connections[input_id].send(image)
        # retrieve results
        results = []
        for input_id in id:
            results.append(self.connections[input_id].recv())
        if len(input) == 1:
            return results[0]
        return results

def landmark_array(mp_lm_obj, norm=True):
    """ Takes a mediapipe landmark object and reformats the data into a 33x4 numpy
    array. Row indices are the same as the mediapipe landmark object indices provided
    by constants in the mp.solutions.pose.PoseLandmark module. 
    e.g. mp.solutions.pose.PoseLandmark.RIGHT_EAR is the index of the right
    ear's landmark.
    
    norm (bool): whether or not to translate and rotate the points so that the 
        middle of the hips is the origin of the coordinate frame and facing the negative
        z direction.
    """
    landmarks = mp_lm_obj.landmark
    lm_array = np.array([
        (lm.x, lm.y, lm.z, lm.visibility) for lm in landmarks
    ])
    if normalize:
        lm_array = normalize(lm_array)
    return lm_array

def weighted_average(landmarks):
    """ Combines multiple landmark numpy arrays using the last column
    (the visibility) as the weight. The idea is to combine multiple views
    of the same person and weight the average by how confident mediapipe
    is that the body part is not occluded.
    """
    weighted_sum = np.sum(
        [(lm[:,:].T * lm[:,-1]).T for lm in landmarks], axis=0
    )
    norm_factors = 1 / np.sum([lm[:,-1] for lm in landmarks], axis=0)
    norm_sum = (weighted_sum.T * norm_factors).T
    return norm_sum

def sideviews(lm_array, ground=True):
    """ Extracts pose landmarks from the image and further breaks them up into
    two 8x3 numpy arrays (for the right and left side of the body). The
    landmarks of the arrays are indexed in the following order:
    (ankle, knee, hip, shoulder, ear, elbow, wrist, index)
    The module contains appropriately named constant attributes if you need
    to index a specific landmark.

    The columns of the arrays are x, y and visibility respectively.

    If ground is True, translate points so that the ankle is the origin.
    
    The return value is a 2-tuple (left, right) where left and right are the
    respective side's numpy arrays.

    if lm_array is None, returns None
    """
    if lm_array is None:
        return lm_array
    idxs = mp.solutions.pose.PoseLandmark
    right = lm_array[[
        idxs.RIGHT_ANKLE, idxs.RIGHT_KNEE, idxs.RIGHT_HIP, idxs.RIGHT_SHOULDER,
        idxs.RIGHT_EAR, idxs.RIGHT_ELBOW, idxs.RIGHT_WRIST, idxs.RIGHT_INDEX
    ]][:,[2,1,3]]
    left = lm_array[[
        idxs.LEFT_ANKLE, idxs.LEFT_KNEE, idxs.LEFT_HIP, idxs.LEFT_SHOULDER,
        idxs.LEFT_EAR, idxs.LEFT_ELBOW, idxs.LEFT_WRIST, idxs.LEFT_INDEX
    ]][:,[2,1,3]]
    #flip vertically
    right[:, 1] = -right[:, 1]
    left[:, 1] = -left[:, 1]
    if ground:
        # make ankle the origin
        right[:, :-1] -= right[ANKLE, :-1]
        left[:, :-1] -= left[ANKLE, :-1]
    return left, right

def normalize(lm_array):
    """ Corrects position and rotation of a 33x4 landmark numpy array such that
    the location and orientation of the camera used to capture the base image
    make no difference on the output landmarks.
    """
    # very important that translation is done before rotation
    lm_array = _normalize_translation(lm_array)
    lm_array = _normalize_rotation(lm_array)
    return lm_array

def _normalize_translation(lm_array):
    """ Takes a 33x4 landmark numpy array and translates the points so that
    the origin (0, 0, 0) is located at the center of the hips.
    """
    assert isinstance(lm_array, np.ndarray) and lm_array.shape == (33, 4), \
        'lm_array is not a proper landmark array (33x4)'
    # find avg hip position
    idxs = mp.solutions.pose.PoseLandmark
    left_hip = lm_array[idxs.LEFT_HIP][:3] # don't need visibility
    right_hip = lm_array[idxs.RIGHT_HIP][:3]
    hip_center = np.mean([left_hip, right_hip], axis=0)
    # subtract center from all points
    lm_array[:,:3] -= hip_center
    return lm_array

def _normalize_rotation(lm_array):
    """ Takes a 33x4 landmark numpy array and rotates the points so that "forward"
    is in the negative z direction (towards the camera).
    """
    assert isinstance(lm_array, np.ndarray) and lm_array.shape == (33, 4)
    # find angle between left hip vector and x axis on the xz plane.
    idxs = mp.solutions.pose.PoseLandmark
    left_hip_vec = lm_array[idxs.LEFT_HIP][[0,2]] # ignore y coord
    x_axis_vec = np.array([1,0])
    theta = -_angle_between_vecs(left_hip_vec, x_axis_vec)
    # flip angle if in third or fourth quandrants
    if left_hip_vec[1] < 0:
        theta *= -1
    # rotate all points by theta_x
    rot_mat = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    lm_array[:,[0,2]] = lm_array[:,[0,2]] @ rot_mat
    return lm_array

def _angle_between_vecs(v1, v2):
    """ returns the angle between two vectors (in randians)
    Reference: https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.dot(v1_u, v2_u))

def _pose_estimator_proc(con, **kwargs):
    """ Function to be started as a subprocess. Parent sends it images which it
    extracts pose estimates from.
    """
    extractor = mp.solutions.pose.Pose(**kwargs)
    try:
        while(True):
            image = con.recv()
            result = extractor.process(image)
            con.send((result.pose_landmarks, result.pose_world_landmarks))
    except KeyboardInterrupt:
        pass
    

if __name__ == '__main__':
    # test landmark extraction
    from PIL import Image
    img = np.array(Image.open('deadlift.jpeg'))
    lme = LandmarkExtractor()
    _, mp_lm_world_obj = lme.extract_landmarks(img)
    lm_array = landmark_array(mp_lm_world_obj)
    print(lm_array)
    left, right = sideviews(lm_array)
    print(left, right)
    avg = weighted_average([left, right])
    print(avg)
    lme.kill_processes()

