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
    def __init__(self, **kwargs):
        self.pose_kwargs = kwargs
        self.pose_extractor_instances = {}

    def parallel_extract(self, images, input_ids, extract_type='all', **kwargs):
        """ uses multiprocessing to execute extraction methods in parallel. 
        images is a list of images that correspond to the input streams identified
        by the corresponding ids in input_ids. They must appear in the same order.
        extract_type (str):
            'all': extract all landmarks
            'side': extract sideview landmarks
        all kwargs are passed to the relevant extraction method
        returns a list of result tuples in the same order as the images and input_ids.

        ex.
        images = [img1, img2, img3]
        input_ids = ['vid1.mp4', 'vid2.mp4', 'vid3.mp4']
        results = le.parallel_extract(images, input_ids, extract_type='side')
        left1, right1, mp_lm_obj1 = results[0]
        left2, right2, mp_lm_obj2 = results[1]
        left3, right3, mp_lm_obj3 = results[2]
        """
        
        if extract_type == 'all':
            target_func = self._extract_landmarks_proc
        elif extract_type == 'side':
            target_func = self._extract_sideview_landmarks_proc
        else:
            raise ValueError('extract_type "{}" is invalid'.format(extract_type))
        processes = []
        queues = []
        print('launching processes')
        for image, input_id in zip(images, input_ids):
            queue = mproc.Queue()
            proc = mproc.Process(target=lambda: target_func(
                queue, image, input_id=input_id, **kwargs
            ))
            proc.start()
            processes.append(proc)
            queues.append(queue)
        results = []
        print('processes started')
        # retrieve all results from child processes
        for queue, input_id in zip(queues, input_ids):
            result, pose_instance = queue.get()
            self.pose_extractor_instances[input_id] = pose_instance
            results.append(result)
        print('results retrieved')
        # wait for all child processes to terminate
        for proc in processes:
            proc.join()
        print('child procs joined')
        return results

    def _extract_landmarks_proc(self, queue, image, input_id, **kwargs):
        print('child {} running'.format(input_id))
        result = self.extract_landmarks(image, input_id=input_id, **kwargs)
        queue.put((result, self.pose_extractor_instances[input_id]))
        print('child {} finished'.format(input_id))

    def _extract_sideview_landmarks_proc(self, queue, image, input_id, **kwargs):
        print('child {} running'.format(input_id))
        result = self.extract_sideview_landmarks(image, input_id=input_id, **kwargs)
        queue.put((result, self.pose_extractor_instances[input_id]))
        print('child {} finished'.format(input_id))


    def extract_landmarks(self, image, input_id='default', world_lms=True):
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

        The columns of the arrays are x, y, z and visibility respectively.

        If the pose estimator failed to detect a person, this method returns None.

        image is expected to be in RGB format
        """
        if input_id not in self.pose_extractor_instances:
            self.pose_extractor_instances[input_id] = mp.solutions.pose.Pose(
                **self.pose_kwargs
            )
        image.flags.writeable = False
        results = self.pose_extractor_instances[input_id].process(image)
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
        if world_lms:
            lm_array = self._normalize_world_landmarks(lm_array)
        mp_lm_obj = results.pose_landmarks
        return lm_array, mp_lm_obj

    def extract_sideview_landmarks(self, image, ground=True, **kwargs):
        """Extracts pose landmarks from the image and further breaks them up into
        two 8x3 numpy arrays (for the right and left side of the body). The
        landmarks of the arrays are indexed in the following order:
        (ankle, knee, hip, shoulder, ear, elbow, wrist, index)
        The module contains appropriately named constant attributes if you need
        to index a specific landmark.

        The columns of the arrays are x, y and visibility respectively.
        
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
        ]][:,[2,1,3]]
        left = lm_array[[
            idxs.LEFT_ANKLE, idxs.LEFT_KNEE, idxs.LEFT_HIP, idxs.LEFT_SHOULDER,
            idxs.LEFT_EAR, idxs.LEFT_ELBOW, idxs.LEFT_WRIST, idxs.LEFT_INDEX
        ]][:,[2,1,3]]
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
        
    def _angle_between_vecs(self, v1, v2):
        """ returns the angle between two vectors (in randians)
        Reference: https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
        """
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        # return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        return np.arccos(np.dot(v1_u, v2_u))

    def _normalize_world_landmarks(self, lm_array):
        """ Take array of world (coordinate frame centered on hips) landmarks
        and rotate so that hips face directly in the -z diraction (towards camera).
        """
        # find angle between left hip vector and x axis on the xz plane.
        idxs = mp.solutions.pose.PoseLandmark
        left_hip_vec = lm_array[idxs.LEFT_HIP][[0,2]] # ignore y coord
        x_axis_vec = np.array([1,0])
        theta = -self._angle_between_vecs(left_hip_vec, x_axis_vec)
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

