"""
Author: Joshua Martin
Email: jmmartin397@protonmail.com
Created: 3/27/2022
Class: CSE 598 Perception in Robotics
Project: Deadlift Critic
Reference: https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
"""
import pickle
import cv2
import glob
import numpy as np


# Defining the dimensions of checkerboard
CHECKERBOARD = (8,6)
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

class Calibrator:
    """ This class allows for easy calibration and undistortion of images
    captured on a given camera. Calibration can eaither be performed at runtime
    or calibration parameters can be loaded from a file.
    """
    def __init__(self):
        self.mtx = None
        self.dist = None

    def load_from_file(self, fname):
        """ Loads pickled calibration parameters from file fname.
        """
        with open(fname, 'rb') as file:
            self.mtx, self.dist = pickle.load(file)

    def load_from_images(self, path_to_dir):
        """ Calculates calibration parameters using images of a 8x6 checkerboard
        pattern captured on the camera of interest.
        """
        # Creating vector to store vectors of 3D points for each checkerboard image
        objpoints = []
        # Creating vector to store vectors of 2D points for each checkerboard image
        imgpoints = [] 
        # Defining the world coordinates for 3D points
        objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        # Extracting path of individual image stored in a given directory
        images = glob.glob('{}/*.png'.format(path_to_dir))
        shape = None
        print(len(images))
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            shape = gray.shape[::-1]
            # Find the chess board corners
            # If desired number of corners are found in the image then ret = true
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + 
                cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            # If desired number of corner are detected, refine the pixel coordinates
            if ret == True:
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), CRITERIA)
                
                imgpoints.append(corners2)
        output = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
        self.mtx, self.dist = output[1:3]

    def save(self, fname):
        assert self.mtx is not None and self.dist is not None, 'Cannot save null \
            params to file. Ensure params have be loaded before saving.'
        with open(fname, 'wb') as file:
            pickle.dump([self.mtx, self.dist], file)

    def undistort(self, image, keep_aspect_ratio=False):
        """ Uses the loaded calibration parameters to undistort the image.
        If keep_aspect_ratio is True, the output image will be of the same 
        resolution as the input, but there may be some black regions around
        the edges. If it is False, the image will be cropped so that these
        black regions are eliminated
        """
        newcameramtx = None
        if not keep_aspect_ratio:
            h,  w = image.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                self.mtx, self.dist, (w,h), 1, (w,h)
            )
        # undistort
        dst = cv2.undistort(image, self.mtx, self.dist, None, newcameramtx)
        # crop the image
        if not keep_aspect_ratio:
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
        return dst


if __name__ == '__main__':
    import vid_man

    cal = Calibrator()
    cal.load_from_file('calibration_code_old/home_cam.pickle')
    #cal.load_from_images('calibration_code_old/home_cam')

    vm = vid_man.VideoManager()
    vm.setup_input(0)

    while vm.is_open():
        suc, image = vm.read()[0]
        undistorted_image = cal.undistort(image, True)
        vm.display(image, fullscreen=False, window_name='original')
        vm.display(undistorted_image, fullscreen=False, window_name='corrected')