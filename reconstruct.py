import cv2
from scipy import linalg
import glob
import numpy as np
import lm_extractor
from lm_extractor import landmark_array
from PIL import Image
import matplotlib.pyplot as plt




class Reconstructor3D:
    def __init__(self, proj_mat1_path, proj_mat2_path):
        self.proj_mat1 = np.loadtxt(proj_mat1_path, dtype=float)
        self.proj_mat2 = np.loadtxt(proj_mat2_path, dtype=float)

    # direct linear transformation  
    def _DLT(self, point1, point2):
        """ Uses direct linear transformation to triangulate a point in R3 given
        two views of that point in R2.
        point[1,2]: 1x2 numpy array (x, y)
        Returns: 1x3 numpy array (x, y, z)
        """
        A = [
            point1[1]*self.proj_mat1[2,:] - self.proj_mat1[1,:],
            self.proj_mat1[0,:] - point1[0]*self.proj_mat1[2,:],
            point2[1]*self.proj_mat2[2,:] - self.proj_mat2[1,:],
            self.proj_mat2[0,:] - point2[0]*self.proj_mat2[2,:]
        ]
        
        A = np.array(A).reshape((4,4))
        B = A.transpose() @ A
        _, _, Vh = linalg.svd(B, full_matrices = False)
    
        triangulated_point = Vh[3,0:3]
        return triangulated_point

    def reconstruct(self, lm_array1, lm_array2):
        """ Uses direct linear transformation to perform 3D reconstruction of landmarks.
        Input:
            lm_array1: 33x4 numpy array of landmarks. Calculated from image taken on camera corresponding
                to the the self.proj_mat1 projection matrix
            lm_array2: Same as lm_array1 except for corresponding to self.proj_mat2
        Returns: reconstructed 33x4 numpy array of landmarks 
        """
        triangulated_points = []
        for point1, point2 in zip(lm_array1, lm_array2):
            triangulation = self._DLT(point1[:2], point2[:2])
            triangulated_points.append(triangulation)
        recon_lm_array = np.ones((33,4))
        recon_lm_array[:,:-1] = np.array(triangulated_points)
        return recon_lm_array


    def cv_triangulate(self, lm_array1, lm_array2):
        """ Uses direct linear transformation to perform 3D reconstruction of landmarks.
        Input:
            lm_array1: 33x4 numpy array of landmarks. Calculated from image taken on camera corresponding
                to the the self.proj_mat1 projection matrix
            lm_array2: Same as lm_array1 except for corresponding to self.proj_mat2
        Returns: reconstructed 33x4 numpy array of landmarks 
        """
        cv_triangulation = []
        for point1, point2 in zip(lm_array1, lm_array2):
            point1 = point1[:2].reshape(2,1)
            point2 = point2[:2].reshape(2,1)
            triangulation = cv2.triangulatePoints(self.proj_mat1, self.proj_mat2, 
                point1, point2)
            output_point = triangulation[:-1] / triangulation[-1]
            cv_triangulation.append(output_point.T)
        recon_lm_array = np.ones((33,4))
        recon_lm_array[:,:-1] = np.vstack(cv_triangulation)
        return recon_lm_array


# testing 
if __name__ == "__main__":
    imagesLeft = sorted(glob.glob('camera_calibration/images/poseLeft/*.png'))
    imagesRight = sorted(glob.glob('camera_calibration/images/poseRIght/*.png'))

    proj_matL = 'camera_calibration/projMatrixL.txt'
    proj_matR = 'camera_calibration/projMatrixR.txt'
    recon = Reconstructor3D(proj_matL, proj_matR)


    lme = lm_extractor.LandmarkExtractor(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5
    )

    for left_image, right_image in zip(imagesLeft,imagesRight):

        image1 = np.array(Image.open(left_image))
        image2 = np.array(Image.open(right_image))
        y_pix_count, x_pix_count, _ = image1.shape

        results1 ,  results2 = lme.extract_landmarks([image1,image2],[1,2])
        lm_array1 = landmark_array(results1[0], norm=False)
        lm_array1[:, 0] *= x_pix_count
        lm_array1[:, 1] *= y_pix_count
        lm_array2 = landmark_array(results2[0], norm=False)
        lm_array2[:, 0] *= x_pix_count
        lm_array2[:, 1] *= y_pix_count

        print(lm_array1)

        #return x,y,z but not visibility
        lm_array = recon.reconstruct(lm_array1, lm_array2)
        print(lm_array)
        input()
        # plot_points(dlt_triangulation_points)

    lme.kill_processes()