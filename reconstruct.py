import cv2
import numpy as np

class Reconstructor3D:
    def __init__(self, proj_mat1_path, proj_mat2_path):
        self.proj_mat1 = np.loadtxt(proj_mat1_path, dtype=float)
        self.proj_mat2 = np.loadtxt(proj_mat2_path, dtype=float)

    def reconstruct(self, lm_array1, lm_array2):
        """ Uses opencv's triangulatePoints function to perform 3D reconstruction of landmarks.
        Input:
            lm_array1: 33x4 numpy array of landmarks. Calculated from image taken on camera corresponding
                to the the self.proj_mat1 projection matrix
            lm_array2: Same as lm_array1 except for corresponding to self.proj_mat2
        Returns: reconstructed 33x4 numpy array of landmarks 
        """
        lm_array1_xy_T = lm_array1[:,[0,1]].T
        lm_array2_xy_T = lm_array2[:,[0,1]].T
        triangulated = cv2.triangulatePoints(self.proj_mat1, self.proj_mat2, 
            lm_array1_xy_T, lm_array2_xy_T).T
        # unhomogenize
        output_lm_array = (triangulated.T / triangulated[:, -1]).T
        # replace ones column with average visibility scores
        output_lm_array[:, -1] = np.mean([lm_array1[:, -1], lm_array2[:, -1]], axis=0)
        return output_lm_array


# testing 
if __name__ == "__main__":
    import glob
    import lm_extractor
    from lm_extractor import landmark_array
    from PIL import Image
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