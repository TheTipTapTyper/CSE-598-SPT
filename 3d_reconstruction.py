import cv2
import mediapipe as mp
import glob
import numpy as np
import lm_extractor
from PIL import Image


def filter_2D_points(left_landmarks, right_landmarks):
    left_points = []
    for landmark in left_landmarks.landmark:
        point = [landmark.x, landmark.y]
        left_points.append(point)
    
    right_points = []
    for landmark in right_landmarks.landmark:
        point = [landmark.x, landmark.y]
        right_points.append(point)
    
    return np.array(left_points), np.array(right_points)

# open cv_triangulate

def helper(P):
    points = [i[0] for i in P]
    return [points[0]/ points[3], points[1]/points[3], points[2]/points[3]]

def cv_triangulate(l_proj_matrix,r_proj_matrix,l_point,r_point):

    cv_triangulation = []
    for index in range(0, 33):
        triangulation = cv2.triangulatePoints(np.array(l_proj_matrix), np.array(r_proj_matrix), l_point[index], r_point[index])
        cv_triangulation.append(helper(triangulation))
    return np.array(cv_triangulation)


# direct linear transformation  
def _DLT(P1, P2, point1, point2):
    
    A = [
        point1[1]*P1[2,:] - P1[1,:],
        P1[0,:] - point1[0]*P1[2,:],
        point2[1]*P2[2,:] - P2[1,:],
        P2[0,:] - point2[0]*P2[2,:]
    ]
    
    A = np.array(A).reshape((4,4))
    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)
 
    return Vh[3,0:3]/Vh[3,3]


def dlt_triangulation(l_point, r_point, l_proj_matrix, r_proj_matrix):

    triangulated_points = []
    for index, point in enumerate(l_point):
        triangulation = _DLT(np.array(l_proj_matrix), np.array(r_proj_matrix), point, r_point[index])
        triangulated_points.append(triangulation)
    return np.array(triangulated_points)


def two_d_to_three_d_reconstruction(l_landmark,r_landmark,left_proj_matrix, right_proj_matrix):

    left_points, right_points = filter_2D_points(l_landmark, r_landmark)

    cv_triangulation_points = cv_triangulate(left_proj_matrix, right_projection_matrix, left_points, right_points)

    dlt_triangulation_points = dlt_triangulation(left_points, right_points, left_proj_matrix, right_proj_matrix)

    # this will return 33x3 landmark np array in world cord
    return cv_triangulation_points, dlt_triangulation_points

# testing 
if __name__ == "__main__":

    imagesLeft = sorted(glob.glob('poseLeft/*.png'))
    imagesRight = sorted(glob.glob('poseRight/*.png'))

    left_camera_matrix = [
        [567.81646729, 0, 489.68407262],
        [0, 536.034729, 294.69114695],
        [0, 0, 1]
    ]

    right_camera_matrix = [
        [653.48419189, 0, 343.6320513],
        [0, 652.83410645, 248.31683002],
        [0, 0, 1]
    ]
    
    left_projection_matrix = [
        [594.43441772,0,291.38330936, 0],
        [0, 594.43441772,264.21261978, 0],
        [0, 0, 1, 0]
 ]
    
    right_projection_matrix = [
        [5.94434418e+02, 0.00000000e+00, 2.91383309e+02, 3.84539378e+05],
        [0.00000000e+00, 5.94434418e+02, 2.64212620e+02, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]
    ]

    lme = lm_extractor.LandmarkExtractor(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5)

    for left_image, right_image in zip(imagesLeft,imagesRight):

        image1 = np.array(Image.open(left_image))
        image2 = np.array(Image.open(right_image))

        land_marks ,  _ = lme.extract_landmarks([image1,image2],[1,2])

        #return x,y,z but not visibility
        cv_triangulation_points, dlt_triangulation_points = two_d_to_three_d_reconstruction(land_marks[0],land_marks[1],left_projection_matrix,right_projection_matrix)
        
        print("CV:" , cv_triangulation_points.shape)
        print("Dlt:" ,dlt_triangulation_points.shape)
        #print(cv_triangulation_points)
        #print(dlt_triangulation_points)
        break
    lme.kill_processes()