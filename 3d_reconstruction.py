import cv2
from scipy import linalg
import glob
import numpy as np
import lm_extractor
from PIL import Image
import matplotlib.pyplot as plt

def filter_2D_points(l_landmarks, r_landmarks):
    """ This function filters out the third coordinate and visibility from the landmarks.
    It takes in two landmark arrays and then returns two arrays consisting of x and y points only 
    """
    
    left_points = []
    for landmark in l_landmarks.landmark:
        point = [landmark.x, landmark.y]
        left_points.append(point)
    
    right_points = []
    for landmark in r_landmarks.landmark:
        point = [landmark.x, landmark.y]
        right_points.append(point)
    
    return np.array(left_points), np.array(right_points)

def _reduce_dimension(P):
    """ This method takes a 33x3x1 array to reduce its dimensions to 33x3
    """
    points = [i[0] for i in P]
    reduced_array = [points[0]/ points[3], points[1]/points[3], points[2]/points[3]]
    return reduced_array

def cv_triangulate(l_proj_matrix,r_proj_matrix,l_point,r_point):
    """ This method uses the OpenCV triangulation method to triangulate points
    It takes in 2 projection matrices and 2 points, and then returns the triangulated point
    """
    
    cv_triangulation = []
    for index in range(0, 33):
        triangulation = cv2.triangulatePoints(np.array(l_proj_matrix), np.array(r_proj_matrix), l_point[index], r_point[index])
        cv_triangulation.append(_reduce_dimension(triangulation))
    return np.array(cv_triangulation)


# direct linear transformation  
def _DLT(l_proj_matrix, r_proj_matrix, l_point, r_point):
    """ This method takes in two projection matrices and two points
    and then uses direct linear transformation to return a triangulated point
    """
    A = [
        l_point[1]*l_proj_matrix[2,:] - l_proj_matrix[1,:],
        l_proj_matrix[0,:] - l_point[0]*l_proj_matrix[2,:],
        r_point[1]*r_proj_matrix[2,:] - r_proj_matrix[1,:],
        r_proj_matrix[0,:] - r_point[0]*r_proj_matrix[2,:]
    ]
    
    A = np.array(A).reshape((4,4))
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices = False)
 
    triangulated_point = Vh[3,0:3]
    return triangulated_point


def dlt_triangulation(l_proj_matrix, r_proj_matrix, l_point, r_point):
    """ This method uses the _DLT function on a series of points to triangulate all of them.
    It takes in two projection matrices and two arrays of points, and then returns an array of triangulated points
    """
    
    triangulated_points = []
    for index, point in enumerate(l_point):
        triangulation = _DLT(np.array(l_proj_matrix), np.array(r_proj_matrix), point, r_point[index])
        triangulated_points.append(triangulation)
    return np.array(triangulated_points)

def add_visibility(points, l_landmark, r_landmark):
    """ 
    this method take 3 variable:
        3d points  
        left landmark
        right landmark
    
    returns a 3d point with avg of the left and right landmark'visibility 
    """

    points_with_visibility  = []
    for index, (l, r) in enumerate(zip(l_landmark.landmark, r_landmark.landmark)):
        points_with_visibility.append(np.append(points[index], (l.visibility + r.visibility) /  2))
    return np.array(points_with_visibility)

def two_d_to_three_d_reconstruction(l_proj_matrix, r_proj_matrix, l_landmark, r_landmark):
    """ 
    this method take 4 variable:
        left  projection matrix
        right projection matrix  
        left landmark 
        right landmark

    return a triangulate 3d point with visibility np array of size 33x4
    """

    # you only need return either cv_triangulation_points_with_visibility or dlt_triangulation_points_with_visibility
    left_points, right_points = filter_2D_points(l_landmark, r_landmark)
    dlt_triangulation_points = dlt_triangulation(l_proj_matrix, r_proj_matrix, left_points, right_points)
    
    return dlt_triangulation_points

def plot_points(points):
    """Given a set of points, this function plots them in 3D"""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(33):
        ax.text(points[i,0], points[i,1], points[i,2], str(i))
        ax.scatter(zs = points[i:i+1,0], ys = points[i:i+1,1], xs = points[i:i+1,2])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(10, -90)
    plt.pause(1000)
    ax.cla()

def get_projection_matrices():
    """Gets the left and right projection matrix of the project"""
    
    left_projection_matrix = np.loadtxt("camera_calibration/projMatrixL.txt",dtype=float)
    right_projection_matrix = np.loadtxt("camera_calibration/projMatrixR.txt", dtype=float)
    
    return left_projection_matrix, right_projection_matrix

# testing 
if __name__ == "__main__":
    imagesLeft = sorted(glob.glob('camera_calibration/images/poseLeft/*.png'))
    imagesRight = sorted(glob.glob('camera_calibration/images/poseRight/*.png'))
    
    left_projection_matrix, right_projection_matrix = get_projection_matrices()

    lme = lm_extractor.LandmarkExtractor(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5
    )

    for left_image, right_image in zip(imagesLeft,imagesRight):

        image1 = np.array(Image.open(left_image))
        image2 = np.array(Image.open(right_image))

        land_marks ,  land_marks1 = lme.extract_landmarks([image1,image2],[1,2])

        #return x,y,z but not visibility
        dlt_triangulation_points = two_d_to_three_d_reconstruction(left_projection_matrix, right_projection_matrix, land_marks[0], land_marks1[0])
        print("dlt_triangulation_points shape", dlt_triangulation_points.shape)
        print("dlt_triangulation_points", dlt_triangulation_points)

        # plot_points(dlt_triangulation_points)
        break

    lme.kill_processes()