import cv2
import mediapipe as mp
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

IMAGE_FILES = []

imagesLeft = sorted(glob.glob('camera_calibration\images\poseLeft\*.png'))
imagesRight = sorted(glob.glob('camera_calibration\images\poseRight\*.png'))

#gets the X and Y values only from the landmarks
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

#Direct Linear Transformation form triangulation
def DLT(P1, P2, point1, point2):
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    
    A = np.array(A).reshape((4,4))

    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)
 
    return Vh[3,0:3]/Vh[3,3]

def triangulation(left_landmark, right_landmark):
    left_points, right_points = filter_2D_points(left_landmark, right_landmark)
    
    # left_camera_matrix = [
    #     [567.81646729, 0, 489.68407262],
    #     [0, 536.034729, 294.69114695],
    #     [0, 0, 1]
    # ]

    # right_camera_matrix = [
    #     [653.48419189, 0, 343.6320513],
    #     [0, 652.83410645, 248.31683002],
    #     [0, 0, 1]
    # ]
    
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


    cv_triangulation = []
    for index in range(0, 33):
        triangulation = cv2.triangulatePoints(np.array(left_projection_matrix), np.array(right_projection_matrix), left_points[index], right_points[index])
        cv_triangulation.append(triangulation[0: -1] / triangulation[-1])

    triangulated_points = []
    for index, point in enumerate(left_points):
        triangulation = DLT(np.array(left_projection_matrix), np.array(right_projection_matrix), point, right_points[index])
        triangulated_points.append(triangulation)

    triangulated_points = np.array(triangulated_points)
    cv_triangulation = np.array(cv_triangulation).reshape(33, 3)
    return triangulated_points, cv_triangulation

def two_d_to_three_d_reconstruction(i1,i2, idx):

    # for stiaic iamges
    BG_COLOR = (192, 192, 192)

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:
        
        image1 = cv2.imread(i1)
        image2 = cv2.imread(i2)

        image1_height, image1_width, _ = image1.shape
        image2_height, image2_width, _ = image2.shape

        # Convert the BGR image to RGB before processing.
        results1 = pose.process(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
        results2 = pose.process(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

        print(
        f'Nose coordinates for image1: ('
        f'{results1.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image1_width}, '
        f'{results1.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image1_height})'
        )

        print(
        f'Nose coordinates for image2: ('
        f'{results2.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image2_width}, '
        f'{results2.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image2_height})'
        )


        #image 1
        annotated_image = image1.copy()
        # Draw segmentation on the image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack((results1.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image1.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        annotated_image = np.where(condition, annotated_image, bg_image)
        # Draw pose landmarks on the image.
        mp_drawing.draw_landmarks(
            annotated_image,
            results1.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        cv2.imwrite('camera_calibration\images\landmarks\image_' + str(idx) + '_L.png', annotated_image)
        # Plot pose world landmarks.
        mp_drawing.plot_landmarks(
            results1.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

        #iamge2
        annotated_image2 = image2.copy()
        # Draw segmentation on the image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack((results2.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image2.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        annotated_image2 = np.where(condition, annotated_image2, bg_image)
        # Draw pose landmarks on the image.
        mp_drawing.draw_landmarks(
            annotated_image2,
            results2.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imwrite('camera_calibration\images\landmarks\image_' + str(idx) + '_R.png', annotated_image2)
        # Plot pose world landmarks.
        # mp_drawing.plot_landmarks(
        #     results2.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

        tri, cv = triangulation(results1.pose_landmarks, results2.pose_landmarks)

        for index, (point1, point2) in enumerate(zip(tri, cv)):
            print("mediapipe variables:", str(results1.pose_landmarks.landmark[index].x) + ", " + str(results1.pose_landmarks.landmark[index].y) + ", " + str(results1.pose_landmarks.landmark[index].z))
            print("DLT variables:", point1)
            print("CV variables:", point2)
            print("-----------------------------------")


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        connections = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,8], [1,9], [2,8], [5,9], [8,9], [0, 10], [0, 11]]
        for _c in connections:
            print(cv[_c[0]])
            print(cv[_c[1]])
            ax.plot(xs = [cv[_c[0],0], cv[_c[1],0]], ys = [cv[_c[0],1], cv[_c[1],1]], zs = [cv[_c[0],2], cv[_c[1],2]], c = 'red')
        
        plt.show()


idx = 1
for i1 , i2 in zip(imagesLeft,imagesRight):
    two_d_to_three_d_reconstruction(i1,i2, idx)
    idx += 1
    break