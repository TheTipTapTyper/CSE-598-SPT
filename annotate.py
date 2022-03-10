import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import os
import pickle
import pose_utils
import time
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


# old code

# def annotate_vid(file_path, complexity=1):
#     # For webcam input:
#     cap = cv2.VideoCapture(file_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     output_vid_path = os.path.join(
#         os.path.dirname(file_path), 
#         'annotated_complexity_{}_'.format(complexity) + os.path.basename(file_path)
#     )
#     output_landmarks_path = os.path.join(
#         os.path.dirname(file_path), 
#         'landmarks_complexity_{}_'.format(complexity) + os.path.basename(file_path).split('.')[0]
#     )
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     width = 1920
#     height = 1080
#     out = cv2.VideoWriter(output_vid_path, fourcc, fps, (width,height))
#     all_landmarks = []
#     all_world_landmarks = []
#     with mp_pose.Pose(
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.75,
#         model_complexity=complexity,
#         smooth_landmarks=False) as pose:
#         while cap.isOpened():
#             success, image = cap.read()
#             if not success:
#                 break

#             # To improve performance, optionally mark the image as not writeable to
#             # pass by reference.
#             image.flags.writeable = False
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             results = pose.process(image)
#             all_landmarks.append(results.pose_landmarks)
#             all_world_landmarks.append(results.pose_world_landmarks)

#             # Draw the pose annotation on the image.
#             image.flags.writeable = True
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#             mp_drawing.draw_landmarks(
#                 image,
#                 results.pose_landmarks,
#                 mp_pose.POSE_CONNECTIONS,
#                 landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
#             image = cv2.resize(image, (1920, 1080))
#             out.write(image)
#             #cv2.imshow('MediaPipe Pose', image)
#     cap.release()
#     out.release()
#     with open(output_landmarks_path, 'wb') as file:
#         pickle.dump({
#             'landmarks': all_landmarks, 
#             'world_landmarks': all_world_landmarks
#         }, file)
#     print('annoted video saved to {}'.format(output_vid_path))
#     print('landmarks saved to {}'.format(output_landmarks_path))


def annotate_image(image, pose_obj, base_width=1280, base_height=720, flip=False, side='right'):
    # Run model inference on image
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_obj.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    if flip:
        image = cv2.flip(image, 1)
    image = cv2.resize(image, (base_width, base_height))

    # Create the sideview image
    if results.pose_world_landmarks is not None:
        side_lms = pose_utils.get_side_landmarks(results.pose_world_landmarks, side=side)
        side_lms = pose_utils.center_side_landmarks(side_lms)
        if flip:
            side_lms = pose_utils.mirror_landmarks(side_lms)
        side_image = pose_utils.side_lms_to_numpy(side_lms)
        side_image = cv2.cvtColor(side_image, cv2.COLOR_BGR2RGB)
        side_image = cv2.resize(side_image, (base_width // 2, base_height))
    else: # no person detected
        side_image = np.ones((base_height, base_width // 2, 3), dtype=np.uint8) * 255
    # stack images together horizontally
    image = np.hstack((image, side_image))

    return image


def annotate_video(input=1, display=True, out_file_path=None, complexity=1, 
             base_width=1280, base_height=720, **kwargs):
    cap = cv2.VideoCapture(input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    is_webcam = (input == 0) # flip if input is webcam
    if out_file_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_file_path, fourcc, fps, (base_width, base_height))
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.75,
        model_complexity=complexity,
        smooth_landmarks=True) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                if is_webcam:
                    continue
                else:
                    break
            image = annotate_image(image, pose, base_width=base_width, 
                base_height=base_height, flip=is_webcam, **kwargs)
            if out_file_path is not None: # save to file
                out.write(image)
            if display:
                cv2.imshow('Deadlift Form Corrector', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
    cap.release()
    if out_file_path is not None:
        out.release()


def compare_landmarks(landmarks_filename):
    with open(landmarks_filename, 'rb') as file:
        lm_dict = pickle.load(file)
    world_lms = lm_dict['world_landmarks']
    lms = lm_dict['landmarks']

    #world_right_lms = [pose_utils.get_side_landmarks(lm, side='right') for lm in world_lms]
    #plot_side_landsmarks(world_right_lms[0])
    fig, ax = plt.subplots()
    ax.set_xlim((-1,1))
    ax.set_ylim((-1,1))
    plt.ion()
    plt.show()

    for lmw, lm in zip(world_lms, lms):
        ax.set_xlim((-1,1))
        ax.set_ylim((-1,1))
        left_world = pose_utils.get_side_landmarks(lmw, side='left')
        right_world = pose_utils.get_side_landmarks(lmw, side='right')
        left = pose_utils.get_side_landmarks(lm, side='left')
        right = pose_utils.get_side_landmarks(lm, side='right')
        pose_utils.draw_side(ax, left_world, 'g')
        pose_utils.draw_side(ax, right_world, 'r')
        pose_utils.draw_side(ax, left, 'b')
        pose_utils.draw_side(ax, right, 'k')
        input()
        ax.clear()



# the following command cuts video starting at 13 seconds and ends at the 26 second mark
# ffmpeg -i output1.mp4 -ss 00:00:13 -to 00:00:26 -c copy output1_trimmed.mp4

if __name__ == '__main__':
    #annotate_vid('./calibration_code/output1_trimmed.mp4', 1)
    #foo('./calibration_code/landmarks_complexity_1_output1_trimmed')
    annotate_video(base_width=640, base_height=480)#input='./calibration_code/output1_trimmed.mp4')