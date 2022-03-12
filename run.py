"""
Author: Joshua Martin
Email: jmmartin397@protonmail.com
Created: 3/12/2022
Class: CSE 598 Perception in Robotics
Project: Deadlift Critic
"""
from renderer import Renderer
from lm_extractor import LandmarkExtractor
from vid_man import VideoManager
import cv2
import numpy as np        

def basic_test_of_sideview():
    re = Renderer()
    le = LandmarkExtractor()
    vm = VideoManager()
    vm.setup_input('input/output1_trimmed.mp4')
    image_id = 'left_right'
    try:
        while(vm.is_open()):
            success, image = vm.read()[0]
            if not success:
                break
            left, right = le.extract_sideview_landmarks(image)
            w_avg = le.weighted_average([left, right])
            side_image = re.render_sideview({
                'left': left,
                'right': right,
                'weighted average': w_avg
            }, image_id=image_id)
            cv2.imshow(image_id, side_image)
            if cv2.waitKey(1) & 0xFF == 27:
                exit()
    finally:
        vm.release()


def multiview_video(width=1920, height=1080):
    re = Renderer()
    le = LandmarkExtractor(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.75,
        smooth_landmarks=True,
        model_complexity=1,
    )
    vm = VideoManager()
    vm.setup_input('input/output1_trimmed.mp4')
    vm.setup_input('input/output2_trimmed.mp4')
    vm.add_input_preprocess(lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    vm.setup_output('output/weighted_average3.mp4', width=width, height=height)
    try:
        while(vm.is_open()):
            ((suc1, img1), (suc2, img2)) = vm.read()
            if not (suc1 and suc2):
                break
            left1, right1 = le.extract_sideview_landmarks(img1)
            top_left_img = re.render_landmarks(img1, le.pose_landmarks)
            left2, right2 = le.extract_sideview_landmarks(img2)
            top_right_img = re.render_landmarks(img2, le.pose_landmarks)
            w_avg = le.weighted_average([left1, right1, left2, right2])
            bottom_left_img = re.render_sideview({
                'left1': left1,
                'right1': right1,
                'left2': left2,
                'right2': right2,
            }, image_id='all_sides')
            bottom_right_img = re.render_sideview({'weighted avg': w_avg}, 
                image_id='w_avg')
            top_left_img = cv2.resize(top_left_img, (width//2, height//2))
            top_right_img = cv2.resize(top_right_img, (width//2, height//2))
            bottom_left_img = cv2.resize(bottom_left_img, (width//2, height//2))
            bottom_right_img = cv2.resize(bottom_right_img, (width//2, height//2))
            image = np.vstack([
                np.hstack([top_left_img, top_right_img]), 
                np.hstack([bottom_left_img, bottom_right_img])
            ])
            cv2.namedWindow('Deadlift Critic', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Deadlift Critic',cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Deadlift Critic', image)
            vm.write(image, 0)
            if cv2.waitKey(1) & 0xFF == 27:
                exit()
    finally:
        vm.release()



if __name__ == '__main__':
    #basic_test_of_sideview()
    #multiview_video(640,480)
    multiview_video()