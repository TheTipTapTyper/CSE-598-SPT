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
import calibrate


def run(inputs, cam_cal_param_files=None, output_fn=None, width=1920, height=1080):
    is_webcam = isinstance(inputs[0], int)
    re = Renderer()
    le = LandmarkExtractor(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.75,
        smooth_landmarks=True,
        model_complexity=1,
    )
    # setup video streams
    vm = VideoManager()
    for idx, input in enumerate(inputs):
        vm.setup_input(input)
        if cam_cal_param_files is not None:
            cal = calibrate.Calibrator()
            cal.load_from_file(cam_cal_param_files[idx])
            vm.add_input_preprocess(lambda im: cal.undistort(im), input)
    # adjust color space for all input streams
    vm.add_input_preprocess(lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    # optionally setup output stream
    if output_fn is not None:
        vm.setup_output(output_fn, width=width, height=height, fps=10)
    # main loop
    print(vm.is_open())
    try:
        while vm.is_open():
            results = [vm.read(id) for id in inputs]
            images = [img for _, img in results]
            success = all(suc for suc, _ in results)
            if not success:
                if is_webcam:
                    print('read failed. skipping frame')
                    continue
                else:
                    print('read failed. exiting')
                    break
            top_row_images = []
            side_view_arrays_dict = dict()
            for idx, image in enumerate(images):
                extraction = le.extract_sideview_landmarks(image)
                if extraction is None: # pose detection failed
                    top_row_images.append(image)
                    break
                left, right, mp_lm_obj = extraction
                top_row_images.append(re.render_landmarks(image, mp_lm_obj))
                side_view_arrays_dict['left{}'.format(idx)] = left
                side_view_arrays_dict['right{}'.format(idx)] = right
            # calculate weighted average of sideviews, if there are any
            w_avg_array_dict = dict()
            if len(side_view_arrays_dict) > 0: # at least one person in view
                w_avg_array_dict['weighted avg'] = le.weighted_average(
                    side_view_arrays_dict.values()
                )
            # render side view images
            left_sv_image = re.render_sideview(side_view_arrays_dict, image_id='left')
            right_sv_image = re.render_sideview(w_avg_array_dict, image_id='right')
            # resize images before stitching them together
            # top row images are squished horizontally so that they all fit
            resized_top_row_images = []
            num_images = len(top_row_images)
            for image in top_row_images:
                resized_top_row_images.append(cv2.resize(image, 
                    (width//num_images, height//2))
                )
            left_sv_image = cv2.resize(left_sv_image, (width//2, height//2))
            right_sv_image = cv2.resize(right_sv_image, (width//2, height//2))
            # stitch together top image and resize to ensure proper dimensions
            top_image = np.hstack(resized_top_row_images)
            top_image = cv2.resize(top_image, (width, height//2))
            # stitch together full image
            bottom_image = np.hstack([left_sv_image, right_sv_image])
            full_image = np.vstack([top_image, bottom_image])
            # display and save (if recording)
            vm.display(full_image, window_name='Deadlift Critic', fullscreen=True)
            if output_fn is not None:
                vm.write(full_image, output_fn)

    finally:
        vm.release()



if __name__ == '__main__':
    inputs = [0]
    run(inputs)