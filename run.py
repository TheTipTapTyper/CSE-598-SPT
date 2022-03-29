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

OUTPUT_FPS = 15


def setup_video_manager(inputs, cam_cal_param_files, output_fn, width, height):
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
        vm.setup_output(output_fn, width=width, height=height, fps=OUTPUT_FPS)
    return vm

def stitch_full_image(top_row_images, left_sv_image, right_sv_image, width, height):
    # if only one top image, add padding to sides
    if len(top_row_images) == 1:
        top_h, top_w, _ = top_row_images[0].shape
        if top_w < width:
            pad_width = width - top_w
            pad = np.zeros((top_h, pad_width // 2, 3), dtype=np.uint8)
            top_row_images[0] = np.hstack([pad, top_row_images[0], pad])
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
    return full_image

def render_images(raw_images, le, re):
    top_row_images = []
    side_view_arrays_dict = dict()
    for idx, image in enumerate(raw_images):
        extraction = le.extract_sideview_landmarks(image)
        if extraction is None: # pose detection failed
            top_row_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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
    return top_row_images, left_sv_image, right_sv_image


def run(inputs, cam_cal_param_files=None, output_fn=None, width=1920, height=1080):
    vm = setup_video_manager(inputs, cam_cal_param_files, output_fn, width, height)
    is_webcam = isinstance(inputs[0], int)
    re = Renderer()
    le = LandmarkExtractor(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.75,
        smooth_landmarks=True,
        model_complexity=1,
    )
    # main loop
    try:
        while vm.is_open():
            results = [vm.read(id) for id in inputs]
            raw_images = [img for _, img in results]
            success = all(suc for suc, _ in results)
            if not success:
                if is_webcam:
                    print('read failed. skipping frame')
                    continue
                else:
                    print('read failed. exiting')
                    break
            top_row_images, left_sv_image, right_sv_image = render_images(
                raw_images, le, re
            )
            full_image = stitch_full_image(
                top_row_images, left_sv_image, right_sv_image, width, height
            )
            # display and save (if recording)
            vm.display(full_image, window_name='Deadlift Critic', fullscreen=True)
            if output_fn is not None:
                vm.write(full_image, output_fn)
    finally:
        vm.release()




def record_2_cams(run_no):
    vm = VideoManager()
    in_no_film = 4
    in_film = 2
    vm.setup_input(in_no_film)
    vm.setup_input(in_film)
    out1 = 'output_3_28_22/cam_no_film_run{}.mp4'.format(run_no)
    out2 = 'output_3_28_22/cam_film_run{}.mp4'.format(run_no)
    vm.setup_output(out1)
    vm.setup_output(out2)
    while vm.is_open():
        suc1, image1 = vm.read(in_no_film)
        suc2, image2 = vm.read(in_film)
        if not suc1 and suc2:
            continue
        vm.display(image1, 'no_film')
        vm.display(image2, 'film')
        vm.write(image1, out1)
        vm.write(image2, out2)


def test_cam_idx(idx):
    vm = VideoManager()
    vm.setup_input(idx)
    while vm.is_open():
        suc, image = vm.read()
        if not suc:
            continue
        vm.display(image)




if __name__ == '__main__':
    run_no = 1
    with_film_fn = 'sample_video_pairs/cam_film_run{}.mp4'.format(run_no)
    no_film_fn = 'sample_video_pairs/cam_no_film_run{}.mp4'.format(run_no)
    inputs = [with_film_fn, no_film_fn]
    
    calibration_params = [
        'with_film_calibration.params',
        'no_film_calibration_params'
    ]
    inputs = [2]
    run(inputs)