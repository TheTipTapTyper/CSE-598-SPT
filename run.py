"""
Author: Joshua Martin
Email: jmmartin397@protonmail.com
Created: 3/12/2022
Class: CSE 598 Perception in Robotics
Project: Deadlift Critic
"""
from renderer import Renderer
import lm_extractor as lm_ex
from lm_extractor import LandmarkExtractor, normalize
from vid_man import VideoManager
import cv2
import numpy as np        
import calibrate
from reconstruct import Reconstructor3D


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
    front_view_arrays_dict = dict()
    results = le.extract_landmarks(raw_images, list(range(len(raw_images))))
    if not isinstance(results, list):
        results = [results]
    for idx, (image, result) in enumerate(zip(raw_images, results)):
        mp_lm_obj, mp_lm_world_obj = result
        if mp_lm_obj is None: # pose detection failed
            top_row_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            break
        lm_array = lm_ex.landmark_array(mp_lm_world_obj, norm=True)
        left, right = lm_ex.sideviews(lm_array, ground=True)
        top_row_images.append(re.render_landmarks(image, mp_lm_obj))
        side_view_arrays_dict['left{}'.format(idx)] = left
        side_view_arrays_dict['right{}'.format(idx)] = right
        front_view_arrays_dict['front{}'.format(idx)] = lm_array
    # calculate weighted average of sideviews, if there are any
    sv_w_avg_array_dict = dict()
    fv_w_avg_array_dict = dict()
    if len(side_view_arrays_dict) > 0: # at least one person in view
        sv_w_avg_array_dict['weighted avg'] = lm_ex.weighted_average(
            side_view_arrays_dict.values()
        )
        fv_w_avg_array_dict['weighted avg'] = lm_ex.weighted_average(
            front_view_arrays_dict.values()
        )
    # render side view images
    #sv1_image = re.render_sideview(side_view_arrays_dict, image_id='sv')
    sv2_image = re.render_sideview(sv_w_avg_array_dict, image_id='sv')
    # fv1_image = re.render_frontview(front_view_arrays_dict, image_id='fv1')
    fv2_image = re.render_frontview(fv_w_avg_array_dict, image_id='fv2')
    return top_row_images, sv2_image, fv2_image

def render_images_3d_recon(raw_images, le, re, recon):
    top_row_images = []
    side_view_arrays_dict = dict()
    front_view_arrays_dict = dict()
    results = le.extract_landmarks(raw_images, list(range(len(raw_images))))
    if not isinstance(results, list):
        results = [results]
    lm_arrays = []
    height, width, _ = raw_images[0].shape
    for idx, (image, result) in enumerate(zip(raw_images, results)):
        mp_lm_obj, mp_lm_world_obj = result
        if mp_lm_obj is None: # pose detection failed
            top_row_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            break
        lm_array = lm_ex.landmark_array(mp_lm_obj, norm=False)
        lm_array[:, 0] *= width
        lm_array[:, 1] *= height
        lm_arrays.append(lm_array)
        top_row_images.append(re.render_landmarks(image, mp_lm_obj))
    lm_array = recon.reconstruct(lm_arrays[0], lm_arrays[1])
    lm_array[:, [0,1,2]] /= 10
    lm_array[:, [0,1]] = lm_array[:, [1,0]]
    lm_array[:,1] = -lm_array[:,1]
    lm_array = normalize(lm_array)
    left, right = lm_ex.sideviews(lm_array, ground=True)
    side_view_arrays_dict['left'] = left
    side_view_arrays_dict['right'] = right
    front_view_arrays_dict['front'] = lm_array
    # calculate weighted average of sideviews, if there are any
    sv_w_avg_array_dict = dict()
    if len(side_view_arrays_dict) > 0: # at least one person in view
        sv_w_avg_array_dict['weighted avg'] = lm_ex.weighted_average(
            side_view_arrays_dict.values()
        )
    # render side view images
    #sv1_image = re.render_sideview(side_view_arrays_dict, image_id='sv')
    sv2_image = re.render_sideview(sv_w_avg_array_dict, image_id='sv')
    fv_image = re.render_frontview(front_view_arrays_dict, image_id='fv')
    return top_row_images, sv2_image, fv_image


def run(inputs, proj_mat1=None, proj_mat2=None, cam_cal_param_files=None, output_fn=None, width=1920, height=1080):
    vm = setup_video_manager(inputs, cam_cal_param_files, output_fn, width, height)
    is_webcam = isinstance(inputs[0], int)
    re = Renderer()
    le = LandmarkExtractor(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.75,
        smooth_landmarks=True,
        model_complexity=1,
    )
    recon = None
    if proj_mat1 is not None and proj_mat2 is not None:
        recon = Reconstructor3D(proj_mat1, proj_mat2)
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
            if recon is None:
                top_row_images, left_sv_image, right_sv_image = render_images(
                    raw_images, le, re
                )
            else:
                top_row_images, left_sv_image, right_sv_image = render_images_3d_recon(
                    raw_images, le, re, recon
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
        le.kill_processes()


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
    run_no = 2
    with_film_fn = 'sample_video_pairs/cam_film_run{}.mp4'.format(run_no)
    no_film_fn = 'sample_video_pairs/cam_no_film_run{}.mp4'.format(run_no)
    inputs = [with_film_fn, no_film_fn]

    inputs = [
        '3d_test_vids/cam0_test.mp4',
        '3d_test_vids/cam1_test.mp4'
    ]


    proj_mat1 = '3d_test_vids/cam0_proj_mat.txt'
    proj_mat2 = '3d_test_vids/cam1_proj_mat.txt'

    calibration_params = [
        'with_film_calibration.params',
        'no_film_calibration.params'
    ]
    #inputs = [1]
    run(inputs, proj_mat1=proj_mat1, proj_mat2=proj_mat2)#, cam_cal_param_files=calibration_params)