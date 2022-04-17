"""
Author: Joshua Martin
Email: jmmartin397@protonmail.com
Created: 3/12/2022
Class: CSE 598 Perception in Robotics
Project: Deadlift Critic
"""
import matplotlib.pyplot as plt
import numpy as np
from lm_extractor import SHOULDER, EAR, ELBOW, INDEX, angle_between_vecs
import cv2
import mediapipe as mp


SIDE_IMG_WIDTH_M = 2
SIDE_IMG_HEIGHT_M = 2
FIG = 'fig'
BACKGROUND = 'bg' # used in blitting (https://matplotlib.org/stable/tutorials/advanced/blitting.html)
AXES = 'ax'
LINES_DICT = 'lines'
BODY_LINE = 'body'
HEAD_LINE = 'line'
BAR_LINE = 'bar'
DEFAULT_COLORS = [
    'black', 'red', 'blue', 'green', 'pink', 'orage', 'yellow', 'teal',
    'lime', 'purple', 'sandybrown', 'cyan', 'deeppink', 'coral', 'wheat', 'grey'
]
SIDE_VIEW = 'SIDE'
FRONT_VIEW = 'front'
CREATE = 'create'
UPDATE = 'update'
BODY_LMS_HEAD_IDX = 11
BODY_LMS_LEFT_HAND_IDX = 6
BODY_LMS_RIGHT_HAND_IDX = 16
FV_BAR_LEN = 1.25 # meters
FV_PLATE_WIDTH = .1
FV_PLATE_HEIGHT = .4


class Renderer:
    def __init__(self, linewidth=3, joint_size=7, joint_edge_width=1.5):
        self.linewidth = linewidth
        self.joint_size = joint_size
        self.joint_edge_width = joint_edge_width
        self.plots = {}
        self.line_objs = {}
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.view_specific_methods = {
            SIDE_VIEW: {
                CREATE: self._create_sideview_artists,
                UPDATE: self._update_sideview_artists
            },
            FRONT_VIEW: {
                CREATE: self._create_frontview_artists,
                UPDATE: self._update_frontview_artists
            }
        }

    def render_landmarks(self, image, mp_lm_obj):
        """ Draws the mediapipe extracted landmarks onto the input image.
        Image is expected to be in RGB format.
        mp_lm_obj is the landmark object returned medeiapipe, not a numpy array.
        """
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.mp_drawing.draw_landmarks(
            image,
            mp_lm_obj,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        return image

    def render_sideview(self, landmarks_dict, **kwargs):
        """Renders one or more sets of sideview landmarks into an image.
        landmarks_dict must be a dictionary mapping label names to sideview landmark
        numpy arrays (8x3).

        The renderer maintains Line2D objects so that subsequent render_sideview
        calls can simply update the data rather than drawing from scratch. However,
        this requires the label names and image_id to remain unchanged between 
        subsequent calls.
        """
        assert isinstance(landmarks_dict, dict) and all(
            isinstance(arr, np.ndarray) and arr.shape == (8,3) 
            for arr in landmarks_dict.values()
        ), 'landmarks_dict must be a dict: {str : (8x3) np.ndarray}'
        return self._render_view(SIDE_VIEW, landmarks_dict, **kwargs)

    def render_frontview(self, landmarks_dict, **kwargs):
        """Renders one or more sets of full body landmarks into an image.
        landmarks_dict must be a dictionary mapping label names to landmark
        numpy arrays (33x4).

        The renderer maintains Line2D objects so that subsequent render_frontview
        calls can simply update the data rather than drawing from scratch. However,
        this requires the label names and image_id to remain unchanged between 
        subsequent calls.
        """
        assert isinstance(landmarks_dict, dict) and all(
            isinstance(arr, np.ndarray) and arr.shape == (33,4) 
            for arr in landmarks_dict.values()
        ), 'landmarks_dict must be a dict: {str : (33x4) np.ndarray}'
        return self._render_view(FRONT_VIEW, landmarks_dict, **kwargs)

    def _render_view(self, view, landmarks_dict, image_id='default', legend=True,
                        width=640, height=480):
        assert view in self.view_specific_methods, '{} is not a recognized view'.format(view)
        colors = DEFAULT_COLORS[:len(landmarks_dict)]
        if image_id not in self.plots:
            self._setup_view_plot(image_id, width, height)
        # if number of landmark sets has changed (or if this is the first render), draw and save background
        if len(landmarks_dict) != len(self.plots[image_id][LINES_DICT]) or \
            self.plots[image_id][BACKGROUND] is None:
            print(landmarks_dict.keys(), self.plots[image_id][LINES_DICT].keys())
            self._refresh_view(image_id, landmarks_dict, colors, legend, view)
        else: # just update the previous draw Line2D objects
            for (label, landmarks), color in zip(sorted(landmarks_dict.items()), colors): 
                self.view_specific_methods[view][UPDATE](label, landmarks, image_id)
        self._restore_background(image_id)
        self._draw_artists(image_id)
        fig = self.plots[image_id][FIG]
        return np.array(fig.canvas.renderer._renderer)[:,:,:3]

    ## methods shared by all views ##
 
    def _restore_background(self, image_id):
        fig = self.plots[image_id][FIG]
        background_img = self.plots[image_id][BACKGROUND]
        fig.canvas.restore_region(background_img)

    def _save_background_for_blitting(self, image_id):
        fig = self.plots[image_id][FIG]
        fig.canvas.draw()
        background_img = fig.canvas.copy_from_bbox(fig.bbox)
        self.plots[image_id][BACKGROUND] = background_img

    def _refresh_view(self, image_id, landmarks_dict, colors, legend, view):
        """ Reinitializes the Line2D objects of the plot, redraws them and saves 
        the background for blitting.
        """
        for artists_dict in self.plots[image_id][LINES_DICT].values():
            for artist in artists_dict.values():
                artist.remove()
        self.plots[image_id][LINES_DICT] = dict()
        for (label, landmarks), color in zip(sorted(landmarks_dict.items()), colors): 
            self.view_specific_methods[view][CREATE](label, landmarks, 
                image_id, color, legend)
        self._save_background_for_blitting(image_id)

    def _setup_view_plot(self, image_id, width, height):
        """ Sets up the figure and axes of the plot.
        """
        px = 1/plt.rcParams['figure.dpi']
        fig, ax = plt.subplots(figsize=(width*px, height*px))            
        x = SIDE_IMG_WIDTH_M / 2
        ax.set_xlim((-x, x))
        ax.set_ylim((0,SIDE_IMG_HEIGHT_M))
        fig.tight_layout(pad=0)
        self.plots[image_id] = {FIG: fig, AXES: ax, LINES_DICT: dict(), BACKGROUND: None}

    def _draw_artists(self, image_id):
        fig = self.plots[image_id][FIG]
        ax = self.plots[image_id][AXES]
        for label in self.plots[image_id][LINES_DICT]:
            line_objs = self.plots[image_id][LINES_DICT][label].values()
            for line_obj in line_objs:
                ax.draw_artist(line_obj)
        fig.canvas.blit(ax.bbox)
        fig.canvas.flush_events()

    ## front view specific methods

    def _create_frontview_artists(self, label, fv_lm_arr, image_id, color, legend):
        # insert additional shoulder before the elbow (after the ear) for proper plotting
        body_lms = self._prep_lms_for_frontview_plot(fv_lm_arr)
        bar_coords = self._front_view_bar_coords(body_lms)
        body, = self.plots[image_id][AXES].plot(
            body_lms[:,0], body_lms[:,1],
            'o-',
            color=color,
            markersize=self.joint_size,
            markerfacecolor='w',
            markeredgewidth=self.joint_edge_width,
            markeredgecolor=color,
            linewidth=self.linewidth,
            label=label,
            animated=True
        )
        head, = self.plots[image_id][AXES].plot(
            (body_lms[BODY_LMS_HEAD_IDX, 0]), 
            (body_lms[BODY_LMS_HEAD_IDX, 1]),
            'o',
            color=color,
            markersize=self.joint_size * 5,
            markerfacecolor='w',
            markeredgewidth=self.joint_edge_width,
            markeredgecolor=color,
            animated=True
        )
        bar, = self.plots[image_id][AXES].plot(
            (bar_coords[:, 0]), 
            (bar_coords[:, 1]),
            color=color,
            animated=True
        )
        if legend:
            self.plots[image_id][AXES].legend()
        self.plots[image_id][LINES_DICT][label] = {
            BODY_LINE: body, HEAD_LINE: head, BAR_LINE: bar
        }

    def _update_frontview_artists(self, label, fv_lm_arr, image_id):
        body_lms = self._prep_lms_for_frontview_plot(fv_lm_arr)
        bar_coords = self._front_view_bar_coords(body_lms, image_id)
        self.plots[image_id][LINES_DICT][label][BODY_LINE].set_data(
            body_lms[:,0],
            body_lms[:,1]
        )
        self.plots[image_id][LINES_DICT][label][HEAD_LINE].set_data(
            (body_lms[BODY_LMS_HEAD_IDX, 0]), 
            (body_lms[BODY_LMS_HEAD_IDX, 1]),
        )
        self.plots[image_id][LINES_DICT][label][BAR_LINE].set_data(
            (bar_coords[:, 0]), 
            (bar_coords[:, 1]),
        )

    def _front_view_bar_coords(self, body_lms, image_id='foo'):
        """ Calculate the ends of the barbell for the front view based on the
        positions of the hands
        """
        left_hand = body_lms[BODY_LMS_LEFT_HAND_IDX, [0,1]]
        right_hand = body_lms[BODY_LMS_RIGHT_HAND_IDX, [0,1]]
        bar_mid = np.mean([left_hand, right_hand], axis=0)
        # calculates coordinates of left plate
        plate_direction = left_hand - bar_mid # direction this half of the bar is pointing
        plate_direction_unit = plate_direction / np.linalg.norm(plate_direction) # unit vec
        plate_direction_scaled = (FV_BAR_LEN / 2) * plate_direction_unit # scaled vec
        plate_location = plate_direction_scaled + bar_mid # move back to correct location
        theta = angle_between_vecs(plate_direction_unit, np.array([1, 0])) # angle between bar and x axis
        if plate_direction_unit[1] < 0:
            theta = -theta
        plate_bot_left = np.array([
            plate_location[0] + (FV_PLATE_HEIGHT / 2) * np.sin(theta),
            plate_location[1] - (FV_PLATE_HEIGHT / 2) * np.cos(theta),
        ])
        plate_top_left = np.array([
            plate_location[0] - (FV_PLATE_HEIGHT / 2) * np.sin(theta),
            plate_location[1] + (FV_PLATE_HEIGHT / 2) * np.cos(theta),
        ])
        plate_bot_right = np.array([
            plate_bot_left[0] + FV_PLATE_WIDTH * np.cos(theta),
            plate_bot_left[1] + FV_PLATE_WIDTH * np.sin(theta)
        ])
        plate_top_right = np.array([
            plate_top_left[0] + FV_PLATE_WIDTH * np.cos(theta),
            plate_top_left[1] + FV_PLATE_WIDTH * np.sin(theta)
        ])
        left_coords = np.vstack([
            plate_location,
            plate_bot_left,
            plate_bot_right,
            plate_top_right,
            plate_top_left,
            plate_location
        ])
        # coords of right plate are simply the inverted coords of the left, after
        # translating so that bar_mid is the origin
        right_coords = -(left_coords - bar_mid) + bar_mid
        bar_coords = np.vstack([left_coords, right_coords])
        return bar_coords

    def _prep_lms_for_frontview_plot(self, lms):
        """ Create the necessary sequence of landmark points in order to draw
        the frontview in one go. Also ensures the figure is in frame.
        """
        idxs = self.mp_pose.PoseLandmark
        lms = lms.copy()
        # flip y coords so that positive y is up
        lms[:,1] = -lms[:,1]
        #center points so what the lower ankle's y coord is y0
        y0 = min(
            lms[idxs.LEFT_ANKLE][1],
            lms[idxs.RIGHT_ANKLE][1]
        )
        lms[:,1] -= y0
        # sequence body landmarks so that they can be drawn in one go using a
        # Line2D object
        mid_shoulder = np.mean([
            lms[idxs.LEFT_SHOULDER],
            lms[idxs.RIGHT_SHOULDER],
        ], axis=0)
        head = np.mean([
            lms[idxs.LEFT_EAR],
            lms[idxs.RIGHT_EAR],
        ], axis=0)
        return np.vstack([
            lms[idxs.LEFT_ANKLE],
            lms[idxs.LEFT_KNEE],
            lms[idxs.LEFT_HIP],
            lms[idxs.LEFT_SHOULDER],
            lms[idxs.LEFT_ELBOW],
            lms[idxs.LEFT_WRIST],
            lms[idxs.LEFT_INDEX],
            lms[idxs.LEFT_WRIST],
            lms[idxs.LEFT_ELBOW],
            lms[idxs.LEFT_SHOULDER],
            mid_shoulder,
            head,
            mid_shoulder,
            lms[idxs.RIGHT_SHOULDER],
            lms[idxs.RIGHT_ELBOW],
            lms[idxs.RIGHT_WRIST],
            lms[idxs.RIGHT_INDEX],
            lms[idxs.RIGHT_WRIST],
            lms[idxs.RIGHT_ELBOW],
            lms[idxs.RIGHT_SHOULDER],
            lms[idxs.RIGHT_HIP],
            lms[idxs.LEFT_HIP],
            lms[idxs.RIGHT_HIP],
            lms[idxs.RIGHT_KNEE],
            lms[idxs.RIGHT_ANKLE],
            # lms[idxs.LEFT_ANKLE], lms[idxs.LEFT_KNEE], lms[idxs.LEFT_HIP],
            # lms[idxs.LEFT_SHOULDER], lms[idxs.LEFT_ELBOW], lms[idxs.LEFT_WRIST],
            # lms[idxs.LEFT_INDEX], lms[idxs.LEFT_WRIST], lms[idxs.LEFT_ELBOW],
            # lms[idxs.LEFT_SHOULDER], mid_shoulder, head, mid_shoulder,
            # lms[idxs.RIGHT_SHOULDER], lms[idxs.RIGHT_ELBOW], lms[idxs.RIGHT_WRIST],
            # lms[idxs.RIGHT_INDEX], lms[idxs.RIGHT_WRIST], lms[idxs.RIGHT_ELBOW],
            # lms[idxs.RIGHT_SHOULDER], lms[idxs.RIGHT_HIP], lms[idxs.LEFT_HIP],
            # lms[idxs.RIGHT_HIP],lms[idxs.RIGHT_KNEE], lms[idxs.RIGHT_ANKLE],
        ])

    ## sideview specific methods ##

    def _create_sideview_artists(self, label, sv_lm_arr, image_id, color, legend):
        # insert additional shoulder before the elbow (after the ear) for proper plotting
        body_lms = self._prep_lms_for_sideview_plot(sv_lm_arr)
        body, = self.plots[image_id][AXES].plot(
            body_lms[:,0], body_lms[:,1],
            'o-',
            color=color,
            markersize=self.joint_size,
            markerfacecolor='w',
            markeredgewidth=self.joint_edge_width,
            markeredgecolor=color,
            linewidth=self.linewidth,
            label=label,
            animated=True
        )
        head, = self.plots[image_id][AXES].plot(
            (sv_lm_arr[EAR, 0]), 
            (sv_lm_arr[EAR, 1]),
            'o',
            color=color,
            markersize=self.joint_size * 5,
            markerfacecolor='w',
            markeredgewidth=self.joint_edge_width,
            markeredgecolor=color,
            animated=True
        )
        bar, = self.plots[image_id][AXES].plot(
            (sv_lm_arr[INDEX, 0]), 
            (sv_lm_arr[INDEX, 1]),
            'o',
            color=color,
            markersize=self.joint_size * 2,
            animated=True
        )
        if legend:
            self.plots[image_id][AXES].legend()
        self.plots[image_id][LINES_DICT][label] = {
            BODY_LINE: body, HEAD_LINE: head, BAR_LINE: bar
        }

    def _update_sideview_artists(self, label, sv_lm_arr, image_id):
        body_lms = self._prep_lms_for_sideview_plot(sv_lm_arr)
        self.plots[image_id][LINES_DICT][label][BODY_LINE].set_data(
            body_lms[:,0],
            body_lms[:,1]
        )
        self.plots[image_id][LINES_DICT][label][HEAD_LINE].set_data(
            (sv_lm_arr[EAR, 0]), 
            (sv_lm_arr[EAR, 1])
        )
        self.plots[image_id][LINES_DICT][label][BAR_LINE].set_data(
            (sv_lm_arr[INDEX, 0]), 
            (sv_lm_arr[INDEX, 1])
        )


    def _prep_lms_for_sideview_plot(self, lms):
        return np.vstack([lms[:ELBOW], lms[SHOULDER], lms[ELBOW:]])
