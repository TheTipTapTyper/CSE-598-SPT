"""
Author: Joshua Martin
Email: jmmartin397@protonmail.com
Created: 3/12/2022
Class: CSE 598 Perception in Robotics
Project: Deadlift Critic
"""
import matplotlib.pyplot as plt
import numpy as np
from lm_extractor import SHOULDER, EAR, ELBOW, INDEX
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

    def render_sideview(self, landmarks_dict, image_id='default', legend=True,
                        width=640, height=480):
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
        colors = DEFAULT_COLORS[:len(landmarks_dict)]
        if image_id not in self.plots:
            self._setup_sideview_plot(image_id, width, height)
        # if number of landmark sets has changed (or if this is the first render), draw and save background
        if len(landmarks_dict) != len(self.plots[image_id][LINES_DICT]) or \
            self.plots[image_id][BACKGROUND] is None:
            self._refresh_sideview(image_id, landmarks_dict, colors, legend)
        else: # just update the previous draw Line2D objects
            for (label, landmarks), color in zip(sorted(landmarks_dict.items()), colors): 
                self._update_sideview_artists(label, landmarks, image_id)
        self._restore_background(image_id)
        self._draw_sideview_artists(image_id)
        fig = self.plots[image_id][FIG]
        return np.array(fig.canvas.renderer._renderer)[:,:,:3]

    def _restore_background(self, image_id):
        fig = self.plots[image_id][FIG]
        background_img = self.plots[image_id][BACKGROUND]
        fig.canvas.restore_region(background_img)

    def _save_background_for_blitting(self, image_id):
        fig = self.plots[image_id][FIG]
        fig.canvas.draw()
        background_img = fig.canvas.copy_from_bbox(fig.bbox)
        self.plots[image_id][BACKGROUND] = background_img

    def _refresh_sideview(self, image_id, landmarks_dict, colors, legend):
        """ This method reinitializes the Line2D objects of the sideview plot
        and redraws and saves the background for blitting.
        """
        for artists_dict in self.plots[image_id][LINES_DICT].values():
            for artist in artists_dict.values():
                artist.remove()
        self.plots[image_id][LINES_DICT] = dict()
        for (label, landmarks), color in zip(sorted(landmarks_dict.items()), colors): 
            self._create_sideview_artists(label, landmarks, image_id, color, legend)
        self._save_background_for_blitting(image_id)

    def _setup_sideview_plot(self, image_id, width, height):
        """ This method sets up the figure and axes of the sideview plot.
        """
        px = 1/plt.rcParams['figure.dpi']
        fig, ax = plt.subplots(figsize=(width*px, height*px))            
        x = SIDE_IMG_WIDTH_M / 2
        ax.set_xlim((-x, x))
        ax.set_ylim((0,SIDE_IMG_HEIGHT_M))
        fig.tight_layout(pad=0)
        self.plots[image_id] = {FIG: fig, AXES: ax, LINES_DICT: dict(), BACKGROUND: None}

    def _create_sideview_artists(self, label, landmarks, image_id, color, legend):
        # insert additional shoulder before the elbow (after the ear) for proper plotting
        body_lms = self._prep_lms_for_body_plot(landmarks)
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
            (landmarks[EAR, 0]), 
            (landmarks[EAR, 1]),
            'o',
            color=color,
            markersize=self.joint_size * 5,
            markerfacecolor='w',
            markeredgewidth=self.joint_edge_width,
            markeredgecolor=color,
            animated=True
        )
        bar, = self.plots[image_id][AXES].plot(
            (landmarks[INDEX, 0]), 
            (landmarks[INDEX, 1]),
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

    def _update_sideview_artists(self, label, landmarks, image_id):
        body_lms = self._prep_lms_for_body_plot(landmarks)
        self.plots[image_id][LINES_DICT][label][BODY_LINE].set_data(
            body_lms[:,0],
            body_lms[:,1]
        )
        self.plots[image_id][LINES_DICT][label][HEAD_LINE].set_data(
            (landmarks[EAR, 0]), 
            (landmarks[EAR, 1])
        )
        self.plots[image_id][LINES_DICT][label][BAR_LINE].set_data(
            (landmarks[INDEX, 0]), 
            (landmarks[INDEX, 1])
        )

    def _prep_lms_for_body_plot(self, lms):
        return np.vstack([lms[:ELBOW], lms[SHOULDER], lms[ELBOW:]])

    def _draw_sideview_artists(self, image_id):
        fig = self.plots[image_id][FIG]
        ax = self.plots[image_id][AXES]
        for label in self.plots[image_id][LINES_DICT]:
            ax.draw_artist(self.plots[image_id][LINES_DICT][label][BODY_LINE])
            ax.draw_artist(self.plots[image_id][LINES_DICT][label][HEAD_LINE])
            ax.draw_artist(self.plots[image_id][LINES_DICT][label][BAR_LINE])
        fig.canvas.blit(ax.bbox)
        fig.canvas.flush_events()

