import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np
import abc
from lm_extractor import SHOULDER, EAR, ELBOW, INDEX


SIDE_IMG_WIDTH_M = 2
SIDE_IMG_HEIGHT_M = 2


class FrameRenderer(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def render_sideview(self, landmarks_dict, image_id='default', color='k'):
        """Renders one or more sets of sideview landmarks into an image.
        landmarks must be a dictionary mapping label names to sideview landmark
        numpy arrays (8x3)."""
        pass

FIG = 'fig'
AXES = 'ax'
LINES_DICT = 'lines'
BODY_LINE = 'body'
HEAD_LINE = 'line'
BAR_LINE = 'bar'

class MPLFrameRenderer(FrameRenderer):
    def __init__(self, linewidth=3, joint_size=7, joint_edge_width=1.5):
        self.linewidth = linewidth
        self.joint_size = joint_size
        self.joint_edge_width = joint_edge_width
        self.plots = {}
        self.line_objs = {}

    def render_sideview(self, landmarks_dict, image_id='default', color='k'):
        """Renders one or more sets of sideview landmarks into an image.
        landmarks must be a dictionary mapping label names to sideview landmark
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
        # if this is a new image_id, create a figure-axes pair for it and a new dict
        # to keep track of Line2D objects
        if image_id not in self.plots:
            fig, ax = plt.subplots()
            x = SIDE_IMG_WIDTH_M / 2
            ax.set_xlim((-x, x))
            ax.set_ylim((0,SIDE_IMG_HEIGHT_M))
            fig.tight_layout(pad=0)
            self.plots[image_id] = {FIG: fig, AXES: ax, LINES_DICT: dict()}
        for label, landmarks in landmarks_dict.items():
            if label not in self.plots[image_id][LINES_DICT]:
                self._draw_sideview(label, landmarks, image_id, color)
            else:
                self._update_sideview(label, landmarks, image_id)
        return mplfig_to_npimage(self.plots[image_id][FIG])

    def _draw_sideview(self, label, landmarks, image_id, color):
        # insert additional shoulder before the elbow (after the ear) for proper plotting
        body_lms = self._prep_lms_for_body_plot(landmarks)
        body, = self.plots[image_id][AXES].plot(
            body_lms[:,0], body_lms[:,1],
            color + 'o-',
            markersize=self.joint_size,
            markerfacecolor='w',
            markeredgewidth=self.joint_edge_width,
            markeredgecolor=color,
            linewidth=self.linewidth,
            label=label
        )
        head, = self.plots[image_id][AXES].plot(
            (landmarks[EAR, 0]), 
            (landmarks[EAR, 1]),
            color + 'o',
            markersize=self.joint_size * 5,
            markerfacecolor='w',
            markeredgewidth=self.joint_edge_width,
            markeredgecolor=color,
        )
        bar, = self.plots[image_id][AXES].plot(
            (landmarks[INDEX, 0]), 
            (landmarks[INDEX, 1]),
            color + 'o',
            markersize=self.joint_size * 2,
        )
        self.plots[image_id][LINES_DICT][label] = {
            BODY_LINE: body, HEAD_LINE: head, BAR_LINE: bar
        }

    def _update_sideview(self, label, landmarks, image_id):
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
        return np.vstack([lms[:ELBOW], lms[SHOULDER], lms[EAR:]])
