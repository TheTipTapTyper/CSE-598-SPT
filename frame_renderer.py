import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
import numpy as np
from lm_extractor import SHOULDER, EAR, ELBOW, INDEX


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


class FrameRenderer:
    def __init__(self, linewidth=3, joint_size=7, joint_edge_width=1.5):
        self.linewidth = linewidth
        self.joint_size = joint_size
        self.joint_edge_width = joint_edge_width
        self.plots = {}
        self.line_objs = {}

    def render_sideview(self, landmarks_dict, image_id='default', colors=None, legend=True):
        """Renders one or more sets of sideview landmarks into an image.
        landmarks must be a dictionary mapping label names to sideview landmark
        numpy arrays (8x3).

        input colors (if given) must be a list of color specifiers (refer to matplotlib
        colors) and it must have the same number of elements as sets of landmarks. If
        not given, default colors will be used, however, if more than 16 inputs are given,
        colors must be supplied manually.

        The renderer maintains Line2D objects so that subsequent render_sideview
        calls can simply update the data rather than drawing from scratch. However,
        this requires the label names and image_id to remain unchanged between 
        subsequent calls.
        """
        assert isinstance(landmarks_dict, dict) and all(
            isinstance(arr, np.ndarray) and arr.shape == (8,3) 
            for arr in landmarks_dict.values()
        ), 'landmarks_dict must be a dict: {str : (8x3) np.ndarray}'
        if colors is None:
            colors = DEFAULT_COLORS[:len(landmarks_dict)]
        assert len(colors) == len(landmarks_dict), 'must have same number of colors as landmark sets'

        # if this is a new image_id, create a figure-axes pair for it and a new dict
        # to keep track of Line2D objects
        if image_id not in self.plots:
            fig, ax = plt.subplots()
            x = SIDE_IMG_WIDTH_M / 2
            ax.set_xlim((-x, x))
            ax.set_ylim((0,SIDE_IMG_HEIGHT_M))
            fig.tight_layout(pad=0)
            self.plots[image_id] = {FIG: fig, AXES: ax, LINES_DICT: dict(), BACKGROUND: None}
        for (label, landmarks), color in zip(sorted(landmarks_dict.items()), colors):
            fig = self.plots[image_id][FIG] 
            if label not in self.plots[image_id][LINES_DICT]:
                self._draw_sideview(label, landmarks, image_id, color, legend)
                
            else:
                self._update_sideview(label, landmarks, image_id)
        # save background without lines drawn for blitting on first draw
        if self.plots[image_id][BACKGROUND] is None:
            fig.canvas.draw()
            self.plots[image_id][BACKGROUND] = fig.canvas.copy_from_bbox(fig.bbox)
        # restore image to original background
        fig.canvas.restore_region(self.plots[image_id][BACKGROUND])
        # draw artists on top of original background
        self._render_artists(image_id)
        return np.array(fig.canvas.renderer._renderer)

    def _draw_sideview(self, label, landmarks, image_id, color, legend):
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

    def _render_artists(self, image_id):
        fig = self.plots[image_id][FIG]
        ax = self.plots[image_id][AXES]
        for label in self.plots[image_id][LINES_DICT]:
            ax.draw_artist(self.plots[image_id][LINES_DICT][label][BODY_LINE])
            ax.draw_artist(self.plots[image_id][LINES_DICT][label][HEAD_LINE])
            ax.draw_artist(self.plots[image_id][LINES_DICT][label][BAR_LINE])
        fig.canvas.blit(ax.bbox)
        fig.canvas.flush_events()

