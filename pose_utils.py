import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import mediapipe as mp
pose_lm = mp.solutions.pose.PoseLandmark


def get_side_landmarks(landmarks, side='right'):
    assert side in ['right', 'left'], 'side must be either "right" or "left"'
    if side == 'right':
        landmarks = {
            'ear': landmarks.landmark[pose_lm.RIGHT_EAR],
            'shoulder': landmarks.landmark[pose_lm.RIGHT_SHOULDER],
            'hip': landmarks.landmark[pose_lm.RIGHT_HIP],
            'knee': landmarks.landmark[pose_lm.RIGHT_KNEE],
            'ankle': landmarks.landmark[pose_lm.RIGHT_ANKLE],
            'elbow': landmarks.landmark[pose_lm.RIGHT_ELBOW],
            'wrist': landmarks.landmark[pose_lm.RIGHT_WRIST],
            'index': landmarks.landmark[pose_lm.RIGHT_INDEX],
        }
        
        
    else:
        landmarks = {
            'ear': landmarks.landmark[pose_lm.LEFT_EAR],
            'shoulder': landmarks.landmark[pose_lm.LEFT_SHOULDER],
            'hip': landmarks.landmark[pose_lm.LEFT_HIP],
            'knee': landmarks.landmark[pose_lm.LEFT_KNEE],
            'ankle': landmarks.landmark[pose_lm.LEFT_ANKLE],
            'elbow': landmarks.landmark[pose_lm.LEFT_ELBOW],
            'wrist': landmarks.landmark[pose_lm.LEFT_WRIST],
            'index': landmarks.landmark[pose_lm.LEFT_INDEX],
        }
    coords = {key: [val.x, val.y, val.z, val.visibility] for key, val in landmarks.items()}
    return coords


def draw_side(ax, lms, color='k', linewidth=3, joint_size=50, inner_joint_size=20):
    #limbs and torso
    ax.plot(
        (lms['ankle'][0], lms['knee'][0], lms['hip'][0], lms['shoulder'][0], lms['ear'][0]),
        (-lms['ankle'][1], -lms['knee'][1], -lms['hip'][1], -lms['shoulder'][1], -lms['ear'][1],),
        color,
        linewidth=linewidth,
        zorder=0,
    )
    ax.plot(
        (lms['shoulder'][0], lms['elbow'][0], lms['wrist'][0], lms['index'][0]),
        (-lms['shoulder'][1], -lms['elbow'][1], -lms['wrist'][1], -lms['index'][1]),
        color,
        linewidth=linewidth,
        zorder=0
    )
    #joints
    ax.scatter(
        (lms['ankle'][0], lms['knee'][0], lms['hip'][0], lms['shoulder'][0]),
        (-lms['ankle'][1], -lms['knee'][1], -lms['hip'][1], -lms['shoulder'][1]),
        s=joint_size,
        c=color,
        zorder=10,
    )
    ax.scatter(
        (lms['ankle'][0], lms['knee'][0], lms['hip'][0], lms['shoulder'][0], lms['ear'][0]),
        (-lms['ankle'][1], -lms['knee'][1], -lms['hip'][1], -lms['shoulder'][1], -lms['ear'][1],),
        s=inner_joint_size,
        c='w',
        zorder=20,
    )
    ax.scatter(
        (lms['shoulder'][0], lms['elbow'][0], lms['wrist'][0]), 
        (-lms['shoulder'][1], -lms['elbow'][1], -lms['wrist'][1]),
        s=joint_size,
        c=color,
        zorder=10,
    )
    ax.scatter(
        (lms['shoulder'][0], lms['elbow'][0], lms['wrist'][0]), 
        (-lms['shoulder'][1], -lms['elbow'][1], -lms['wrist'][1]),
        s=inner_joint_size,
        c='w',
        zorder=20,
    )
    #head
    ax.scatter(
        (lms['ear'][0]), 
        (-lms['ear'][1]),
        s=joint_size * 10,
        c=color,
        zorder=20,
    )
    #bar
    ax.scatter(
        (lms['index'][0]), 
        (-lms['index'][1]),
        s=joint_size * 1.5,
        c=color,
        zorder=20,
    )


def center_side_landmarks(side_lms):
    ankle_x = side_lms['ankle'][0]
    ankle_y = side_lms['ankle'][1]
    for key_point in side_lms:
        side_lms[key_point][0] -= ankle_x
        side_lms[key_point][1] -= ankle_y
    return side_lms


def mirror_landmarks(side_lms):
    for key_point in side_lms:
        side_lms[key_point][0] = -side_lms[key_point][0]
    return side_lms 


def side_lms_to_numpy(side_lms):
    fig, ax = plt.subplots()
    ax.set_xlim((-.75,.75))
    ax.set_ylim((0,1.5))
    #ax.set_axis_off()
    fig.tight_layout(pad=0)
    draw_side(ax, side_lms)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data
