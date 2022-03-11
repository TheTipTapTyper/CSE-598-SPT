import matplotlib.pyplot as plt

import mediapipe as mp
pose_lm = mp.solutions.pose.PoseLandmark

# the following command cuts video starting at 13 seconds and ends at the 26 second mark
# ffmpeg -i output1.mp4 -ss 00:00:13 -to 00:00:26 -c copy output1_trimmed.mp4

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


def draw_side(ax, lms, color='k', linewidth=3, joint_size=7, joint_edge_width=1.5):
    #limbs and trunk
    body, = ax.plot(
        (lms['ankle'][0], lms['knee'][0], lms['hip'][0], lms['shoulder'][0], lms['ear'][0], 
            lms['shoulder'][0], lms['elbow'][0], lms['wrist'][0], lms['index'][0]),
        (-lms['ankle'][1], -lms['knee'][1], -lms['hip'][1], -lms['shoulder'][1], -lms['ear'][1], 
            -lms['shoulder'][1], -lms['elbow'][1], -lms['wrist'][1], -lms['index'][1]),
        color + 'o-',
        markersize=joint_size,
        markerfacecolor='w',
        markeredgewidth=joint_edge_width,
        markeredgecolor=color,
        linewidth=linewidth,

    )
    # #head
    head, = ax.plot(
        (lms['ear'][0]), 
        (-lms['ear'][1]),
        color + 'o',
        markersize=joint_size * 5,
        markerfacecolor='w',
        markeredgewidth=joint_edge_width,
        markeredgecolor=color,
    )
    #bar
    bar, = ax.plot(
        (lms['index'][0]), 
        (-lms['index'][1]),
        color + 'o',
        markersize=joint_size * 2,
    )
    return body, head, bar

def update_side_drawing(lms, body, head, bar):
    body.set_data(
        (lms['ankle'][0], lms['knee'][0], lms['hip'][0], lms['shoulder'][0], lms['ear'][0], 
            lms['shoulder'][0], lms['elbow'][0], lms['wrist'][0], lms['index'][0]),
        (-lms['ankle'][1], -lms['knee'][1], -lms['hip'][1], -lms['shoulder'][1], -lms['ear'][1], 
            -lms['shoulder'][1], -lms['elbow'][1], -lms['wrist'][1], -lms['index'][1])
    )
    head.set_data(
        (lms['ear'][0]), 
        (-lms['ear'][1]),
    )
    bar.set_data(
        (lms['index'][0]), 
        (-lms['index'][1]),
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

