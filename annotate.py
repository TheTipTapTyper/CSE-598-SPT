import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pose_utils
from PIL import Image
from moviepy.video.io.bindings import mplfig_to_npimage


SIDE_IMG_WIDTH_M = 2
SIDE_IMG_HEIGHT_M = 2


class Annotater:
    def __init__(self, side='right', base_width=1280, base_height=720, **kwargs):
        self.fig, self.ax = plt.subplots()
        x = SIDE_IMG_WIDTH_M / 2
        self.ax.set_xlim((-x, x))
        self.ax.set_ylim((0,SIDE_IMG_HEIGHT_M))
        self.fig.tight_layout(pad=0)
        self.side = side
        self.base_width = base_width
        self.base_height = base_height
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(**kwargs)
        self.line_objs = None

    def side_lms_to_numpy(self,side_lms):
        if self.line_objs is None:
            self.line_objs = pose_utils.draw_side(self.ax, side_lms)
        else:
            pose_utils.update_side_drawing(side_lms, *self.line_objs)
        data = mplfig_to_npimage(self.fig)
        return data

    def annotate_image(self, image, flip=False):
        if isinstance(image, str): #filename
            image = np.array(Image.open(image))
        # Run model inference on image
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        if flip:
            image = cv2.flip(image, 1)
        image = cv2.resize(image, (self.base_width, self.base_height))

        # Create the sideview image
        if results.pose_world_landmarks is not None:
            side_lms = pose_utils.get_side_landmarks(results.pose_world_landmarks, side=self.side)
            side_lms = pose_utils.center_side_landmarks(side_lms)
            if flip:
                side_lms = pose_utils.mirror_landmarks(side_lms)
            side_image = self.side_lms_to_numpy(side_lms)
            side_image = cv2.cvtColor(side_image, cv2.COLOR_RGB2BGR)
            side_image = cv2.resize(side_image, (self.base_width // 2, self.base_height))
        else: # no person detected
            side_image = np.ones((self.base_height, self.base_width // 2, 3), dtype=np.uint8) * 255
        # stack images together horizontally
        image = np.hstack((image, side_image))

        return image

    def annotate_video(self, input=1, output=None, display=True):
        flip = is_webcam = isinstance(input, int)
        cap = cv2.VideoCapture(input)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if output is not None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output, fourcc, fps, (int(self.base_width * 1.5), self.base_height))
        with self.pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    if is_webcam:
                        continue
                    else:
                        break
                image = self.annotate_image(image, flip=flip)
                if output is not None: # save to file
                    out.write(image)
                if display:
                    cv2.namedWindow('Deadlift Form Corrector', cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty('Deadlift Form Corrector',cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN)
                    cv2.imshow('Deadlift Form Corrector', image)
                    if cv2.waitKey(5) & 0xFF == 27:
                        break
        cap.release()
        if output is not None:
            out.release()


if __name__ == '__main__':
    #### example running on an image ###
    # anno = Annotater(base_width=640, base_height=480)
    # im = anno.annotate_image('deadlift.jpeg')
    # Image.fromarray(im).show()

    ### example running on a video ###
    # anno = Annotater(base_width=640, base_height=480)
    # anno.annotate_video(
    #     input='./calibration_code/output1_trimmed.mp4', 
    # )

    ### example running on a video and saving to file without displaying###
    # anno = Annotater(base_width=640, base_height=480)
    # anno.annotate_video(
    #     input='./calibration_code/output1_trimmed.mp4', 
    #     output='./output/output1_trimmed_anno.mp4',
    #     display=False
    # )

    ### example running on webcam ###
    anno = Annotater(base_width=640, base_height=480)
    anno.annotate_video()
