"""
Author: Joshua Martin
Email: jmmartin397@protonmail.com
Created: 3/12/2022
Class: CSE 598 Perception in Robotics
Project: Deadlift Critic
"""
import cv2
import os


class VideoManager:
    """ Interface for managing one or more video input streams and/or one or more
    mp4 file output streams.

    Supports use of mp4 files as well as webcams for input.
    Suports writing to mp4 files.
    """
    def __init__(self):
        self.input_preprocess_funcs = dict()
        self.output_postprocess_funcs = dict()
        self.output_path = None
        self.caps = None
        self.outs = None
        self.input_ids = [] # stored to preserve order of setup
        self.output_ids = [] # stored to preserve order of setup

    def is_open(self):
        """ Convenience method for verifying all input streams are still open.
        """
        assert self.caps is not None, 'Must first call setup_input before calling is_open'
        return all(cap.isOpened() for cap in self.caps.values())

    def release(self):
        """ Convenience method for releasing any and all open input and output streams.
        """
        if self.caps is not None:
            for cap in self.caps.values():
                cap.release()
            self.caps = None
        if self.outs is not None:
            for out in self.outs.values():
                out.release()
            self.outs = None

    def add_input_preprocess(self, func, input_id=None):
        """ preprocessing functions are applied to read images in the order they are added
        before images are returned from the read method.

        If input_id is specified, this func will only be applied to that input
        stream, otherwise it will be applied to all input streams.

        Note that adding a preprocessing func to an input_id that has not yet been
        setup will error. Additionally, adding a preprocessing func to be applied
        to all input streams will only be applied to input streams setup before
        add_input_preprocess is called.

        e.g.
        if vm.read() would normally return image, then the following:
            vm.add_input_preprocess(func1)
            vm.add_input_preprocess(func2)
            vm.add_input_preprocess(func3)
        would result in vm.read() returning:
            func3(func2(func1(image)))

        example use case: resizing images and converting their color space
        vm.add_input_preprocess(labmda im: cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        vm.add_input_preprocess(labmda im: cv2.resize(im, (1280, 720)))
        """
        if input_id is None:
            ids = self.input_ids
        else:
            ids = [input_id]
        for id in ids:
            assert id in self.input_ids, 'no input with input_id "{}" has been setup'.format(id)
            if id not in self.input_preprocess_funcs:
                self.input_preprocess_funcs[id] = []
            self.input_preprocess_funcs[id].append(func)

    def add_output_postprocess(self, func, output_id=None):
        """ postprocessing functions are applied to images in the order they are added
        before images are written to file by the write method.

        If output_id is specified, this func will only be applied to that output
        stream, otherwise it will be applied to all output streams.

        Note that adding a postprocessing func to an output_id that has not yet been
        setup will error. Additionally, adding a postprocessing func to be applied
        to all output streams will only be applied to output streams setup before
        add_output_postprocess is called. 
        
        e.g.
        if vm.write() would normally write image to file, then the following:
            vm.add_output_postprocess(func1)
            vm.add_output_postprocess(func2)
            vm.add_output_postprocess(func3)
        would result in vm.write() writing the following to file:
            func3(func2(func1(image)))

        example use case: resizing images and converting their color space
        vm.add_output_postprocess(labmda im: cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        vm.add_output_postprocess(labmda im: cv2.resize(im, (1280, 720)))
        """
        if output_id is None:
            ids = self.output_ids
        else:
            ids = [output_id]
        for id in ids:
            assert id in self.output_ids, 'no output with output_id "{}" has been setup'.format(id)
            if id not in self.output_postprocess_funcs:
                self.output_postprocess_funcs[id] = []
            self.output_postprocess_funcs[id].append(func)

    def setup_input(self, input):
        """ Prepares opencv to capture video input. 
        input can be one of the following:
        1) an integer, interpreted as a webcam index
        2) a string, interpreted as a path to a mp4 file
        """
        if not isinstance(input, int): # filename
            if not os.path.isfile(input):
                raise FileNotFoundError(input)
        if self.caps is None:
            self.caps = dict()
        self.caps[input] = cv2.VideoCapture(input)
        self.input_ids.append(input)

    def setup_output(self, output, fps=None, width=None, height=None):
        """ Prepares opencv to write to an mp4 file.
        output must be a string and is the path the mp4 file will be written to.
        If fps, width and or Height are not specified, setup will attempt to use
        the values of the first input method setup. However, if being used only
        to record and no input has been setup, these values must be specified.

        value of output parameter is used in self.write to identify the output
        stream, so it is advised this value be saved in a variable for later
        reference. e.g.

        my_out = 'path/to/my/output/file.pm4'
        vm.setup_output(my_out)
        ...
        vm.write(image, my_out)
        """
        assert self.caps is not None or not (fps is None or width is None or height is None), \
            'Input must be specified or all params must be given (fps, width, height)'
        if len(self.caps) > 0:
            first_cap = self.caps[self.input_ids[0]]
        fps = fps or first_cap.get(cv2.CAP_PROP_FPS)
        width = width or first_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = height or first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if self.outs is None:
            self.outs = dict()
        self.outs[output] = cv2.VideoWriter(output, fourcc, fps, (width, height))
        self.output_ids.append(output)

    def read(self, input_id=None):
        """ captures image from the specified input stream (or first input stream
        setup if input_id is not specified) and apply any specified preprocessing.
        returns: (success, image) tuple.
        """
        assert self.caps is not None, 'Must first call setup_input before calling read'
        if input_id is None:
            input_id = self.input_ids[0]
        cap = self.caps[input_id]
        success, image = cap.read()
        # apply preprocessing functions
        if success:
            if input_id in self.input_preprocess_funcs:
                for func in self.input_preprocess_funcs[input_id]:
                    image = func(image)
        return success, image

    def write(self, image, output_id=None):
        """ write image to the specified output stream (or the first output stream
        setup if output_id is not specified) and apply any specified postprocessing.
        """
        assert self.outs is not None, 'Must first call setup_output before calling write'
        if output_id is None:
            output_id = self.output_ids[0]
        if output_id in self.output_postprocess_funcs:
            for func in self.output_postprocess_funcs[output_id]:
                image = func(image)
        self.outs[output_id].write(image)

    def display(self, image, window_name='default', fullscreen=False):
        """ Display the given image and name the window window_name. Closes
        window if ESC is pressed.
        """
        if fullscreen:
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name,cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_name, image)
        if cv2.waitKey(1) & 0xFF == 27:
            exit()

    def record(self, is_webcam=True, display=True):
        """Convenience method for directly writing video(s) to file. Number and order of
        input and output streams must be the same. Skips image on read failure if
        is_webcam is set to True, otherwise exits (end of mp4 file reached).

        if display is set to True, preview windows will show video as it is recorded.

        One use case of this method is to apply pre or post processing steps to an mp4
        file by setting it as the input and another mp4 file as the output. e.g. resizing
        or changing contrast.

        Press escape while any display window is open to exit
        """
        assert len(self.input_ids) == len(self.output_ids), \
            'mismatch in number of input ({}) and output ({}) streams'.format(
                len(self.input_ids), len(self.output_ids)
            )
        names = ['stream_{}'.format(i) for i in range(len(self.caps))]
        try:
            while(self.is_open()):
                results = [self.read(id) for id in self.input_ids]
                success = all(suc for suc, _ in results)
                images = [img for _, img in results]
                if not success:
                    if is_webcam:
                        continue
                    else:
                        break
                if display:
                    for name, image in zip(names, images):
                        self.display(image, name, fullscreen=False)
                for i in range(len(images)):
                    self.write(images[i], self.output_ids[i])
        finally:
            self.release()

def example():
    # example usage for reading from one webcam, resizing, displaying
    # and recording to video. In this case the display and output video
    # are set to different resolutions and the output video mirrored
    display_width = 1920
    display_height = 1080
    output_width = 1280
    output_height = 720
    
    vm = VideoManager()
    vm.setup_input(0)
    vm.setup_output('example_vm_output.mp4', width=output_width, height=output_height)
    vm.add_input_preprocess(lambda im: cv2.resize(im, (display_width, display_height)))
    vm.add_output_postprocess(lambda im: cv2.resize(im, (output_width, output_height)))
    vm.add_output_postprocess(lambda im: cv2.flip(im, 1))
    vm.record()

if __name__ == '__main__':
    example()