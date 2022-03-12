import cv2

class VideoManager:
    """ Interface for managing one or more video input streams and/or one or more
    mp4 file output streams.

    Supports use of mp4 files as well as webcams for input.
    Suports writing to mp4 files.
    """
    def __init__(self):
        self.input_preprocess_funcs = []
        self.output_postprocess_funcs = []
        self.output_path = None
        self.caps = None
        self.outs = None

    def is_open(self):
        """ Convenience method for verifying all input streams are still open.
        """
        assert self.caps is not None, 'Must first call setup_input before calling is_open'
        return all(cap.isOpened() for cap in self.caps)

    def release(self):
        """ Convenience method for releasing any and all open input and output streams.
        """
        if self.caps is not None:
            for cap in self.caps:
                cap.release()
            self.caps = None
        if self.outs is not None:
            for out in self.outs:
                out.release()
            self.outs = None

    def add_input_preprocess(self, func):
        """ preprocessing functions are applied to read images in the order they are added
        before images are returned from the read method.
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
        self.input_preprocess_funcs.append(func)

    def add_output_postprocess(self, func):
        """ postprocessing functions are applied to images in the order they are added
        before images are written to file by the write method.
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
        self.output_postprocess_funcs.append(func)

    def setup_input(self, input):
        """ Prepares opencv to capture video input. 
        input can be one of the following:
        1) an integer, interpreted as a webcam index
        2) a list of integer, interpreted as multiple webcam indices
        """
        if self.caps is None:
            self.caps = []
        self.caps.append(cv2.VideoCapture(input))

    def setup_multi_inputs(self, inputs):
        """ Convenience method for setting up multiple inputs.
        """
        for input in inputs:
            self.setup_input(input)

    def setup_output(self, output, fps=None, width=None, height=None):
        """ Prepares opencv to write to an mp4 file.
        output must be a string and is the path the mp4 file will be written to.
        If fps, width and or Height are not specified, setup will attempt to use
        the values of the first input method specified. However, if being used only
        to record and no input has been setup, these values must be specified.
        """
        assert self.caps is not None or not (fps is None or width is None or height is None), \
            'Input must be specified or all params must be given (fps, width, height)'
        fps = fps or self.caps[0].get(cv2.CAP_PROP_FPS)
        width = width or self.caps[0].get(cv2.CAP_PROP_FRAME_WIDTH)
        height = height or self.caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if self.outs is None:
            self.outs = []
        self.outs.append(cv2.VideoWriter(output, fourcc, fps, (width, height)))

    def setup_multi_outputs(self, outputs, **kwargs):
        """ Convenience function for setting up multiple output streams at once 
        if they all have the same fps, width and height.
        """
        for output in outputs:
            self.setup_output(output, **kwargs)

    def read(self):
        """ captures image(s) from input streams and applies any specified preprocessing.
        returns: list of (success, image) tuples. list indeces correspond input streams.
        """
        assert self.caps is not None, 'Must first call setup_input before calling read'
        results = []
        for cap in self.caps:
            success, image = cap.read()
            # apply preprocessing functions
            if success:
                for func in self.input_preprocess_funcs:
                    image = func(image)
            results.append((success, image))
        return results

    def write(self, image, output_idx):
        """ applies any specified postprocessing and then writes images to the
        output stream specified by output_idx. This index is returned by the
        setup_output method.
        """
        assert self.outs is not None, 'Must first call setup_output before calling write'
        for func in self.output_postprocess_funcs:
            image = func(image)
        self.outs[output_idx].write(image)

    def write_all(self, images):
        """ Convenience function for writing to all output streams at once. Assumes images
        is an iterable where the elements are the images to be written and are in the same
        order as their corresponding output streams were added via subsequent calls to
        the setup_output method or a single call to the setup_multi_oututs method.
        """
        for idx, image in enumerate(images):
            self.write(image, idx)

    def record(self, is_webcam=True, display=True):
        """Convenience function for directly writing video(s) to file. Number and order of
        input and output streams must be the same. Skips image on read failure if
        is_webcam is set to True, otherwise exits (end of mp4 file reached).

        if display is set to True, preview windows will show video as it is recorded.

        One use case of this method is to apply pre or post processing steps to an mp4
        file by setting it as the input and another mp4 file as the output. e.g. resizing
        or changing contrast.

        Press escape while any display window is open to exit
        """
        names = ['stream_{}'.format(i) for i in range(len(self.caps))]
        try:
            while(self.is_open()):
                results = self.read()
                success = all(suc for suc, _ in results)
                images = [img for _, img in results]
                if not success:
                    if is_webcam:
                        continue
                    else:
                        break
                if display:
                    for name, image in zip(names, images):
                        cv2.imshow(name, image)
                        if cv2.waitKey(1) & 0xFF == 27:
                            exit()
                self.write_all(images)
        finally:
            self.release()


if __name__ == '__main__':
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