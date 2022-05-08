from threading import Thread
import cv2


class CamVideoStream:
    def __init__(self, src=0, name="CamVideoStream", live_mode=True):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the thread name
        self.name = name

        # Streaming source
        self.src = src

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

        # live mode, make it False to process video files
        self.live_mode = live_mode

    def start(self):
        if self.live_mode:
            # start the thread to read frames from the video stream
            t = Thread(target=self.update, name=self.name, args=())
            #t.daemon = True
            t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        if self.live_mode:
            # return the frame most recently read
            return self.frame
        else:
            _, self.frame = self.stream.read()
            return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
    
    def reconnect(self):
        # Try to reconnect stream if empty frame
        #self.stream.release()
        self.stream = cv2.VideoCapture(self.src)