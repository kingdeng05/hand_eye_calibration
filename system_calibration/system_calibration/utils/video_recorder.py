import cv2

class VideoRecorder(object):
    def __init__(self, framerate, filename='output.mp4', fourcc_str='mp4v'):
        self.framerate = framerate
        self.filename = filename
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        self.video_writer = None
        self.frame_size = None

    def add_frame(self, frame, text=None, text_position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, 
                  font_scale=1, color=(255, 0, 0), thickness=2):
        """
        Add a frame to the video. Optionally overlay it with text.
        """
        if len(frame.shape) == 2:  # If grayscale image
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if self.video_writer is None:
            h, w = frame.shape[:2]
            self.frame_size = (w,h)
            self.video_writer = cv2.VideoWriter(self.filename, self.fourcc, self.framerate, self.frame_size)

        if text:
            cv2.putText(frame, text, text_position, font, font_scale, color, thickness)
        
        self.video_writer.write(frame)

    def stop(self):
        """
        Finalize and save the video to the disk.
        """
        if self.video_writer:
            self.video_writer.release()

    def __del__(self):
        self.stop()