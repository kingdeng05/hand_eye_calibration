import numpy as np
import cv2 as cv 

class ArucoDetector(object):
    def __init__(self, vis=False):
        dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
        parameters =  cv.aruco.DetectorParameters()
        self.detector = cv.aruco.ArucoDetector(dictionary, parameters) 
        self.vis = vis

    def detect(self, img):
        """
        Args:
            img - np.ndarray, (height, width, channel)
        Returns:
            corners - [n, 4, 2], n is the number of the markers
            ids - [n, ], n is the number of the markers
        """
        corners, ids, _ = self.detector.detectMarkers(img)
        corners = self.refine_corners(img, corners)

        if self.vis and ids is not None:
            img_copy = img.copy()
            # Perform subpixel refinement for detected markers
            # Draw detected markers after subpixel refinement
            cv.aruco.drawDetectedMarkers(img_copy, corners, ids, borderColor=(0, 255, 0))
            cv.imshow('ArUco with Subpixel Refinement', cv.resize(img_copy, (int(img_copy.shape[1]/2), int(img_copy.shape[0]/2)))) 
            cv.waitKey(0)
        if ids is None:
            return np.array([]), np.array([])
        return np.array([corner[0] for corner in corners]), ids.flatten()

    def refine_corners(self, img, corners):
        if len(img.shape) == 3:
            image_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            image_gray = img
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        for corner in corners:
            cv.cornerSubPix(image_gray, corner, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria)
        return corners
