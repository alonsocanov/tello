import cv2
import imutils
import numpy as np
import sys


# color normalization of HSV to OpenCV HSV
def hsv2cvhsv(hsv: np.array) -> np.array:
    # For HSV, Hue range is [0,179], Saturation range is [0,255]
    # and Value range is [0,255]. Different software use different scales.
    # So if you are comparinn in OpenCV values with them, you need to normalize these ranges.
    hsv_cv = np.array([179, 255, 255])
    hsv_orig = np.array([360, 100, 100])
    cv_hsv = np.divide((hsv * hsv_cv), hsv_orig)
    return cv_hsv

def tolerance(dim, factor=.2):
    return np.array([dim[0] * factor, dim[1] * factor, dim[1] * .1], dtype=np.int32)

def imgCenter(dim):
    return np.array([dim[0] / 2, dim[1] / 2, .15 * dim[1]], dtype=np.int32)

def imgResizeDimension(img, factor=None):
    h, w = img.shape[:2]
    if not factor:
        factor = 400 / w
    return int(factor * w), int(factor * h)

def drawTolerance(img, tolerance):
    h, w = img.shape[:2]
    cx, cy, _ = imgCenter((w, h))
    green = (0, 255, 0)
    thickness = 3
    x_min, y_min = int(cx - tolerance[0] / 2), int(cy - tolerance[1] / 2)
    x_max, y_max = int(cx + tolerance[0] / 2), int(cy + tolerance[1] / 2)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), green, thickness)

def drawObjectPosition(img, pos, radius):
    red = (0, 0, 255)
    green = (0, 255, 0)
    thickness = 4
    pixel = (int(pos[0]), int(pos[1]))
    cv2.circle(img, pixel, int(radius), green, thickness)
    cv2.circle(img, pixel, 1, red, thickness)

def velocityChange(unit_vector, scale=40):
    unit_vector[1] = -1 * unit_vector[1]
    unit_vector[2] = -1 * unit_vector[2]
    return (unit_vector * scale).astype(int).tolist()

def colorTracking(img):
    # lower and upper range of hsv color
    hsv_lower = hsv2cvhsv(np.array([45, 40, 20]))
    hsv_upper = hsv2cvhsv(np.array([65, 100, 100]))
    width, height = imgResizeDimension(img)
    img = cv2.resize(img, (width, height))
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    mask = cv2.erode(mask, None, iterations=1)
    tol = tolerance((width, height))
    img_center = imgCenter((width, height))
    drawTolerance(img, tol)
    

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    unit_dist = np.array([0, 0, 0])

    if len(cnts) > 0:        
        c = max(cnts, key=cv2.contourArea)
        ((x, y), r) = cv2.minEnclosingCircle(c)
        obj_pos = np.array([x, y, r], dtype=np.int32)
        if r > .05 * height:
            drawObjectPosition(img, (x, y), r)
            pixel_dist = (obj_pos - img_center)
                        
            abs_dist = np.absolute(pixel_dist)
            if np.any(abs_dist > tol / 2):
                unit_dist = pixel_dist / np.linalg.norm(pixel_dist)
            
        

    return img, unit_dist
