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

def toleranceCircle(dim:tuple, factor:float = .2):
    return np.array([dim[0] * factor, dim[1] * factor, dim[1] * .1], dtype=np.int32)

def toleranceFace(dim:tuple, factor:float = .2):
    return np.array([dim[0] * factor, dim[1] * factor, dim[1] * factor], dtype=np.int32)

def imgCenter(dim:tuple):
    return np.array([dim[0] * .5, dim[1] * .5, dim[1] * .2], dtype=np.int32)



def imgResizeDimension(img:np.ndarray, factor:float = None):
    h, w = img.shape[:2]
    if not factor:
        factor = 500 / w
    return int(factor * w), int(factor * h)

def drawTolerance(img:np.ndarray, tolerance:np.ndarray):
    h, w = img.shape[:2]
    cx, cy, _ = imgCenter((w, h))
    green = (0, 255, 0)
    thickness = 3
    x_min, y_min = cx - tolerance[0], cy - tolerance[1]
    x_max, y_max = cx + tolerance[0], cy + tolerance[1]
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), green, thickness)

def drawObjectPosition(img:np.ndarray, pos:tuple, radius:float):
    red = (0, 0, 255)
    green = (0, 255, 0)
    thickness = 2
    pixel = (int(pos[0]), int(pos[1]))
    cv2.circle(img, pixel, int(radius), green, thickness)
    cv2.circle(img, pixel, 1, red, thickness)

def unitToPixel(dim:tuple, unit_values:np.ndarray):
    w, h = dim
    return np.array([w * unit_values[0], h * unit_values[1], w * h * unit_values[2]], dtype=np.int32)

def drawFacePosition(img:np.ndarray, pos:np.ndarray, dim:tuple):
    x_min, y_min = pos[0] - int(dim[0] / 2), pos[1] - int(dim[1] / 2)
    x_max, y_max = pos[0] + int(dim[0] / 2), pos[1] + int(dim[1] / 2)
    thickness = 2
    blue = (255, 0, 0)
    red = (0, 0, 255)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), blue, thickness)
    cv2.circle(img, (pos[0], pos[1]), 1, red, thickness)

def velocityChange(unit_vector:np.ndarray, scale:int = 40):
    return (unit_vector * scale).astype(int).tolist()

def colorTracking(img:np.ndarray):
    # lower and upper range of hsv color
    hsv_lower = hsv2cvhsv(np.array([45, 40, 20]))
    hsv_upper = hsv2cvhsv(np.array([65, 90, 90]))
    width, height = imgResizeDimension(img)
    img = cv2.resize(img, (width, height))
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    mask = cv2.erode(mask, None, iterations=2)
    tol = toleranceCircle((width, height))
    img_center = imgCenter((width, height))



    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    unit_dist = np.array([0, 0, 0, 0])

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
                unit_dist[1] = -1 * unit_dist[1]
                unit_dist[2] = -1 * unit_dist[2]
                unit_dist = np.append(unit_dist, [0], axis=0)
    drawTolerance(img, tol)
    return img, unit_dist

def HaarFaceTracking(img:np.ndarray):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    width, height = imgResizeDimension(img)
    img = cv2.resize(img, (width, height))
    # image is taken as unit dimension, and area unit
    img_center = np.array([0.5, 0.5, .12], dtype=np.float64)
    tolerance = np.array([0.1, 0.1, .05], dtype=np.float64)
    img_dim = np.array([width, height, width * height], dtype=np.float64)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    unit_dist = np.array([0, 0, 0, 0])



    if isinstance(faces, np.ndarray):
        idx_max_area = np.argmax(faces[:, 2] * faces[:, 3])
        x, y, w, h = faces[idx_max_area, :]
        face_center = np.array([(x + w / 2) , y + h /2, w * h]) / img_dim

        dist = face_center - img_center
        abs_dist = np.absolute(dist)
        if np.any(abs_dist > tolerance):
            unit_dist = dist / np.linalg.norm(dist)
            unit_dist[1] = -1 * unit_dist[1]
            unit_dist[2] = -1 * unit_dist[2]
            unit_dist = np.append(unit_dist, [0], axis=0)
        pixel_face = unitToPixel((width, height), face_center)
        drawFacePosition(img, pixel_face, (w, h))
    pixel_tolerance = unitToPixel((width, height), tolerance)
    drawTolerance(img, pixel_tolerance)

    return img, unit_dist


def manualCommand(img:np.ndarray):
    width, height = imgResizeDimension(img)
    img = cv2.resize(img, (width, height))

    unit_vector = np.array([0, 0, 0, 0])

    key = cv2.waitKey(1) & 0xff
    if key == ord('w'):
        # move up
        unit_vector[2] = 1.0
    elif key == ord('s'):
        # move down
        unit_vector[2] = -1.0
    elif key == ord('a'):
        # move left
        unit_vector[3] = -1.0
    elif key == ord('d'):
        # move right
        unit_vector[3] = 1.0
    elif key == ord('e'):
        # clockwise
        unit_vector[0] = -1.0
    elif key == ord('q'):
        # counter clockwise
        unit_vector[0] = 1.0
    elif key == ord('r'):
        # move up
        unit_vector[1] = 1.0
    elif key == ord('f'):
        # move down
        unit_vector[1] = -1.0
    elif key == ord('q'):
        pass
    return img, unit_vector
