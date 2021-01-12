from djitellopy import Tello
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


def imageAnalysis(frame):
    # lower and upper range of hsv color
    hsv_lower = hsv2cvhsv(np.array([45, 40, 20]))
    hsv_upper = hsv2cvhsv(np.array([65, 100, 100]))
    width = 320
    height = 240

    img = cv2.resize(frame, (width, height))
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    mask = cv2.erode(mask, None, iterations=1)

    img_center = np.array([width * .5, height * .5, .25 * height])

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    unit_dist = np.array([0, 0, 0])
   

    if len(cnts) > 0:
        try:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), r) = cv2.minEnclosingCircle(c)
            pos = np.array([x, y, r])
            M = cv2.moments(c)
            obj_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            if r > .06 * height:
                cv2.circle(img, (int(x), int(y)), int(r), (0, 255, 0), 4)
                cv2.circle(img, obj_center, 1, (0, 0, 255), 5)

                pixel_dist = (pos - img_center)
                unit_dist = pixel_dist / np.linalg.norm(pixel_dist)
        except:
            pass

    return img, unit_dist


def velocityChange(unit_vector, scale=10):
    return (unit_vector * scale).astype(int).tolist()



def main():

    start_counter = 0

    drone = Tello()
    if not drone.connect():
        print('Unable to connecrt with drone')
        sys.exit(0)

    
    drone.forward_back_velocity = 0
    drone.left_right_velocity = 0
    drone.up_down_velocity = 0
    drone.yaw_velocity = 0
    drone.speed = 0

    
    print(drone.get_battery())

    drone.streamoff()
    drone.streamon()

    win_name = 'Frame'
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, 20, 20)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = .5
    fontColor = (255, 255, 255)
    lineType = 2

    


    while True:
        frame_read = drone.get_frame_read()
        frame = frame_read.frame

        img, unit_vector = imageAnalysis(frame)
        
        # drone.forward_back_velocity = 0
        # drone.left_right_velocity = 0
        # drone.up_down_velocity = 0
        # drone.yaw_velocity = 0
        
        

        if start_counter == 0:
            drone.takeoff()
            start_counter = 1
        try:
            drone.yaw_velocity, drone.up_down_velocity, drone.forward_backward_velocity = velocityChange(unit_vector)
            drone.send_rc_control(drone.left_right_velocity, drone.forward_backward_velocity, drone.up_down_velocity, drone.yaw_velocity)
        except TypeError:
            print('Type error for output')

        cv2.imshow(win_name, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            drone.land()
            cv2.destroyAllWindows()
            drone.streamoff()
            break

if __name__ == '__main__':
    main()