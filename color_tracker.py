from djitellopy import Tello
import cv2
import imutils
import numpy as np
import sys
from utils import hsv2cvhsv, tolerance, imgCenter, imgDimensions, drawTolerance, drawObjectPosition, velocityChange, colorTracking









def main():

    start_counter = 0

    drone = Tello()
    if not drone.connect():
        print('Unable to connecrt with drone')
        sys.exit(0)

    
    forward_back_velocity = 0
    left_right_velocity = 0
    up_down_velocity = 0
    yaw_velocity = 0
    speed = 0

    
    print(drone.get_battery().strip('dm\r\n'))

    drone.streamoff()
    drone.streamon()

    win_name = 'Frame'
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, 20, 20)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = .5
    fontColor = (255, 255, 255)
    lineType = 2

    unit_dist = np.array([0, 0, 0])
    

    while True:
        frame_read = drone.get_frame_read()
        frame = frame_read.frame

        img, unit_vector = colorTracking(frame)
        
        forward_back_velocity = 0
        left_right_velocity = 0
        up_down_velocity = 0
        yaw_velocity = 0
        
        
        if start_counter == 0:
            drone.takeoff()
            start_counter = 1
        try:
            yaw_velocity, up_down_velocity, forward_backward_velocity = velocityChange(unit_vector)
            drone.send_rc_control(left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity)
        except TypeError:
            forward_back_velocity = 0
            left_right_velocity = 0
            up_down_velocity = 0
            yaw_velocity = 0
            drone.send_rc_control(left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity)
        
        cv2.imshow(win_name, img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            drone.land()
            drone.streamoff()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()