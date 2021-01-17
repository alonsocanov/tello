from djitellopy import Tello
import cv2
import argparse
import sys
from utils import velocityChange, colorTracking



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('track', type=str, help="Write the kind of tracking 'color' or 'face' or manual")
    parser.add_argument('takeoff', type=bool, help='True to takeoff False to just show video feed')
    args = parser.parse_args()

    takeoff = args.takeoff

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

    

    while True:
        frame_read = drone.get_frame_read()
        frame = frame_read.frame

        if args.track == 'color':
            img, unit_vector = colorTracking(frame)
        elif args.track == 'face':
            img, unit_vector = HaarFaceTracking(frame)
        elif args.track == 'manual':
            img, unit_vector = manualCommand(frame)
        elif args.track == 'command':
            pass
        else:
            pass
        
        forward_back_velocity = 0
        left_right_velocity = 0
        up_down_velocity = 0
        yaw_velocity = 0
        
        
        if not takeoff:
            drone.takeoff()
            takeoff = True
        try:
            yaw_velocity, up_down_velocity, forward_backward_velocity, left_right_velocity = velocityChange(unit_vector)
            drone.send_rc_control(left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity)
        except TypeError:
            drone.send_rc_control(0, 0, 0, 0)
        
        cv2.imshow(win_name, img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            drone.land()
            drone.streamoff()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()