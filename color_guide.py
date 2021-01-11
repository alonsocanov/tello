from djitellopy import Tello
import cv2


def manualControl():
    key = cv2.waitKey(1) & 0xff
    if key == 27: # ESC
        break
    elif key == ord('w'):
        tello.move_forward(30)
    elif key == ord('s'):
        tello.move_back(30)
    elif key == ord('a'):
        tello.move_left(30)
    elif key == ord('d'):
        tello.move_right(30)
    elif key == ord('e'):
        tello.rotate_clockwise(30)
    elif key == ord('q'):
        tello.rotate_counter_clockwise(30)
    elif key == ord('r'):
        tello.move_up(30)
    elif key == ord('f'):
        tello.move_down(30)


width = 320
height = 240
start_counter = 0

drone = Tello()
drone.connect()
drone.for_back_velocity = 0
drone.left_right_velocity = 0
drone.up_down_velocity = 0
drone.yaw_velocity = 0
drone.speed = 0

print(drone.get_battery())

drone.streamoff()
drone.streamon()


while True:
    frame_read = drone.get_frame_read()
    frame = frame_read.frame
    img = cv2.resize(frame, (width, height))

    if start_counter == 0:
        drone.takeoff()
        start_counter = 1

    cv2.imshow('Image', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        drone.land()
        cv2.destroyAllWindows()
        drone.streamoff()
        break