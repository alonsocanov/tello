from djitellopy import Tello
import cv2

def main():
    width = 320
    height = 240
    start_counter = 0

    # connet to tello drone
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

    print('w: forward')
    print('s: back')
    print('a: left')
    print('d: right')
    print('e: rotate clockwise')
    print('q: rotate counter clockwise')
    print('r: up')
    print('f: down')


    while True:
        frame_read = drone.get_frame_read()
        frame = frame_read.frame
        img = cv2.resize(frame, (width, height))

        if start_counter == 0:
            drone.takeoff()
            start_counter = 1

        key = cv2.waitKey(1) & 0xff
        if key == 27 or drone.get_battery() < 10:
            drone.land()
            cv2.destroyAllWindows()
            drone.streamoff()
            break
        elif key == ord('w'):
            drone.move_forward(30)
        elif key == ord('s'):
            drone.move_back(30)
        elif key == ord('a'):
            drone.move_left(30)
        elif key == ord('d'):
            drone.move_right(30)
        elif key == ord('e'):
            drone.rotate_clockwise(30)
        elif key == ord('q'):
            drone.rotate_counter_clockwise(30)
        elif key == ord('r'):
            drone.move_up(30)
        elif key == ord('f'):
            drone.move_down(30)

        cv2.imshow('Image', img)
    