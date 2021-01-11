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



    while True:


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
    