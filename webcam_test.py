import cv2
import argparse
from utils import  colorTracking, HaarFaceTracking, velocityChange
import sys



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('track', type=str, help="Write the kind of tracking 'color' or 'face'")
    args = parser.parse_args()

    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        webcam.release()
        sys.exit("Error opening webcam")
    ret, frame = webcam.read()
    if not ret:
        webcam = cv2.VideoCapture(1)
        ret, frame = webcam.read()



    win_name = 'Frame'
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, 20, 20)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = .5
    fontColor = (255, 255, 255)
    lineType = 2

    while webcam.isOpened():
        ret, frame = webcam.read()

        if args.track == 'color':
            img, unit_vector = colorTracking(frame)
        elif args.track == 'face':
            img, unit_vector = HaarFaceTracking(frame)

        yaw_velocity, up_down_velocity, forward_backward_velocity, left_right_velocity = velocityChange(unit_vector)
        print(yaw_velocity, up_down_velocity, forward_backward_velocity, left_right_velocity)


        cv2.imshow(win_name, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()