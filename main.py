import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np


def detect_trafficlight_color(video_frame: np.ndarray) -> str:

    #HARD-CODED POSITION, CHANGE THEM IF NEEDED
    rect = (1810, 160, 110, 250)
    x, y, w, h = rect

    #crop the traffic light box
    traffic_light = video_frame[y - int(h / 2): y + int(h / 2), x - int(w / 2):x + int(w / 2)].copy()
    h, w, _ = traffic_light.shape

    #Crop the 3 traffic light section
    red = traffic_light[:int(h / 3), :]
    yellow = traffic_light[int(h / 3):h - int(h / 3), :]
    green = traffic_light[h - int(h / 3):, :]

    colors = {
        "yellow": yellow,
        "green": green,
        "red": red,
    }

    tl_color = ""
    for name, arr in colors.items():

        #apply tresholding to detect the active color (the red one do not pass the thresholding)
        gris = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        pimg = cv2.medianBlur(gris, 7)
        _, thresholded_image = cv2.threshold(pimg, 110, 255, cv2.THRESH_BINARY)

        #count the white pixel (if there are some, it means that that color is active)
        white_pixels = np.sum(thresholded_image == 255, )

        if white_pixels > 500:
            tl_color = name

        #the red one is the only one that do not pass the thresholding, (no white pixels when active)
        #is active whene either green and yellow are not active
        if name == "red" and tl_color == "":
            tl_color = name

    return tl_color


if __name__ == "__main__":
    video_path = "traffic_video_modified.mp4"

    cap = cv2.VideoCapture(video_path)


    if cap.isOpened():
        while True:

            ret, frame = cap.read()

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            traffic_light_color = detect_trafficlight_color(frame)
            cv2.rectangle(frame, (1500 -10, 100+10), (1500 + 210, 100 -50), (255,255,255), -1)
            cv2.putText(frame, f'{traffic_light_color.upper()}',(1550,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2 )
            cv2.imshow("video", frame)


            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()