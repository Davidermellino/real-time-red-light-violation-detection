import re
import threading
from time import sleep

import cv2


class OCR_thread(threading.Thread):

    def __init__(self, violating_boxes_pipeline,violating_plates_text_pipeline, reader):
        threading.Thread.__init__(self, daemon=True)

        self.violating_boxes_pipeline = violating_boxes_pipeline
        self.violating_plates_text_pipeline = violating_plates_text_pipeline
        self.reader = reader

        self.stopped = False

    def run(self):

        while not self.stopped:

            plate_cropped, box_info = self.violating_boxes_pipeline.get_message(block=True)


            #preprocess img
            gris = cv2.cvtColor(plate_cropped, cv2.COLOR_BGR2GRAY)

            _, thresholded_plate = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            res = self.reader.readtext(thresholded_plate)

            for (bbox, text, prob) in res:
                if prob > 0.7 and re.match(r"^[A-Z]{2}\s[0-9]{3,4}$", text):
                    self.violating_plates_text_pipeline.set_message((text, box_info))

            sleep(0.03)

    def stop(self):
        self.stopped = True