
import cv2
import easyocr

from threads.OCR_thread import OCR_thread
from threads.frame_producer import FrameProducer
from threads.processor_thread import FrameProcessor

from threads.pipeline import Pipeline
import pandas as pd


if __name__ == "__main__":

    #create the buffers
    frame_pipeline = Pipeline()
    processed_pipeline = Pipeline()
    violating_boxes_pipeline = Pipeline()
    violating_plates_text_pipeline = Pipeline()

    #create the threads
    t1 = FrameProducer(frame_pipeline, "traffic_video_modified.mp4")
    t2 = FrameProcessor(frame_pipeline, processed_pipeline,violating_boxes_pipeline, batch_dim=8)

    reader = easyocr.Reader(['en'])
    t3 = OCR_thread(violating_boxes_pipeline, violating_plates_text_pipeline, reader)


    t1.start()
    t2.start()
    t3.start()


    plates_text = []
    violation_info = []
    while True:
        ret, frame = processed_pipeline.get_message()

        if not ret:  # Video finito
            break

        violating_plate_text_box_info = violating_plates_text_pipeline.get_message(block=False)

        if violating_plate_text_box_info is not None:
            violating_plate_text, box_info = violating_plate_text_box_info
            #add text to list
            plates_text.append(violating_plate_text)
            violation_info.append(box_info)
        if len(plates_text) > 1:
            for i, text in enumerate(plates_text):
                cv2.rectangle(frame, (int(700 * t2.scale_factor), int(40 * t2.scale_factor + i*70 )),(int(1400 * t2.scale_factor), int(120 * t2.scale_factor + i*70 )), (0, 0, 0), -1)
                cv2.putText(frame, f"violation detected -> {text}", (int(720*t2.scale_factor),int(100*t2.scale_factor+(i*70))), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)

        cv2.imshow("c", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            t1.stop()
            t2.stop()
            break

    cv2.destroyAllWindows()


    #write the results on csv
    df = pd.DataFrame(violation_info)
    df["plate"] = plates_text
    df.to_csv("violations_data.csv", index=False)


