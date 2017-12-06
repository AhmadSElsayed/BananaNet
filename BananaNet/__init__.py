import mxnet as mx
from FaceDetector import FaceDetector
import cv2
import os
import time

detector = FaceDetector(model_folder='model', ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False)
camera = cv2.VideoCapture(0)
while True:
    grab, frame = camera.read()
    img = cv2.resize(frame, (320,180))

    t1 = time.time()
    results = detector.detect_face(img)
    print('time: ',time.time() - t1)

    cv2.imshow("Camera Feed", img)

    if results is None:
        continue

    total_boxes = results[0]
    points = results[1]
    
    draw = img.copy()
    for b in total_boxes:
        cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))

    for p in points:
        for i in range(5):
            cv2.circle(draw, (p[i], p[i + 5]), 1, (255, 0, 0), 2)
    cv2.imshow("detection result", draw)

    if cv2.waitKey(1)&0xFF == ord('x'):
        break


camera.release()
cv2.destroyAllWindows()
