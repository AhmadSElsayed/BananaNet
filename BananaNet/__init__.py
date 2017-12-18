import mxnet as mx
import swap
import numpy as np
from FaceDetector import FaceDetector
import cv2
import os
import time

def main():
	detector = FaceDetector(model_folder='model', ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False)
	
	OriginFace = cv2.imread('face.jpg',cv2.IMREAD_COLOR )
	ResultOrigin = detector.detect_face(OriginFace)
	P = ResultOrigin[1][0]
	origin_points = np.array([(int(P[0]), int(P[5])), (int(P[1]), int(P[6])), (int(P[2]), int(P[7])), (int(P[3]), int(P[8])), (int(P[4]), int(P[9]))])
	swap.initSwappingModule(OriginFace, origin_points)

	camera = cv2.VideoCapture(0)
	while True:
		grab, frame = camera.read()
		img = cv2.resize(frame, (320,180))

		t1 = time.time()
		results = detector.detect_face(img)
		print('time: ',time.time() - t1)

		if results is None:
			continue

		total_boxes = results[0]
		points = results[1]
		
		draw = img.copy()
		for b in total_boxes:
			cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))

		result = draw.copy()
		
		for p in points:
			for i in range(5):
				cv2.circle(draw, (p[i], p[i + 5]), 1, (255, 0, 0), 2)

			origin_points = np.array([(int(p[0]), int(p[5])), (int(p[1]), int(p[6])), (int(p[2]), int(p[7])), (int(p[3]), int(p[8])), (int(p[4]), int(p[9]))])
			result = swap.swap(result, origin_points)
			cv2.imshow("FaceSwap", result)

		cv2.imshow("detection result", draw)

		if cv2.waitKey(1)&0xFF == ord('x'):
			break
	camera.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
