

import cv2


cap = cv2.VideoCapture(4)
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    
    cv2.imshow("Test", frame)
    cv2.waitKey(3)