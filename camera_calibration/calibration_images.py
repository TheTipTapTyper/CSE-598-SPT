import cv2
import os
cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

num = 0

print("About to make dir")

while cap.isOpened():

    succes, img = cap.read()
    succes2, img2 = cap2.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('camera_calibration\images\human_pose\imageL' + str(num) + '.png', img)
        cv2.imwrite('camera_calibration\images\human_pose\imageR' + str(num) + '.png', img2)
        print("images saved!")
        num += 1

    cv2.imshow('Img 1',img)
    cv2.imshow('Img 2',img2)

# Release and destroy all windows before termination
cap.release()
cap2.release()

cv2.destroyAllWindows()