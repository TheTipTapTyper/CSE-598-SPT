import numpy as np
import cv2 as cv
import glob


################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (9,7)
frameSize = (640,480)


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 25
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.

imagesLeft = sorted(glob.glob('camera_calibration\images\stereoLeft\*.png'))
imagesRight = sorted(glob.glob('camera_calibration\images\stereoRight\*.png'))

print(len(imagesRight))

for imgLeft, imgRight in zip(imagesLeft, imagesRight):

    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if retL and retR == True:

        objpoints.append(objp)

        cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        imgpointsL.append(cornersL)

        cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpointsR.append(cornersR)

        # Draw and display the corners
        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.imshow('img left', imgL)
        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        cv.imshow('img right', imgR)
        cv.waitKey(1000)


cv.destroyAllWindows()




############## CALIBRATION #######################################################

retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
heightL, widthL, channelsL = imgL.shape
newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
heightR, widthR, channelsR = imgR.shape
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))



########## Stereo Vision Calibration #############################################

flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same 

criteria_stereo= (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)



########## Stereo Rectification #################################################

rectifyScale= 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))
print("projMatrixR", projMatrixR)
print("projMatrixL", projMatrixL)

# stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
# stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

# print("Saving parameters!")
# cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)

# cv_file.write('stereoMapL_x',stereoMapL[0])
# cv_file.write('stereoMapL_y',stereoMapL[1])
# cv_file.write('stereoMapR_x',stereoMapR[0])
# cv_file.write('stereoMapR_y',stereoMapR[1])

# cv_file.release()


# Camera parameters to undistort and rectify images
# cv_file = cv2.FileStorage()
# cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

# stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
# stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
# stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
# stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()


# Open both cameras
# cap_right = cv.VideoCapture(1, cv.CAP_DSHOW)                    
# cap_left =  cv.VideoCapture(0, cv.CAP_DSHOW)


# while(cap_right.isOpened() and cap_left.isOpened()):

#     succes_right, frame_right = cap_right.read()
#     succes_left, frame_left = cap_left.read()

#     # Undistort and rectify images
#     frame_left = cv.undistort(frame_left, cameraMatrixL, distL, None, newCameraMatrixL)
#     frame_right = cv.undistort(frame_right, cameraMatrixR, distR, None, newCameraMatrixR)
                     
#     # Show the frames
#     cv.imshow("frame right", frame_right) 
#     cv.imshow("frame left", frame_left)


#     # Hit "q" to close the window
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break


# # Release and destroy all windows before termination
# cap_right.release()
# cap_left.release()

# cv.destroyAllWindows()