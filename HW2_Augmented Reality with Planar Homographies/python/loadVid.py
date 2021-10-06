import numpy as np
import cv2


def loadVid(path):
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(path)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    i = 0
    # Read until video is completed
    frames = []
    while (cap.isOpened()):
        # Capture frame-by-frame
        i += 1
        ret, frame = cap.read()
        if ret == True:
            # Store the resulting frame
            frames.append(frame[np.newaxis, ...])
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    frames = np.concatenate(frames, axis=0)
    frames = np.squeeze(frames)

    return frames