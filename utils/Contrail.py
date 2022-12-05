# pip install --upgrade imutils 필요시 설차


# import the necessary packages
from collections import deque
import numpy as np
import argparse
import cv2


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--buffer", type=int, default=45)
args = vars(ap.parse_args())
# pts = deque(maxlen=args["buffer"])

def tracking_tail(xycenter, frame, color):
    

    # # update the points queue
    # pts.appendleft(center)

    # loop over the set of tracked points
    for i in range(1, len(xycenter)):
        # if either of the tracked points are None, ignore
        # them
        if xycenter[i - 1] is None or xycenter[i] is None:
            continue
        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, xycenter[i - 1], xycenter[i], color, thickness)
        # cv2.imwrite("test_tracking_frame.png",frame)
    # show the frame to our screen
    # cv2.imshow("Frame", frame)
    
    # cv2.imwrite("test_tracking_result.png",frame)
    return (frame)

    # python tracking.py --video ball_tracking_example.mp4