# pip install --upgrade imutils 필요시 설차


# import the necessary packages
from collections import deque
import numpy as np
import argparse
import cv2
from utils.PixelMapper import pm1


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--buffer", type=int, default=45)
args = vars(ap.parse_args())
# pts = deque(maxlen=args["buffer"])

def tracking_tail(xycenter, frame, color, meal_amount, water_intake):
    if meal_amount == 1:
        color = (98, 232, 236)

    if water_intake == 1:
        color = (155, 95, 41)

            # print('test')
        # c1 = xycenter[0]
        # c2 = xycenter[1]
        # c1 = pm1.pixel_to_lonlat(c1)
        # c2 = pm1.pixel_to_lonlat(c2)
        # c1 = [int(c1[0][0]), int(c1[0][1])]
        # c2 = [int(c2[0][0]), int(c2[0][1])]

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
        cv2.line(frame, xycenter[i - 1], xycenter[i], color, thickness + 3)
        # cv2.imwrite("test_tracking_frame.png",frame)
    # show the frame to our screen
    # cv2.imshow("Frame", frame)
    
    # cv2.imwrite("test_tracking_result.png",frame)
    return (frame)

    # python tracking.py --video ball_tracking_example.mp4

def minimap_tail(xycenter, frame, color, meal_amount, water_intake):
    if meal_amount == 1:
        color = (98, 232, 236)

    if water_intake == 1:
        color = (155, 95, 41)



    result = list()
    for k in range(len(xycenter)):
        center = pm1.pixel_to_lonlat(xycenter[k])
        center = (int(center[0][0]), int(center[0][1]))
        result.append(center)

    for i in range(1, len(result)):
        # if either of the tracked points are None, ignore
        # them
        if result[i - 1] is None or result[i] is None:
            continue
        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, result[i - 1], result[i], color, thickness + 3)
        # cv2.imwrite("test_tracking_frame.png",frame)
    # show the frame to our screen
    # cv2.imshow("Frame", frame)
    
    # cv2.imwrite("test_tracking_result.png",frame)
    return (frame)