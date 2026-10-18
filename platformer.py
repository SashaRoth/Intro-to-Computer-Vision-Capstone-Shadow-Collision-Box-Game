"""
1. Capture frames --> pass frame as single image
2. Edge detect image
3. Use edge detected image to set bounding box on video game side
4. Video game backend, wasd platformer
"""

import cv2
import numpy as np
import queue

#Set up queue
q = queue.Queue(maxsize=10)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
else:
    print("Video stream opened successfully.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Error: Could not read frame from video stream.")
        exit()

    image.flags.writeable = False
    cv2.imshow('Live Stream', cv2.flip(image, 1))

    # Exit stream command    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
