import cv2
import numpy as np
import time
import math
import HandTrackingModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cam_width, cam_height = 1280, 720

pTime = 0
cTime = 0

cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

detector = htm.HandDetector(detection_confidence=0.7, max_hands=1)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume_range = volume.GetVolumeRange()
min_volume = volume_range[0]
max_volume = volume_range[1]
vol_bar = 400
vol_percentage = 0
area = 0
colour_vol = (255, 0, 0)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Find Hand
    img = detector.find_hands(img)
    lm_list, bbox = detector.find_position(img, draw=False)
    if len(lm_list) != 0:

        # Filter based on size
        cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
        if 500 < area < 1500:

            # Find distance between index and thumb

            length, img, line_info = detector.find_distance(4, 8, img)

            # Convert volume

            # Hand range 50 - 280
            # Volume range min - max

            vol_bar = np.interp(length, [50, 280], [400, 150])
            vol_percentage = np.interp(length, [50, 280], [0, 100])
            

            # Reduce resolution to make it smoother

            increment = 5
            vol_percentage = increment * round(vol_percentage / increment)

            # Check what fingers are up
            fingers = detector.fingers_up()
            print(fingers)

            # If little finger down then set volume
            if fingers[-1] == 0:
                volume.SetMasterVolumeLevelScalar(vol_percentage / 100, None)
                colour_vol = (0, 255, 0)
            else:
                colour_vol = (255, 0, 0)

            # If volume 0 notify user is deafened        
            if length < 50:
                cv2.circle(img, (line_info[-2], line_info[-1]), 15, (0, 255, 0), cv2.FILLED)
                cv2.putText(img, 'DEAFENED', (1100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                

    # Drawings
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(vol_percentage)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    curr_vol = int(volume.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(img, f'Current Volume: {int(curr_vol)}', (900, 50), cv2.FONT_HERSHEY_COMPLEX, 1, colour_vol, 3)
    
    # Frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {str(int(fps))}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Img", img)
    key = cv2.waitKey(1)

    if key == ord('q'):
            break