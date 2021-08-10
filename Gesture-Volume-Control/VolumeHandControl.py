import cv2
import numpy as np
import time
import math
import HandTrackingModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#######################################
cam_width, cam_height = 1280, 720
#######################################

pTime = 0
cTime = 0

cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

detector = htm.HandDetector(detection_confidence=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume_range = volume.GetVolumeRange()
min_volume = volume_range[0]
max_volume = volume_range[1]
vol_bar = 400
vol_percentage = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.find_hands(img)
    lm_list, _ = detector.find_position(img, draw=False)
    if len(lm_list) != 0:

        x1, y1, = lm_list[4][1], lm_list[4][2]
        x2, y2, = lm_list[8][1], lm_list[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (255, 0 ,255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0 ,255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0 ,255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        # Hand range 50 - 280
        # Volume range min - max

        vol = np.interp(length, [50, 280], [min_volume, max_volume])
        vol_bar = np.interp(length, [50, 280], [400, 150])
        vol_percentage = np.interp(length, [50, 280], [0, 100])

        print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
            cv2.putText(img, 'DEAFENED', (1100, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(vol_percentage)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {str(int(fps))}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Img", img)
    key = cv2.waitKey(1)

    if key == ord('q'):
            break