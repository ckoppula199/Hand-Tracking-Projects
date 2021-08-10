import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

#######################
brushThickness = 25
eraserThickness = 100
########################


folder_path = "Header"
my_list = os.listdir(folder_path)
print(my_list)
overlay_list = []
for img_path in my_list:
    image = cv2.imread(f'{folder_path}/{img_path}')
    overlay_list.append(image)
print(len(overlay_list))
header = overlay_list[0]
draw_color = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.HandDetector(detection_confidence=0.85)
xp, yp = 0, 0
img_canvas = np.zeros((720, 1280, 3), np.uint8)

while True:

    # Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Find Hand Landmarks
    img = detector.find_hands(img)
    lm_list, _ = detector.find_position(img, draw=False)

    if len(lm_list) != 0:

        x1, y1 = lm_list[8][1], lm_list[8][2]
        x2, y2 = lm_list[12][1], lm_list[12][2]

        # Check which fingers are up
        fingers = detector.fingers_up()

        # If Selection Mode - Two finger are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlay_list[0]
                    draw_color = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlay_list[1]
                    draw_color = (255, 100, 0)
                elif 800 < x1 < 950:
                    header = overlay_list[2]
                    draw_color = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlay_list[3]
                    draw_color = (0, 0, 0)

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), draw_color, cv2.FILLED)


        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if draw_color == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), draw_color, eraserThickness)
                cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), draw_color, brushThickness)
                cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, brushThickness)

            xp, yp = x1, y1

        # Clear Canvas when all fingers are up
        # if all (x >= 1 for x in fingers):
        #     img_canvas = np.zeros((720, 1280, 3), np.uint8)

    imgGray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,img_canvas)


    # Setting header image
    img[0:125, 0:1280] = header
    cv2.imshow("Img", img)
    # cv2.imshow("Img canvas", img_canvas)
    key = cv2.waitKey(1)

    if key == ord('q'):
            break