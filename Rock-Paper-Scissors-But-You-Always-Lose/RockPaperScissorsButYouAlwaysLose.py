import cv2
import HandTrackingModule as htm

cam_width, cam_height = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

detector = htm.HandDetector(detection_confidence=0.7, max_hands=1)

while True:
    # Read Image from webcam then horizontally flip image
    success, img = cap.read()
    cv2.flip(img, 1)

    # Find hand
    img = detector.find_hands(img)
    lm_list, _ = detector.find_position(img, draw=False)

    if len(lm_list) != 0:
        fingers = detector.fingers_up()
        # print(fingers)

        if fingers[1] and fingers[2] and not fingers[0] and not fingers[3] and not fingers[4]:
            # Scissors
            cv2.putText(img, 'Computer chooses ROCK', (800, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, 'You Lose!', (1050, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        elif all(fingers):
            # Paper
            cv2.putText(img, 'Computer chooses SCISSORS', (700, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, 'You Lose!', (1050, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        elif fingers[0] and not all(fingers[1:]):
            # Rock
            cv2.putText(img, 'Computer chooses PAPER', (750, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, 'You Lose!', (1050, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)


    cv2.imshow("Img", img)
    key = cv2.waitKey(1)

    if key == ord('q'):
            break
