import cv2

cap = cv2.VideoCapture('../Videos/passageway1-c1.avi')

while cap.isOpened():
    _, frame = cap.read()

    cv2.imshow("inter", frame)

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
