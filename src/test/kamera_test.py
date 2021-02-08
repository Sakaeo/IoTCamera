import cv2

vid_capture = cv2.VideoCapture(0)

while True:
    _, frame = vid_capture.read()

    # if end of video
    if frame is None:
        break

    cv2.imshow("Test", frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:  # Esc
        break

cv2.destroyAllWindows()
vid_capture.release()
