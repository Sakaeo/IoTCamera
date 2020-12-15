import cv2

# https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/
# vid_capture = cv2.VideoCapture('../Videos/passageway1-c1.avi')
vid_capture = cv2.VideoCapture(0)

FRAME_WIDTH = int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

ret, frame1 = vid_capture.read()
ret, frame2 = vid_capture.read()

while vid_capture.isOpened():
    # Difference between 1st and 2nd Frame
    diff = cv2.absdiff(frame1, frame2)
    cv2.imshow("diff", diff)
    # Make Grey
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # Filter Noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Filters out Pixels outside the Threshold
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    # Fills out Contours
    dilated = cv2.dilate(thresh, None, iterations=3)
    cv2.imshow("dilated", dilated)
    # Get Contours
    _, contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) > 1000:
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("src", frame1)
    # Replace frame1 and load new Frame2
    frame1 = frame2
    ret, frame2 = vid_capture.read()

    k = cv2.waitKey(5) & 0xFF
    if k == 27:  # Esc
        break

cv2.destroyAllWindows()
vid_capture.release()
