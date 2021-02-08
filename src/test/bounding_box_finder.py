import cv2

# vid_capture = cv2.VideoCapture("../../videos/example_01.mp4")
# vid_capture = cv2.VideoCapture("../../videos/passageway1-c1.avi")
# vid_capture = cv2.VideoCapture("../../videos/traffic.mp4")
vid_capture = cv2.VideoCapture(0)

ret, img = vid_capture.read()
bounding_boxes = cv2.selectROIs("Tracking", img, False, False)

for (i, (x, y, w, h)) in enumerate(bounding_boxes):
    print("Box: {} = {}".format(i, (x, y, w, h)))


def draw_box(frame, box):
    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (0, 255, 0), 2)


while True:
    _, frame = vid_capture.read()

    # if end of video
    if frame is None:
        break

    for (i, (x, y, w, h)) in enumerate(bounding_boxes):
        draw_box(frame, bounding_boxes[i])

        text = "ID {}".format(i)
        x = int(x + w / 2)
        y = int(y + h / 2)
        cv2.putText(frame, text, (x + 10, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    cv2.imshow("Tracking", frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:  # Esc
        break

cv2.destroyAllWindows()
vid_capture.release()
