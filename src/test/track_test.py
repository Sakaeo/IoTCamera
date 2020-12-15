import cv2

vid_capture = cv2.VideoCapture(0)

# tracker = cv2.TrackerMOSSE_create()  # faster
tracker = cv2.TrackerCSRT_create()  # more accurate

# Start Tracker with Region of Interest (ROI)
ret, img = vid_capture.read()
bounding_box = cv2.selectROI("Tracking", img, False)
tracker.init(img, bounding_box)


def draw_box(frame, box):
    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (0, 255, 0), 2)
    cv2.putText(frame, "Tracking", (75, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def main():
    while vid_capture.isOpened():
        ret, frame = vid_capture.read()
        # Tracker Update
        success, update_box = tracker.update(frame)

        if success:
            draw_box(frame, update_box)
        else:
            cv2.putText(frame, "lost", (75, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Tracking", frame)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:  # Esc
            break

    cv2.destroyAllWindows()
    vid_capture.release()


if __name__ == '__main__':
    main()
