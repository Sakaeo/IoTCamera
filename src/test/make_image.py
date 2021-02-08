import cv2
import dlib
import numpy as np

W = None
H = None
rois = []
frame_nbr = 0
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

target = "car"

# Video Source
# vid_capture = cv2.VideoCapture("../img/in/campus4-c1.avi")
# vid_capture = cv2.VideoCapture("../img/in/20201115_133334.mp4")
vid_capture = cv2.VideoCapture("../img/in/traffic.mp4")

# Load Model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("../project/camera/mobilenet_ssd/MobileNetSSD_deploy.prototxt",
                               "../project/camera/mobilenet_ssd/MobileNetSSD_deploy.caffemodel")
print("Done")


def do_rec(frame):
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
    net.setInput(blob)
    detections = net.forward()

    boxes = []

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated
        # with the prediction
        confidence = detections[0, 0, i, 2]

        if confidence > 0.4:
            # if the class label is not a person, ignore it
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != target:
                continue

            # compute the (x, y)-coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            pos = dlib.rectangle(start_x, start_y, end_x, end_y)

            # unpack the position object
            x = int(pos.left())
            y = int(pos.top())
            w = int(pos.right() - pos.left())
            h = int(pos.bottom() - pos.top())

            boxes.append((x, y, w, h))
    return boxes


def draw_roi(frame, name, box, rgb):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), ((x + w), (y + h)), rgb, 2)

    x = int(x + w / 2)
    y = int(y + h / 2)
    text = "{}".format(name)
    cv2.putText(frame, text, (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 2)
    cv2.circle(frame, (x, y), 1, rgb, -1)


def draw_centroid(frame, name, centroid):
    text = "{}".format(name)
    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.circle(frame, (centroid[0], centroid[1]), 1, (0, 255, 0), -1)


while True:
    _, frame = vid_capture.read()
    frame_nbr += 1
    print(frame_nbr)

    # if end of video
    if frame is None:
        print("no Frame")
        error = True
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    key = cv2.waitKey(0) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    if key == ord("e"):
        roi = cv2.selectROI("ROI select", frame, False)
        rois.append(roi)

    for roi in rois:
        draw_roi(frame, "ROI", roi, (0, 0, 255))
    boxes = do_rec(frame)
    for rec in boxes:
        draw_roi(frame, target, rec, (0, 255, 0))

    cv2.imshow("imgTaker", frame)

    if key == ord("s"):
        cv2.imwrite("../img/out/{}.png".format(frame_nbr), frame)

cv2.destroyAllWindows()
vid_capture.release()
print("Camera stopped")
