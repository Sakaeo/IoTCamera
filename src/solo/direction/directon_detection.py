from collections import OrderedDict
from solo.direction.helper.centroid_tracker import CentroidTracker
from solo.direction.helper.trackable_object import TrackableObject
import cv2
import dlib
import numpy as np
from imutils.video import FPS

# Start Arguments
skip_frame = 5
min_confidence = 0.4

# Video Source and ROI selection
# vid_capture = cv2.VideoCapture("../../videos/example_01.mp4")
vid_capture = cv2.VideoCapture("../../videos/passageway1-c1.avi")
# vid_capture = cv2.VideoCapture(0)

ret, img = vid_capture.read()
bounding_boxes = cv2.selectROIs("Tracking", img, False, False)

# Classes the net Model recognises
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# centroid tracker and some inits
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}
W = None
H = None

# countable variables
totalFrames = 0


def draw_box(frame, box):
    x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (0, 255, 0), 2)


def draw_centroid(frame, name, centroid):
    text = "{}".format(name)
    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.circle(frame, (centroid[0], centroid[1]), 1, (0, 255, 0), -1)


def main():
    global W, H, trackers, totalFrames

    # init key centroid array
    key_centroids = OrderedDict()

    # loop over the bounding box rectangles and calculate centroid
    for (i, (x, y, w, h)) in enumerate(bounding_boxes):
        cX = int((x + w / 2))
        cY = int((y + h / 2))
        key_centroids[i] = (cX, cY)
    ct.update_key_centroids(key_centroids)

    # Load Model
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe("helper/mobilenet_ssd/MobileNetSSD_deploy.prototxt",
                                   "helper/mobilenet_ssd/MobileNetSSD_deploy.caffemodel")
    # Start FPS counter
    fps = FPS().start()

    while True:
        _, frame = vid_capture.read()

        # if end of video
        if frame is None:
            break

        # resize and convert to rgb for dlib
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # set frame dimensions
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # init bounding box rectangles
        rects = []

        # Only search for objects every 5 frames
        if totalFrames % skip_frame == 0:
            # init new set of trackers
            trackers = []

            # convert the frame to a blob and pass the blob through the
            # network and obtain the detections
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated
                # with the prediction
                confidence = detections[0, 0, i, 2]

                if confidence > min_confidence:
                    # if the class label is not a person, ignore it
                    idx = int(detections[0, 0, i, 1])
                    if CLASSES[idx] != "person":
                        continue

                    # compute the (x, y)-coordinates of the bounding box
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    # make a dlib rectangle object and start the dlib tracker
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    trackers.append(tracker)

        # use tracker during skipped frames, not object recognition
        else:
            for tracker in trackers:
                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # add the box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))

        # use centroids to match old and new centroids and then loop over them
        objects = ct.update(rects)
        for (objectID, centroid) in objects.items():
            # check if object is in the object list
            t_object = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if t_object is None:
                t_object = TrackableObject(objectID, centroid)

            # Else if movement direction is required TODO

            # store the trackable object in our dictionary
            trackableObjects[objectID] = t_object

            # draw Centroid
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        for box in bounding_boxes:
            draw_box(frame, box)
        for (i, centroid) in enumerate(key_centroids.values()):
            draw_centroid(frame, i, centroid)

        cv2.imshow("Tracking", frame)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:  # Esc
            break

        totalFrames += 1
        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    cv2.destroyAllWindows()
    vid_capture.release()


if __name__ == '__main__':
    main()
