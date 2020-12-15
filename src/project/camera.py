import base64
import json
import math
import time
from collections import OrderedDict

import cv2
import dlib
import numpy as np
from imutils.video import FPS

from project import mqtt_publisher
from project.helper.centroid_tracker import CentroidTracker
from project.helper.trackable_object import TrackableObject


def draw_roi(frame, name, box):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (0, 255, 0), 2)

    x = int(x + w / 2)
    y = int(y + h / 2)
    text = "{}".format(name)
    cv2.putText(frame, text, (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


def draw_centroid(frame, name, centroid):
    text = "{}".format(name)
    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.circle(frame, (centroid[0], centroid[1]), 1, (0, 255, 0), -1)


class Camera:
    def __init__(self, publisher: mqtt_publisher, skip_frame=5, min_confidence=0.4):
        # Start Arguments
        self.skip_frame = skip_frame
        self.min_confidence = min_confidence

        self.publisher: mqtt_publisher = publisher

        # Video Source and ROI selection
        # self.vid_capture = cv2.VideoCapture("../../videos/passageway1-c1.avi")
        # self.vid_capture = cv2.VideoCapture("../../videos/example_01.mp4")
        # vid_capture = cv2.VideoCapture("../../videos/traffic.mp4")
        self.vid_capture = cv2.VideoCapture(0)

        # Classes the net Model recognises
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]

        # centroid tracker and some inits
        self.ct = CentroidTracker(publisher, maxDisappeared=40, maxDistance=50, )
        self.trackers = []
        self.trackableObjects = {}
        self.targets = []
        self.rois = OrderedDict()
        self.W = None
        self.H = None
        self.class_list = []
        self.take_snap = False

        # countable variables
        self.totalFrames = 0

    def read_config(self):
        self.targets = []
        self.rois = OrderedDict()
        with open("config.json") as json_file:
            data = json.load(json_file)
            for t in data["target"]:
                self.targets.append(t)
            for roi in data["ROI"]:
                name = roi["name"]
                (x, y, w, h) = roi["coordinates"]
                self.rois[name] = (int(x), int(y), int(w), int(h))
        self.ct.update_key_centroids(self.rois)

    def reload_config(self):
        self.read_config()

    def snapshot(self):
        self.take_snap = True

    def publish_snap(self, img):
        val, buffer = cv2.imencode(".jpg", img)
        encoded = base64.b64encode(buffer)

        packet_size = 3000
        start = 0
        end = packet_size
        length = len(encoded)
        pic_id = "snapshot_{}".format(self.totalFrames)
        pos = 0
        packet_number = math.ceil(length / packet_size) - 1

        while start <= length:
            data = {
                "data": str(encoded[start:end]),
                "pic_id": pic_id,
                "pos": pos,
                "packet_number": packet_number
            }
            print("sending {}/{}".format(pos, packet_number))

            self.publisher.publish(json.dumps(data), "test/test/snapshot")
            end += packet_size
            start += packet_size
            pos = pos + 1
            time.sleep(0.2)
        print("ok")

    def run_camera(self):
        self.read_config()

        # Load Model
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe("helper/mobilenet_ssd/MobileNetSSD_deploy.prototxt",
                                       "helper/mobilenet_ssd/MobileNetSSD_deploy.caffemodel")
        # Start FPS counter
        fps = FPS().start()

        while True:
            _, frame = self.vid_capture.read()

            # if end of video
            if frame is None:
                break

            # resize and convert to rgb for dlib
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # set frame dimensions
            if self.W is None or self.H is None:
                (self.H, self.W) = frame.shape[:2]

            # snapshot
            if self.take_snap:
                self.publish_snap(frame)
                self.take_snap = False
                continue

            # init bounding box rectangles
            rects = []

            # Only search for objects every 5 frames
            if self.totalFrames % self.skip_frame == 0:
                totalFrames = 0
                # init new set of trackers
                self.trackers = []
                self.class_list = []

                # convert the frame to a blob and pass the blob through the
                # network and obtain the detections
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (self.W, self.H), 127.5)
                net.setInput(blob)
                detections = net.forward()

                # loop over the detections
                for i in np.arange(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated
                    # with the prediction
                    confidence = detections[0, 0, i, 2]

                    if confidence > self.min_confidence:
                        # if the class label is not a person, ignore it
                        idx = int(detections[0, 0, i, 1])
                        target = self.CLASSES[idx]
                        if not self.targets.__contains__(target):
                            continue

                        # compute the (x, y)-coordinates of the bounding box
                        box = detections[0, 0, i, 3:7] * np.array([self.W, self.H, self.W, self.H])
                        (startX, startY, endX, endY) = box.astype("int")

                        # make a dlib rectangle object and start the dlib tracker
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(startX, startY, endX, endY)
                        tracker.start_track(rgb, rect)

                        self.trackers.append(tracker)
                        self.class_list.append(target)

            # use tracker during skipped frames, not object recognition
            else:
                for tracker in self.trackers:
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
            (objects, class_dict) = self.ct.update(rects, self.class_list)

            for (object_id, centroid) in objects.items():
                # check if object is in the object list
                t_object = self.trackableObjects.get(object_id, None)

                # if there is no existing trackable object, create one
                if t_object is None:
                    t_object = TrackableObject(object_id, centroid, class_dict[object_id])

                else:
                    t_object.centroids.append(centroid)

                # store the trackable object in our dictionary
                self.trackableObjects[object_id] = t_object

                # draw Centroid
                text = "ID {}".format(object_id)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            for roi in self.rois:
                draw_roi(frame, roi, self.rois[roi])

            # Debugging
            # cv2.imshow("Tracking", frame)

            k = cv2.waitKey(5) & 0xFF
            if k == 27:  # Esc
                break

            self.totalFrames += 1
            fps.update()

        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        cv2.destroyAllWindows()
        self.vid_capture.release()
