from collections import OrderedDict

import numpy as np
from scipy.spatial import distance as dist


class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        # Object ID, Start with 0
        self.next_object_id = 0
        # Ordered dictionaries with current and disappeared Centroids in it
        # ID as key, (x,y) as value
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # number of Frames a object can disappear before it gets deregistered
        self.max_disappeared = maxDisappeared

        # max distance a object can move between frames and still be recognised as the same object
        self.max_distance = maxDistance

        self.key_centroids = np.zeros(1)
        self.key_names = []

    def find_closest_key(self, centroid):
        d = dist.cdist(self.key_centroids, centroid.reshape((1, 2)))
        rows = d.min(axis=1).argsort()
        return self.key_names[rows[0]]

    def register(self, centroid):
        id = self.next_object_id
        self.objects[id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        start = self.find_closest_key(centroid)

        # TODO MQTT
        print("ID: {} starts in {}".format(id, start))

    def deregister(self, object_id):
        end = self.find_closest_key(self.objects[object_id])

        # TODO MQTT
        print("ID: {} ends in {}".format(object_id, end))

        del self.objects[object_id]
        del self.disappeared[object_id]

    def update_key_centroids(self, key_centroids):
        self.key_names = list(key_centroids.keys())
        self.key_centroids = np.zeros((len(key_centroids.values()), 2), dtype="int")
        for (i, (x, y)) in enumerate(key_centroids.values()):
            self.key_centroids[i] = (x, y)

    def update(self, rects):  # Input list of bounding boxes (startX,startY,endX,endY)
        # if the input is empty, loop over all current objects an count them as disappeared
        # and deregister them if they reached mDisappeared
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # initialize an array for the current frame
        # Rows = each Centroid, Cols = (X,Y) for Centroid
        input_centroids = np.zeros((len(rects), 2), dtype="int")

        # loop over the bounding box rectangles and calculate centroid
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        # if no objects are tracked, register the input centroids
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])

        # otherwise, match tracked objects with input
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # compute the distance between each object and input centroid
            # D has dimensions: len(objectCentroids) by len(inputCentroids)
            # with the Distance between i nad j in ij
            d = dist.cdist(np.array(object_centroids), input_centroids)

            # rows contains a 1xn Array,
            # starting with the row index of the smallest value in d
            rows = d.min(axis=1).argsort()

            # same for cols
            cols = d.argmin(axis=1)[rows]

            # if a pair is found, the centroid cant be used anymore
            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):  # the (row,col) tuple starts with the min value in D
                # skip used values
                if row in used_rows or col in used_cols:
                    continue

                # skip if distance bigger than maxDistance
                if d[row, col] > self.max_distance:
                    continue

                # else Update object and reset disappeared counter
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                # add to used
                used_rows.add(row)
                used_cols.add(col)

            # compute witch cols and rows are unused
            unused_rows = set(range(0, d.shape[0])).difference(used_rows)
            unused_cols = set(range(0, d.shape[1])).difference(used_cols)

            # if tracked objects >= input centroids, then check if objects have disappeared
            if d.shape[0] >= d.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1

                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)

            # else register new object
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        # return the set of trackable objects
        return self.objects
