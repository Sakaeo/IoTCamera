class TrackableObject:

    def __init__(self, object_id, centroid, object_class):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.object_id = object_id
        self.centroids = [centroid]
        # initialize a boolean used to indicate if the object has
        # already been counted or not
        self.counted = False

        self.object_class = object_class

    def __repr__(self):
        return "ID: {}, Class: {}, Centroids: {}".format(self.object_id, self.object_class, self.centroids)
