class TrackableObject:
    def __init__(self, objectID, centroid):
        # Object ID
        self.objectID = objectID
        # List of current and past Centroids
        self.centroids = [centroid]
