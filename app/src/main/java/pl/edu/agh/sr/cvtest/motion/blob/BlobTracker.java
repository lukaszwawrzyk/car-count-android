package pl.edu.agh.sr.cvtest.motion.blob;

import org.opencv.core.Point;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

public class BlobTracker {
    private List<Blob> trackedBlobs = new ArrayList<>();
    private int currentId = 1;

    public List<Blob> blobs() {
        return Collections.unmodifiableList(trackedBlobs);
    }

    public void update(List<Blob> visibleBlobs) {
        if (trackedBlobs.isEmpty()) {
            addAsNewBlobs(visibleBlobs);
        } else {
            matchBlobs(visibleBlobs);
        }
    }

    private void addAsNewBlobs(List<Blob> visibleBlobs) {
        for (Blob visibleBlob : visibleBlobs) {
            addNewBlob(visibleBlob);
        }
    }

    private void matchBlobs(List<Blob> visibleBlobs) {
        for (Blob trackedBlob : trackedBlobs) {
            trackedBlob.matchFoundOrIsNew = false;
            trackedBlob.updatePredictedPosition();
        }
        for (Blob currentFrameBlob : visibleBlobs) {
            addAsNewOrUpdate(currentFrameBlob);
        }
        Iterator<Blob> iterator = trackedBlobs.iterator();
        while (iterator.hasNext()) {
            if (iterator.next().disappeared()) {
                iterator.remove();
            }
        }
    }

    private void addAsNewOrUpdate(Blob newBlob) {
        Blob closestExistingBlob = trackedBlobs.get(0);
        double minDistance = 10000000;

        for (Blob existingBlob : trackedBlobs) {
            double distance = distance(newBlob.position(), existingBlob.predictedPosition);
            if (distance < minDistance) {
                minDistance = distance;
                closestExistingBlob = existingBlob;
            }
        }

        newBlob.matchFoundOrIsNew = true;
        if (newBlob.isCloseEnough(minDistance)) {
            closestExistingBlob.updateFrom(newBlob);
        } else {
            addNewBlob(newBlob);
        }
    }

    private double distance(Point point1, Point point2) {
        double xDiff = Math.abs(point1.x - point2.x);
        double yDiff = Math.abs(point1.y - point2.y);
        return Math.sqrt(Math.pow(xDiff, 2) + Math.pow(yDiff, 2));
    }

    private void addNewBlob(Blob newBlob) {
        newBlob.id = nextId();
        trackedBlobs.add(newBlob);
    }

    private int nextId() {
        return currentId++;
    }

}
