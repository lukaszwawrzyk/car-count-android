package pl.edu.agh.sr.cvtest.counting;

import android.support.annotation.NonNull;
import org.opencv.core.*;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.opencv.imgproc.Imgproc;
import org.opencv.core.Core;

public class MainLoop {

    private static final Drawing draw = new Drawing();

    private Mat storedPreviousFrame;
    private Mat storedCurrentFrame;

    private List<Blob> trackedBlobs = new ArrayList<>();
    private CrossingLine crossingLine;
    private BlobDetector blobDetector;
    private int blobCount = 0;

    public Mat getFrame(Mat newFrame) {
        if (notInitialized()) {
            initFromFirstFrame(newFrame);
            return newFrame;
        }
        shiftFrames(newFrame);

        Mat prevFrame = storedPreviousFrame;
        Mat currFrame = storedCurrentFrame.clone();
        List<Blob> currentFrameBlobs = blobDetector.detect(prevFrame, currFrame);
        updateTrackedBlobs(currentFrameBlobs);
        boolean lineCrossed = countLineCrossingBlobs();

        Mat output = storedCurrentFrame.clone();
        draw.finalFrame(output, trackedBlobs, crossingLine, lineCrossed, blobCount);
        return output;
    }

    private boolean countLineCrossingBlobs() {
        boolean crossed = false;
        for (Blob blob : trackedBlobs) {
            if (blob.crossed(crossingLine)) {
                blobCount++;
                crossed = true;
            }
        }
        return crossed;
    }

    private void updateTrackedBlobs(List<Blob> currentFrameBlobs) {
        if (trackedBlobs.isEmpty()) {
            addAsNewBlobs(currentFrameBlobs);
        } else {
            matchBlobs(currentFrameBlobs);
        }
    }

    private void addAsNewBlobs(List<Blob> currentFrameBlobs) {
        for (Blob currentFrameBlob : currentFrameBlobs) {
            addNewBlob(currentFrameBlob);
        }
    }

    private void matchBlobs(List<Blob> currentFrameBlobs) {
        for (Blob existingBlob : trackedBlobs) {
            existingBlob.matchFoundOrIsNew = false;
            existingBlob.updatePredictedPosition();
        }
        for (Blob currentFrameBlob : currentFrameBlobs) {
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
            double distance = distanceBetweenPoints(newBlob.position(), existingBlob.predictedPosition);
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

    private void addNewBlob(Blob newBlob) {
        newBlob.id = nextId();
        trackedBlobs.add(newBlob);
    }

    private int currentId = 1;
    private int nextId() {
        return currentId++;
    }

    private double distanceBetweenPoints(Point point1, Point point2) {
        double xDiff = Math.abs(point1.x - point2.x);
        double yDiff = Math.abs(point1.y - point2.y);
        return Math.sqrt(Math.pow(xDiff, 2) + Math.pow(yDiff, 2));
    }

    private boolean notInitialized() {
        return storedPreviousFrame == null && storedCurrentFrame == null;
    }

    private void shiftFrames(Mat newFrame) {
        if (storedPreviousFrame != null) storedPreviousFrame.release();
        storedPreviousFrame = storedCurrentFrame;
        storedCurrentFrame = newFrame.clone();
    }

    private void initFromFirstFrame(Mat newFrame) {
        storedCurrentFrame = newFrame.clone();
        blobDetector = new BlobDetector(new Mat(newFrame.size(), newFrame.type()));
        crossingLine = new CrossingLine(newFrame.size());
    }

}
