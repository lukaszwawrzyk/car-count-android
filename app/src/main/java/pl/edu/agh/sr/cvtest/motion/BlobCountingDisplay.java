package pl.edu.agh.sr.cvtest.motion;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import pl.edu.agh.sr.cvtest.motion.blob.Blob;
import pl.edu.agh.sr.cvtest.motion.blob.BlobDetector;
import pl.edu.agh.sr.cvtest.motion.blob.BlobTracker;
import pl.edu.agh.sr.cvtest.motion.counting.CrossingBlobsCounter;
import pl.edu.agh.sr.cvtest.motion.crossing.CrossingLine;
import pl.edu.agh.sr.cvtest.motion.drawing.Drawing;

import java.util.List;

public class BlobCountingDisplay {

    private Mat storedPreviousFrame;
    private Mat storedCurrentFrame;

    private Drawing draw;
    private CrossingLine crossingLine;
    private BlobDetector blobDetector;
    private BlobTracker blobTracker;
    private CrossingBlobsCounter blobCounter;

    public void updateLinePosition(Point start, Point end) {
        crossingLine.updateEnds(start, end);
    }

    public void resetCounter() {
        blobCounter.reset();
    }

    public Mat getFrame(Mat newFrame) {
        if (notInitialized()) {
            initFromFirstFrame(newFrame);
            return newFrame;
        }
        shiftFrames(newFrame);

        Mat prevFrame = storedPreviousFrame;
        Mat currFrame = storedCurrentFrame.clone();
        return processFrame(prevFrame, currFrame);
    }

    private boolean notInitialized() {
        return storedPreviousFrame == null && storedCurrentFrame == null;
    }

    private void initFromFirstFrame(Mat newFrame) {
        storedCurrentFrame = newFrame.clone();
        blobDetector = new BlobDetector(new Mat(newFrame.size(), newFrame.type()));
        crossingLine = new CrossingLine(newFrame.size());
        blobTracker = new BlobTracker();
        blobCounter = new CrossingBlobsCounter(crossingLine);
        draw = new Drawing();
    }

    private void shiftFrames(Mat newFrame) {
        if (storedPreviousFrame != null) storedPreviousFrame.release();
        storedPreviousFrame = storedCurrentFrame;
        storedCurrentFrame = newFrame.clone();
    }

    private Mat processFrame(Mat prevFrame, Mat currFrame) {
        List<Blob> currentBlobs = blobDetector.detect(prevFrame, currFrame);
        blobTracker.update(currentBlobs);
        boolean isLineCrossed = blobCounter.count(blobTracker.blobs());
        return drawFrame(isLineCrossed);
    }

    private Mat drawFrame(boolean isLineCrossed) {
        Mat output = storedCurrentFrame.clone();
        draw.finalFrame(output, blobTracker.blobs(), crossingLine, isLineCrossed, blobCounter.value());
        return output;
    }

}
