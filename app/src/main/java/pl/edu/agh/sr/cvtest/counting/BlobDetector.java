package pl.edu.agh.sr.cvtest.counting;

import android.support.annotation.NonNull;
import org.opencv.core.*;

import java.util.ArrayList;
import java.util.List;

import org.opencv.imgproc.Imgproc;
import org.opencv.core.Core;

public class BlobDetector {

    private static final Drawing draw = new Drawing();

    private Mat storedPrevFrame;
    private Mat storedCurrentFrame;

    private Mat transformedFrame;

    private List<Blob> blobs = new ArrayList<>();
    private Point[] crossingLine;
    private int crossingLinePosition;

    private int carCount = 0;

    public Mat getMovingObjects(Mat newFrame) {
        if (notInitialized()) {
            initFromFirstFrame(newFrame);
            return newFrame;
        }
        shiftFrames(newFrame);


        Mat prevFrame = storedPrevFrame;
        Mat currFrame = storedCurrentFrame.clone();
        bwBlur(prevFrame);
        bwBlur(currFrame);
        diffWithThreshold(prevFrame, currFrame, transformedFrame);
        dilate(transformedFrame);

        FourWayDisplay display = new FourWayDisplay(transformedFrame);
        display.put1(transformedFrame);

        List<Blob> currentFrameBlobs = getBlobs(transformedFrame);
        draw.blobs(currentFrameBlobs, transformedFrame);
        display.put2(transformedFrame);

        updateBlobs(currentFrameBlobs);
        draw.blobs(blobs, transformedFrame);
        display.put3(transformedFrame);

        boolean lineCrossed = countLineCrossingBlobs();
        Mat output = currFrame; // storedCurrentFrame.clone();
        draw.finalFrame(output, blobs, crossingLine, lineCrossed, carCount);
        display.put4(output);
        return display.getOutputImg();
    }

    private boolean countLineCrossingBlobs() {
        boolean crossed = false;
        for (Blob blob : blobs) {
            if (blob.isStillTracked && blob.horizontalLineCrossedFromBottom(crossingLinePosition)) {
                carCount++;
                crossed = true;
            }
        }
        return crossed;
    }

    private void updateBlobs(List<Blob> currentFrameBlobs) {
        if (blobs.isEmpty()) {
            blobs.addAll(currentFrameBlobs);
        } else {
            matchBlobs(blobs, currentFrameBlobs);
        }
    }

    private void matchBlobs(List<Blob> existingBlobs, List<Blob> currentFrameBlobs) {
        for (Blob existingBlob : existingBlobs) {
            existingBlob.matchFoundOrIsNew = false;
            existingBlob.updatePredictedPosition();
        }
        for (Blob currentFrameBlob : currentFrameBlobs) {
            addAsNewOrUpdate(existingBlobs, currentFrameBlob);
        }
        for (Blob existingBlob : existingBlobs) {
            existingBlob.updateTrackStatus();
        }
    }

    private void addAsNewOrUpdate(List<Blob> existingBlobs, Blob newBlob) {
        Blob closestExistingBlob = existingBlobs.get(0);
        double minDistance = 10000000;

        for (Blob existingBlob : existingBlobs) {
            if (existingBlob.isStillTracked) {
                double distance = distanceBetweenPoints(newBlob.currPosition(), existingBlob.predictedPosition);
                if (distance < minDistance) {
                    minDistance = distance;
                    closestExistingBlob = existingBlob;
                }
            }
        }

        newBlob.matchFoundOrIsNew = true;
        newBlob.isStillTracked = true;
        if (newBlob.isCloseEnough(minDistance)) {
            closestExistingBlob.updateFrom(newBlob);
        } else {
            existingBlobs.add(newBlob);
        }
    }

    private double distanceBetweenPoints(Point point1, Point point2) {
        double xDiff = Math.abs(point1.x - point2.x);
        double yDiff = Math.abs(point1.y - point2.y);
        return Math.sqrt(Math.pow(xDiff, 2) + Math.pow(yDiff, 2));
    }

    private boolean notInitialized() {
        return storedPrevFrame == null && storedCurrentFrame == null;
    }

    private void shiftFrames(Mat newFrame) {
        if (storedPrevFrame != null) storedPrevFrame.release();
        storedPrevFrame = storedCurrentFrame;
        storedCurrentFrame = newFrame.clone();
    }

    private void initFromFirstFrame(Mat newFrame) {
        storedCurrentFrame = newFrame.clone();
        transformedFrame = new Mat(newFrame.size(), newFrame.type());

        crossingLinePosition = (int)Math.round(newFrame.rows() * 0.35);

        crossingLine = new Point[2];
        crossingLine[0] = new Point();
        crossingLine[1] = new Point();

        crossingLine[0].x = 0;
        crossingLine[1].x = newFrame.cols() - 1;

        crossingLine[0].y = crossingLinePosition;
        crossingLine[1].y = crossingLinePosition;
    }

    private void bwBlur(Mat prevFrame) {
        toBW(prevFrame);
        blur(prevFrame);
    }

    private void dilate(Mat frame) {
        Mat structuringElement5x5 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.dilate(frame, frame, structuringElement5x5);
    }

    @NonNull
    private List<Blob> getBlobs(Mat frame) {
        List<MatOfPoint> convexHulls = getConvexHullsOfContours(frame);
        List<Blob> blobs = new ArrayList<>();
        for (MatOfPoint convexHull : convexHulls) {
            Blob possibleBlob = new Blob(convexHull);
            if (possibleBlob.isValid()) {
                blobs.add(possibleBlob);
            }
        }
        return blobs;
    }

    @NonNull
    private List<MatOfPoint> getConvexHullsOfContours(Mat frame) {
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(frame, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        List<MatOfPoint> convexHulls = new ArrayList<>(contours.size());
        MatOfInt hull = new MatOfInt();
        for (int i = 0; i < contours.size(); i++) {
            Imgproc.convexHull(contours.get(i), hull);
            MatOfPoint hullContour = hull2Points(hull, contours.get(i));
            convexHulls.add(hullContour);
        }
        hull.release();
        return convexHulls;
    }

    private void diffWithThreshold(Mat prevFrame, Mat nextFrame, Mat output) {
        Core.absdiff(prevFrame, nextFrame, output);
        Imgproc.threshold(output, output, 30, 255, Imgproc.THRESH_BINARY);
    }

    private void blur(Mat prevFrame) {
        Imgproc.GaussianBlur(prevFrame, prevFrame, new Size(5, 5), 0);
    }

    private void toBW(Mat prevFrame) {
        Imgproc.cvtColor(prevFrame, prevFrame, Imgproc.COLOR_BGR2GRAY);
    }

    private MatOfPoint hull2Points(MatOfInt hull, MatOfPoint contour) {
        List<Integer> indexes = hull.toList();
        List<Point> points = new ArrayList<>();
        MatOfPoint point = new MatOfPoint();
        for (Integer index : indexes) {
            points.add(contour.toList().get(index));
        }
        point.fromList(points);
        return point;
    }

}
