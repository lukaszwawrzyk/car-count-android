package pl.edu.agh.sr.cvtest.counting;

import android.support.annotation.NonNull;
import android.util.Log;
import org.opencv.core.*;

import java.util.ArrayList;
import java.util.List;

import org.opencv.imgproc.Imgproc;
import org.opencv.core.Core;

public class BlobDetector {

    private Mat storedPrevFrame;
    private Mat storedCurrentFrame;

    private Mat transformedFrame;

    private Scalar COLOR_BLACK = new Scalar(0, 0, 0);
    private Scalar COLOR_WHITE = new Scalar(255, 255, 255);
    private Scalar COLOR_RED = new Scalar(0, 0, 255);
    private Scalar COLOR_BLUE = new Scalar(255, 0, 0);
    private Scalar COLOR_GREEN = new Scalar(0, 255, 0);

    private boolean isFirstFrame = true;

    private List<Blob> blobs = new ArrayList<>();

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
        List<Blob> currentFrameBlobs = getBlobs();

        if (isFirstFrame) {
            blobs.addAll(currentFrameBlobs);
            isFirstFrame = false;
        } else {
            matchBlobs(blobs, currentFrameBlobs);
        }

        Mat output = storedCurrentFrame.clone();
        drawBlobRects(output, blobs);
        return output;
    }

    private void matchBlobs(List<Blob> existingBlobs, List<Blob> currentFrameBlobs) {
        for (Blob existingBlob : existingBlobs) {
            existingBlob.matchFoundOrIsNew = false;
            existingBlob.predictNextPosition();
        }
        for (Blob currentFrameBlob : currentFrameBlobs) {
            int leastDistanceIndex = 0;
            double leastDistance = 100000;

            for (int i = 0; i < existingBlobs.size(); i++) {
                if (existingBlobs.get(i).isStillTracked) {
                    double distance = distanceBetweenPoints(currentFrameBlob.position(), existingBlobs.get(i).predictedPosition);
                    if (distance < leastDistance) {
                        leastDistance = distance;
                        leastDistanceIndex = i;
                    }
                }
            }

            if (leastDistance < currentFrameBlob.diagonalSize * 1.15) {
                updateExistingBlob(currentFrameBlob, existingBlobs.get(leastDistanceIndex));
            } else {
                addNew(currentFrameBlob, existingBlobs);
            }
        }

        for (Blob existingBlob : existingBlobs) {
            if (!existingBlob.matchFoundOrIsNew) {
                existingBlob.consecutiveFramesWithoutAMatch++;
            }
            if (existingBlob.consecutiveFramesWithoutAMatch >= 5) {
                existingBlob.isStillTracked = false;
            }
        }
    }

    private void updateExistingBlob(Blob currentFrameBlob, Blob closestExistingBlob) {
        closestExistingBlob.contour = currentFrameBlob.contour;
        closestExistingBlob.boundingRect = currentFrameBlob.boundingRect;
        closestExistingBlob.positionHistory.add(currentFrameBlob.position());
        closestExistingBlob.diagonalSize = currentFrameBlob.diagonalSize;
        closestExistingBlob.aspectRatio = currentFrameBlob.aspectRatio;
        closestExistingBlob.isStillTracked = true;
        closestExistingBlob.matchFoundOrIsNew = true;
    }

    private void addNew(Blob currentFrameBlob, List<Blob> existingBlobs) {
        currentFrameBlob.matchFoundOrIsNew = true;
        existingBlobs.add(currentFrameBlob);
    }

    private double distanceBetweenPoints(Point point1, Point point2) {
        double intX = Math.abs(point1.x - point2.x);
        double intY = Math.abs(point1.y - point2.y);

        return Math.sqrt(Math.pow(intX, 2) + Math.pow(intY, 2));
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
    }

    private void drawBlobRects(Mat frame, List<Blob> blobs) {
        drawBlobRects(frame, frame, blobs);
    }

    private void drawBlobRects(Mat frame, Mat dest, List<Blob> blobs) {
        if (frame != dest) {
            frame.copyTo(dest);
        }
        for (int i = 0; i < blobs.size(); i++) {
            Blob blob = blobs.get(i);
            if (blob.isStillTracked) {
                Imgproc.rectangle(dest, blob.boundingRect.tl(), blob.boundingRect.br(), COLOR_BLUE, 2);
                Imgproc.circle(dest, blob.position(), 3, COLOR_GREEN, -1);
                int fontFace = Core.FONT_HERSHEY_SIMPLEX;
                double fontScale = blob.diagonalSize / 60;
                int fontThickness = (int)Math.round(fontScale);
                Imgproc.putText(dest, String.valueOf(i), blob.position(), fontFace, fontScale, COLOR_RED, fontThickness);
            }
        }
    }

    private void drawBlobs(List<Blob> blobs, Mat out) {
        out.setTo(COLOR_BLACK);
        List<MatOfPoint> hullsOfBlobs = new ArrayList<>();
        for (Blob blob : blobs) {
            hullsOfBlobs.add(blob.contour);
        }
        Imgproc.drawContours(out, hullsOfBlobs, -1, COLOR_WHITE, -1);
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
    private List<Blob> getBlobs() {
        List<MatOfPoint> convexHulls = getConvexHullsOfContours();
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
    private List<MatOfPoint> getConvexHullsOfContours() {
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(transformedFrame, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
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

    private void dumpMat(String msg, Mat threshold) {
        StringBuilder res = new StringBuilder(msg + "\n");
        for (int col = 0; col < threshold.cols(); col++) {
            for (int row = 0; row < threshold.rows(); row++) {
                byte[] x = new byte[4];
                threshold.get(row, col, x);
                res.append(x[0] + x[1] + x[2] + x[3]);
                res.append(" ");
            }
            res.append("\n");
        }
        Log.d("AAA", res.toString());
    }

}
