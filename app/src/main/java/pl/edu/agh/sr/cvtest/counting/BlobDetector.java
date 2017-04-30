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

    private Scalar SCALAR_BLACK = new Scalar(0, 0, 0);
    private Scalar SCALAR_WHITE = new Scalar(255, 255, 255);
    private Scalar SCALAR_RED = new Scalar(0, 0, 255);
    private Scalar SCALAR_BLUE = new Scalar(255, 0, 0);
    private Scalar SCALAR_GREEN = new Scalar(0, 255, 0);

    public Mat getMovingObjects(Mat newFrame) {
        if (notInitialized()) {
            initFromFirstFrame(newFrame);
            return newFrame;
        }
        shiftFrames(newFrame);

        FourWayDisplay display = new FourWayDisplay(transformedFrame);
        Mat prevFrame = storedPrevFrame;
        Mat currFrame = storedCurrentFrame.clone();
        bwBlur(prevFrame);
        bwBlur(currFrame);
        display.put1(currFrame);
        diffWithThreshold(prevFrame, currFrame, transformedFrame);
        dilate(transformedFrame);
        display.put2(transformedFrame);
        List<Blob> blobs = getBlobs();
        drawBlobs(blobs, transformedFrame);
        display.put3(transformedFrame);
        drawBlobRects(currFrame, transformedFrame, blobs);
        display.put4(transformedFrame);
        return display.getOutputImg();
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

    private void drawBlobRects(Mat currFrame, Mat dest, List<Blob> blobs) {
        currFrame.copyTo(dest);
        for (Blob blob : blobs) {
            Imgproc.rectangle(dest, blob.boundingRect.tl(), blob.boundingRect.br(), SCALAR_BLUE, 2);
            Imgproc.circle(dest, blob.centerPosition, 3, SCALAR_GREEN, -1);
        }
    }

    private void drawBlobs(List<Blob> blobs, Mat out) {
        out.setTo(SCALAR_BLACK);
        List<MatOfPoint> hullsOfBlobs = new ArrayList<>();
        for (Blob blob : blobs) {
            hullsOfBlobs.add(blob.contour);
        }
        Imgproc.drawContours(out, hullsOfBlobs, -1, SCALAR_WHITE, -1);
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
            if (isValid(possibleBlob)) {
                blobs.add(possibleBlob);
            }
        }
        return blobs;
    }

    private boolean isValid(Blob blob) {
        return blob.boundingRect.area() > 100 &&
                blob.aspectRatio >= 0.2 &&
                blob.aspectRatio <= 1.2 &&
                blob.boundingRect.width > 15 &&
                blob.boundingRect.height > 20 &&
                blob.diagonalSize > 30;
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
