package pl.edu.agh.sr.cvtest.counting;

import android.util.Log;
import org.opencv.core.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.opencv.imgproc.Imgproc.*;
import static org.opencv.core.Core.*;

public class BlobDetector {

    private Mat frame1;
    private Mat frame2;

    private Mat difference;
    private Mat threshold;

    private Scalar SCALAR_BLACK = new Scalar(0.0, 0.0, 0.0);
    private Scalar SCALAR_WHITE = new Scalar(255.0, 255.0, 255.0);
    private Scalar SCALAR_RED = new Scalar(0, 0, 255.0);
    private Scalar SCALAR_GREEN = new Scalar(0, 200, 0);

    public Mat getMovingObjects(Mat newFrame) {
        if (frame1 == null && frame2 == null) {
            frame2 = newFrame.clone();
            difference = new Mat(newFrame.size(), newFrame.type());
            threshold = new Mat(newFrame.size(), newFrame.type());
            return newFrame;
        }
        if (frame1 != null) frame1.release();
        frame1 = frame2;
        frame2 = newFrame.clone();

        Mat frame1Copy = frame1.clone();
        Mat frame2Copy = frame2.clone();
        cvtColor(frame1Copy, frame1Copy, COLOR_BGR2GRAY);
        cvtColor(frame2Copy, frame2Copy, COLOR_BGR2GRAY);
        GaussianBlur(frame1Copy, frame1Copy, new Size(5, 5), 0);
        GaussianBlur(frame2Copy, frame2Copy, new Size(5, 5), 0);
        absdiff(frame1Copy, frame2Copy, difference);
        threshold(difference, threshold, 30, 255, THRESH_BINARY);

        Mat structuringElement3x3 = getStructuringElement(MORPH_RECT, new Size(3, 3));
        Mat structuringElement5x5 = getStructuringElement(MORPH_RECT, new Size(5, 5));
        Mat structuringElement7x7 = getStructuringElement(MORPH_RECT, new Size(7, 7));
        Mat structuringElement9x9 = getStructuringElement(MORPH_RECT, new Size(9, 9));
        dilate(threshold, threshold, structuringElement5x5);
        dilate(threshold, threshold, structuringElement5x5);
        erode(threshold, threshold, structuringElement5x5);

        Mat thresholdCopy = threshold.clone();
        List<MatOfPoint> contours = new ArrayList<>();
        findContours(thresholdCopy, contours, new Mat(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        Mat imgContours = new Mat(threshold.size(), CvType.CV_8UC3, SCALAR_BLACK);
        drawContours(imgContours, contours, -1, SCALAR_WHITE, -1);

        List<MatOfPoint> convexHulls = new ArrayList<>(contours.size());

        MatOfInt hull = new MatOfInt();
        for (int i = 0; i < contours.size(); i++) {
            convexHull(contours.get(i), hull);
            MatOfPoint hullContour = hull2Points(hull, contours.get(i));
            convexHulls.add(hullContour);
        }

        List<Blob> blobs = new ArrayList<>();
        for (MatOfPoint convexHull : convexHulls) {
            Blob possibleBlob = new Blob(convexHull);

            if (possibleBlob.boundingRect.area() > 100 &&
                    possibleBlob.aspectRatio >= 0.2 &&
                    possibleBlob.aspectRatio <= 1.2 &&
                    possibleBlob.boundingRect.width > 15 &&
                    possibleBlob.boundingRect.height > 20 &&
                    possibleBlob.diagonalSize > 30) {
                blobs.add(possibleBlob);
            }
        }

        Mat imgConvexHulls = new Mat(threshold.size(), CvType.CV_8UC3, SCALAR_BLACK);
        convexHulls.clear();
        for (Blob blob : blobs) {
            convexHulls.add(blob.contour);
        }

        drawContours(imgConvexHulls, convexHulls, -1, SCALAR_WHITE, -1);

        Mat frame2CopyAgain = frame2Copy.clone();

        for (Blob blob : blobs) {
            rectangle(frame2CopyAgain, blob.boundingRect.tl(), blob.boundingRect.br(), SCALAR_RED, 2);
            circle(frame2CopyAgain, blob.centerPosition, 3, SCALAR_GREEN, -1);
        }

        return frame2CopyAgain;
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

    private void dump(String msg, Mat threshold) {
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

    /*List<Blob> getBlobs(Mat newFrame) {
        if (frame1 == null && frame2 == null) {
            frame2 = newFrame;
            return Collections.emptyList();
        }
        frame1 = frame2;
        frame2 = newFrame;

        List<Blob> blobs = new ArrayList<>();
        Mat frame1Copy = frame1.clone();
        Mat frame2Copy = frame2.clone();
        cvtColor(frame1Copy, frame1Copy, COLOR_BGR2GRAY);
        cvtColor(frame2Copy, frame2Copy, COLOR_BGR2GRAY);
        GaussianBlur(frame1Copy, frame1Copy, new Size(5, 5), 0);
        GaussianBlur(frame2Copy, frame2Copy, new Size(5, 5), 0);
        absdiff(frame1Copy, frame2Copy, difference);
        threshold(difference, threshold, 30, 255, THRESH_BINARY);
    }*/

}
