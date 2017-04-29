package pl.edu.agh.sr.cvtest.counting;

import android.util.Log;
import org.opencv.core.*;

import java.util.ArrayList;
import java.util.List;

import static org.opencv.imgproc.Imgproc.*;
import static org.opencv.core.Core.*;

public class BlobDetector {

    private Mat storedPrevFrame;
    private Mat storedCurrentFrame;

    private Mat threshold;

    private Scalar SCALAR_BLACK = new Scalar(0.0, 0.0, 0.0);
    private Scalar SCALAR_WHITE = new Scalar(255.0, 255.0, 255.0);
    private Scalar SCALAR_RED = new Scalar(0, 0, 255.0);
    private Scalar SCALAR_GREEN = new Scalar(0, 200, 0);

    public Mat getMovingObjects(Mat newFrame) {
        if (storedPrevFrame == null && storedCurrentFrame == null) {
            storedCurrentFrame = newFrame.clone();
            threshold = new Mat(newFrame.size(), newFrame.type());
            return newFrame;
        }
        if (storedPrevFrame != null) storedPrevFrame.release();
        storedPrevFrame = storedCurrentFrame;
        storedCurrentFrame = newFrame.clone();

        Mat frame1 = storedPrevFrame;
        Mat frame2 = storedCurrentFrame.clone();
        cvtColor(frame1, frame1, COLOR_BGR2GRAY);
        cvtColor(frame2, frame2, COLOR_BGR2GRAY);
        GaussianBlur(frame1, frame1, new Size(5, 5), 0);
        GaussianBlur(frame2, frame2, new Size(5, 5), 0);
        absdiff(frame1, frame2, threshold);
        threshold(threshold, threshold, 30, 255, THRESH_BINARY);

        Mat structuringElement5x5 = getStructuringElement(MORPH_RECT, new Size(5, 5));
        dilate(threshold, threshold, structuringElement5x5);
        dilate(threshold, threshold, structuringElement5x5);
        erode(threshold, threshold, structuringElement5x5);

        List<MatOfPoint> contours = new ArrayList<>();
        findContours(threshold, contours, new Mat(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

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

        Mat currentFrameCopy = storedCurrentFrame.clone();

        for (Blob blob : blobs) {
            rectangle(currentFrameCopy, blob.boundingRect.tl(), blob.boundingRect.br(), SCALAR_RED, 2);
            circle(currentFrameCopy, blob.centerPosition, 3, SCALAR_GREEN, -1);
        }

        return currentFrameCopy;
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


}
