package pl.edu.agh.sr.cvtest.counting;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

class BlobDetector {
    private final Mat transformedFrame;

    BlobDetector(Mat transformedFrame) {
        this.transformedFrame = transformedFrame;
    }

    List<Blob> detect(Mat prevFrame, Mat currFrame) {
        bwBlur(prevFrame);
        bwBlur(currFrame);
        diffWithThreshold(prevFrame, currFrame, transformedFrame);
        dilate(transformedFrame);
        return getBlobs(transformedFrame);
    }

    private void bwBlur(Mat frame) {
        toBW(frame);
        blur(frame);
    }

    private void toBW(Mat frame) {
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2GRAY);
    }

    private void blur(Mat frame) {
        Imgproc.GaussianBlur(frame, frame, new Size(5, 5), 0);
    }

    private void diffWithThreshold(Mat prevFrame, Mat nextFrame, Mat output) {
        Core.absdiff(prevFrame, nextFrame, output);
        Imgproc.threshold(output, output, 30, 255, Imgproc.THRESH_BINARY);
    }

    private void dilate(Mat frame) {
        Mat structuringElement5x5 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.dilate(frame, frame, structuringElement5x5);
    }

    private List<Blob> getBlobs(Mat frame) {
        List<MatOfPoint> contours = getContours(frame);
        List<MatOfPoint> convexHulls = getConvexHulls(contours);
        return createBlobs(convexHulls);
    }

    private List<MatOfPoint> getContours(Mat frame) {
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(frame, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        return contours;
    }

    private List<MatOfPoint> getConvexHulls(List<MatOfPoint> contours) {
        List<MatOfPoint> convexHulls = new ArrayList<>(contours.size());
        MatOfInt hull = new MatOfInt();
        for (int i = 0; i < contours.size(); i++) {
            Imgproc.convexHull(contours.get(i), hull);
            MatOfPoint hullContour = convertHullToPoints(hull, contours.get(i));
            convexHulls.add(hullContour);
        }
        hull.release();
        return convexHulls;
    }

    private MatOfPoint convertHullToPoints(MatOfInt hull, MatOfPoint contour) {
        List<Integer> indexes = hull.toList();
        List<Point> points = new ArrayList<>();
        MatOfPoint point = new MatOfPoint();
        for (Integer index : indexes) {
            points.add(contour.toList().get(index));
        }
        point.fromList(points);
        return point;
    }

    private List<Blob> createBlobs(List<MatOfPoint> convexHulls) {
        List<Blob> blobs = new ArrayList<>();
        for (MatOfPoint convexHull : convexHulls) {
            Blob possibleBlob = new Blob(convexHull);
            if (possibleBlob.hasExpectedSizeAndShape()) {
                blobs.add(possibleBlob);
            }
        }
        return blobs;
    }

}
