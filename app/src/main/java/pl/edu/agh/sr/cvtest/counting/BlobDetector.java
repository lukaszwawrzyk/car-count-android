package pl.edu.agh.sr.cvtest.counting;

import android.util.Log;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;

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

    public Mat getThreshold(Mat newFrame) {
        if (frame1 == null && frame2 == null) {
            frame2 = newFrame.clone();
            difference = new Mat(newFrame.rows(), newFrame.cols(), newFrame.type());
            threshold = new Mat(newFrame.rows(), newFrame.cols(), newFrame.type());
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
        return threshold.clone();
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
