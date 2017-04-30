package pl.edu.agh.sr.cvtest.counting;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

class Drawing {
    private static final int fontFace = Core.FONT_HERSHEY_SIMPLEX;

    void blobs(List<Blob> blobs, Mat out) {
        out.setTo(Colors.BLACK);
        List<MatOfPoint> hullsOfBlobs = new ArrayList<>();
        for (Blob blob : blobs) {
            hullsOfBlobs.add(blob.contour);
        }
        Imgproc.drawContours(out, hullsOfBlobs, -1, Colors.WHITE, -1);
    }

    void finalFrame(Mat output, List<Blob> blobs, Point[] crossingLine, boolean lineCrossed, int carCount) {
        drawBlobBoundingRects(output, blobs);
        drawCrossingLine(output, crossingLine, lineCrossed);
        drawCrossingBlobsCount(output, carCount);
    }

    private void drawBlobBoundingRects(Mat dest, List<Blob> blobs) {
        for (Blob blob : blobs) {
            Imgproc.rectangle(dest, blob.boundingRect.tl(), blob.boundingRect.br(), Colors.ORANGE, 2);
            Imgproc.circle(dest, blob.currPosition(), 3, Colors.GREEN, -1);
            double fontScale = blob.diagonalSize / 100;
            int fontThickness = (int)Math.round(fontScale);
            Imgproc.putText(dest, String.valueOf(blob.id), blob.currPosition(), fontFace, fontScale, Colors.GREEN, fontThickness);
        }
    }

    private void drawCrossingLine(Mat output, Point[] crossingLine, boolean lineCrossed) {
        if (lineCrossed) {
            drawLine(output, crossingLine, Colors.GREEN);
        } else {
            drawLine(output, crossingLine, Colors.RED);
        }
    }

    private void drawLine(Mat output, Point[] line, Scalar color) {
        Imgproc.line(output, line[0], line[1], color, 2);
    }

    private void drawCrossingBlobsCount(Mat output, int carCount) {
        double fontScale = (output.rows() * output.cols()) / 300000.0;
        int fontThickness = (int) Math.round(fontScale * 1.5);
        Size textSize = Imgproc.getTextSize(String.valueOf(carCount), fontFace, fontScale, fontThickness, null);
        Point blPosition = new Point();
        blPosition.x = output.cols() - 1 - (int)(textSize.width * 1.25);
        blPosition.y = (int)(textSize.height * 1.25);
        Imgproc.putText(output, String.valueOf(carCount), blPosition, fontFace, fontScale, Colors.GREEN, fontThickness);
    }



}
