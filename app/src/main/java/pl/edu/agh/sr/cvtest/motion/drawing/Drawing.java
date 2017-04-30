package pl.edu.agh.sr.cvtest.motion.drawing;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import pl.edu.agh.sr.cvtest.motion.blob.Blob;
import pl.edu.agh.sr.cvtest.motion.crossing.CrossingLine;

import java.util.ArrayList;
import java.util.List;

public class Drawing {
    private static final int fontFace = Core.FONT_HERSHEY_SIMPLEX;

    public void blobs(List<Blob> blobs, Mat out) {
        out.setTo(Colors.BLACK);
        List<MatOfPoint> hullsOfBlobs = new ArrayList<>();
        for (Blob blob : blobs) {
            hullsOfBlobs.add(blob.contour);
        }
        Imgproc.drawContours(out, hullsOfBlobs, -1, Colors.WHITE, -1);
    }

    public void finalFrame(Mat output, List<Blob> blobs, CrossingLine crossingLine, boolean lineCrossed, int carCount) {
        drawBlobBoundingRects(output, blobs);
        drawCrossingLine(output, crossingLine, lineCrossed);
        drawCrossingBlobsCount(output, carCount);
    }

    private void drawBlobBoundingRects(Mat dest, List<Blob> blobs) {
        for (Blob blob : blobs) {
            drawBoundingRect(dest, blob);
            drawPositionTrace(dest, blob);
            drawCurrentPosition(dest, blob);
            drawBlobId(dest, blob);
        }
    }

    private void drawBoundingRect(Mat dest, Blob blob) {
        Imgproc.rectangle(dest, blob.boundingRect.tl(), blob.boundingRect.br(), Colors.ORANGE, 2);
    }

    private void drawPositionTrace(Mat dest, Blob blob) {
        for (Point historicalPos : blob.positionHistory) {
            drawPoint(dest, historicalPos, Colors.BLUE);
        }
    }

    private void drawCurrentPosition(Mat dest, Blob blob) {
        drawPoint(dest, blob.position(), Colors.GREEN);
    }

    private void drawPoint(Mat dest, Point center, Scalar color) {
        Imgproc.circle(dest, center, 3, color, -1);
    }

    private void drawBlobId(Mat dest, Blob blob) {
        double fontScale = blob.diagonalSize / 100;
        int fontThickness = (int)Math.round(fontScale);
        Imgproc.putText(dest, String.valueOf(blob.id), blob.position(), fontFace, fontScale, Colors.GREEN, fontThickness);
    }

    private void drawCrossingLine(Mat output, CrossingLine crossingLine, boolean lineCrossed) {
        if (lineCrossed) {
            drawLine(output, crossingLine.points(), Colors.GREEN);
        } else {
            drawLine(output, crossingLine.points(), Colors.RED);
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
