package pl.edu.agh.sr.cvtest.motion.blob;

import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;
import pl.edu.agh.sr.cvtest.motion.crossing.CrossingLine;

import java.util.ArrayList;
import java.util.List;

public final class Blob {
    public int id;
    public MatOfPoint contour;
    public Rect boundingRect;
    public double diagonalSize;
    private double aspectRatio;
    public List<Point> positionHistory;
    boolean matchFoundOrIsNew;
    private int consecutiveFramesWithoutAMatch;
    Point predictedPosition;

    Blob(MatOfPoint contour) {
        positionHistory = new ArrayList<>();
        predictedPosition = new Point();

        this.contour = contour;
        boundingRect = Imgproc.boundingRect(contour);

        double centerX = boundingRect.x + boundingRect.width / 2;
        double centerY = boundingRect.y + boundingRect.height / 2;
        Point position = new Point(centerX, centerY);
        positionHistory.add(position);

        diagonalSize = Math.sqrt(Math.pow(boundingRect.width, 2) + Math.pow(boundingRect.height, 2));
        aspectRatio = (float)boundingRect.width / (float)boundingRect.height;

        matchFoundOrIsNew = true;
        consecutiveFramesWithoutAMatch = 0;
    }

    public Point position() { return positionHistory.get(positionHistory.size() - 1); }

    private Point prevPosition() { return positionHistory.get(positionHistory.size() - 2); }

    boolean hasExpectedSizeAndShape() {
        return aspectRatio >= 0.4 &&
               aspectRatio <= 3.0 &&
               boundingRect.width > 35 &&
               boundingRect.height > 35 &&
               (Imgproc.contourArea(contour) / boundingRect.area()) > 0.50;
    }

    void updatePredictedPosition() {
        predictedPosition = PositionPrediction.predictNext(positionHistory);
    }

    void updateFrom(Blob currentFrameBlob) {
        contour = currentFrameBlob.contour;
        boundingRect = currentFrameBlob.boundingRect;
        diagonalSize = currentFrameBlob.diagonalSize;
        aspectRatio = currentFrameBlob.aspectRatio;
        positionHistory.add(currentFrameBlob.position());
        matchFoundOrIsNew = currentFrameBlob.matchFoundOrIsNew;
    }

    boolean disappeared() {
        if (!matchFoundOrIsNew) {
            consecutiveFramesWithoutAMatch++;
        }
        return consecutiveFramesWithoutAMatch >= 5;
    }

    boolean isCloseEnough(double distance) {
        return distance < diagonalSize * 1.15;
    }

    public boolean crossed(CrossingLine crossingLine) {
        return positionHistory.size() >= 2 && crossingLine.crossed(prevPosition(), position());
    }
}
