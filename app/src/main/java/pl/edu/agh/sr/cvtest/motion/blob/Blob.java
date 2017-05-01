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
    public boolean matchFoundOrIsNew;
    private int consecutiveFramesWithoutAMatch;
    public Point predictedPosition;

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
        return boundingRect.area() > 400 &&
               aspectRatio >= 0.2 &&
               aspectRatio <= 4.0 &&
               boundingRect.width > 30 &&
               boundingRect.height > 30 &&
               diagonalSize > 60 &&
               (Imgproc.contourArea(contour) / boundingRect.area()) > 0.50;
    }

    void updatePredictedPosition() {
        int lastIndex = positionHistory.size() - 1;
        Point lastPosition = positionHistory.get(lastIndex);
        int numbersOfDiffsToCalculate = Math.min(lastIndex, 4);

        int diffWeight = numbersOfDiffsToCalculate;
        int totalDiffWeight = 0;
        double totalXChangesWeighted = 0;
        double totalYChangesWeighted = 0;
        for (int i = 0; i < numbersOfDiffsToCalculate; i++) {
            totalXChangesWeighted += (positionHistory.get(lastIndex - i).x - positionHistory.get(lastIndex - i - 1).x) * diffWeight;
            totalYChangesWeighted += (positionHistory.get(lastIndex - i).y - positionHistory.get(lastIndex - i - 1).y) * diffWeight;
            totalDiffWeight += diffWeight;
            diffWeight -= 1;
        }
        int deltaX = totalDiffWeight == 0 ? 0 : (int)Math.round(totalXChangesWeighted / totalDiffWeight);
        int deltaY = totalDiffWeight == 0 ? 0 : (int)Math.round(totalYChangesWeighted / totalDiffWeight);

        predictedPosition.x = lastPosition.x + deltaX;
        predictedPosition.y = lastPosition.y + deltaY;
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
