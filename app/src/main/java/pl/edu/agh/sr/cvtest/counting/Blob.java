package pl.edu.agh.sr.cvtest.counting;

import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

final class Blob {
    int id;
    MatOfPoint contour;
    Rect boundingRect;
    double diagonalSize;
    double aspectRatio;
    List<Point> positionHistory;
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

    Point currPosition() { return positionHistory.get(positionHistory.size() - 1); }
    Point prevPosition() { return positionHistory.get(positionHistory.size() - 2); }

    boolean isValid() {
        return boundingRect.area() > 400 &&
               aspectRatio >= 0.2 &&
               aspectRatio <= 4.0 &&
               boundingRect.width > 30 &&
               boundingRect.height > 30 &&
               diagonalSize > 60 &&
               (Imgproc.contourArea(contour) / boundingRect.area()) > 0.50;
    }

    void updatePredictedPosition() {
        int numPositions = positionHistory.size();
        Point lastPosition = positionHistory.get(numPositions - 1);

        if (numPositions == 1) {
            predictedPosition.x = lastPosition.x;
            predictedPosition.y = lastPosition.y;
        } else if (numPositions == 2) {
            double deltaX = positionHistory.get(1).x - positionHistory.get(0).x;
            double deltaY = positionHistory.get(1).y - positionHistory.get(0).y;
            predictedPosition.x = lastPosition.x + deltaX;
            predictedPosition.y = lastPosition.y + deltaY;
        } else if (numPositions == 3) {
            double sumOfXChanges = ((positionHistory.get(2).x - positionHistory.get(1).x) * 2) +
                    ((positionHistory.get(1).x - positionHistory.get(0).x) * 1);
            int deltaX = (int)Math.round(sumOfXChanges / 3.0);

            double sumOfYChanges = ((positionHistory.get(2).y - positionHistory.get(1).y) * 2) +
                    ((positionHistory.get(1).y - positionHistory.get(0).y) * 1);
            int deltaY = (int)Math.round(sumOfYChanges / 3.0);

            predictedPosition.x = lastPosition.x + deltaX;
            predictedPosition.y = lastPosition.y + deltaY;
        } else if (numPositions == 4) {
            double sumOfXChanges = ((positionHistory.get(3).x - positionHistory.get(2).x) * 3) +
                    ((positionHistory.get(2).x - positionHistory.get(1).x) * 2) +
                    ((positionHistory.get(1).x - positionHistory.get(0).x) * 1);
            int deltaX = (int)Math.round(sumOfXChanges / 6.0);

            double sumOfYChanges = ((positionHistory.get(3).y - positionHistory.get(2).y) * 3) +
                    ((positionHistory.get(2).y - positionHistory.get(1).y) * 2) +
                    ((positionHistory.get(1).y - positionHistory.get(0).y) * 1);
            int deltaY = (int)Math.round(sumOfYChanges / 6.0);

            predictedPosition.x = lastPosition.x + deltaX;
            predictedPosition.y = lastPosition.y + deltaY;
        } else if (numPositions >= 5) {
            double sumOfXChanges = ((positionHistory.get(numPositions - 1).x - positionHistory.get(numPositions - 2).x) * 4) +
                    ((positionHistory.get(numPositions - 2).x - positionHistory.get(numPositions - 3).x) * 3) +
                    ((positionHistory.get(numPositions - 3).x - positionHistory.get(numPositions - 4).x) * 2) +
                    ((positionHistory.get(numPositions - 4).x - positionHistory.get(numPositions - 5).x) * 1);

            int deltaX = (int)Math.round(sumOfXChanges / 10.0);

            double sumOfYChanges = ((positionHistory.get(numPositions - 1).y - positionHistory.get(numPositions - 2).y) * 4) +
                    ((positionHistory.get(numPositions - 2).y - positionHistory.get(numPositions - 3).y) * 3) +
                    ((positionHistory.get(numPositions - 3).y - positionHistory.get(numPositions - 4).y) * 2) +
                    ((positionHistory.get(numPositions - 4).y - positionHistory.get(numPositions - 5).y) * 1);

            int deltaY = (int)Math.round(sumOfYChanges / 10.0);

            predictedPosition.x = lastPosition.x + deltaX;
            predictedPosition.y = lastPosition.y + deltaY;
        } else {
            // should never get here
        }
    }

    void updateFrom(Blob currentFrameBlob) {
        contour = currentFrameBlob.contour;
        boundingRect = currentFrameBlob.boundingRect;
        diagonalSize = currentFrameBlob.diagonalSize;
        aspectRatio = currentFrameBlob.aspectRatio;
        positionHistory.add(currentFrameBlob.currPosition());
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

    boolean isHorizontalLineCrossedFromBottom(int crossingLinePosition) {
        return positionHistory.size() >= 2 &&
                prevPosition().y > crossingLinePosition &&
                currPosition().y <= crossingLinePosition;
    }
}
