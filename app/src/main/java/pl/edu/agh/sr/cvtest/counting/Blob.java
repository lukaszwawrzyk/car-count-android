package pl.edu.agh.sr.cvtest.counting;

import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;

final class Blob {
    MatOfPoint contour;
    Rect boundingRect;
    Point centerPosition;
    double diagonalSize;
    double aspectRatio;

    Blob(MatOfPoint contour) {
        this.contour = contour;
        boundingRect = Imgproc.boundingRect(contour);

        double centerX = boundingRect.x + boundingRect.width / 2;
        double centerY = boundingRect.y + boundingRect.height / 2;
        centerPosition = new Point(centerX, centerY);

        diagonalSize = Math.sqrt(Math.pow(boundingRect.width, 2) + Math.pow(boundingRect.height, 2));
        aspectRatio = (float)boundingRect.width / (float)boundingRect.height;
    }
}
