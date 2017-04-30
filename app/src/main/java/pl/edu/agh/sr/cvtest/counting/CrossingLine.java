package pl.edu.agh.sr.cvtest.counting;

import org.opencv.core.Point;
import org.opencv.core.Size;

class CrossingLine {
    private Point[] startAndEnd;
    private int position;

    CrossingLine(Size frameSize) {
        position = (int)Math.round(frameSize.height * 0.35);

        startAndEnd = new Point[2];
        startAndEnd[0] = new Point();
        startAndEnd[1] = new Point();

        startAndEnd[0].x = 0;
        startAndEnd[1].x = frameSize.width - 1;

        startAndEnd[0].y = position;
        startAndEnd[1].y = position;
    }

    boolean crossed(Point previous, Point current) {
        return previous.y > position && current.y <= position;
    }

    Point[] points() {
        return startAndEnd;
    }
}
