package pl.edu.agh.sr.cvtest.motion.crossing;

import org.opencv.core.Point;
import org.opencv.core.Size;

public class CrossingLine {
    private Point[] startAndEnd;

    public CrossingLine(Size frameSize) {
        int position = (int)Math.round(frameSize.height * 0.35);

        startAndEnd = new Point[2];
        startAndEnd[0] = new Point();
        startAndEnd[1] = new Point();

        startAndEnd[0].x = 0;
        startAndEnd[1].x = frameSize.width - 1;

        startAndEnd[0].y = position;
        startAndEnd[1].y = position;
    }

    public boolean crossed(Point previous, Point current) {
        return LineSegmentIntersection.intersect(startAndEnd[0], startAndEnd[1], previous, current);
    }

    public Point[] points() {
        return startAndEnd;
    }

    public void updateEnds(Point start, Point end) {
        startAndEnd[0] = start;
        startAndEnd[1] = end;
    }
}
