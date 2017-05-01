package pl.edu.agh.sr.cvtest.motion.crossing;

import org.opencv.core.Point;

class LineSegmentIntersection {

    private static final double THRESHOLD = 0.001;

    static boolean intersect(Point a, Point b, Point c, Point d) {
        return get_line_intersection(a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y);
    }

    // http://stackoverflow.com/a/14795484/5123895
    private static boolean get_line_intersection(double p0_x, double p0_y, double p1_x, double p1_y,
                                         double p2_x, double p2_y, double p3_x, double p3_y) {
        double s02_x, s02_y, s10_x, s10_y, s32_x, s32_y, s_numer, t_numer, denom;
        s10_x = p1_x - p0_x;
        s10_y = p1_y - p0_y;
        s32_x = p3_x - p2_x;
        s32_y = p3_y - p2_y;

        denom = s10_x * s32_y - s32_x * s10_y;
        if (eq(denom, 0))
            return false; // Collinear
        boolean denomPositive = denom > 0;

        s02_x = p0_x - p2_x;
        s02_y = p0_y - p2_y;
        s_numer = s10_x * s02_y - s10_y * s02_x;
        if ((s_numer < 0) == denomPositive)
            return false; // No collision

        t_numer = s32_x * s02_y - s32_y * s02_x;
        if ((t_numer < 0) == denomPositive)
            return false; // No collision

        if (((s_numer > denom) == denomPositive) || ((t_numer > denom) == denomPositive))
            return false; // No collision

        // Collision detected
        return true;
    }

    private static boolean eq(double a, double b) {
        return Math.abs(a - b) < THRESHOLD;
    }
}
