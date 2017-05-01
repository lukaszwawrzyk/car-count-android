package pl.edu.agh.sr.cvtest.motion.blob;

import org.opencv.core.Point;

import java.util.List;

class PositionPrediction {

    private static final int MAX_POSITIONS_TO_LOOK_BACK = 4;

    static Point predictNext(List<Point> positionHistory) {
        int lastIndex = positionHistory.size() - 1;
        int numbersOfDiffsToCalculate = Math.min(lastIndex, MAX_POSITIONS_TO_LOOK_BACK);

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

        Point lastPosition = positionHistory.get(lastIndex);
        return new Point(lastPosition.x + deltaX, lastPosition.y + deltaY);
    }

}
