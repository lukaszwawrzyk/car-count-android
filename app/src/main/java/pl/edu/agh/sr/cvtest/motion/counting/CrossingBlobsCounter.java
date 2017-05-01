package pl.edu.agh.sr.cvtest.motion.counting;

import pl.edu.agh.sr.cvtest.motion.blob.Blob;
import pl.edu.agh.sr.cvtest.motion.crossing.CrossingLine;

import java.util.List;

public class CrossingBlobsCounter {
    private final CrossingLine crossingLine;
    private int counter;

    public CrossingBlobsCounter(CrossingLine crossingLine) {
        this.crossingLine = crossingLine;
        this.counter = 0;
    }

    public int value() { return counter; }

    public boolean count(List<Blob> blobs) {
        boolean crossed = false;
        for (Blob blob : blobs) {
            if (blob.crossed(crossingLine)) {
                counter++;
                crossed = true;
            }
        }
        return crossed;
    }

    public void reset() {
        counter = 0;
    }
}
