package pl.edu.agh.sr.cvtest.util;

import android.util.Log;

public class Stopwatch {

    private long lastMeasuredTime;

    public static Stopwatch start() {
        return new Stopwatch();
    }

    private Stopwatch() {
        lastMeasuredTime = System.nanoTime();
    }

    public void register(String phaseName) {
        long currentMeasuredTime = System.nanoTime();
        long differenceNs = currentMeasuredTime - lastMeasuredTime;
        double differenceMs = differenceNs / 1_000_000;
        Log.d("STOPWATCH", "Phase: " + phaseName + " took " + differenceMs + " ms");
        lastMeasuredTime = currentMeasuredTime;
    }

}
