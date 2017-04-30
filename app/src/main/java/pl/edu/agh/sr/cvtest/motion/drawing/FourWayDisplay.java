package pl.edu.agh.sr.cvtest.motion.drawing;

import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class FourWayDisplay {
    private Size halfSize;
    private Mat tmpMat;
    private Mat outputImg;
    private Rect tl;
    private Rect tr;
    private Rect bl;
    private Rect br;

    public FourWayDisplay(Mat templateFrame) {
        outputImg = new Mat(templateFrame.size(), templateFrame.type());
        halfSize = new Size(outputImg.size().width / 2, outputImg.size().height / 2);
        tmpMat = new Mat(halfSize, templateFrame.type());
        tl = new Rect(0, 0, (int) halfSize.width, (int) halfSize.height);
        tr = new Rect((int) halfSize.width, 0, (int) halfSize.width, (int) halfSize.height);
        bl = new Rect(0, (int) halfSize.height, (int) halfSize.width, (int) halfSize.height);
        br = new Rect((int) halfSize.width, (int) halfSize.height, (int) halfSize.width, (int) halfSize.height);
    }

    public Mat getOutputImg() { return outputImg; }

    void put1(Mat mat) { put(mat, tl); }
    void put2(Mat mat) { put(mat, tr); }
    void put3(Mat mat) { put(mat, bl); }
    void put4(Mat mat) { put(mat, br); }

    private void put(Mat mat, Rect rect) {
        Imgproc.resize(mat, tmpMat, halfSize);
        tmpMat.copyTo(outputImg.submat(rect));
    }

}