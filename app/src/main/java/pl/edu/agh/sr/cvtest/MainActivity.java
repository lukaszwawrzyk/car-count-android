package pl.edu.agh.sr.cvtest;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.WindowManager;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import pl.edu.agh.sr.cvtest.motion.BlobCountingDisplay;

public class MainActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "MAIN";

    // 176x144 640x480 1024x768 1280x720
    private static final int PREVIEW_WIDTH = 640;
    private static final int PREVIEW_HEIGHT = 480;

    private CameraBridgeViewBase cameraView;
    private BlobCountingDisplay loop;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        cameraView = (CameraBridgeViewBase) findViewById(R.id.HelloOpenCvView);
        cameraView.setMaxFrameSize(PREVIEW_WIDTH, PREVIEW_HEIGHT);
        cameraView.setVisibility(SurfaceView.VISIBLE);
        cameraView.setCvCameraViewListener(this);
    }

    private BaseLoaderCallback openCvLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    Log.i(TAG, "OpenCV loaded successfully");
                    loop = new BlobCountingDisplay();
                    cameraView.enableView();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (event.getPointerCount() == 2) {
            MotionEvent.PointerCoords coords = new MotionEvent.PointerCoords();
            event.getPointerCoords(0, coords);
            Point start = mapToCameraViewCoordinates(coords);
            event.getPointerCoords(1, coords);
            Point end = mapToCameraViewCoordinates(coords);
            loop.updateLinePosition(start, end);
        }
        return true;
    }

    private Point mapToCameraViewCoordinates(MotionEvent.PointerCoords coords) {
        double xFactor = (double) PREVIEW_WIDTH / cameraView.getWidth();
        double yFactor = (double) PREVIEW_HEIGHT / cameraView.getHeight();
        return new Point(coords.x * xFactor, coords.y * yFactor);
    }


    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, openCvLoaderCallback);
    }

    @Override
    public void onPause() {
        super.onPause();
        disableCameraView();
    }

    public void onDestroy() {
        super.onDestroy();
        disableCameraView();
    }

    private void disableCameraView() {
        if (cameraView != null) {
            cameraView.disableView();
        }
    }

    @Override public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        return loop.getFrame(inputFrame.rgba());
    }

    @Override public void onCameraViewStarted(int width, int height) { }
    @Override public void onCameraViewStopped() { }
}
