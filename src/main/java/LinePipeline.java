
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;


class LinePipeline
{
    private static final Scalar red = new Scalar(0, 0, 255);
    private static final Scalar black = new Scalar(0, 0, 0);
    private static final Scalar white = new Scalar(255, 255, 255);

    // Dynamic setting of Threshold values;
    private static double[] hsvThresholdHue = {0.0, 255.0};
    private static double[] hsvThresholdSaturation = {0.0, 50.0};
    private static double[] hsvThresholdValue = {200.0, 255.0};

    private static double filterContoursMinArea = 100.0;
    private static double[] rectRatio = {0.1, 0.30};

    static void setThresholdHue(double min, double max)
    {
        hsvThresholdHue[0] = min;
        hsvThresholdHue[1] = max;
    }

    static double[] getThresholdHue()
    {
        return hsvThresholdHue;
    }

    static void setThresholdSaturation(double min, double max)
    {
        hsvThresholdSaturation[0] = min;
        hsvThresholdSaturation[1] = max;
    }

    static double[] getThresholdSaturation()
    {
        return hsvThresholdSaturation;
    }

    static void setThresholdValue(double min, double max)
    {
        hsvThresholdValue[0] = min;
        hsvThresholdValue[1] = max;
    }

    static double[] getThresholdValue()
    {
        return hsvThresholdValue;
    }

    static void setContoursMinArea(double min)
    {
        filterContoursMinArea = min;
    }

    static double getfilterContoursMinArea()
    {
        return filterContoursMinArea;
    }

    static void setRotatedRectRatio(double min, double max)
    {
        rectRatio[0] = min;
        rectRatio[1] = max;
    }

    static double[] getRotatedRectRatio()
    {
        return rectRatio;
    }

    // Inputs
    private Rect crop = new Rect();

    //Outputs
    private Mat hsvThresholdOutput = new Mat();
    private ArrayList<MatOfPoint> findContoursOutput = new ArrayList<>();
    private ArrayList<MatOfPoint> filterContoursOutput = new ArrayList<>();
    private ArrayList<RotatedRect> findRotatedRectsOutput = new ArrayList<>();
    private double lineAngle = Double.NaN;
    private double lineMinY = Double.NaN;

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

//    void process(Mat source0, TapeInfo ti)
//    {
//        double targetWidth = ti.getMaxX() - ti.getMinX();
//        double minX = Math.max(ti.getMinX() - targetWidth / 2, 0.0);
//        double maxX = Math.min(ti.getMaxX() + targetWidth / 2, source0.width());
//        double width = maxX - minX;
//
//        this.crop = new Rect((int) minX, source0.height() / 2, (int) width, source0.height() / 2);
//        this.ti = ti;
//
//        Mat subImage = source0.submat(crop);
//        process(subImage);
//    }

    public void process(Mat source0)
    {
        hsvThreshold(source0, hsvThresholdHue, hsvThresholdSaturation, hsvThresholdValue, hsvThresholdOutput);
        findContours(hsvThresholdOutput, false, findContoursOutput);
        findRotatedRects(findContoursOutput, findRotatedRectsOutput);
    }

    Mat hsvThresholdOutput()
    {
        return hsvThresholdOutput;
    }

    ArrayList<MatOfPoint> findContoursOutput()
    {
        return findContoursOutput;
    }

    ArrayList<MatOfPoint> filterContoursOutput()
    {
        return filterContoursOutput;
    }

    ArrayList<RotatedRect> findRotatedRectsOutput()
    {
        return findRotatedRectsOutput;
    }

    Rect getCrop()
    {
        return crop;
    }

    double getLineAngle()
    {
        return lineAngle;
    }

    double getLineMinY()
    {
        return lineMinY;
    }


    /**
     * Segment an image based on hue, saturation, and value ranges.
     *
     * @param input The image on which to perform the HSL threshold.
     * @param hue   The min and max hue
     * @param sat   The min and max saturation
     * @param val   The min and max value
     * @param out   The image in which to store the output.
     */
    void hsvThreshold(Mat input, double[] hue, double[] sat, double[] val, Mat out)
    {
        Imgproc.cvtColor(input, out, Imgproc.COLOR_BGR2HSV);
        Core.inRange(out, new Scalar(hue[0], sat[0], val[0]),
                new Scalar(hue[1], sat[1], val[1]), out);
    }


    @SuppressWarnings("SameParameterValue")
    static void findContours(Mat input, boolean externalOnly, List<MatOfPoint> contours)
    {
        Mat hierarchy = new Mat();
        contours.clear();

        int mode = (externalOnly ? Imgproc.RETR_EXTERNAL : Imgproc.RETR_LIST);
        int method = Imgproc.CHAIN_APPROX_SIMPLE;
        Imgproc.findContours(input, contours, hierarchy, mode, method);
    }

    static void findRotatedRects(List<MatOfPoint> inputContours, List<RotatedRect> outputRotatedRects)
    {
        MatOfPoint2f mat2f = new MatOfPoint2f();
        outputRotatedRects.clear();

        for (MatOfPoint inputContour : inputContours) {
            // Find MinAreaRect for each contour and consider if it meets criteria
            inputContour.convertTo(mat2f, CvType.CV_32F);
            RotatedRect rect = Imgproc.minAreaRect(mat2f);

            outputRotatedRects.add(rect);
        }
    }

    static void renderContours(List<RotatedRect> rects, Mat output, int offsetX, int offsetY, boolean debug)
    {
        double fontScale = (output.width() > 352 ? 1.0 : 0.7);

        for (RotatedRect rect : rects) {
            Point[] vertices = new Point[4];
            rect.points(vertices);
            // add offsets first
            for (int i = 0; i < 4; i++) {
                vertices[i].x += offsetX;
                vertices[i].y += offsetY;
            }

            for (int i = 0; i < 4; i++)
                Imgproc.line(output, vertices[i], vertices[(i + 1) % 4], red);

            Point p = rect.center.clone();
            p.x += offsetX - 20.0;  // center text over the center of the line, should calc test length
            p.y += offsetY;

            double angle = (Math.round((rect.size.width < rect.size.height) ? rect.angle + 90 : rect.angle) * 10.0) / 10.0;

            double dy = 15;
            p.y -= dy;
            Imgproc.putText(output, "/" + angle, p, Core.FONT_HERSHEY_PLAIN, fontScale, black, 3);
            Imgproc.putText(output, "/" + angle, p, Core.FONT_HERSHEY_PLAIN, fontScale, white, 1);

            if (debug) {
                p.y += dy;
                String detail = "(" + (int)Math.round(rect.center.x + offsetX) + ","
                        + (int)Math.round(rect.center.y + offsetY) + ","
                        + (int)rect.size.width + ","
                        + (int)rect.size.height + ")";
                Imgproc.putText(output, detail, p, Core.FONT_HERSHEY_PLAIN, fontScale, black, 3);
                Imgproc.putText(output, detail, p, Core.FONT_HERSHEY_PLAIN, fontScale, white, 1);
            }
        }
    }
}

