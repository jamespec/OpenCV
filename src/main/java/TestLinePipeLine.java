import org.opencv.core.Point;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.util.ArrayList;
import java.util.List;

import static org.opencv.core.Core.inRange;
import static org.opencv.videoio.Videoio.CAP_DSHOW;

class TestLinePipeLine {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

//    private final static Scalar minGreen = new Scalar(24, 44, 57, 0);
//    private final static Scalar maxGreen = new Scalar(71, 159, 161, 255);
//    private final static Scalar minYellow = new Scalar(13, 99, 126, 0);
//    private final static Scalar maxYellow = new Scalar(25, 255, 233, 255);
//    private final static Scalar minWhite = new Scalar(0, 0, 180, 0);
//    private final static Scalar maxWhite = new Scalar(180, 27, 255, 255);
//    private final static Scalar minPurple = new Scalar(131, 28, 85, 0);
//    private final static Scalar maxPurple = new Scalar(157, 144, 216, 255);
    private final static Scalar minGreen = new Scalar(36, 0, 48, 0);
    private final static Scalar maxGreen = new Scalar(79, 153, 255, 255);
    private final static Scalar minYellow = new Scalar(13, 112, 158, 0);
    private final static Scalar maxYellow = new Scalar(25, 255, 255, 255);
    private final static Scalar minWhite = new Scalar(0, 0, 180, 0);
    private final static Scalar maxWhite = new Scalar(180, 27, 255, 255);
    private final static Scalar minPurple = new Scalar(131, 28, 71, 0);
    private final static Scalar maxPurple = new Scalar(157, 116, 255, 255);

    private static BufferedImage Mat2BufferedImage(Mat m) {
        // Fastest code
        // output can be assigned either to a BufferedImage or to an Image

        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (m.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        int bufferSize = m.channels() * m.cols() * m.rows();
        byte[] b = new byte[bufferSize];
        m.get(0, 0, b); // get all the pixels
        BufferedImage image = new BufferedImage(m.cols(), m.rows(), type);
        //DataBufferByte a;
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(b, 0, targetPixels, 0, b.length);
        return image;
    }

    private static JLabel image;

    private static void CreateImageFrame(Image img2, String title)
    {
        ImageIcon icon = new ImageIcon(img2);
        JFrame frame = new JFrame();
        frame.setLayout(new FlowLayout());
        frame.setSize(img2.getWidth(null) + 50, img2.getHeight(null) + 50);
        image = new JLabel();
        image.setIcon(icon);
        frame.add(image);
        frame.setVisible(true);
        frame.setTitle(title);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    private static void UpdateImage(Image img2)
    {
        ImageIcon icon = new ImageIcon(img2);
        image.setIcon(icon);
    }

    public static void main(String[] args) {
        VideoCapture capture = new VideoCapture(0, CAP_DSHOW);

        Mat mat = new Mat();
        Mat hsvMat = new Mat();
        Mat colorMat = new Mat();

        capture.read(mat);
        Imgproc.cvtColor(mat, hsvMat, Imgproc.COLOR_BGR2HSV);
        BufferedImage bi1 = Mat2BufferedImage(hsvMat);
        CreateImageFrame(bi1, "Original");

        List<MatOfPoint> contours = new ArrayList<>();
        Point center = new Point();
        float[] radius = new float[2];
        MatOfPoint2f mat2f = new MatOfPoint2f();

        while (true) {
            capture.read(mat);
            Imgproc.cvtColor(mat, hsvMat, Imgproc.COLOR_BGR2HSV);

            blur(hsvMat, 4.0, hsvMat);

            inRange(hsvMat, minGreen, maxGreen, colorMat);
            Mat totalColorMat = colorMat.clone();

            inRange(hsvMat, minYellow, maxYellow, colorMat);
            Core.add(totalColorMat, colorMat, totalColorMat);
            inRange(hsvMat, minWhite, maxWhite, colorMat);
            Core.add(totalColorMat, colorMat, totalColorMat);
            inRange(hsvMat, minPurple, maxPurple, colorMat);
            Core.add(totalColorMat, colorMat, totalColorMat);

            int contourCount = 0;
            LinePipeline.findContours(totalColorMat, true, contours);
            for (int i = 0; i < contours.size(); ++i) {
                MatOfPoint contour = contours.get(i);
                contour.convertTo(mat2f, CvType.CV_32F);
                Imgproc.minEnclosingCircle(mat2f, center, radius);

                if (radius[0] > 15 && center.y > mat.height()/2.0) {
                    contourCount++;
                    // draw the contour and center of the shape on the image
                    Imgproc.drawContours(mat, contours, i, new Scalar(0, 255, 0), 2);

                    String s = String.format("(%d, %d)", (int) center.x, (int) center.y);
                    center.x += 30.0;
                    Imgproc.putText(mat, s, center, Core.FONT_HERSHEY_PLAIN, 1.0, new Scalar(255, 255, 255), 1);
                }
            }
            String s = String.format("Contours: %d", contourCount );
            Imgproc.putText(mat, s, new Point(100,100), Core.FONT_HERSHEY_PLAIN, 1.0, new Scalar(255, 255, 255), 1);

            bi1 = Mat2BufferedImage(mat);
            UpdateImage(bi1);
        }
    }

    private static void blur(Mat input, double doubleRadius, Mat output) {
        int radius = (int) (doubleRadius + 0.5);
        int kernelSize;

        kernelSize = 2 * radius + 1;
        Imgproc.blur(input, output, new Size(kernelSize, kernelSize));
    }
}
