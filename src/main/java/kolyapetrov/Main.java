package kolyapetrov;

import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.text.DecimalFormat;
import java.util.concurrent.TimeUnit;

public class Main {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.out.println("Version: " + Core.VERSION);
    }

    public static void main(String[] args) {
        long time = System.currentTimeMillis();
        int imageNum = 2;

//        findImageZoomed(imageNum);
//        findImageRotated(imageNum);
        createZoomedImages();

        System.out.println("All time: " + TimeUnit.SECONDS.convert(System.currentTimeMillis() - time,
                TimeUnit.MILLISECONDS) + " seconds.");
        HighGui.waitKey();
    }

    private static void findZoomedImage(int imageNum) {
        DecimalFormat decimalFormat = new DecimalFormat("0.000");
        Mat img = Imgcodecs.imread("src/main/resources/original/" + imageNum + ".png");

        for (double coeff = 0.9; coeff <= 1.1; coeff += 0.025) {
            Mat searchableFragment = Imgcodecs.imread("src/main/resources/zoomed_images/" + imageNum + "/"
                    + decimalFormat.format(coeff) + ".png");
            findImage(img, searchableFragment, decimalFormat.format(coeff));
        }
        Mat searchableFragment = Imgcodecs.imread("src/main/resources/fragment/" + imageNum + ".png");
        findImage(img, searchableFragment, "0.0");
    }

    private static void findRotatedImage(int imageNum) {
        DecimalFormat decimalFormat = new DecimalFormat("0.0");
        Mat img = Imgcodecs.imread("src/main/resources/original/" + imageNum + ".png");

        for (double angle = 2.0; angle <= 10.0; angle += 2.0) {
            Mat searchableFragment = Imgcodecs.imread("src/main/resources/rotated_images/" + imageNum + "/"
                    + decimalFormat.format(angle) + ".png");

            findImage(img, searchableFragment, decimalFormat.format(angle));
        }
        Mat searchableFragment = Imgcodecs.imread("src/main/resources/fragment/" + imageNum + ".png");
        findImage(img, searchableFragment, "1.000");
    }

    private static void findImage(Mat img, Mat searchableFragment, String param) {
        long time = System.currentTimeMillis();

        Mat grayImage = new Mat();
        Mat graySearchableFragment = new Mat();

        // Sharpening
        Mat img2 = new Mat();
        Imgproc.GaussianBlur(img, img2, new Size(9, 9), 0);
        Core.addWeighted(img, 4, img2, -3, 0, img);
        Mat img3 = new Mat();
        Imgproc.GaussianBlur(searchableFragment, img3, new Size(9, 9), 0);
        Core.addWeighted(searchableFragment, 4, img3, -3, 0, searchableFragment);

        Imgproc.cvtColor(img, grayImage, Imgproc.COLOR_RGB2GRAY);
        Imgproc.cvtColor(searchableFragment, graySearchableFragment, Imgproc.COLOR_RGB2GRAY);
        centeringByBrightness(grayImage);
        centeringByBrightness(graySearchableFragment);

        int imageWidth = img.cols();
        int imageHeight = img.rows();
        int fragmentWidth = searchableFragment.cols();
        int fragmentHeight = searchableFragment.rows();

        Point sizePoint = new Point(imageWidth - fragmentWidth + 1, imageHeight - fragmentHeight + 1);
        Size size = new Size(sizePoint);
        System.out.println("Result image size: " + size);
        System.out.println("Iteration for param: " + param);

        Mat result = new Mat(size, CvType.CV_32FC1);

        for (int y = 0; y < result.rows(); ++y) {
            for (int x = 0; x < result.cols(); ++x) {
                double cor = 0;
                for (int l = 0; l < fragmentHeight; ++l) {
                    for (int k = 0; k < fragmentWidth; ++k) {
                        double brightnessForImage = grayImage.get(y + l, x + k)[0];
                        double brightnessForFragment = graySearchableFragment.get(l, k)[0];

                        cor += brightnessForImage * brightnessForFragment;
                    }
                }
                result.put(y, x, cor);
            }
            System.out.println(y);
        }

        Core.MinMaxLocResult mmr = Core.minMaxLoc(result);
        Point maxLoc = mmr.maxLoc;

        Scalar color = new Scalar(0, 0, 255);
        Imgproc.rectangle(img, maxLoc, new Point(maxLoc.x + searchableFragment.cols(),
                        maxLoc.y + searchableFragment.rows()),
                color, 2);

        Core.normalize(result, result, 0, 255, Core.NORM_MINMAX);
        result.convertTo(result, CvType.CV_8UC1);
        HighGui.imshow(param, result);

//        Imgcodecs.imwrite("src/main/resources/compare2/" + param + ".png", result);
        HighGui.imshow("Result: ", img);
        HighGui.imshow("Frg: ", searchableFragment);

        System.out.println("Time: " + TimeUnit.SECONDS.convert(System.currentTimeMillis() - time,
                TimeUnit.MILLISECONDS) + " seconds.");
    }

    private static void centeringByBrightness(Mat mat) {
        double avg = Core.mean(mat).val[0]; // search for the avg value among  all elements
        Core.subtract(mat, new Scalar(avg), mat); // subtract avg value from all elements
    }

    private static Mat rotateImage(Mat image, double angle) {
        Mat rotationMat = Imgproc.getRotationMatrix2D(new Point((double) image.width() / 2,
                (double) image.height() / 2), angle, 1);
        Mat newImage = new Mat();
        Imgproc.warpAffine(image, newImage, rotationMat, image.size());
        return newImage;
    }

    private static Mat zoomImage(Mat image, double coeff) {
        long newWidth = (long) (image.width() * coeff);
        long newHeight = (long) (image.height() * coeff);

        Mat resizedImage = new Mat();
        Imgproc.resize(image, resizedImage, new Size(newWidth, newHeight));

        return resizedImage;
    }

    private static void zoomImageAndWriteToMemory(Mat mat, String name, double limit) {
        for (double coeff = 0.9; coeff <= limit; coeff += 0.025) {
            if (coeff == 1.0) continue;
            Mat newImage = zoomImage(mat, coeff);
            DecimalFormat decimalFormat = new DecimalFormat("0.000");
            Imgcodecs.imwrite("src/main/resources/zoomed_images/" + name + "/" + decimalFormat.format(coeff)
                            + ".png", newImage);
        }
    }

    private static void rotateImageAndWriteToMemory(Mat mat, String name, double limit) {
        for (double angle = 2.0; angle <= limit; angle += 2.0) {
            Mat newImage = rotateImage(mat, angle);
            Imgcodecs.imwrite("src/main/resources/rotated_images/" + name + "/" + angle + ".png",
                    newImage);
        }
    }

    private static void createRotatedImages(double limit) {
        for (int i = 1; i <= 3; ++i) {
            Mat fragment = Imgcodecs.imread("src/main/resources/fragment/" + i + ".png");
            double angleLimit = 10.0;
            rotateImageAndWriteToMemory(fragment, "" + i, angleLimit);
        }
    }

    private static void createZoomedImages() {
        for (int i = 1; i <= 3; ++i) {
            Mat fragment = Imgcodecs.imread("src/main/resources/fragment/" + i + ".png");
            double zoomCoeffLimit = 1.5;
            zoomImageAndWriteToMemory(fragment, "" + i, zoomCoeffLimit);
        }
    }
}
