#include <opencv2\core.hpp>
#include <opencv2\imgcodecs.hpp> // imread
#include <opencv2\highgui.hpp> // imshow, waitKey
using namespace cv;
int main() {
    // Load image as a single channel grayscale Mat.
    Mat img = imread("pic1.bmp", IMREAD_GRAYSCALE);
    // Mat is a thin template wrapper on top of the Mat class.
    // Mat_::operator()(y, x) does the same thing as Mat::at(y, x).
    Mat_<uchar> imgWrp(img);
    Mat_<uchar> smallImgWrp(img.size() / 2);
    for (int rowIndex = 0; rowIndex != smallImgWrp.rows; ++rowIndex)
        for (int colIndex = 0; colIndex != smallImgWrp.cols; ++colIndex)
            smallImgWrp(rowIndex, colIndex) =
                imgWrp(rowIndex * 2, colIndex * 2);
    Mat result(img.rows + smallImgWrp.rows, img.cols, CV_8U, Scalar(0));
    imgWrp.copyTo(result(Rect(0, 0, imgWrp.cols, imgWrp.rows)));
    smallImgWrp.copyTo(
        result(Rect(0, imgWrp.rows, smallImgWrp.cols, smallImgWrp.rows)));
    cv::imshow("Hi", result);
    cv::waitKey(); // Wait for the user to press a key.
    return 0;
}