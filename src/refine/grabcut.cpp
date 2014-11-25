#include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat segment(const Mat& img, const Mat& sparseSeg);

int main() {
    Mat I = imread("0005.jpg");
    Mat S = imread("0005.png", CV_LOAD_IMAGE_GRAYSCALE);
    Mat S2 = segment(I, S);
    normalize(S2, S2, 0, 255, NORM_MINMAX, CV_8UC1);
    imwrite("result.jpg", S2);
    return 0;
}

Mat segment(const Mat& img, const Mat& sparseSeg) {
    Mat mask = Mat(img.rows, img.cols, CV_8UC1);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (sparseSeg.at<uchar>(Point2d(j, i)) > 0) {
                mask.at<uchar>(Point2d(j, i)) = GC_PR_FGD;
            } else {
                mask.at<uchar>(Point2d(j, i)) = GC_PR_BGD;
            }
        }
    }
    //to visualize
    //normalize(mask, mask, 0, 255, NORM_MINMAX, CV_8UC1);
    //imshow("tremp", mask);
    //waitKey(0);
    grabCut(img, mask, Rect(), Mat(), Mat(), 100);
    return mask;
}

