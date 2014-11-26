#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat segment(const Mat& img, const Mat& sparseSeg, const Mat&, const Rect&);
void threshold(Mat& img);

Rect transformRect(const Rect& orig, float wd, float ht, float towd, float toht) {
    float xmin, ymin, newwd, newht;
    xmin = orig.x * (towd / wd);
    ymin = orig.y * (toht / ht);
    newwd = orig.width * (towd / wd);
    newht = orig.height * (toht / ht);
    return Rect(xmin, ymin, newwd, newht);
}

int main(int argc, char* argv[]) {
    if (argc <= 3) {
        cerr << "Usage: " << argv[0] << " input_directory img_names_fpath gc_dir" << endl;
        return -1;
    }
    ifstream infile(argv[2]);
    string fname;
    Mat I, S, S2;
    float xmin, xmax, ymin, ymax;
    while (infile >> fname >> xmin >> ymin >> xmax >> ymax) {
        fname = fname.substr(0, fname.length() - 4);
        I = imread(string(argv[1]) + "/" + fname + ".jpg");
        S = imread(string(argv[1]) + "/" + fname + ".png", CV_LOAD_IMAGE_GRAYSCALE);
        
        float dsize = 10;
        Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2*dsize + 1, 2*dsize+1), Point(dsize, dsize));
        Mat S_eroded;
        erode(S, S_eroded, element);
//        imshow("temp", S); waitKey();
        S2 = segment(I, S, S_eroded, 
                transformRect(Rect(xmin, ymin, xmax - xmin, ymax - ymin),
                    227, 227,
                    I.cols, I.rows));
        threshold(S2);
        normalize(S2, S2, 0, 255, NORM_MINMAX, CV_8UC1);
        imwrite(string(argv[3]) + "/" + fname + ".png", S2);
        cout << "Done for " << fname << endl;
    }
    infile.close();
    return 0;
}

Mat segment(const Mat& img, const Mat& sparseSeg, const Mat& sureSeg, const Rect& bbox) {
    Mat mask = Mat(img.rows, img.cols, CV_8UC1);
    float xmin=9999, xmax=0, ymin=9999, ymax=0;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            int x = j, y = i;
            if (sparseSeg.at<uchar>(Point2d(x, y)) > 0) {
                mask.at<uchar>(Point2d(x, y)) = GC_PR_FGD;
                
                if (x > xmax) xmax = x;
                if (x < xmin) xmin = x;
                if (y > ymax) ymax = y;
                if (y < ymin) ymin = y;
                
            } else {
                mask.at<uchar>(Point2d(x, y)) = GC_PR_BGD;
            }
            if (sureSeg.at<uchar>(Point2d(x, y)) > 0) {
                mask.at<uchar>(Point2d(x, y)) = GC_PR_FGD;
            }
        }
    }
    //to visualize
    //normalize(mask, mask, 0, 255, NORM_MINMAX, CV_8UC1);
    //imshow("tremp", mask);
    //waitKey(0);
    //Rect bbox2(xmin, ymin, xmax - xmin, ymax - ymin);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (j < xmin || j > xmax || i < ymin || i > ymax) {
                mask.at<uchar>(Point2d(j, i)) = GC_BGD;
            }
        }
    }

    Mat bgdmodel, fgdmodel;
    grabCut(img, mask, bbox, bgdmodel, fgdmodel, 10, GC_INIT_WITH_MASK | GC_INIT_WITH_RECT);
    rectangle(mask, bbox, Scalar(1));
    return mask;
}

void threshold(Mat& img) {
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img.at<uchar>(Point2d(j, i)) == GC_PR_BGD) {
                img.at<uchar>(Point2d(j, i)) = GC_BGD;
            }
        }
    }
}

