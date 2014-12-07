/*
#define OUT_DIR "/exports/cyclops/work/001_Selfies/002_Segmentation/src/Caffe_Segmentation/results/backpage/ImagesNevada/segs/imgs"
 * createFinalRes.cpp
 *
 *  Created on: Dec 1, 2014
 *      Author: rohitgirdhar
 */

#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <algorithm>
#include <libgen.h>

#include <opencv2/opencv.hpp>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;
using std::string;

using namespace cv;
using namespace std;

#define SZ_X 227
#define SZ_Y 227


int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
    if (argc < 4) {
        LOG(ERROR) << "Usage: " << argv[0] << " LOC_RES_FILE SEG_IMG_DIR OUT_DIR";
        return -1;
    }
    char *LOC_RES_FILE = argv[1];
    char *SEG_IMG_DIR = argv[2];
    char *OUT_DIR = argv[3];
    string fname;
    float xmin, xmax, ymin, ymax;
    ifstream infile(LOC_RES_FILE);
    if (!infile.is_open()) {
        LOG(ERROR) << "Unable to open file: " << LOC_RES_FILE;
        return -1;
    }
    Mat I, S;
    I = Mat(SZ_X, SZ_Y, CV_8UC1);
    while (infile >> fname >> xmin >> ymin >> xmax >> ymax) {
        I.setTo(Scalar(0));
        char *fbasename = basename(strdup(fname.c_str()));
        xmin = std::max(xmin, (float) 0);
        ymin = std::max(ymin, (float) 0);
        xmax = std::min(xmax, (float) I.cols);
        ymax = std::min(ymax, (float) I.rows);
        S = imread(string(SEG_IMG_DIR) + "/" + fbasename, CV_LOAD_IMAGE_GRAYSCALE);
        if (!S.data) {
            LOG(ERROR) << "Unable to read image " << string(SEG_IMG_DIR) + fbasename;
            continue;
        }
        resize(S, S, Size(xmax - xmin, ymax - ymin));
        S.copyTo(I(Rect(xmin, ymin, xmax - xmin, ymax - ymin)));
        imwrite(string(OUT_DIR) + "/" + fbasename, I);
    }
	return 0;
}

