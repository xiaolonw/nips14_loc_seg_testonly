/*
 * cropDet.cpp
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

#define SEG_RES_FILE "/exports/cyclops/work/001_Selfies/002_Segmentation/src/Caffe_Segmentation/results/backpage/ImagesNevada/segs/segResult.txt"
#define OUT_DIR "/exports/cyclops/work/001_Selfies/002_Segmentation/src/Caffe_Segmentation/results/backpage/ImagesNevada/segs/imgs"
#define DIM 55

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
    string fname;
    ifstream infile(SEG_RES_FILE);
    if (!infile.is_open()) {
        LOG(ERROR) << "Unable to open file: " << SEG_RES_FILE;
        return -1;
    }
    Mat S(DIM, DIM, CV_32FC1);
    while (infile >> fname) {
        for (int i = 0; i < DIM; i++) {
            for (int j = 0; j < DIM; j++) {
                infile >> S.at<float>(Point2f(j, i));
            }
        }
        Mat C2;
        S.convertTo(C2, CV_8UC1);
        equalizeHist(C2, C2);
        char *fname_char = strdup(fname.c_str());
        imwrite(string(OUT_DIR) + string("/") + basename(fname_char), C2);
    }
	return 0;
}

