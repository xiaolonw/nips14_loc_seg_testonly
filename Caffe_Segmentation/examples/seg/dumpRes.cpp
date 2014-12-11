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
#include <boost/filesystem.hpp>

#include <opencv2/opencv.hpp>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;
using std::string;

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
    if (argc < 4) {
        LOG(ERROR) << "Usage: " << argv[0] << " SEG_RES_FILE OUT_DIR IMG_DIM";
        return -1;
    }
    char* SEG_RES_FILE = argv[1];
    char* OUT_DIR = argv[2];
    int DIM = atof(argv[3]);
    boost::filesystem::create_directory(OUT_DIR);

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
        string fpath = string(OUT_DIR) + string("/") + fname;

        boost::filesystem::create_directory(dirname(strdup(fpath.c_str())));
        imwrite(fpath, C2);
    }
	return 0;
}

