/*
 * cropDet.cpp
 *
 *  Created on: Nov 30, 2014
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
    if (argc != 4) {
        LOG(ERROR) << "Usage: " << argv[0] << " LOC_RES_FILE IMGS_DIR OUT_DIR";
        return -1;
    }
    char* LOC_RES_FILE = argv[1];
    char* IMGS_DIR = argv[2];
    char* OUT_DIR = argv[3];
    string fname;
    float xmin, ymin, xmax, ymax;
    ifstream infile(LOC_RES_FILE);
    if (!infile.is_open()) {
        LOG(ERROR) << "Unable to open file: " << LOC_RES_FILE;
        return -1;
    }
    Mat I, C;
    while (infile >> fname >> xmin >> ymin >> xmax >> ymax) {
        string fpath = string(IMGS_DIR) + string("/") + fname;
        I = imread(fpath.c_str());
        if (!I.data) {
            LOG(ERROR) << "Unable to read image : " << fpath;
            continue;
        }
        resize(I, I, Size(SZ_X, SZ_Y)); 
        xmin = std::max(xmin, (float) 0);
        ymin = std::max(ymin, (float) 0);
        xmax = std::min(xmax, (float) I.cols);
        ymax = std::min(ymax, (float) I.rows);
        C = I(Rect(xmin, ymin, xmax - xmin, ymax - ymin));
        imwrite(OUT_DIR + string("/") + fname.substr(string("ImagesNevada/").length()), C);
    }
	return 0;
}

