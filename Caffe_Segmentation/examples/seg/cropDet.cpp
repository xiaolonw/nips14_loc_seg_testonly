/*
 * cropDet.cpp
 *
 *  Created on: Nov 30, 2014
 *      Author: rohitgirdhar
 */

#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <cstdio>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <algorithm>
#include <libgen.h> // for dirname
#include <boost/filesystem.hpp>

#include <opencv2/opencv.hpp>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;
using std::string;

using namespace cv;
using namespace std;

#define SZ_X 256
#define SZ_Y 256

#define OFFSET ((256-227)/2)

int CreateDir(const char*, int);

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
        xmin = std::min(std::max(xmin + OFFSET, (float) 0), (float) SZ_X-1);
        ymin = std::min(std::max(ymin + OFFSET, (float) 0), (float) SZ_Y-1);
        xmax = std::min(std::max(xmax + OFFSET, (float) 0), (float) SZ_X-1);
        ymax = std::min(std::max(ymax + OFFSET, (float) 0), (float) SZ_Y-1);
        C = I(Rect(xmin, ymin, xmax - xmin, ymax - ymin));
        string filename = OUT_DIR + string("/") + fname;
        boost::filesystem::create_directory(dirname(strdup(filename.c_str())));
        imwrite(filename, C);
        LOG(ERROR)<<filename<<" "<<xmin<<" "<<ymin<<" "<<xmax<<" "<<ymax;
    }
	return 0;
}

