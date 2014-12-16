/*
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

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
	if (argc < 4) {
		LOG(ERROR)<< "Usage: " << argv[0] << " LOC_RES_FILE SEG_IMG_DIR IMG_DIR OUT_DIR";
		return -1;
	}
	char *LOC_RES_FILE = argv[1];
	char *SEG_IMG_DIR = argv[2];
	char *IMG_DIR = argv[3];
	char *OUT_DIR = argv[4];
	boost::filesystem::create_directory(OUT_DIR);

	string fname;
	float xmin, xmax, ymin, ymax;
	ifstream infile(LOC_RES_FILE);
	if (!infile.is_open()) {
		LOG(ERROR)<< "Unable to open file: " << LOC_RES_FILE;
		return -1;
	}
	Mat I, S;
	Mat I2;
	I = Mat(SZ_X, SZ_Y, CV_8UC1);
	while (infile >> fname >> xmin >> ymin >> xmax >> ymax) {
		I.setTo(0);
		xmin = std::min(std::max(xmin + OFFSET, (float) 0), (float) SZ_X-1);
		ymin = std::min(std::max(ymin + OFFSET, (float) 0), (float) SZ_Y-1);
		xmax = std::min(std::max(xmax + OFFSET, (float) 0), (float) SZ_X-1);
		ymax = std::min(std::max(ymax + OFFSET, (float) 0), (float) SZ_Y-1);
		int x1 = xmin;
		int y1 = ymin;
		int x2 = xmax;
		int y2 = ymax;
		//LOG(ERROR)<<fname<<" "<<x1<<" "<<y1<<" "<<x2<<" "<<y2;

		S = imread(string(SEG_IMG_DIR) + "/" + fname, CV_LOAD_IMAGE_GRAYSCALE);
		if (!S.data) {
			LOG(ERROR)<< "Unable to read image " << string(SEG_IMG_DIR) + "/" + fname;
			continue;
		}
		int height = MAX(1, y2 - y1 + 1);
		int width = MAX(1, x2 - x1 + 1);
		resize(S, S, Size(width, height));

		Mat extractedImage = I(Rect(x1, y1, width, height));
		S.copyTo(extractedImage);
		string fpath = string(OUT_DIR) + "/" + fname;
		boost::filesystem::create_directory(dirname(strdup(fpath.c_str())));
		imwrite(fpath, I);
		I2 = imread(string(IMG_DIR) + "/" + fname);
		I2 = I2 * 0.5; 
		resize(I2, I2, Size(SZ_X, SZ_Y));
		Mat channels[3];
		split(I2, channels);
		addWeighted(channels[2], 0.5, I, 1.0, 0.0, channels[2]);
		merge(channels, 3, I2);
		fpath = string(OUT_DIR) + "/" + fname + ".segmask.jpg";
		imwrite(fpath, I2);
	}
	return 0;
}

