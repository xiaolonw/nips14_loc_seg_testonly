/*
 * convert_normal.cpp
 *
 *  Created on: Aug 11, 2014
 *      Author: dragon123
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

#include <opencv2/opencv.hpp>
//#include <cv.h>
//#include <highgui.h>
//#include <cxcore.h>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;
using std::string;

using namespace cv;

#define RESIZE_LEN 55
#define LABEL_LEN 3025

// use float label, if changed, one should change caffe.proto label too

struct Seg_Anno {
	string filename_;
	std::vector<int> pos_;
};

bool MyReadImageToDatum(const string& filename, const std::vector<int> & label,
    const int height, const int width, Datum* datum)
{
	cv::Mat cv_img_origin = cv::imread(filename, CV_LOAD_IMAGE_COLOR), cv_img;
	if (!cv_img_origin.data) {
		LOG(ERROR) << "Could not open or find file " << filename;
		return false;
    }

	if (height > 0 && width > 0) {
		cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
	} else {
		cv_img = cv_img_origin;
	}

	datum->set_channels(3);
	datum->set_height(cv_img.rows);
	datum->set_width(cv_img.cols);
	datum->clear_data();
	datum->clear_float_data();

	string* datum_string = datum->mutable_data();
	for (int c = 0; c < 3; ++c) {
		for (int h = 0; h < cv_img.rows; ++h) {
		  for (int w = 0; w < cv_img.cols; ++w) {
			datum_string->push_back(
				static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
		  }
		}
	}

	//datum->set_label(label);
	datum->clear_label();
	for(int i = 0; i < label.size(); i ++)
		datum->add_label(label[i]);

	return true;
}


int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
	if (argc < 5) {
		printf(
				"Convert a set of images to the leveldb format used\n"
						"as input for Caffe.\n"
						"Usage:\n"
						"    convert_imageset ROOTFOLDER/ ANNOTATION DB_NAME"
						" MODE[0-train, 1-val, 2-test] RANDOM_SHUFFLE_DATA[0 or 1, default 1] RESIZE_WIDTH[default 256] RESIZE_HEIGHT[default 256](0 indicates no resize)\n"
						"The ImageNet dataset for the training demo is at\n"
						"    http://www.image-net.org/download-images\n");
		return 0;
	}
	std::ifstream infile(argv[2]);
	std::vector<Seg_Anno> annos;
	std::set<string> fNames;
	string filename;
	int cc = 0;
	while (infile >> filename)
	{
		if (cc % 1000 == 0)
		LOG(INFO)<<filename;
		cc ++;

		Seg_Anno seg_Anno;
		seg_Anno.filename_ = filename;
		for (int i = 0; i < LABEL_LEN; i++)
		{
			//infile >> prop;
			seg_Anno.pos_.push_back(0);
		}
		if (fNames.find(filename)== fNames.end())
		{
			fNames.insert(filename);
			annos.push_back(seg_Anno);
		}
		//debug
		//if(annos.size() == 10)
		//	break;
	}
	if (argc < 6 || argv[5][0] != '0') {
		// randomly shuffle data
		LOG(INFO)<< "Shuffling data";
		std::random_shuffle(annos.begin(), annos.end());
	}
	LOG(INFO)<< "A total of " << annos.size() << " images.";

	leveldb::DB* db;
	leveldb::Options options;
	options.error_if_exists = true;
	options.create_if_missing = true;
	options.write_buffer_size = 268435456;
	LOG(INFO)<< "Opening leveldb " << argv[3];
	leveldb::Status status = leveldb::DB::Open(options, argv[3], &db);
	CHECK(status.ok()) << "Failed to open leveldb " << argv[3];

	string root_folder(argv[1]);
	Datum datum;
	int count = 0;
	const int maxKeyLength = 256;
	char key_cstr[maxKeyLength];
	leveldb::WriteBatch* batch = new leveldb::WriteBatch();
	int data_size;
	bool data_size_initialized = false;

	// resize to height * width
    int width = RESIZE_LEN;
    int height = RESIZE_LEN;
    if (argc > 6) width = atoi(argv[6]);
    if (argc > 7) height = atoi(argv[7]);
    if (width == 0 || height == 0)
        LOG(INFO) << "NO RESIZE SHOULD BE DONE";
    else
        LOG(INFO) << "RESIZE DIM: " << width << "*" << height;

	for (int anno_id = 0; anno_id < annos.size(); ++anno_id)
	{
		if (!MyReadImageToDatum(root_folder + "/" + annos[anno_id].filename_,
				annos[anno_id].pos_, height, width, &datum))
		{
			continue;
		}
		if (!data_size_initialized)
		{
			data_size = datum.channels() * datum.height() * datum.width();
			data_size_initialized = true;
		}
		else
		{
			const string& data = datum.data();
			CHECK_EQ(data.size(), data_size)<< "Incorrect data field size " << data.size();
		}

		// sequential
		snprintf(key_cstr, maxKeyLength, "%07d_%s", anno_id, annos[anno_id].filename_.c_str());
		string value;
		// get the value
		datum.SerializeToString(&value);
		batch->Put(string(key_cstr), value);
		if (++count % 1000 == 0)
		{
			db->Write(leveldb::WriteOptions(), batch);
			LOG(ERROR)<< "Processed " << count << " files.";
			delete batch;
			batch = new leveldb::WriteBatch();
		}
	}
	// write the last batch
	if (count % 1000 != 0) {
		db->Write(leveldb::WriteOptions(), batch);
		LOG(ERROR)<< "Processed " << count << " files.";
	}

	delete batch;
	delete db;
	return 0;
}
