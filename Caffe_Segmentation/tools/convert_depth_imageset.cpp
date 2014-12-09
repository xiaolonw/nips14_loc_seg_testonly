/*
 * convert_depth_imageset.cpp
 *
 *  Created on: Nov 19, 2014
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
#include <sstream>
#include <string>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;
using std::string;

using namespace std;

#define HT 228
#define WD 304
#define RESIZE_HEIGHT 55
#define RESIZE_WIDTH 74
#define LABEL_LEN (55 * 74 * 1)

// use float label, if changed, one should change caffe.proto label too

struct Seg_Anno {
	string filename_;
	std::vector<float> pos_;
};

bool MyReadImageToDatum(const string& fpath, Datum* datum)
{
	datum->set_channels(3);
	datum->set_height(HT);
	datum->set_width(WD);
	datum->clear_data();
	datum->clear_float_data();
	datum->clear_label();

	string* datum_string = datum->mutable_data();

    ifstream infile(fpath.c_str());
    if (!infile) {
        cerr << "Unable to open " << fpath << endl;
        return false;
    }
    string line;
    int lno = 0;
    while (getline(infile, line)) {
        istringstream iss(line);
        string token;
        if (lno < 3) {
            while (getline(iss, token, ',')) {
                datum_string->push_back(static_cast<char>(atoi(token.c_str())));
            }
        } else {
            while (getline(iss, token, ',')) {
                datum->add_label(static_cast<float>(atof(token.c_str())));
            }
        }
        lno ++;
    }

	return true;
}


int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
	
	leveldb::DB* db;
	leveldb::Options options;
	options.error_if_exists = true;
	options.create_if_missing = true;
	options.write_buffer_size = 268435456;
	LOG(INFO)<< "Opening leveldb ";
	leveldb::Status status = leveldb::DB::Open(options, "examples/depth/depth_train_leveldb", &db);
	CHECK(status.ok()) << "Failed to open leveldb ";

	Datum datum;
	int count = 0;
	const int maxKeyLength = 256;
	char key_cstr[maxKeyLength];
	leveldb::WriteBatch* batch = new leveldb::WriteBatch();
	int data_size;
	bool data_size_initialized = false;


	for (int anno_id = 1; anno_id <= 1449; ++anno_id)
	{
        char temp[100000];
        sprintf(temp, "data/NYUv2Depth/%d.txt", anno_id);
        string fpath = string(temp);
		if (!MyReadImageToDatum(fpath, &datum))
		{
			continue;
		}
		if (!data_size_initialized)
		{
			data_size = datum.channels() * datum.height() * datum.width() ;
			data_size_initialized = true;
		}
		else
		{
			const string& data = datum.data();
			CHECK_EQ(data.size(), data_size)<< "Incorrect data field size " << data.size();
		}

		// sequential
		snprintf(key_cstr, maxKeyLength, "%07d_%s", anno_id, fpath.c_str());
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
