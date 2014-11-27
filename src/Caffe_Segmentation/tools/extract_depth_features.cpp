// 输出分割的结果

/*#include <cuda_runtime.h> */

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

//#include <cv.h>
//#include <highgui.h>
//#include <cxcore.h>

#include <cstring>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <utility>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <leveldb/db.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"

#include <string>
#include <fstream>

using namespace std;
using namespace cv;
using namespace caffe;
using std::vector;
#define LABEL_LEN 40
#define LABEL_SIZE 13
#define IMG_LENGTH 256

#define GEN_VIS 1 // set as 0 to stop generating visualization
#define OUTPUT_DIR string("examples/depth/output/")
#define GT_DIR string("examples/depth/gt/")

int CreateDir(const char *sPathName, int beg) {
	char DirName[256];
	strcpy(DirName, sPathName);
	int i, len = strlen(DirName);
	if (DirName[len - 1] != '/')
		strcat(DirName, "/");

	len = strlen(DirName);

	for (i = beg; i < len; i++) {
		if (DirName[i] == '/') {
			DirName[i] = 0;
			if (access(DirName, 0) != 0) {
				CHECK(mkdir(DirName, 0755) == 0)<< "Failed to create folder "<< sPathName;
			}
			DirName[i] = '/';
		}
	}

	return 0;
}

void imagesc(Mat& input);

char buf[101000];
int main(int argc, char** argv)
{

	//cudaSetDevice(0);
	Caffe::set_phase(Caffe::TEST);
	//Caffe::set_mode(Caffe::CPU);

	if (argc == 8 && strcmp(argv[7], "CPU") == 0) {
		LOG(ERROR) << "Using CPU";
		Caffe::set_mode(Caffe::CPU);
	} else {
		LOG(ERROR) << "Using GPU";
		Caffe::set_mode(Caffe::GPU);
	}
	Caffe::SetDevice(1);

	//Caffe::set_mode(Caffe::CPU);
	NetParameter test_net_param;
	ReadProtoFromTextFile(argv[1], &test_net_param);
	Net<float> caffe_test_net(test_net_param);
	NetParameter trained_net_param;
	ReadProtoFromBinaryFile(argv[2], &trained_net_param);
	caffe_test_net.CopyTrainedLayersFrom(trained_net_param);

	vector<shared_ptr<Layer<float> > > layers = caffe_test_net.layers();
	const DataLayer<float> *datalayer = dynamic_cast<const DataLayer<float>* >(layers[0].get());
	CHECK(datalayer);
    
	string labelFile(argv[3]);
	int data_counts = 0;
	FILE * file = fopen(labelFile.c_str(), "r");
	while(fgets(buf,100000,file) > 0)
	{
		data_counts++;
	}
	fclose(file);

	vector<Blob<float>*> dummy_blob_input_vec;
	string rootfolder(argv[4]);
	rootfolder.append("/");
	CreateDir(rootfolder.c_str(), rootfolder.size() - 1);
	string folder;
	string fName;

	int counts = 0;

	file = fopen(labelFile.c_str(), "r");

	Blob<float>* c1 = (*(caffe_test_net.bottom_vecs().rbegin()))[0];
    int c2 = c1->num();
	int batchCount = std::ceil(data_counts / (floor)(c2));//(test_net_param.layers(0).layer().batchsize()));//                (test_net_param.layers(0).layer().batchsize() ));

	string resulttxt = rootfolder + "3dNormalResult.txt";
	string gttxt = rootfolder + "gt.txt";
	FILE * resultfile = fopen(resulttxt.c_str(), "w");

	for (int batch_id = 0; batch_id < batchCount; ++batch_id)
	{
		LOG(INFO)<< "processing batch :" << batch_id+1 << "/" << batchCount <<"...";

		const vector<Blob<float>*>& result = caffe_test_net.Forward(dummy_blob_input_vec);
		Blob<float>* bboxs = (*(caffe_test_net.bottom_vecs().rbegin()))[0];
		int bsize = bboxs->num();

		const Blob<float>* labels = (*(caffe_test_net.bottom_vecs().rbegin()))[1];
		for (int i = 0; i < bsize && counts < data_counts; i++, counts++)
		{
			char fname[1010];
			fscanf(file, "%s", fname);
			/*for(int w = 0; w < LABEL_SIZE; w ++)
				for(int h = 0; h < LABEL_SIZE; h ++)
				{
					int lbl;
					fscanf(file,"%d",&lbl);
				}*/
//			fprintf(resultfile, "%s ", fname);
			int len = (74 * 55);
            Mat output(55, 74, CV_32FC1);
			for(int j = 0; j < len; j ++)
			{
				fprintf(resultfile, "%f ", (float)(bboxs->data_at(i, j, 0, 0)) );
#ifdef GEN_VIS
                output.at<float>(Point(j % 74, j / 74)) = ((float) bboxs->data_at(i, j, 0, 0));
#endif
			}
#ifdef GEN_VIS
            imagesc(output);
            imwrite((OUTPUT_DIR + string(fname) + ".jpg").c_str(), output);
#endif
			fprintf(resultfile, "\n");

#ifdef GEN_VIS
            Mat gt(55, 74, CV_32FC1);
			for(int j = 0; j < len; j ++)
			{
                gt.at<float>(Point(j % 74, j / 74)) = ((float) labels->data_at(i, j, 0, 0));
			}
            imagesc(gt);
            imwrite((GT_DIR + string(fname) + ".jpg").c_str(), gt);
#endif

		}
	}

	fclose(resultfile);
	fclose(file);


	return 0;
}

void imagesc(Mat& input) {
    double min;
    double max;
    minMaxIdx(input, &min, &max);
    Mat adjMap;
    // Histogram Equalization
    float scale = 255 / (max-min);
    input.convertTo(adjMap, CV_8UC1, scale, -min*scale);


    input = adjMap;
}

