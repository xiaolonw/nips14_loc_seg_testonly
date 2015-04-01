#include <opencv2/opencv.hpp>

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
#define LABEL_LEN 1
#define LABEL_SIZE 4

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



char buf[101000];
int main(int argc, char** argv)
{

    Caffe::set_phase(Caffe::TEST);

    if (argc == 6 && strcmp(argv[5], "CPU") == 0) {
        LOG(ERROR) << "Using CPU";
        Caffe::set_mode(Caffe::CPU);
    } else {
        LOG(ERROR) << "Using GPU";
        Caffe::set_mode(Caffe::GPU);
    }

    NetParameter test_net_param;
    ReadProtoFromTextFile(argv[1], &test_net_param);
    Net<float> caffe_test_net(test_net_param);
    NetParameter trained_net_param;
    ReadProtoFromBinaryFile(argv[2], &trained_net_param);
    caffe_test_net.CopyTrainedLayersFrom(trained_net_param);

    vector<shared_ptr<Layer<float> > > layers = caffe_test_net.layers();

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

    boost::shared_ptr<Blob<float> > c1 = caffe_test_net.blob_by_name("data");
    int c2 = c1->num();
    int batchCount = std::ceil(data_counts / (floor)(c2));

    string resulttxt = rootfolder + "locResult.txt";
    FILE * resultfile = fopen(resulttxt.c_str(), "w");

    for (int batch_id = 0; batch_id < batchCount; ++batch_id)
    {
        LOG(INFO)<< "processing batch :" << batch_id+1 << "/" << batchCount <<"...";

        /* const vector<Blob<float>*>& result = */ caffe_test_net.Forward(dummy_blob_input_vec);
        boost::shared_ptr<Blob<float> > bboxs = caffe_test_net.blob_by_name("fc8_loc");
        int bsize = bboxs->num();

        //labels needed if you want  to do evaluations
        //const Blob<float>* labels = (*(caffe_test_net.bottom_vecs().rbegin()))[1];
        for (int i = 0; i < bsize && counts < data_counts; i++, counts++)
        {
            char fname[1010];
            fscanf(file, "%s", fname);
            fprintf(resultfile, "%s ", fname);
            int len = LABEL_SIZE * LABEL_LEN;
            for(int j = 0; j < len; j ++)
            {
                fprintf(resultfile, "%f ", (float)(bboxs->data_at(i, j, 0, 0)) );
            }
            fprintf(resultfile, "\n");
        }
    }

    fclose(resultfile);
    fclose(file);
    return 0;
}

