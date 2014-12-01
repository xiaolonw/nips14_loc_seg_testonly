#!/usr/bin/env sh

rootfolder=/exports/cyclops/work/001_Selfies/002_Segmentation/src/Caffe_Segmentation

GLOG_logtostderr=1 $rootfolder/build/examples/seg/convert_normal_test.bin /exports/cyclops/work/003_Backpage/dataset/backpage/corpus/ /exports/cyclops/work/003_Backpage/dataset/backpage/TrainSet_1K.txt /exports/cyclops/work/003_Backpage/dataset/backpage/leveldb/ImagesNevada 0 0 227 227
