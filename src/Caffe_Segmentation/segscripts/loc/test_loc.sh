#!/usr/bin/env sh                                                                                                

# test_net_seg.bin test_proto pre_train_model label.txt outputfolder [CPU/GPU]

ROOTFILE=/exports/cyclops/work/001_Selfies/002_Segmentation/src/Caffe_Segmentation

GLOG_logtostderr=1  $ROOTFILE/build/tools/test_net_loc.bin \
    $ROOTFILE/segscripts/loc/imagenet_test.prototxt  \
    $ROOTFILE/data/seg/loc.caffemodel \
    /exports/cyclops/work/003_Backpage/dataset/backpage/TrainSet_10K.txt \
    $ROOTFILE/results/backpage/ImagesNevada/loc \
    CPU


