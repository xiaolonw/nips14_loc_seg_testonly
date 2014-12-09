#!/usr/bin/env sh                                                                                                

# test_net_seg.bin test_proto pre_train_model label.txt outputfolder [CPU/GPU]

ROOTFILE=/exports/cyclops/work/001_Selfies/002_Segmentation/src/Caffe_Segmentation

GLOG_logtostderr=1  $ROOTFILE/build/tools/test_net_3dnormal_seg.bin  \
    $ROOTFILE/segscripts/seg/seg_test.prototxt  \
    $ROOTFILE/data/seg/seg__iter_200000  \
    /exports/cyclops/work/003_Backpage/dataset/backpage/TrainSet_10K.txt \
    /exports/cyclops/work/001_Selfies/002_Segmentation/src/Caffe_Segmentation/results/backpage/ImagesNevada/segs

