#!/usr/bin/env sh

rootfolder=/exports/cyclops/work/001_Selfies/002_Segmentation/src/Caffe_Segmentation

GLOG_logtostderr=1 $rootfolder/build/examples/seg/convert_normal_test_seg.bin \
    $rootfolder/results/backpage/ImagesNevada/crops/ \
    /exports/cyclops/work/003_Backpage/dataset/backpage/TrainSet_10K.txt  \
    /exports/cyclops/work/001_Selfies/002_Segmentation/src/Caffe_Segmentation/results/backpage/ImagesNevada/cropped_leveldb \
    0 0 55 55
