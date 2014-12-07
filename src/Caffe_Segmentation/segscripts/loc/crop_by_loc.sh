#!/bin/bash

rootfolder=/exports/cyclops/work/001_Selfies/002_Segmentation/src/Caffe_Segmentation

GLOG_logtostderr=1 $rootfolder/build/examples/seg/cropDet.bin \
    /exports/cyclops/work/001_Selfies/002_Segmentation/src/Caffe_Segmentation/results/backpage/ImagesNevada/loc/locResult.txt \
    /exports/cyclops/work/003_Backpage/dataset/backpage/corpus \
    /exports/cyclops/work/001_Selfies/002_Segmentation/src/Caffe_Segmentation/results/backpage/ImagesNevada/crops

