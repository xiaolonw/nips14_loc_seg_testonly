#!/usr/bin/env sh

GLOG_logtostderr=1 $CAFFEROOT/build/examples/seg/convert_normal_test_seg.bin \
    $SEGSCRDIR/data/corpus \
    $SEGSCRDIR/data/ImgsList.txt  \
    $SEGSCRDIR/data/seg_leveldb \
    0 0 256 256
