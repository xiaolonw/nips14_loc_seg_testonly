#!/usr/bin/env sh                                                                                                
GLOG_logtostderr=1  $CAFFEROOT/build/tools/test_net_loc.bin \
    $SEGSCRDIR/loc/imagenet_test.prototxt  \
    $SEGSCRDIR/models/loc.caffemodel \
    $SEGSCRDIR/data/ImgsList.txt \
    $SEGSCRDIR/data \
    CPU

