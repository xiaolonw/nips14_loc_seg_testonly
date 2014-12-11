#!/usr/bin/env sh

GLOG_logtostderr=1 $CAFFEROOT/build/examples/seg/convert_normal_test_loc.bin \
    $SEGSCRDIR/data/corpus/ \
    $SEGSCRDIR/data/ImgsList.txt \
    $SEGSCRDIR/data/loc_leveldb 0 0 227 227

