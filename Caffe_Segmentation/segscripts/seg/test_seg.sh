#!/usr/bin/env sh                                                                                                
GLOG_logtostderr=1  $CAFFEROOT/build/tools/test_net_3dnormal_seg.bin  \
    $SEGSCRDIR/seg/seg_test.prototxt  \
    $SEGSCRDIR/models/seg.caffemodel \
    $SEGSCRDIR/data/ImgsList.txt \
    $SEGSCRDIR/data/

