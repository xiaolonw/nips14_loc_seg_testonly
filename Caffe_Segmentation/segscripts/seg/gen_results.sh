#!/usr/bin/env sh                                                                                                
GLOG_logtostderr=1  $CAFFEROOT/build/examples/seg/dumpRes.bin  \
    $SEGSCRDIR/data/segResult.txt \
    $SEGSCRDIR/data/seg_imgs \
    50

GLOG_logtostderr=1  $CAFFEROOT/build/examples/seg/createFinalRes.bin  \
    $SEGSCRDIR/data/locResult.txt \
    $SEGSCRDIR/data/seg_imgs \
    $SEGSCRDIR/data/final_segmentations

