#!/bin/bash

GLOG_logtostderr=1 $CAFFEROOT/build/examples/seg/cropDet.bin \
    $SEGSCRDIR/data/locResult.txt \
    $SEGSCRDIR/data/corpus \
    $SEGSCRDIR/data/crops

