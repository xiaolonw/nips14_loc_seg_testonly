if [ $# -lt 1 ]; then
    echo 'Usage: ' $0 ' <CAFFE_PATH>'
    exit
fi
export CAFFEROOT=$1
export SEGSCRDIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

echo 'Setting up data for localization'
rm -rf $SEGSCRDIR/data/loc_leveldb
bash loc/convert_loc_test.sh $SEGSCRDIR/data

echo 'Running localization'
bash loc/test_loc.sh

echo 'Cropping based on localization'
bash loc/crop_by_loc.sh

echo 'Create leveldb for segmentaion'
rm -rf $SEGSCRDIR/data/seg_leveldb
bash seg/convert_seg_test.sh

echo 'Running Segmentation'
bash seg/test_seg.sh

echo 'Generating final results'
bash seg/gen_results.sh

