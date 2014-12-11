mkdir -p models
cd models

wget /path/to/seg.caffemodel models/seg.caffemodel
wget /path/to/loc.caffemodel models/loc.caffemodel
wget /path/to/seg_mean.binaryproto models/seg_mean.binaryproto
wget /path/to/loc_mean.binaryproto models/loc_mean.binaryproto

