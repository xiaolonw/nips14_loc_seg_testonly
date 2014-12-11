mkdir -p models
cd models

wget https://www.dropbox.com/sh/t851g3hxzgisgit/AABNm6PjYi5rvrr8ADOUufX7a/seg.caffemodel
wget https://www.dropbox.com/sh/t851g3hxzgisgit/AAAG5OXDbnFmD1cXNWsrxQyFa/loc.caffemodel
wget https://www.dropbox.com/sh/t851g3hxzgisgit/AAC7dCiBM31IdM2ir83rZo67a/seg_mean.binaryproto
wget https://www.dropbox.com/sh/t851g3hxzgisgit/AAD3YBBXEcUHRrGvGCwq03IFa/imagenet_mean.binaryproto loc_mean.binaryproto

