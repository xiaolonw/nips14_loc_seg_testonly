Deep Learning for Segmentation (Testing Code)
============================================

Based on        
**Xiaolong Wang**, Liliang Zhang, Liang Lin, Zhujin Liang, and Wangmeng Zuo. Joint Task Learning via Deep Neural Networks with Application to Generic Object Extraction. Proc. of Advances in Neural Information Processing Systems (NIPS), 2014

Forked from [this](https://github.com/BVLC/caffe/tree/c18d22eb92488f02c0256a3fe4ac20a8ad827596) 
commit of [Caffe](https://github.com/BVLC/caffe).

Demo
----

This code comes with some sample images for a demo. Copy the `segscripts` directory
and change paths to data etc in bash scripts and prototxts to run for your own data.

To run the provided demo

```bash
$ cd segscripts
$ bash get_seg_loc_models.sh # downloads models from internet
$ bash run_seg.sh ../ # runs the segmentation. The argument is the path to CAFFE
```

Generates segmentaion of images in `segscripts/data/corpus` into `segscripts/data/final_segmentations`.

Downloads
----

The models trained on the training set of [Object Extraction dataset](http://objectextraction.github.io/) are available to be downloaded from dropbox via the script "get_seg_loc_models.sh". If dropbox is blocked in your district, you can try to download from[ Baidu Cloud ](http://pan.baidu.com/s/1sjyk2Wt)




