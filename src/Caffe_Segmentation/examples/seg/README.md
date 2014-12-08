Steps to run
------------

```bash
$ cd $CAFFEROOT/segscripts
```

1. Localization
    
    ```bash
    $ cd loc
    $ bash convert_loc_test.sh # to generate `leveldb` data for segmentation
    $ bash test_loc.sh # to run the localization on the images
    $ bash crop_by_loc.sh # to crop the images
    ```

2. Segmentation

    ```bash
    $ cd ../seg
    $ bash convert_seg_test.sh
    $ bash test_seg.sh
    $ bash gen_results.sh
    ```

