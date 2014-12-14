#include <string>

#include "caffe/data_transformer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <fstream>
#include <iostream>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

namespace caffe {

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const int batch_item_id,
		const Datum& datum, const Dtype* mean, Dtype* transformed_data) {
	const string& data = datum.data();
	const int channels = datum.channels();
	const int height = datum.height();
	const int width = datum.width();
	const int size = datum.channels() * datum.height() * datum.width();

	const int crop_size = param_.crop_size();
	const bool mirror = param_.mirror();
	const Dtype scale = param_.scale();

	if (mirror && crop_size == 0) {
		LOG(FATAL)<< "Current implementation requires mirror and crop_size to be "
		<< "set at the same time.";
	}

	if (crop_size) {
		CHECK(data.size()) << "Image cropping only support uint8 data";
		int h_off, w_off;
		// We only do random crop when we do training.
		if (phase_ == Caffe::TRAIN) {
			h_off = Rand() % (height - crop_size);
			w_off = Rand() % (width - crop_size);
		} else {
			h_off = (height - crop_size) / 2;
			w_off = (width - crop_size) / 2;
		}
		if (mirror && Rand() % 2) {
			// Copy mirrored version
			for (int c = 0; c < channels; ++c) {
				for (int h = 0; h < crop_size; ++h) {
					for (int w = 0; w < crop_size; ++w) {
						int data_index = (c * height + h + h_off) * width + w
								+ w_off;
						int top_index = ((batch_item_id * channels + c)
								* crop_size + h) * crop_size
								+ (crop_size - 1 - w);
						Dtype datum_element =
								static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
						transformed_data[top_index] = (datum_element
								- mean[data_index]) * scale;
					}
				}
			}
		} else {
			// Normal copy
			for (int c = 0; c < channels; ++c) {
				for (int h = 0; h < crop_size; ++h) {
					for (int w = 0; w < crop_size; ++w) {
						int top_index = ((batch_item_id * channels + c)
								* crop_size + h) * crop_size + w;
						int data_index = (c * height + h + h_off) * width + w
								+ w_off;
						Dtype datum_element =
								static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
						transformed_data[top_index] = (datum_element
								- mean[data_index]) * scale;
					}
				}
			}
		}
	} else {
		// we will prefer to use data() first, and then try float_data()
		if (data.size()) {
			for (int j = 0; j < size; ++j) {
				Dtype datum_element =
						static_cast<Dtype>(static_cast<uint8_t>(data[j]));
				transformed_data[j + batch_item_id * size] = (datum_element
						- mean[j]) * scale;
			}
		} else {
			for (int j = 0; j < size; ++j) {
				transformed_data[j + batch_item_id * size] = (datum.float_data(
						j) - mean[j]) * scale;
			}
		}
	}
}

#define OFFSET ((256-227)/2)
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const int idx, const int batch_item_id,
		const Datum& datum, const Dtype* mean, Dtype* transformed_data) {
	const string& data = datum.data();
	const int channels = datum.channels();
	const int height = datum.height();
	const int width = datum.width();
	//const int size = datum.channels() * datum.height() * datum.width();

	const int crop_size = param_.crop_size();
	const bool mirror = param_.mirror();
	const Dtype scale = param_.scale();

	if (mirror && crop_size == 0) {
		LOG(FATAL)<< "Current implementation requires mirror and crop_size to be "
		<< "set at the same time.";
	}

	// 坐标映射回原图
	int x1 = MIN(width-1, MAX(0, loc_result_[idx][0] + OFFSET));
	int y1 = MIN(height-1, MAX(0, loc_result_[idx][1] + OFFSET));
	int x2 = MIN(width-1, MAX(0, loc_result_[idx][2] + OFFSET));
	int y2 = MIN(height-1, MAX(0, loc_result_[idx][3] + OFFSET));
	//LOG(INFO)<<x1<<" "<<y1<<" "<<x2<<" "<<y2;

	// 先读取图片数据
	Mat im_resize(crop_size, crop_size, CV_8UC3);
	{
		Mat im_pred(MAX(1, MIN(height, y2 - y1 + 1)),
				MAX(1, MIN(width, x2 - x1 + 1)), CV_8UC3);
		for (int c = 0; c < channels; c++) {
			for (int h = y1, h_i = 0; h <= y2 && h < height; h++, h_i++) {
				for (int w = x1, w_i = 0; w <= x2 && w < width; w++, w_i++) {
					im_pred.at<Vec3b>(h_i, w_i)[c] = (uint8_t) data[(c * height
							+ h) * width + w];
				}
			}
		}
		resize(im_pred, im_resize, Size(crop_size, crop_size));
	}

	// 读取相应的mean
	Mat im_mean_resize(crop_size, crop_size, CV_8UC3);
	{
		Mat im_mean(MAX(1, MIN(height, y2 - y1 + 1)),
				MAX(1, MIN(width, x2 - x1 + 1)), CV_8UC3);
		for (int c = 0; c < channels; c++) {
			for (int h = y1, h_i = 0; h <= y2 && h < height; h++, h_i++) {
				for (int w = x1, w_i = 0; w <= x2 && w < width; w++, w_i++) {
					im_mean.at<Vec3b>(h_i, w_i)[c] = (uint8_t) mean[(c * height
							+ h) * width + w];
				}
			}
		}
		resize(im_mean, im_mean_resize, Size(crop_size, crop_size));
	}

	for (int c = 0; c < channels; ++c) {
		for (int h = 0; h < crop_size; ++h) {
			for (int w = 0; w < crop_size; ++w) {
				transformed_data[((batch_item_id * channels + c) * crop_size + h) * crop_size
						+ crop_size - 1 - w] = (static_cast<Dtype>(im_resize.at<
						Vec3b>(h, w)[c] - im_mean_resize.at<Vec3b>(h, w)[c]))
						* scale;
			}
		}
	}
}

template<typename Dtype>
void DataTransformer<Dtype>::InitRand() {
	const bool needs_rand = (phase_ == Caffe::TRAIN)
			&& (param_.mirror() || param_.crop_size());
	if (needs_rand) {
		const unsigned int rng_seed = caffe_rng_rand();
		rng_.reset(new Caffe::RNG(rng_seed));
	} else {
		rng_.reset();
	}
}

template<typename Dtype>
unsigned int DataTransformer<Dtype>::Rand() {
	CHECK(rng_);
	caffe::rng_t* rng = static_cast<caffe::rng_t*>(rng_->generator());
	return (*rng)();
}

template<typename Dtype>
void DataTransformer<Dtype>::SetUpLocResultFromText() {
	ifstream infile(param_.loc_result().c_str());
	if (!infile.is_open()) {
		LOG(ERROR)<< "Unable to open file: " << param_.loc_result();
		return;
	}
	string fname;
	Dtype xmin, xmax, ymin, ymax;
	while (infile >> fname >> xmin >> ymin >> xmax >> ymax) {
		vector<Dtype> bbox;
		bbox.push_back(xmin);
		bbox.push_back(ymin);
		bbox.push_back(xmax);
		bbox.push_back(ymax);
		loc_result_.push_back(bbox);
	}
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
