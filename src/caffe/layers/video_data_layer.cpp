#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/benchmark.hpp"
#include <omp.h>

#ifdef USE_MPI
#include "mpi.h"
#include <boost/filesystem.hpp>
using namespace boost::filesystem;
#endif

namespace caffe{
template <typename Dtype>
VideoDataLayer<Dtype>:: ~VideoDataLayer<Dtype>(){
	this->JoinPrefetchThread();
}

template <typename Dtype>
void VideoDataLayer<Dtype>:: DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const int new_height  = this->layer_param_.video_data_param().new_height();
	const int new_width  = this->layer_param_.video_data_param().new_width();
	const int new_length  = this->layer_param_.video_data_param().new_length();
	const int num_segments = this->layer_param_.video_data_param().num_segments();
	const string& source = this->layer_param_.video_data_param().source();
	string root_folder = this->layer_param_.video_data_param().root_folder();
	const bool flow_is_color = this->layer_param_.video_data_param().flow_is_color();

	LOG(INFO) << "Opening file: " << source;
	std:: ifstream infile(source.c_str());
	string filename;
	int label;
	int length;
	while (infile >> filename >> length >> label){
		lines_.push_back(std::make_pair(filename,label));
		lines_duration_.push_back(length);
	}
	if (this->layer_param_.video_data_param().shuffle()){
		const unsigned int prefectch_rng_seed = caffe_rng_rand();
		prefetch_rng_1_.reset(new Caffe::RNG(prefectch_rng_seed));
		prefetch_rng_2_.reset(new Caffe::RNG(prefectch_rng_seed));
		ShuffleVideos();
	}

	LOG(INFO) << "A total of " << lines_.size() << " videos.";
	lines_id_ = 0;

	//if (this->phase_==TRAIN)
		CHECK(num_segments == 1) << "num_segments should be 1 for training";
	
	const int len_vid = int(lines_duration_[lines_id_]);
	CHECK_GE(len_vid, new_length) << "video length should be larger than new_length";

	vector<int> offsets(1, 0);
	Datum datum;
	if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_FLOW)
		CHECK(ReadSegmentFlowToDatum(root_folder + lines_[lines_id_].first, lines_[lines_id_].second, offsets, new_height, new_width, new_length, &datum, flow_is_color));
	else
		CHECK(ReadSegmentRGBToDatum(root_folder + lines_[lines_id_].first, lines_[lines_id_].second, offsets, new_height, new_width, new_length, &datum, true));
	const int crop_size = this->layer_param_.transform_param().crop_size();
	const int batch_size = this->layer_param_.video_data_param().batch_size();
	if (crop_size > 0){
		top[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
		this->prefetch_data_.Reshape(batch_size, datum.channels(), crop_size, crop_size);
	} else {
		top[0]->Reshape(batch_size, datum.channels(), datum.height(), datum.width());
		this->prefetch_data_.Reshape(batch_size, datum.channels(), datum.height(), datum.width());
	}
	LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();

	top[1]->Reshape(batch_size, 1, 1, 1);
	this->prefetch_label_.Reshape(batch_size, 1, 1, 1);

	vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
	this->transformed_data_.Reshape(top_shape);
}

template <typename Dtype>
void VideoDataLayer<Dtype>::ShuffleVideos(){
	caffe::rng_t* prefetch_rng1 = static_cast<caffe::rng_t*>(prefetch_rng_1_->generator());
	caffe::rng_t* prefetch_rng2 = static_cast<caffe::rng_t*>(prefetch_rng_2_->generator());
	shuffle(lines_.begin(), lines_.end(), prefetch_rng1);
	shuffle(lines_duration_.begin(), lines_duration_.end(),prefetch_rng2);
}


inline void Linspace(const int s, const int e, const int num, vector<int> &ids)
{
	CHECK(e >= s);
	ids.clear();
	if (1==num)
		ids.push_back((s+e)/2);
	else
	{
		float step = float(e - s) / (num-1);
		for (int i = 0; i < num; i++)
			ids.push_back(int(s + step * i));
	}
        
}

template <typename Dtype>
void VideoDataLayer<Dtype>::InternalThreadEntry(){

	CHECK(this->prefetch_data_.count());
	Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
	Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
	VideoDataParameter video_data_param = this->layer_param_.video_data_param();
	const int batch_size = video_data_param.batch_size();
	const int new_height = video_data_param.new_height();
	const int new_width = video_data_param.new_width();
	CHECK(new_height > 0 && new_width > 0) << "size_resize should be set";
	const int new_length = video_data_param.new_length();
	const int num_segments = video_data_param.num_segments();
	string root_folder = video_data_param.root_folder();
	const bool flow_is_color = this->layer_param_.video_data_param().flow_is_color();
	const int lines_size = lines_.size();

	CHECK(lines_size >= batch_size) << "too small number of data, will cause problem in parallel reading";

	CPUTimer timer;
	timer.Start();
	omp_set_num_threads(8);
#pragma omp parallel for   
	for (int item_id = 0; item_id < batch_size; ++item_id){		
		Datum datum;
		const int lines_id_loc = (lines_id_ + item_id) % lines_size;
		CHECK_GT(lines_size, lines_id_loc);
		const int len_vid = lines_duration_[lines_id_loc];
		CHECK_GT(len_vid, new_length) << "video length should be larger than new_length";

		vector<int> offsets;
		for (int i = 0; i < num_segments; ++i){
			if (this->phase_==TRAIN){
				caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());				
				int offset = (*frame_rng)() % (len_vid - new_length + 1);
				offsets.push_back(offset);
			} else{
				Linspace(0, len_vid-new_length, num_segments, offsets);
			}
		}
		
		if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_FLOW){
			if(!ReadSegmentFlowToDatum(root_folder + lines_[lines_id_loc].first, lines_[lines_id_loc].second, offsets, new_height, new_width, new_length, &datum, flow_is_color))				
				continue;
		} else{		
			if(!ReadSegmentRGBToDatum(root_folder + lines_[lines_id_loc].first, lines_[lines_id_loc].second, offsets, new_height, new_width, new_length, &datum, true))
				continue;
		}

		int offset1 = this->prefetch_data_.offset(item_id);

		Blob<Dtype> transformed_data_loc;
		vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
		transformed_data_loc.Reshape(top_shape);
		transformed_data_loc.set_cpu_data(top_data + offset1);
		const int chn_flow_single = flow_is_color ? 3 : 1;
		this->data_transformer_->Transform(datum, &(transformed_data_loc), chn_flow_single);

		top_label[item_id] = lines_[lines_id_loc].second;
	}

	//next iteration
	lines_id_ += batch_size;
	if (lines_id_ >= lines_size) {
		DLOG(INFO) << "Restarting data prefetching from start.";
		lines_id_ = 0;
		if(this->layer_param_.video_data_param().shuffle()){
			ShuffleVideos();
		}
	}

}

INSTANTIATE_CLASS(VideoDataLayer);
REGISTER_LAYER_CLASS(VideoData);
}
