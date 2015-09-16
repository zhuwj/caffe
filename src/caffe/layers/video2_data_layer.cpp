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
Video2DataLayer<Dtype>:: ~Video2DataLayer<Dtype>(){
	this->JoinPrefetchThread();
}

template <typename Dtype>
void Video2DataLayer<Dtype>:: DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const int new_height  = this->layer_param_.video2_data_param().new_height();
	const int new_width  = this->layer_param_.video2_data_param().new_width();
	const int new_length_rgb  = this->layer_param_.video2_data_param().new_length_rgb();
	const int new_length_flow = this->layer_param_.video2_data_param().new_length_flow();
	const int new_length_max = std::max(new_length_rgb, new_length_flow);
	const int num_segments = this->layer_param_.video2_data_param().num_segments();
	const string& source = this->layer_param_.video2_data_param().source();
	string root_folder_rgb = this->layer_param_.video2_data_param().root_folder_rgb();        
	string root_folder_flow = this->layer_param_.video2_data_param().root_folder_flow();
	const bool flow_is_color = this->layer_param_.video2_data_param().flow_is_color();

	LOG(INFO) << "Opening file: " << source;
	std:: ifstream infile(source.c_str());
	string filename;
	int label;
	int length;
	while (infile >> filename >> length >> label){
		lines_.push_back(std::make_pair(filename,label));
		lines_duration_.push_back(length);
	}
	if (this->layer_param_.video2_data_param().shuffle()){
		const unsigned int prefectch_rng_seed = caffe_rng_rand();
		prefetch_rng_1_.reset(new Caffe::RNG(prefectch_rng_seed));
		prefetch_rng_2_.reset(new Caffe::RNG(prefectch_rng_seed));
		ShuffleVideos();
	}

	LOG(INFO) << "A total of " << lines_.size() << " videos.";
	lines_id_ = 0;

	const unsigned int frame_prefectch_rng_seed = caffe_rng_rand();
	frame_prefetch_rng_.reset(new Caffe::RNG(frame_prefectch_rng_seed));
	int average_duration = (int) lines_duration_[lines_id_]/num_segments;

	vector<int> offsets_rgb, offsets_flow;	
	for (int i = 0; i < num_segments; ++i){
		caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
		int offset = (*frame_rng)() % (average_duration - new_length_max + 1);
		offsets_rgb.push_back(offset+i*average_duration + new_length_max / 2 - new_length_rgb / 2);
		offsets_flow.push_back(offset + i*average_duration + new_length_max / 2 - new_length_flow / 2);
	}

	Datum datum_flow, datum_rgb;
	CHECK(ReadSegmentRGBToDatum(root_folder_rgb + lines_[lines_id_].first, lines_[lines_id_].second, offsets_rgb, new_height, new_width, new_length_rgb, &datum_rgb, true));
	CHECK(ReadSegmentFlowToDatum(root_folder_flow + lines_[lines_id_].first, lines_[lines_id_].second, offsets_flow, new_height, new_width, new_length_flow, &datum_flow, flow_is_color));

	const int crop_size = this->layer_param_.transform_param().crop_size();
	const int batch_size = this->layer_param_.video2_data_param().batch_size();
	if (crop_size > 0){
		top[0]->Reshape(batch_size, datum_rgb.channels(), crop_size, crop_size);
		this->prefetch_data_rgb_.Reshape(batch_size, datum_rgb.channels(), crop_size, crop_size);
		top[1]->Reshape(batch_size, datum_flow.channels(), crop_size, crop_size);
		this->prefetch_data_flow_.Reshape(batch_size, datum_flow.channels(), crop_size, crop_size);
	} else {
		top[0]->Reshape(batch_size, datum_rgb.channels(), datum_rgb.height(), datum_rgb.width());
		this->prefetch_data_rgb_.Reshape(batch_size, datum_rgb.channels(), datum_rgb.height(), datum_rgb.width());
		top[1]->Reshape(batch_size, datum_flow.channels(), datum_flow.height(), datum_flow.width());
		this->prefetch_data_flow_.Reshape(batch_size, datum_flow.channels(), datum_flow.height(), datum_flow.width());
	}
	LOG(INFO) << "rgb output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();
	LOG(INFO) << "flow output data size: " << top[1]->num() << "," << top[1]->channels() << "," << top[1]->height() << "," << top[1]->width();

	top[2]->Reshape(batch_size, 1, 1, 1);
	this->prefetch_label_.Reshape(batch_size, 1, 1, 1);

	this->transformed_data_rgb_.Reshape(this->data_transformer_->InferBlobShape(datum_rgb));
	this->transformed_data_flow_.Reshape(this->data_transformer_->InferBlobShape(datum_flow));

}

template <typename Dtype>
void Video2DataLayer<Dtype>::ShuffleVideos(){
	caffe::rng_t* prefetch_rng1 = static_cast<caffe::rng_t*>(prefetch_rng_1_->generator());
	caffe::rng_t* prefetch_rng2 = static_cast<caffe::rng_t*>(prefetch_rng_2_->generator());
	shuffle(lines_.begin(), lines_.end(), prefetch_rng1);
	shuffle(lines_duration_.begin(), lines_duration_.end(),prefetch_rng2);
}

template <typename Dtype>
void Video2DataLayer<Dtype>::InternalThreadEntry(){
	CHECK(this->prefetch_data_rgb_.count());
	CHECK(this->prefetch_data_flow_.count());
	Dtype* top_data_rgb = this->prefetch_data_rgb_.mutable_cpu_data();
	Dtype* top_data_flow = this->prefetch_data_flow_.mutable_cpu_data();
	Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
	Video2DataParameter video2_data_param = this->layer_param_.video2_data_param();
	const int batch_size = video2_data_param.batch_size();
	const int new_height = video2_data_param.new_height();
	const int new_width = video2_data_param.new_width();
	CHECK(new_height > 0 && new_width > 0) << "size_resize should be set";
	const int new_length_rgb = video2_data_param.new_length_rgb();
	const int new_length_flow = video2_data_param.new_length_flow();
	const int new_length_max = std::max(new_length_rgb, new_length_flow);
	const int num_segments = video2_data_param.num_segments();
	string root_folder_rgb = video2_data_param.root_folder_rgb();
	string root_folder_flow = video2_data_param.root_folder_flow();
	const bool flow_is_color = this->layer_param_.video2_data_param().flow_is_color();
	const int lines_size = lines_.size();

	CHECK(lines_size >= batch_size) << "too small number of data, will cause problem in parallel reading";

	CPUTimer timer;
	timer.Start();
#pragma omp parallel for   
	for (int item_id = 0; item_id < batch_size; ++item_id){
		const int lines_id_loc = (lines_id_ + item_id) % lines_size;
		CHECK_GT(lines_size, lines_id_loc);		
		CHECK_GT(lines_duration_[lines_id_loc], 0) << "0 duration for video" << lines_[lines_id_loc].first;

		int average_duration = (int) lines_duration_[lines_id_loc] / num_segments;
		CHECK(average_duration - new_length_max >= 0) << "average_duration should be larger than new_length_max (" << average_duration <<  " v.s. " << new_length_max << ")";		
		const int chn_flow_single = flow_is_color ? 3 : 1;
		vector<int> offsets_rgb, offsets_flow;
		for (int i = 0; i < num_segments; ++i){
			int offset;
			if (this->phase_==TRAIN){
				caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());				
				offset = (*frame_rng)() % (average_duration - new_length_max + 1);
			} else{
				offset = int((average_duration-new_length_max+1) / 2);
			}
			offsets_rgb.push_back(offset + i*average_duration + new_length_max / 2 - new_length_rgb / 2);
			offsets_flow.push_back(offset + i*average_duration + new_length_max / 2 - new_length_flow / 2);
		}
		Datum datum_flow, datum_rgb;
		CHECK(ReadSegmentRGBToDatum(root_folder_rgb + lines_[lines_id_loc].first, lines_[lines_id_loc].second, offsets_rgb, new_height, new_width, new_length_rgb, &datum_rgb, true));
		CHECK(ReadSegmentFlowToDatum(root_folder_flow + lines_[lines_id_loc].first, lines_[lines_id_loc].second, offsets_flow, new_height, new_width, new_length_flow, &datum_flow, flow_is_color));

		int offset1 = this->prefetch_data_rgb_.offset(item_id);
		Blob<Dtype> transformed_data_rgb_loc;
		vector<int> top_shape = this->data_transformer_->InferBlobShape(datum_rgb);
		transformed_data_rgb_loc.Reshape(top_shape);
		transformed_data_rgb_loc.set_cpu_data(top_data_rgb + offset1);
		//this->data_transformer_->Transform(datum_rgb, &(transformed_data_rgb_loc));

		offset1 = this->prefetch_data_flow_.offset(item_id);
		Blob<Dtype> transformed_data_flow_loc;
		top_shape = this->data_transformer_->InferBlobShape(datum_flow);
		transformed_data_flow_loc.Reshape(top_shape);
		transformed_data_flow_loc.set_cpu_data(top_data_flow + offset1);
		//this->data_transformer_->Transform(datum_flow, &(transformed_data_flow_loc), chn_flow_single);
		this->data_transformer_->Transform(datum_rgb, datum_flow, &(transformed_data_rgb_loc), &(transformed_data_flow_loc), chn_flow_single);

		top_label[item_id] = lines_[lines_id_loc].second;
	}
//	printf("read images cost %.f ms\n", timer.MicroSeconds()/1000);

	//next iteration
	lines_id_ += batch_size;
	if (lines_id_ >= lines_size) {
		DLOG(INFO) << "Restarting data prefetching from start.";
		lines_id_ = 0;
		if(this->layer_param_.video2_data_param().shuffle()){
			ShuffleVideos();
		}
	}
}

INSTANTIATE_CLASS(Video2DataLayer);
REGISTER_LAYER_CLASS(Video2Data);
}
