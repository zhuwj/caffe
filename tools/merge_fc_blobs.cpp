#include <glog/logging.h>

#include <cstring>
#include <string>
#include <vector>
#include "caffe/util/io.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/math_functions.hpp"

using caffe::string;
using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::vector;


using namespace caffe;  // NOLINT(build/namespaces)
//using namespace std;

int MatchString(const string &query, const vector<string> &names)
{
  int i = 0;
  for (; i < int(names.size()); ++i)
  {
    if (query == names[i])
      return i;
  }
  return -1;
}

int merge_fc_blobs(const vector<shared_ptr<Blob<float> > > &blobs1, const vector<shared_ptr<Blob<float> > >&blobs2, const vector<shared_ptr<Blob<float> > >&blobs_dst)
{
  if (blobs1.size() ==0 || blobs1.size() != blobs2.size() || blobs1.size() != blobs_dst.size()) {
    LOG(INFO) << "blobs size do not match! ( " << blobs1.size() <<" vs. " << blobs2.size() << " vs. " << blobs_dst.size() << " )";
    return -1;
  }
  vector<int> weight_shape_src = blobs1[0].get()->shape();
  vector<int> weight_shape_dst = blobs_dst[0].get()->shape();
  LOG(INFO) << "source weights shape : " << weight_shape_src[0] << " " << weight_shape_src[1];
  LOG(INFO) << "destination weights shape : " << weight_shape_dst[0] << " " << weight_shape_dst[1];
    
// merge weights blob (blob0)
  Blob<float> *blob1_1 = blobs1[0].get();
  Blob<float> *blob2_1 = blobs2[0].get();
  Blob<float> *blob_dst_1 = blobs_dst[0].get();
  const float* blob1_1_data = blob1_1->cpu_data();
  const float* blob2_1_data = blob2_1->cpu_data();
  float* blob_dst_1_data = blob_dst_1->mutable_cpu_data();
  const int cols = weight_shape_src[1];
  for (int row = 0; row != weight_shape_src[0]; ++row){
    caffe_copy(cols, blob1_1_data + row * cols, blob_dst_1_data + row * cols * 2);
    caffe_copy(cols, blob2_1_data + row * cols, blob_dst_1_data + row * cols * 2 + cols);
  }
  
  // merge bias blob (blob1)
  Blob<float> *blob1_2 = blobs1[1].get();
  Blob<float> *blob2_2 = blobs2[1].get();
  Blob<float> *blob_dst_2 = blobs_dst[1].get();
  const float* blob1_2_data = blob1_2->cpu_data();
  const float* blob2_2_data = blob2_2->cpu_data();
  float* blob_dst_2_data = blob_dst_2->mutable_cpu_data();
  const int count(blob1_2->count());
  LOG(INFO) << "bias shape: " << blob1_2->count();
  caffe_set(count, 0.f, blob_dst_2_data);
  caffe_add(count, blob1_2_data, blob2_2_data, blob_dst_2_data);
  caffe_scal(count, float(0.5),  blob_dst_2_data);

  return 0;
}

int main(int argc, char** argv) {
  if (argc != 11){
    LOG(INFO) << "Error usage!";
    return -1;
  }
  const string deploy1(argv[1]);
  const string model1(argv[2]);
  const string layer1_name(argv[3]);
  const string deploy2(argv[4]);
  const string model2(argv[5]);
  const string layer2_name(argv[6]);
  const string deploy_dst(argv[7]);
  const string model_dst(argv[8]);
  const string layer_name_dst(argv[9]);
  const string model_out(argv[10]);

  //load model
  Net<float> net1(deploy1, caffe::TEST);
  net1.CopyTrainedLayersFrom(model1);
  Net<float> net2(deploy2, caffe::TEST);
  net2.CopyTrainedLayersFrom(model2);
  Net<float> net_dst(deploy_dst, caffe::TEST);
  net_dst.CopyTrainedLayersFrom(model_dst);
  Net<float> net_out(deploy_dst, caffe::TEST);
  net_out.CopyTrainedLayersFrom(model_dst);
  const vector<shared_ptr<Layer<float> > > &layers1 = net1.layers();
  const vector<shared_ptr<Layer<float> > > &layers2 = net2.layers();
  const vector<shared_ptr<Layer<float> > > &layers_out = net_out.layers();
  const vector<string>& layer_names1 = net1.layer_names();
  const vector<string>& layer_names2 = net2.layer_names();
  const vector<string>& layer_names_out = net_out.layer_names();
 
  
  const int id_name1 = MatchString(layer1_name, layer_names1);
  const int id_name2 = MatchString(layer2_name, layer_names2);
  const int id_name_out = MatchString(layer_name_dst, layer_names_out);
  LOG(INFO) << "layer id :" << id_name1 << " "  <<id_name2 <<" " << id_name_out;
  const vector<shared_ptr<Blob<float> > >&blobs1 = layers1[id_name1].get()->blobs();
  const vector<shared_ptr<Blob<float> > >&blobs2 = layers2[id_name2].get()->blobs();
  const vector<shared_ptr<Blob<float> > >&blobs_out = layers_out[id_name_out].get()->blobs();
 
  if (merge_fc_blobs(blobs1, blobs2, blobs_out) == -1) {
    return -1;
  }
  //save model
  caffe::NetParameter net_param;
  net_out.ToProto(&net_param, false);
  WriteProtoToBinaryFile(net_param, model_out.c_str());
  LOG(INFO) << "Sucefully generate fc_merged model!";
  return 0;
}
