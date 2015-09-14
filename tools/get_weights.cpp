#include <glog/logging.h>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "caffe/util/io.hpp"

using std::string;
using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::vector;

int main(int argc, char** argv){
  //ReadNetParamsFromTextFileOrDie(FLAGS_model, &param);
  if (argc != 4){
    LOG(INFO) << "usage: ./build/tools/get_weights train_val.prototxt trained.caffemodel weights.txt";
  }
  const string deploy_file(argv[1]);
  const string model_file(argv[2]);
  const string weights_file(argv[3]);
  Net<float> caffe_net(deploy_file, caffe::TRAIN);
  std::ofstream fout;
  fout.open(weights_file.c_str());
  //caffe::NetParameter net_param; 
  //ReadNetParamsFromBinaryFileOrDie(FLAGS_model, &net_param);

  caffe_net.CopyTrainedLayersFrom(model_file);
  const vector<shared_ptr<Layer<float> > > &layers = caffe_net.layers();
  for (int i = 0; i < layers.size();++i){
     vector<shared_ptr<Blob<float> > >&blobs = layers[i].get()->blobs();
     for(int j = 0; j < blobs.size(); ++j){
          fout << layers[i].get()->layer_param().name() <<" ";
          fout << "blob " << j <<": ";
        for(int c = 0; c < blobs[j].get()->count(); ++c){
          fout << blobs[j].get()->cpu_data()[c];
          fout << "  ";
      }
     fout<< "\n";
    }
    fout<<"\n";
  }
  fout.close();
  return 0;
}
