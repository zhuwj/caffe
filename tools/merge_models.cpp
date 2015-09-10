#include <cstring>
#include <string>
#include "caffe/caffe.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/net.hpp"

using caffe::Caffe;
using caffe::Net;
using caffe::NetParameter;
using std::string;

int main(int argc, char** argv) {
  const string deploy(argv[1]);
  const string model_dst(argv[argc-1]);
  NetParameter net_param;
  Net<float> caffe_net(deploy, caffe::TEST);
  for (int i = 2; i < argc - 1; ++i){
    caffe_net.CopyTrainedLayersFrom(string(argv[i]));
  }
  caffe_net.ToProto(&net_param,false);
  WriteProtoToBinaryFile(net_param, model_dst.c_str());
  LOG(INFO) <<"\n" << "Successfully creat test model called "<< model_dst <<"\n";
  return 0;
}

