#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;

int MatchString(const string &query, const vector<string> &names)
{
  int i = 0;
  for (; i < int(names.size()); ++i)
  {
    if (query == names[i])
      return i;
  }
  return i;
}

int main(int argc, char** argv) {
  const string deploy_src(argv[1]);
  const string model_src(argv[2]);
  const string deploy_dst(argv[3]);
  const string model_dst(argv[4]);

  //load model
  Net<float> net_src(deploy_src, caffe::TEST);
  net_src.CopyTrainedLayersFrom(model_src);
  Net<float> net_dst(deploy_dst, caffe::TEST);
  net_dst.CopyTrainedLayersFrom(model_src);
  const vector<shared_ptr<Layer<float> > > &layers_src = net_src.layers();
  const vector<shared_ptr<Layer<float> > > &layers_dst = net_dst.layers();
  const vector<string>& layer_names_src = net_src.layer_names();
  const vector<string>& layer_names_dst = net_dst.layer_names();
  if (layers_src.size() != layer_names_src.size())
  {
    cout << "layers_src and layer_names_src does not match" << endl;
    return -1;
  }
  if (layers_dst.size() != layer_names_dst.size())
  {
    cout << "layers_dst and layer_names_dst does not match" << endl;
    return -1;
  }

  //copy fc layer to conv layer
  const int num_layer_src = int(layers_src.size());
  const int num_layer_dst = int(layers_dst.size());
  for (int i = 0; i < num_layer_dst; ++i)
  {
    int id_src = MatchString(layer_names_dst[i], layer_names_src);
    if (id_src < num_layer_src) //find it
    {
     continue;
    }
    else //did not find it
    {
      string name_ori = layer_names_dst[i].substr(0, layer_names_dst[i].find_last_of("_")); //check for fc layers
      id_src = MatchString(name_ori, layer_names_src);
      if (id_src >= num_layer_src) //not matched
      {
        //cout << "ERROR: Could not find " << layer_names_dst[i] << " layer in source netword" << endl;
        //return -1;
        continue;
      }
      else //matched
      {
        vector<shared_ptr<Blob<float> > >blobs_src = layers_src[id_src].get()->blobs();
        vector<shared_ptr<Blob<float> > >blobs_dst = layers_src[i].get()->blobs();
        for (int id_blob = 0; id_blob < blobs_src.size(); ++id_blob){
           caffe_copy(blobs_src[id_blob].get()->count(), blobs_src[id_blob].get()->cpu_data(), blobs_dst[id_blob].get()->mutable_cpu_data());
        } 
      }
    }
  }

  //save model
  caffe::NetParameter net_param;
  net_dst.ToProto(&net_param, false);
  WriteProtoToBinaryFile(net_param, model_dst.c_str());
  return 0;
}
