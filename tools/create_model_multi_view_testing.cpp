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

bool FC_2_Conv(const boost::shared_ptr<caffe::Layer<float> > layer_fc, const boost::shared_ptr<caffe::Layer<float> > layer_conv)
{
  //check size
  if (layer_fc->blobs()[0]->count() != layer_conv->blobs()[0]->count())
  {
    cout << "fc count = " << layer_fc->blobs()[0]->count() << endl;
    cout << "conv count = " << layer_conv->blobs()[0]->count() << endl;
    return false;
  }

  if (layer_fc->blobs().size() != 2 || layer_conv->blobs().size() != 2)
  {
    cout << "ERROR: layer_fc or layer_conv has other than 2 blobs" << endl;
    return false;
  }
  //copy the first blob
  caffe_copy(layer_fc->blobs()[0]->count(), layer_fc->blobs()[0]->cpu_data(), layer_conv->blobs()[0]->mutable_cpu_data());
  caffe_copy(layer_fc->blobs()[1]->count(), layer_fc->blobs()[1]->cpu_data(), layer_conv->blobs()[1]->mutable_cpu_data());
  //dopy the bias blob
  return true;
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
      if (string(layers_src[id_src]->type()) == string("InnerProduct"))
      {
        cout << "ERROR: FC layer " << layer_names_dst[i] << " exists in the conv net" << endl;
        return -1;
      }
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
        if (string(layers_src[id_src]->type()) != string("InnerProduct"))
        {
          cout << "ERROR: " << layer_names_src[id_src] << " should be FC layer" << endl;
          return -1;
        }
        if (string(layers_dst[i]->type()) != string("Convolution"))
        {
          cout << "ERROR: " << layer_names_dst[i] << " should be Convolution layer" << endl;
          return -1;
        }

        if (!FC_2_Conv(layers_src[id_src], layers_dst[i]))
        {
           cout << "ERROR: Failed to convert fc layer to conv layer, size does not match for " << layer_names_dst[i] << endl;
           return -1;
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