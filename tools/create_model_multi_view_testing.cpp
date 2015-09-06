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
using std::string;

int main(int argc, char** argv) {
  const string model_file(argv[1]);
  const string trained_file(argv[2]);
  const string mode(argv[3]);
  const string trained_file_new(argv[4]);

  Phase mode_phase;
  if (0 == strcmp(mode, "train"))
  {
    mode_phase = caffe::TRAIN;
    cout << "train phase" << endl;
  }
  else if (0 == strcmp(mode, "test"))
  {
    mode_phase = caffe::TEST;
    cout << "test phase" << endl;
  }
  else
  {
    cout << "Unknow phase " << mode << endl;
    return -1;
  }

  //load model
  Net<float> net(model_file, mode_phase);
  net.CopyTrainedLayersFrom(trained_file);

  //save model
  caffe::NetParameter net_param;
  net.ToProto(&net_param, false);
  WriteProtoToBinaryFile(net_param, trained_file_new.c_str());

  return 0;
}
