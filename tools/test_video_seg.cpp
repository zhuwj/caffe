#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <fstream>
//#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"

using namespace caffe;

int id_by_name(const string &query, const vector<string> &names)
{
  int i = 0;
  for (; i < int(names.size()); ++i)
  {
    if (query == names[i])
      return i;
  }
  return -1;
}

string trim(string& s)
{
  if (s.empty()) return s;
  s.erase(0,s.find_first_not_of("\n"));
  s.erase(s.find_last_not_of("\n") + 1);

  s.erase(0,s.find_first_not_of("\r"));
  s.erase(s.find_last_not_of("\r") + 1);

  s.erase(0,s.find_first_not_of("\t"));
  s.erase(s.find_last_not_of("\t") + 1);

  s.erase(0,s.find_first_not_of(" "));
  s.erase(s.find_last_not_of(" ") + 1);
  return s;
}

int main(int argc ,char **argv) {
  
  const string deploy = argv[1];
  const string model = argv[2];
  const string test_list = argv[3];
  const string saved_folder = argv[4];
  const string softmax_name = argv[5];
  const int iterations = atoi(argv[6]);
  if (argc == 8) {
    const int gpu = atoi(argv[7]);
    // Set device id and mode
    if (gpu >= 0) {
      LOG(INFO) << "Use GPU with device ID " << gpu;
      Caffe::SetDevice(gpu);
      Caffe::set_mode(Caffe::GPU);
    } else {
      LOG(INFO) << "Use CPU.";
      Caffe::set_mode(Caffe::CPU);
    }
  }
  // Read test_list
   vector<string> video_names;
  std::ifstream infile(test_list.c_str());
  char str[100];
  LOG(INFO) << "Begin to read list";
  while(infile.getline(str,sizeof(str))){
    string line(str);
    size_t index = line.find_first_of(" ");
    string video_name =  line.substr(0,index);
    LOG(INFO) << trim(video_name);
    video_names.push_back(trim(video_name));
  }
  infile.close();
  // Instantiate the caffe net.
  Net<float> caffe_net(deploy, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(model);

  LOG(INFO) << "Finished instanitiate net!";
  const vector<shared_ptr<Layer<float> > > &layers = caffe_net.layers();

  int num_seg;
  int batch_size;
  if (layers[0].get()->layer_param().has_video_data_param()){
    num_seg = layers[0].get()->layer_param().video_data_param().num_segments();
    batch_size = layers[0].get()->layer_param().video_data_param().batch_size();
  }
  else if(layers[0].get()->layer_param().has_video2_data_param()){
    num_seg = layers[0].get()->layer_param().video2_data_param().num_segments();
    batch_size = layers[0].get()->layer_param().video2_data_param().batch_size();
  }else{
    LOG(INFO) << "First layer does not data layer ! ";
    return -1;
  }
  CHECK((batch_size % num_seg) == 0);
  int batch_video = batch_size / num_seg;
  LOG(INFO) << "batch_size = " << batch_size << " num_segments = " << num_seg << " #video in batch = " << batch_video;

  int num_class =  caffe_net.blob_by_name(softmax_name).get()->count() / batch_size;
  int num_each_video = num_class * num_seg;
  LOG(INFO) << "num_class :" << num_class <<" num_each_video: " << num_each_video;
  LOG(INFO) << "Running for " << iterations << " iterations."; 
  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result = caffe_net.Forward(bottom_vec, &iter_loss);
    //save softmax output
    const float* softmax_cpu_data = caffe_net.blob_by_name(softmax_name).get()->cpu_data();
    LOG(INFO) << "Get softmax output";
    for ( int j = 0; j < batch_video; ++j){
      std::ofstream outfile((saved_folder + "/" + video_names[i * (batch_video) + j] + ".txt").c_str());
      LOG(INFO) << "Begin to write "<< video_names[i * (batch_video) + j] <<".txt";
      for (int id_seg = 0; id_seg < num_seg; ++id_seg){
        for(int id_class = 0; id_class < num_class; ++id_class){
          LOG(INFO) << j * num_each_video + id_seg * num_class + id_class << " ";
          outfile << softmax_cpu_data[j * num_each_video + id_seg * num_class + id_class] << " ";
        }
        outfile << "\n";
      }
      outfile.close();
    }
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= iterations;
  LOG(INFO) << "Loss: " << loss;
  
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net.blob_loss_weights()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }

  return 0;
}
