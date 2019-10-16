#ifndef _G_M_S_
#define _G_M_S_
#include <algorithm>
#include <string>
#include <vector>
#include "boost/scoped_ptr.hpp"
#include "glog/logging.h"
#include <caffe/caffe.hpp>
#include <unordered_map>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>
#include "convert_seq_to_lmdb.cpp"
#include "dump_seq_from_db.cpp"
#include "pickup.h"

using namespace caffe;
using namespace std;
using namespace boost;



boost::shared_ptr<Net<float>> initialize_inference_net(string model_path, string prototxt_path){
  boost::shared_ptr<Net<float>> net;
  Caffe::set_mode(Caffe::CPU);
  net.reset(new Net<float>(prototxt_path, TEST));
  net->CopyTrainedLayersFrom(model_path);
  const vector<int> shape_vec{1, 1, 1, 120};
  Blob<float>* input_layer = net->input_blobs()[0];
  input_layer->Reshape(1, 1, 1, 120);
  net->Reshape();
  return net;
}

struct statistic_record get_model_statistics(string model_path, string data_path){
  unordered_map<string, string> config = get_config();
  struct statistic_record record;
  string deploy = config.at("ROOT_DIR") + config.at("DEPLOY_PROTOTXT_NAME");
  boost::shared_ptr<Net<float>> candidate_net = initialize_inference_net(model_path, deploy);
  Blob<float>* input_layer = candidate_net->input_blobs()[0];
//  string negfolder = "/data/training/sarah/chunck0";
  vector<string> filenames = get_filenames(data_path);
  cout << filenames.size() << endl;
  if (filenames.size() == 0){
    cout << "no data in target '" + data_path + "'" << endl;
    return record;
  }
  set<string> key_s;
  string key;
  unordered_map<string, float> pred_map;
  unordered_map<string, float> aod_pred_map;
  unordered_map<string, float>::iterator it;
  int neg_count = 0;
  float neg_correct = 0.0;
  ostringstream ss;
  ss << filenames.size();
  // LOGI(("filenames: " + ss.str()).c_str());
  int size = filenames.size();
  if (size > atoi(config.at("NB_MAX_INFERENCE").c_str())){
    size = atoi(config.at("NB_MAX_INFERENCE").c_str());
  }
  for (int i = 0; i < filenames.size(); i++){
    int index = filenames[i].find_last_of("/");
    if (filenames[i].substr(index+2, 1) == "_"){
      key = filenames[i].substr(index+3, filenames[i].find("-") - (index+3));
      vector<float> seq = getSequence(filenames[i]);
      float* input_data = input_layer->mutable_cpu_data();
      for (int i = 0; i < seq.size(); i++){
        input_data[i] = seq[i];
      }
      candidate_net->Forward();
      Blob<float>* output_layer = candidate_net->output_blobs()[0];
      const float* begin = output_layer->cpu_data();
      // cout << "result is " << begin[0] << endl;
      if (filenames[i].substr(index+1, 1) != "2"){
        it = pred_map.begin();
        it = pred_map.find(key);
        if (it == pred_map.end()){
          pred_map[key] = 0;
        }
        if (begin[0] > 0.5){
          pred_map[key] = 1;
        }
      } else{
        it = aod_pred_map.begin();
        it = aod_pred_map.find(key);
        if (it == aod_pred_map.end()){
          aod_pred_map[key] = 0;
        }        
        if (begin[0] > 0.5){
          aod_pred_map[key] = 1;
        }
      }
    } else {
      vector<float> seq = getSequence(filenames[i]);
      float* input_data = input_layer->mutable_cpu_data();
      for (int i = 0; i < seq.size(); i++){
        input_data[i] = seq[i];
      }
      candidate_net->Forward();
      Blob<float>* output_layer = candidate_net->output_blobs()[0];
      const float* begin = output_layer->cpu_data();
      if (begin[0] <= 0.5){
        neg_correct++;
      }
      neg_count++;
    }
  }
  float non_aod_correct = 0.0;
  for (unordered_map<string, float>::iterator iter = pred_map.begin(); iter != pred_map.end(); ++iter){
    if (iter->second == 1) non_aod_correct += 1;
  }
  float aod_correct = 0.0;
  for (unordered_map<string, float>::iterator iter = aod_pred_map.begin(); iter != aod_pred_map.end(); ++iter){
    if (iter->second == 1) aod_correct += 1;
  }  
  cout << "pos acc: " << non_aod_correct / pred_map.size() << endl;
  cout << "aod pos acc: " << aod_correct / aod_pred_map.size() << endl;
  cout << "neg acc: " << neg_correct / neg_count << endl;
  record.aod_amount = aod_pred_map.size();
  record.aod_pred_correct = aod_correct;
  record.train_amount = pred_map.size();
  record.train_pred_correct = non_aod_correct;
  // if (neg_count == 0){
  //   record.neg_acc = -1;
  // } else
  record.neg_acc = neg_correct / neg_count;
  // int last_val_amount;
  // int last_val_pred_correct;
  return record;
}
#endif