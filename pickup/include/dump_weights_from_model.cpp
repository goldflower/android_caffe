#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <ostream>
#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include <iomanip>
#include <caffe/caffe.hpp>
#include <chrono>
#include <unordered_map>
//#include "../params.h"
using namespace caffe;
using namespace std;
using namespace std::chrono;
//static const string root_dir = "/data/test_sql/";
void dump_weights_from_model() {
  std::unordered_map<std::string, std::string> config;
  std::fstream fin;
//  cout << "try to open ../ini/sarah_setting.ini" << endl;
  fin.open("sarah_setting.config", std::ios::in);
//  cout << "ok" << endl;
  std::string line;
  while (std::getline(fin, line)){
    std::istringstream iss(line);
    std::string key;
    std::string value;
    if (iss >> key >> value){
      config.insert(std::pair<std::string, std::string>(key, value));
    }
  }

  Caffe::set_mode(Caffe::CPU);
  
  //Net<float> fc(argv[1], caffe::TEST);
  Net<float> fc(config.at("TRAIN_PROTOTXT_PATH"), caffe::TRAIN);
//  cout << "ok2" << endl;
  //fc.CopyTrainedLayersFrom(argv[2]);
  string num = config.at("MAX_ITER");
  while (num.length() < 8){
    num = "0" + num;
  }
  cout << "trying to load " << config.at("CKPT_PREFIX") + num + "_loss_inf_.caffemodel" << endl;
  fc.CopyTrainedLayersFrom(config.at("CKPT_FOLDER") + config.at("CKPT_PREFIX") + "_iter_" + num + "_loss_inf_.caffemodel");
  cout << "load succesfully" << endl;
  vector<Blob<float>*> blobs = fc.learnable_params();
  //string weight_file_name = argv[3]; 
  milliseconds ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
  string weight_file_name = config.at("WB_PREFIX");
  ofstream f(weight_file_name);
  f << "version=" << ms.count() << "\n";
  f << "axis='xyzxgygzg'\n2\n84 120 1 84\n";
  for (int i = 0; i < blobs.size(); i++){
    string shape_info = blobs[i]->shape_string();
    vector<int> shape_info_vec = blobs[i]->shape();
    if (shape_info_vec.size() == 1){
      shape_info_vec.push_back(1);
    }
    float* data = blobs[i]->mutable_cpu_data();
    for (int j = 0; j < shape_info_vec[0]; j ++){
	for (int k = 0; k < shape_info_vec[1]; k++){
           f << data[j * shape_info_vec[1] + k] << " ";
           //cout << data[j * shape_info_vec[1] + k] << " ";
	}
	f << "\n";
	
    }
    
//    cout << data[0] << endl;
//    cout << shape_info << endl;
  }
  /*
  cout << blobs.size() << " " << endl;
  string shape_info = blobs[0]->shape_string();
  cout << shape_info << endl;
  
  for (int i = 0; i < blobs.size(); i++){
    Blob<type>* blob = blobs[i];
    for (int j = 0; j < blobs[i].size(); j++){
      cout << blobs[i][j] << " ";
    }
    cout << endl;
  }
  */
  return;
}
