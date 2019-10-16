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
#include "pickup.h"
//#include <boost/filesystem.hpp>
#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS
using namespace caffe;
using namespace std;
using namespace std::chrono;
//static const string root_dir = "/data/test_sql/";
void dump_weights_from_model(string target_model) {
  unordered_map<string, string> config = get_config();

  Caffe::set_mode(Caffe::CPU);
  //Net<float> fc(argv[1], caffe::TEST);
  Net<float> fc(config.at("ROOT_DIR") + config.at("TRAIN_PROTOTXT_NAME"), caffe::TRAIN);
//  cout << "ok2" << endl;
  //fc.CopyTrainedLayersFrom(argv[2]);
  string num = config.at("MAX_ITER");
  while (num.length() < 8){
    num = "0" + num;
  }
  cout << "trying to load " << target_model << endl;
  fc.CopyTrainedLayersFrom(target_model);
  cout << "load succesfully" << endl;
  vector<Blob<float>*> blobs = fc.learnable_params();
  //string weight_file_name = argv[3]; 
  seconds secs = duration_cast<seconds>(system_clock::now().time_since_epoch());
  string weight_file_name = config.at("ROOT_DIR") + config.at("WB_PREFIX");
  ofstream f(weight_file_name);
  f << "version=" << secs.count() << "\n";
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
  }
  f.close();  
  ifstream infile(weight_file_name);
  ofstream outfile("/mnt/vendor/persist/sensors/pickup_nn_wb.txt");
  outfile << infile.rdbuf();
  infile.close();
  outfile.close();
  /*
  ofstream out2("/data/training/sarah/TRAIN_SUCCESS");
  out2.close();
  */
  return;
}
