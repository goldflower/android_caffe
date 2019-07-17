#include <fstream>
#include <string>
#include <iostream>
#include <unordered_map>
#include <sstream>

using namespace std;

void create_prototxt(){
  
  unordered_map<string, string> config;
  fstream fin;
  fin.open("sarah_setting.config", std::ios::in);
  string line;
  while (getline(fin, line)){
    istringstream iss(line);
    string key;
    string value;
    if (iss >> key >> value){
      config.insert(std::pair<std::string, std::string>(key, value));
    }
  }
  ofstream out(config["TRAIN_PROTOTXT_PATH"]);
  const char* model_char1 = R"V0G0N(
name: "fc"
layer{
  name: "data"
  type: "Data"
  top: "data"
  top: "label" 
  include{
    phase: TRAIN
  }
  data_param{
    source: )V0G0N";
  string model_str1(model_char1);
  out << model_str1 << "\"" << config["ROOT_DIR"] << config["LMDB_DIR"] << "\"";
  
  const char* model_str3 = R"V0G0N(
    batch_size: 64
    backend: LMDB
  }
}
layer{
  name: "fc1"
  type: "InnerProduct"
  bottom: "data"
  top: "fc1"
  inner_product_param{
    num_output: 84
    weight_filler{
      type: "gaussian"
      std: 0.01
    }
    bias_filler{
      type: "constant"
      value: 0.01
    }
  }
}
layer{
  name: "relu1"
  type: "ReLU"
  bottom: "fc1"
  top: "fc1"
}
layer{
  name: "out"
  type: "InnerProduct"
  bottom: "fc1"
  top: "out"
  inner_product_param{
    num_output: 1
    weight_filler{
      type: "gaussian"
      std: 0.01
    }
    bias_filler{
      type: "constant"
      value: 0.01
    }
  }
}
layer{
  name: "loss"
  type: "SigmoidCrossEntropyLoss"
  top: "loss"
  bottom: "out"
  bottom: "label"
}
)V0G0N";
  out << model_str3;
  out.close();


  ofstream out2(config["SOLVER_PROTOTXT"]);
  string model_char2 = "net: \"" + config["ROOT_DIR"] + config["TRAIN_PROTOTXT_PATH"] + "\"\n";
  const char* model_char3 = R"V0G0N(
# 100775 // 100
# test_iter: 1007
# Carry out testing every 500 training iterations.
# test_interval: 500
# All parameters are from the cited paper above
base_lr: 0.001
momentum: 0.9
momentum2: 0.999
# since Adam dynamically changes the learning rate, we set the base learning
# rate to a fixed value
lr_policy: "fixed"
# Display every 100 iterations
display: 2000
# snapshot intermediate results
snapshot: 10000
# solver mode: CPU or GPU
type: "Adam"
solver_mode: CPU
)V0G0N";
  string model_char4 = "snapshot_prefix: \"" + config["ROOT_DIR"] + config["CKPT_FOLDER"] + config["CKPT_PREFIX"] + "\"\n";
  string model_char5 = "max_iter: " + config["MAX_ITER"];
  out2 << model_char2 << model_char3 << model_char4 << model_char5;
  out2.close();

}
