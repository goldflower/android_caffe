#include <fstream>
#include <string>
#include <iostream>
#include <unordered_map>
#include <sstream>

using namespace std;

void create_prototxt(){
  
  unordered_map<string, string> config = get_config();
  ofstream out("/data/training/sarah/" + config["TRAIN_PROTOTXT_NAME"]);
  const char* model_char1 = R"V0G0N(
name: "fc"
layer{
  name: "data"
  type: "Data"
  top: "data"
  top: "label" 
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


  ofstream out2(config.at("ROOT_DIR") + config.at("SOLVER_PROTOTXT_NAME"));
  string model_char2 = "net: \"" + config.at("ROOT_DIR") + config.at("TRAIN_PROTOTXT_NAME") + "\"\n";
  char* model_char3 = R"V0G0N(
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
snapshot: )V0G0N";
  //out2 << model_char2 << model_char3 << config.at("MAX_ITER") << "\n";
  out2 << model_char2 << model_char3 << config.at("CKPT_ITER") << "\n";
  const char* model_char4 = R"V0G0N(
# solver mode: CPU or GPU
type: "Adam"
solver_mode: CPU
)V0G0N";
  string model_char5 = "snapshot_prefix: \"" + config.at("ROOT_DIR") + config.at("CKPT_DIR") + config["CKPT_PREFIX"] + "\"\n";
  string model_char6 = "max_iter: " + config["MAX_ITER"];
  out2 << model_char4 << model_char5 << model_char6;
  out2.close();


  ofstream out3(config.at("ROOT_DIR") + "sarah_fc_deploy.prototxt");
  char* model_char7 = R"V0G0N(
name: "fc"
layer{
  name: "data"
  type: "Input"
  top: "data"
  input_param{
    shape: {dim: 1 dim: 1 dim: 1 dim: 120}
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
  name: "prob"
  type: "Sigmoid"
  bottom: "out"
  top: "prob"
}
)V0G0N";
  out3 << model_char7;
  out3.close();
}
