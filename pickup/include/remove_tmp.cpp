#include "boost/filesystem.hpp"
#include <fstream>
#include <string>
#include <iostream>
#include <unordered_map>
#include <sstream>

using namespace std;

void free_space(){
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
  boost::filesystem::remove_all(config.at("ROOT_DIR") + config.at("LMDB_DIR"));
  boost::filesystem::remove(config.at("ROOT_DIR") + config.at("SOLVER_PROTOTXT"));
  boost::filesystem::remove(config.at("ROOT_DIR") + config.at("TRAIN_PROTOTXT_PATH"));
  string ckpt_path = config.at("CKPT_PREFIX");

  boost::filesystem::remove(config.at("ROOT_DIR") + ckpt_path.substr(0, ckpt_path.length() - ckpt_path.find("/")));

}
