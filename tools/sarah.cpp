#define TAG "PICKUP_FLOW"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "../pickup/include/convert_seq_to_lmdb.cpp"
#include "../pickup/include/dump_seq_from_db.cpp"
#include "../pickup/include/dump_weights_from_model.cpp"
#include "../pickup/include/sarah_train.cpp"
#include "../pickup/include/create_prototxt.cpp"
#include "../pickup/include/remove_tmp.cpp"
#include <random>
//#include "../pickup/params.h"
//static const std::string root_dir = "/data/test_sql/";

using namespace std;
int main(int argc, char *argv[]){
  LOGE("start data preprocessing");
  srand(time(NULL));
  int target_chunk_num = rand() % 4;
  std::ostringstream ss;
  ss << target_chunk_num;
  string target_chunk(ss.str());
  
  cout << "create prototxt" << endl;
  create_prototxt();
  cout << "fetch db" << endl;
//  system("./libs/dump_seq_from_db");
  dump_seq_from_db(target_chunk);
  cout << "convert to lmdb" << endl;
  convert_seq_to_lmdb(target_chunk);
//  system("./libs/convert_seq_to_lmdb");
  LOGE("start training model");
  cout << "train model" << endl;
  sarah_train();
//  system("./libs/sarah_train");
  cout << "write weights" << endl;
  dump_weights_from_model();
//  system("./libs/dump_weights_from_model");
  //cout << "delete temp data" << endl;
  //free_space();
}

