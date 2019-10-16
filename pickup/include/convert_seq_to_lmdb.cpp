#ifndef _CONVERTSEQ_H_
#define _CONVERTSEQ_H_
#include <algorithm>
#include <fstream>

#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <sstream>
#include <iterator>
#include <numeric>
#include <unordered_map>
#include <cstdlib>
#include "boost/scoped_ptr.hpp"
// #include "gflags/gflags.h"
// #include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
// caffe include
#include <caffe/common.hpp>
#include <caffe/caffe.hpp>
// lmdb related
#include "caffe/proto/caffe.pb.h"
#include "boost/filesystem.hpp"
#include <random>
#include <time.h>
#include "pickup.h"
using namespace std;
using namespace caffe;
using boost::scoped_ptr;
using std::pair;
#include <android/log.h>

// #define TAG_C "PICKUP_CONVERT"
#define LOGE_C(...)  __android_log_print(ANDROID_LOG_ERROR,"PICKUP_CONVERT",__VA_ARGS__)
#define LOGD_C(...)  __android_log_print(ANDROID_LOG_DEBUG,"PICKUP_CONVERT",__VA_ARGS__)
#define LOGI_C(...)  __android_log_print(ANDROID_LOG_INFO,"PICKUP_CONVERT",__VA_ARGS__)
#define FLAGS_shuffle true

// const int accel_x = 0;
// const int accel_y = WINDOW_SIZE;
// const int accel_z = WINDOW_SIZE*2;
// const int gyro_x = WINDOW_SIZE*3;
// const int gyro_y = WINDOW_SIZE*4;
// const int gyro_z = WINDOW_SIZE*5;

#define ARR_SIZE(arr) sizeof(arr)/sizeof(arr[0])
#define accel_x        algo->accel_gyro_buffer[0]
#define accel_x_len    ARR_SIZE(accel_x)
#define accel_y        algo->accel_gyro_buffer[1]
#define accel_y_len    ARR_SIZE(accel_y)
#define accel_z        algo->accel_gyro_buffer[2]
#define accel_z_len    ARR_SIZE(accel_z)
#define gyro_x         algo->accel_gyro_buffer[3]
#define gyro_x_len     ARR_SIZE(gyro_x)
#define gyro_y         algo->accel_gyro_buffer[4]
#define gyro_y_len     ARR_SIZE(gyro_y)
#define gyro_z         algo->accel_gyro_buffer[5]
#define gyro_z_len     ARR_SIZE(gyro_z)


// head is the newest
float head(float *buffer){
  return buffer[0];
}
// tail is the earliest
float tail(float *buffer){
  return buffer[WINDOW_SIZE - 1];
}

// buffer_len should not be 0
static float _get_max(float *buffer, int buffer_len)
{
    float ret = buffer[0];
    for (int i = 1; i < buffer_len; i++)
        if (ret < buffer[i])
            ret = buffer[i];
    return ret;
}
#define get_max(buffer) _get_max(buffer, buffer##_len)

// buffer_len should not be 0
static float _get_min(float *buffer, int buffer_len)
{
    float ret = buffer[0];
    for (int i = 1; i < buffer_len; i++)
        if (ret > buffer[i])
            ret = buffer[i];
    return ret;
}
#define get_min(buffer) _get_min(buffer, buffer##_len)

static float _sum(float *buffer, int buffer_len)
{
    float ret = 0.0f;
    for (int i = 0; i < buffer_len; i++)
        ret += buffer[i];
    return ret;
}
#define sum(buffer) _sum(buffer, buffer##_len)
#define mean(buffer) (_sum(buffer, buffer##_len) / buffer##_len)


void display_progress(float progress)
{
  int barWidth = 70;
  cout << "[";
  int pos = barWidth * progress;
  for (int i = 0; i < barWidth; i++)
  {
    if (i < pos)
      cout << "=";
    else if (i == pos)
      cout << ">";
    else
      cout << " ";
  }
  cout << "] " << int(progress * 100.0) << " %\r";
  cout.flush();
}




bool put_down_check(sns_pickup_nn_algo_t *algo)
{
    if (sum(gyro_x) < PUT_DOWN_GYROX_SUM_TH && tail(accel_z) > PUT_DOWN_ACCZ_INITIAL_TH){
        return true;
    }
    return false;
}


bool steady_check_for_any_two_axis(sns_pickup_nn_algo_t *algo, const float th)
{
    float My = get_max(accel_y);
    float my = get_min(accel_y);
    float Mx = get_max(accel_x);
    float mx = get_min(accel_x);
    float Mz = get_max(accel_z);
    float mz = get_min(accel_z);
    bool cond2 = My - my < th;
    bool cond1 = Mx - mx < th;
    bool cond3 = Mz - mz < th;
    bool r = false;
    if (cond1 && cond2){
      r = true;
    } else if (cond1 && cond3){
      r = true;
    } else if (cond2 && cond3){
      r = true;
    }
    return r;
}

bool steady_check(sns_pickup_nn_algo_t *algo)
{
    float th = THREE_AXES_STEADY_TH;
    int len = THREE_AXES_STEADY_SEQUENCE_LEN;
    if (_get_max(accel_y, len) - _get_min(accel_y, len) < th &&
        _get_max(accel_z, len) - _get_min(accel_z, len) < th)
    {
        return true;
    }
    return false;
}

bool head_down_check(sns_pickup_nn_algo_t *algo)
{
    bool pointing_down = accel_y[0] < HEAD_DOWN_ACCY_TH;
    bool not_facing_up_or_down = HEAD_DOWN_FACING_DOWNWARD_TH < accel_z[0] && accel_z[0] < HEAD_DOWN_FACING_UPWARD_TH;
    if (pointing_down && not_facing_up_or_down){
      return true;
    }
    return false;
}


bool fuzzy_check(sns_pickup_nn_algo_t *algo)
{
    float mean_value = mean(accel_y);
    int cross_count = 0;
    for (int i=0; i<accel_y_len-1; i++)
    {
        if (accel_y[i] < mean_value && accel_y[i+1] > mean_value)
        {
            cross_count += 1;
        }
        else if (accel_y[i] > mean_value && accel_y[i+1] < mean_value)
        {
            cross_count += 1;
        }
    }
    if (cross_count > FUZZY_ACCY_CROSS_COUNT_TH)
    {
        return true;
    }
    return false;
}


bool gyro_fuzzy_check(sns_pickup_nn_algo_t *algo, int count_th, const float lower_bound)
{
    float mean_value = 0;
    int cross_count = 0;
    float above_lower_bound_flag = false;
    for (int i=0; i < gyro_x_len-1; i++)
    {
        if (gyro_x[i] < mean_value && gyro_x[i+1] >= mean_value)
        {
            cross_count += 1;
        }
        else if (gyro_x[i] > mean_value && gyro_x[i+1] <= mean_value)
        {
            cross_count += 1;
        }
        if (gyro_x[i] > lower_bound){
            above_lower_bound_flag = true;
        }
    }
    if (cross_count > count_th && above_lower_bound_flag)
    {
        return true;
    }
    return false;
}


bool gyro_out_of_boundary_check(sns_pickup_nn_algo_t *algo)
{
    bool upper_flag = false;
    bool lower_flag = false;
    for (int i=0; i < gyro_x_len-1; i++)
    {
        if (gyro_x[i] > GYROX_UPPER_BOUND)
        {
            upper_flag = true;
        }
        else if (gyro_x[i] < GYROX_LOWER_BOUND)
        {
            lower_flag = true;
        }
    }
    if (upper_flag && lower_flag)
    {
        return true;
    }
    return false;
}



bool max_range_check(sns_pickup_nn_algo_t *algo)
{
    if (get_max(accel_z) - get_min(accel_z) > MAX_RANGE_ACCZ_TH)
    {
        return true;
    }
    return false;
}

bool slip_check(sns_pickup_nn_algo_t *algo){
  bool facing_up = true;
  bool facing_down = true;

  for (int i = 0; i < accel_z_len/2; i++){
    if (accel_z[i] < SLIP_ACCZ_POSITIVE_TH){
      facing_up = false;
      break;
    }
  }
  for (int i = 0; i < accel_z_len/2; i++){
    if (accel_z[i] > SLIP_ACCZ_NEGATIVE_TH){
      facing_down = false;
      break;
    }
  }
  if (facing_up || facing_down){
    return true;
  }
  return false;
}



bool heuristic_check(sns_pickup_nn_algo_t *algo)
{
  //  float gyro_steady_th = 2.0;
    return false
        || steady_check(algo)
        || (put_down_check(algo) && !max_range_check(algo))
        || head_down_check(algo)
        || (max_range_check(algo) && fuzzy_check(algo))
        || (gyro_out_of_boundary_check(algo) && gyro_fuzzy_check(algo, FUZZY_GYROX_OOB_COUNT_TH, FUZZY_GYROX_OOB_LOWER_BOUND))
        || gyro_fuzzy_check(algo, FUZZY_GYROX_IB_COUNT_TH, FUZZY_GYROX_IB_LOWER_BOUND)
        || steady_check_for_any_two_axis(algo, ANY_TWO_AXES_STEADY_TH)
        || slip_check(algo)
        // || cover_check(algo)
        ;
}

vector<float> getSequence(const string &filename)
{
  // 1-d sequence
  // acc_x1, ... acc_xn, acc_y1, ... acc_zn, gyro_x1, ... gyro_zn
  ifstream infile(filename);
  string line;
  vector<float> vec;
  vector<float> result;
  while (getline(infile, line))
  {
    stringstream s(line);
    float num;
    while (s >> num)
    {
      vec.push_back(num);
      if (s.peek() == ',')
      {
        s.ignore();
      }
    }
  }
  for (int i = 0; i < 6; i++)
  { // for NN input
    for (int j = i; j < vec.size(); j += 6)
    {
      result.push_back(vec[j]);
    }
  }
  return result;
}

// just convert 1-d sequence to datum, not for general purpose
bool ReadSensorsToDatum(const string &filename, int label,
                        bool do_check, Datum *datum)
{
  vector<float> vec = getSequence(filename);
  if (vec.size() != WINDOW_SIZE * 6)
  {
    boost::filesystem::remove(filename);
    return false;
  }
  sns_pickup_nn_algo_t algo;
  for (int i = 0; i < NUM_AXES; i++){
    for (int j = 0; j < WINDOW_SIZE; j++){
      algo.accel_gyro_buffer[i][j] = vec[i*WINDOW_SIZE + j];
    }
  }
  if (do_check && label == 1 && heuristic_check(&algo))
  {
    boost::filesystem::remove(filename);
    return false;
  }
  datum->set_channels(1);
  datum->set_height(1);
  datum->set_width(WINDOW_SIZE * 6);
  google::protobuf::RepeatedField<float> *datumFloatData = datum->mutable_float_data();
  for (int i = 0; i < vec.size(); i++)
  {
    datumFloatData->Add(vec[i]);
  }
  datum->set_label(label);
  return true;
}

int convert_seq_to_lmdb(string data_folder, int neg_per_pos, string v2_neg_file_list_name)
{
  /*   
      return: -1: no valid data 
              -2: config data with wrong format
               else: valid data amount (the whole sequence, not sliding window)
  */
  // reading data / label pairs,
  // => [(filename1, label1), (...), ...]
  // according to this pairs we can get the information of each data

  unordered_map<string, string> config = get_config();
  if (boost::filesystem::exists(config.at("ROOT_DIR") + config.at("LMDB_DIR"))){
    boost::filesystem::remove_all(config.at("ROOT_DIR") + config.at("LMDB_DIR"));
  }    
  ifstream infile;
  // sql + v2 negative
  cout << "file listed in: " << config.at("FILE_LIST_ALL") << endl;
  ofstream file;
  file.open(config.at("ROOT_DIR") + config.at("FILE_LIST_ALL"));
  ifstream v2_file;
  v2_file.open(v2_neg_file_list_name);
  vector<string> v2_lines;
  string line;
  while (std::getline(v2_file, line))
  {
    if (line.substr(line.length() - 2, 1) == "0")
    {
      file << line << "\n";
      v2_lines.push_back(line);
    }
  }
  v2_file.close();

  ifstream sql_file;
  sql_file.open(config.at("ROOT_DIR") + config.at("FILE_LIST_SQL"));
  vector<string> sql_lines;
  while (std::getline(sql_file, line))
  {
    file << line << "\n";
    sql_lines.push_back(line);
  }

  infile.open(config.at("ROOT_DIR") + config.at("FILE_LIST_ALL"));
  vector<pair<string, int>> lines;
  size_t pos;
  int label;
  string f_name = "";
  while (getline(infile, line))
  {
    if (line.substr(0, 1) == "2"){ // only trained with evt 0 and evt 1
      continue;
    }
    pos = line.find_last_of(' ');
    label = atoi(line.substr(pos + 1).c_str());
    
    lines.push_back(make_pair(line.substr(0, pos), label));
  }
  if (FLAGS_shuffle)
  {
    // randomly shuffle data
    shuffle(lines.begin(), lines.end());
  }
  // LOGI_C("create new db");
  // create new DB
  scoped_ptr<db::DB> db(db::GetDB("lmdb"));

  db->Open(config.at("ROOT_DIR") + config.at("LMDB_DIR"), db::NEW);
  cout << "db name: " << config.at("LMDB_DIR") << endl;
  scoped_ptr<db::Transaction> txn(db->NewTransaction());
  cout << "root folder: " << data_folder << endl;
  int count = 0;
  int valid_seq_amount = 0;
  vector<int> neg_ok_line_ids;
  vector<int> pos_ok_line_ids;
  int neg_size = 0;
  for (int line_id = 0; line_id < lines.size(); ++line_id)
  {
    /* 
    if (line_id % 100 == 0){
      display_progress(line_id / (lines.size()+0.0f));
    }
    */
    bool status;
    Datum datum;
    status = ReadSensorsToDatum(data_folder + '/' + lines[line_id].first, lines[line_id].second, true, &datum);
    if (!status)
      continue;
    if (lines[line_id].second == 1){
      pos_ok_line_ids.push_back(line_id);
      valid_seq_amount++;
    } else if (lines[line_id].second == 0){
      neg_ok_line_ids.push_back(line_id);
      neg_size++;
    }
    string key_str = caffe::format_int(line_id) + "_" + lines[line_id].first;
    string out;
    //    CHECK(datum.SerializeToString(&out)); // error if on mobile
    out = datum.SerializeAsString();
    txn->Put(key_str, out);
    if (++count % 5000 == 0)
    {
      // commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
    }
    // if (valid_seq_amount > atoi(config.at("MAX_CURRENT_SAMPLE_AMOUNT").c_str())){
    //   break;
    // }    
  }
  if (valid_seq_amount == 0)
  {
    return -1;
  }
  // write the last batch
  if (count % 5000 != 0)
  {
    txn->Commit();
  }
  random_shuffle(pos_ok_line_ids.begin(), pos_ok_line_ids.end());
  random_shuffle(neg_ok_line_ids.begin(), neg_ok_line_ids.end());
  int diff = pos_ok_line_ids.size() * neg_per_pos - neg_ok_line_ids.size();
  cout << "pos size before rebalance " << pos_ok_line_ids.size() << endl;
  cout << "neg size before rebalance " << neg_ok_line_ids.size() << endl;
  cout << "diff: " << diff << endl;
  count = 0;
  if (diff < 0)
  {
    int rebalance_num_for_pos = neg_ok_line_ids.size() / neg_per_pos + neg_ok_line_ids.size() % neg_per_pos - pos_ok_line_ids.size();
    int idx = 0;
    int round = 0;
    while (rebalance_num_for_pos > 0)
    {
      Datum datum;
      ReadSensorsToDatum(data_folder + '/' + lines[pos_ok_line_ids[idx]].first, lines[pos_ok_line_ids[idx]].second, true, &datum);      
      string key_str = caffe::format_int(pos_ok_line_ids[idx]) + caffe::format_int(round) + "_" + lines[pos_ok_line_ids[idx]].first;
      string out;
      out = datum.SerializeAsString();
      txn->Put(key_str, out);
      rebalance_num_for_pos--;
      idx++;
      if (idx == pos_ok_line_ids.size()){
        round++;
      }
      idx = idx % pos_ok_line_ids.size();
      valid_seq_amount++;
    }
  }
  else if (diff > 0)
  {
    int idx = 0;
    int round = 0;
    while (diff > 0)
    {
      Datum datum;
      ReadSensorsToDatum(data_folder + '/' + lines[neg_ok_line_ids[idx]].first, lines[neg_ok_line_ids[idx]].second, true, &datum);            
      string key_str = caffe::format_int(neg_ok_line_ids[idx]) + caffe::format_int(round) + "_" + lines[neg_ok_line_ids[idx]].first;
      string out;
      out = datum.SerializeAsString();
      txn->Put(key_str, out);
      diff--;
      idx++;
      if (idx == pos_ok_line_ids.size()){
        round++;
      }      
      idx = idx % neg_ok_line_ids.size();
      neg_size++;
    }
  }  
  if (count != 0){
    txn->Commit();
  }
  cout << "pos size after rebalance: " << valid_seq_amount << endl;
  cout << "neg size after rebalance: " << neg_size << endl;
  return valid_seq_amount;
}
#endif
