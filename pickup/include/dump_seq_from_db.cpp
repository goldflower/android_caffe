//#define TAG    "PICKUP_DUMP"
// caffe include
#include <caffe/common.hpp>
#include <caffe/caffe.hpp>
// stl include
#include <vector>
#include <iostream>
#include <string>
#include <unordered_map>
#include <map>
#include <tuple>
#include <fstream>

#include <sstream>
// linux c includes
#include <sys/types.h>
#include <dirent.h>
// lmdb related
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "boost/scoped_ptr.hpp"
#include "boost/filesystem.hpp"
#include "boost/algorithm/string.hpp"
#include <boost/functional/hash.hpp>
// sqlite include
#include "sqlite3.h"
// for system property
#include <dlfcn.h>
// for file op
#include <boost/algorithm/string.hpp>
//#include "convert_pickup_seq.cpp"
//#include "../params.h"

//static const std::string root_dir = "/data/test_sql/";

const char* select_sequences(){
  return "SELECT OP_payload FROM OP_payload_upload_table WHERE OP_payload LIKE \'\{\"en\":\"sarah\%'";
}

using statement = std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)>;
statement create_statement(sqlite3* db, std::string sql){
  sqlite3_stmt* stmt = nullptr;
  int rc = sqlite3_prepare_v2(db, sql.c_str(), sql.length(), &stmt, nullptr);
  if (rc != SQLITE_OK){
    //LOGE("Unable to create statement '%s': %s", sql.c_str(), sqlite3_errmsg(db));
    sqlite3_close(db);
    std::exit(EXIT_FAILURE);
  }
  //LOGE("create statement successfully");
  return statement(stmt, sqlite3_finalize);
}


bool dump_current_row_to_map(sqlite3_stmt* stmt, std::unordered_map<std::tuple<std::string, std::string, std::string>, std::vector<std::string>, boost::hash<std::tuple<std::string, std::string, std::string>>>* M, std::set<std::string>* t){
  std::string head_str = "data";
  if (sqlite3_column_type(stmt, 0) == SQLITE_TEXT){
     auto first = sqlite3_column_text(stmt, 0);
     std::size_t s = sqlite3_column_bytes(stmt, 0);
     std::string record = std::string((const char*)first, s);
     std::size_t data_start_loc = record.find(head_str) + 7; // len(data":") = 7
     std::size_t data_end_loc = record.find("\",", data_start_loc);
     std::size_t sensor_start_loc = record.find("sensor", data_end_loc) + 9;
     std::size_t sensor_end_loc = record.find("\",", sensor_start_loc);
     std::size_t type_start_loc = record.find("type", sensor_end_loc) + 7;
     std::size_t type_end_loc = record.find("\"}", type_start_loc);
     std::string data_str = record.substr(data_start_loc, data_end_loc-data_start_loc);
     std::string sensor_type_str = record.substr(sensor_start_loc, sensor_end_loc-sensor_start_loc);
     if (sensor_type_str == "10"){
       return false;
     }
     std::string evt_type_str = record.substr(type_start_loc, type_end_loc-type_start_loc);
     std::vector<std::string> data_split_by_time_;
     std::string time_key;
     boost::split(data_split_by_time_, data_str, boost::is_any_of("\\n"));
     std::vector<std::string> data_split_by_time;
     for (std::string str: data_split_by_time_){
       if (str.length() > 3) data_split_by_time.push_back(str);
     }
     
     int sample_rate = data_split_by_time.size();
//     std::cout << "sample rate " << sample_rate << std::endl;
     std::vector<std::string> concate_vec(sample_rate*3);
     bool flag = true;
     // x1, x2, ..., y1, y2, ... ,z1, ...
     for (int i = 0; i < sample_rate; i++){
       std::string str = data_split_by_time[i];
       std::vector<std::string> vec;
       boost::split(vec, str, boost::is_any_of(","));
       if (vec.size() == 1){
         continue;
       }
       for (int j = 0; j < 3; j++){
         concate_vec[j*sample_rate + i] = vec[j+2];
       }
       if (flag){
         time_key = vec[0];
         flag = false;
	 t->insert(time_key);
       }
     }

     //std::cout << "time stamp " << time_key << " sensor " << sensor_type_str << std::endl ;
     M->insert(std::make_pair(std::make_tuple(evt_type_str, time_key, sensor_type_str), concate_vec));
  }
  //LOGE("\n");
  return true;
}


using database = std::unique_ptr<sqlite3, decltype(&sqlite3_close)>;
database open_database(const char* name){
  sqlite3* db = nullptr;
  auto rc = sqlite3_open(name, &db);
  if (rc != SQLITE_OK){
    //LOGE("Unable to open database '%s': %s", name, sqlite3_errmsg(db));
    sqlite3_close(db);
    std::exit(EXIT_FAILURE);
  }
  return database{db, sqlite3_close};
}


void dump_sensor_data_to_file(std::string dest, std::vector<std::string> vec1, std::vector<std::string> vec2, int target_window_size, int begin_idx, int stride, int sample_rate, std::ofstream& file_list_ss, int idx){
    /* only accept 2 types of sample rate:, we down-sample 49 into 25
     *  1. 24-25, 2. 49
     */
    std::string seq = "";
    if (vec1.size() != 49*3 && vec1.size() != 25*3 && vec1.size() != 24*3) return;
    for (int i = begin_idx; i < sample_rate; i+=stride){
      for (int j = 0; j < 3; j++){
        seq += vec1[sample_rate*j+i] + ",";
      }
      for (int j = 0; j < 2; j++){
        seq += vec2[sample_rate*j+i] + ",";
      }
      seq += vec2[sample_rate*2 + i];
      if (i + stride < sample_rate){
        seq += "\n";
      }
    }
    std::vector<std::string> lines;
    boost::split(lines, seq, boost::is_any_of("\n"));
    //std::cout << "lines size: " << lines.size() << std::endl;
    int diff = lines.size() - target_window_size;
    std::ofstream f;
    std::string pickup_seq;
    for (int i = 0; i < diff; i++){
      std::ostringstream ss;
      ss << i;
      std::string file_name = dest + ss.str();
      f.open(file_name);
      pickup_seq = "";
      for (int j = 0; j < target_window_size; j++){
       //std::cout << lines[i+j] << std::endl;
       pickup_seq += lines[i+j] + "\n";
      }
      f << pickup_seq;
      file_list_ss << file_name.substr(idx, file_name.length()-idx) << " 1\n";
      f.close();
    }
}

void dump_seq_from_db(std::string target_chunk){
  std::unordered_map<std::string, std::string> config;
  std::fstream fin;
  fin.open("sarah_setting.config", std::ios::in);
  std::string line;
  while (std::getline(fin, line)){
    std::istringstream iss(line);
    std::string key;
    std::string value;
    if (iss >> key >> value){
      config.insert(std::pair<std::string, std::string>(key, value));
    }
  }
  std::string file_name = config.at("DB_NAME");
  
  auto db = open_database(file_name.c_str());
  auto stmt = create_statement(db.get(), select_sequences());
  int rc;
  int count = 0;
  std::vector<std::string> evt_types = {"0", "1", "2"};
  std::vector<std::string> sensor_types = {"1", "4"};
  std::unordered_map<std::tuple<std::string, std::string, std::string>, std::vector<std::string>, boost::hash<std::tuple<std::string, std::string, std::string>>> evt_time_sensor_data_map;
  std::set<std::string> unique_times;
  while (true){
    rc = sqlite3_step(stmt.get());
    if (rc != SQLITE_ROW){
      //LOGE("break");
      break;
    }
    dump_current_row_to_map(stmt.get(), &evt_time_sensor_data_map, &unique_times);
    count++;
  }

  std::vector<std::tuple<std::string, std::string, std::string>> keys;
  keys.reserve(evt_time_sensor_data_map.size());
  std::vector<std::vector<std::string>> vals;
  vals.reserve(evt_time_sensor_data_map.size());
  for (auto kv: evt_time_sensor_data_map){
    keys.push_back(kv.first);
    vals.push_back(kv.second);
  }
  /*
  for (auto key: keys){
    LOGE("%s, %s, %s", std::get<0>(key).c_str(), std::get<1>(key).c_str(), std::get<2>(key).c_str());
  }
  */
 std::string dest = "chunk" + target_chunk + "/";
 //std::string dest = "test_folder/";
 // int target_window_size = std::stoi(config["WINDOW_SIZE"]);
 int target_window_size;
 std::istringstream(config.at("WINDOW_SIZE")) >> target_window_size;
 std::ofstream file_list_ss;
 file_list_ss.open(config.at("FILE_LIST_SQL"));
 for (std::string t: unique_times){
   for (std::string evt: evt_types){
     // if there is anyone of the sensors missing, skip
     try{
       std::tuple<std::string, std::string, std::string> key1(evt, t, "1");
       std::tuple<std::string, std::string, std::string> key2(evt, t, "4");
       std::vector<std::string> val1 = evt_time_sensor_data_map.at(key1);
       std::vector<std::string> val2 = evt_time_sensor_data_map.at(key2);
       //std::cout << "val1 length " << val1.size() << std::endl;;
       //std::cout << "val2 length " << val2.size() << std::endl;;
       
       if (val1.size() != val2.size()){
	 //std::cout << t + " continue" << std::endl;
         continue;
       } 
       int sample_rate = val1.size() / 3;
       std::ostringstream ss;
       ss << 0;
       int name_idx = dest.length();
       dump_sensor_data_to_file(dest + evt + "_" + t + ss.str(), val1, val2, target_window_size, 0, 2, sample_rate, file_list_ss, name_idx);
       ss.str("");
       ss << 1;
       dump_sensor_data_to_file(dest + evt + "_" + t + ss.str(), val1, val2, target_window_size, 1, 2, sample_rate, file_list_ss, name_idx);
     } catch (const std::out_of_range& oor){
       //LOGE("Out of Range error: %s", oor.what());
       continue;
     }
   }
 }
 file_list_ss.close();
}
