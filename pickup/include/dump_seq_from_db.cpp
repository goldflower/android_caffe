#ifndef _DUMP_SEQ_H_
#define _DUMP_SEQ_H_
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
// #include <boost/iostreams/filtering_stream.hpp>
// #include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
// #include <boost/iostreams/filter/gzip.hpp>
// #include <boost/iostreams/device/back_inserter.hpp>
#include "sqlite3.h"
// for system property
#include <dlfcn.h>
// for file op
#include <boost/algorithm/string.hpp>
#include "pickup.h"
using namespace boost::iostreams;
using namespace std;




using statement = unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)>;
statement create_statement(sqlite3* db, string sql){
  sqlite3_stmt* stmt = nullptr;
  int rc = sqlite3_prepare_v2(db, sql.c_str(), sql.length(), &stmt, nullptr);
  if (rc != SQLITE_OK){
    //LOGE("Unable to create statement '%s': %s", sql.c_str(), sqlite3_errmsg(db));
    sqlite3_close(db);
    exit(EXIT_FAILURE);
  }
  //LOGE("create statement successfully");
  return statement(stmt, sqlite3_finalize);
}


bool dump_current_row_to_map(sqlite3_stmt* stmt, unordered_map<tuple<string, string, string>, vector<string>, 
                             boost::hash<tuple<string, string, string>>>* M, set<string>* t){
  string head_str = "data";
  string data_str;
  string time_key;
  string sensor_type_str;
  string evt_type_str;
  for (int i = 0; i < sqlite3_column_count(stmt); ++i){
    auto first = sqlite3_column_text(stmt, i);
    size_t s = sqlite3_column_bytes(stmt, i);
    switch (i){
      case 0:
        time_key = string((const char*)first, s);
        break;
      case 1:
        evt_type_str = string((const char*)first, s);
        break;
      case 2:
        sensor_type_str = string((const char*)first, s);
        break;
      case 3:
        data_str = string((const char*)first, s);
        break;
      default:
        cout << "WTF" << endl;
    }
  }
  vector<string> data_split_by_time; 
  boost::split(data_split_by_time, data_str, boost::is_any_of("\n")); // ["x0,y0,z0","x1,y1,z1", ...]
  int sample_rate = data_split_by_time.size() - 1; 
//  cout << "sample rate: " << sample_rate << endl;
  vector<string> concate_vec(sample_rate*3);
  for (int i = 0; i < sample_rate; i++){
    string str = data_split_by_time[i];
//    cout << str << endl;
    vector<string> vec;
    boost::split(vec, str, boost::is_any_of(","));

    if (vec.size() == 1) continue;
    for (int j = 0; j < 3; j++){
      concate_vec[j*sample_rate + i] = vec[j];
    }
    t->insert(time_key);
  }
  M->insert(make_pair(make_tuple(evt_type_str, time_key, sensor_type_str), concate_vec));
  return true;
}


using database = unique_ptr<sqlite3, decltype(&sqlite3_close)>;
database open_database(const char* name){
  sqlite3* db = nullptr;
  auto rc = sqlite3_open(name, &db);
  if (rc != SQLITE_OK){
    //LOGE("Unable to open database '%s': %s", name, sqlite3_errmsg(db));
    sqlite3_close(db);
    exit(EXIT_FAILURE);
  }
  return database{db, sqlite3_close};
}


void dump_sensor_data_to_file(string dest, vector<string> vec1, vector<string> vec2, int target_window_size, 
                              int begin_idx, int stride, int sample_rate, ofstream& file_list_ss, int idx){
    /*  sample_rate to window size, no down sampling process for now
     */
    string seq = "";
    // i++ for assuming sample_rate = 25
    for (int i = begin_idx; i < sample_rate; i++){ 
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
    vector<string> lines;
    boost::split(lines, seq, boost::is_any_of("\n"));
    int diff = lines.size() - target_window_size;
    ofstream f;
    string pickup_seq;
    for (int i = 0; i < diff/2; i++){
      ostringstream ss;
      ss << i;
      string file_name = dest + ss.str();
      f.open(file_name);
      pickup_seq = "";
      for (int j = 0; j < target_window_size; j++){
       pickup_seq += lines[i+j] + "\n";
      }
      f << pickup_seq;
      file_list_ss << file_name.substr(idx, file_name.length()-idx) << " 1\n";
      f.close();
    }
}

unsigned int dump_seq_from_db(string dest, unsigned long long time_begin, unsigned long long time_end){
//  string dest = config.at("ROOT_DIR") + "data/chunk" + target_chunk + "/";
  unordered_map<string, string> config = get_config();
  string db_name = config.at("DB_NAME");
  auto db = open_database((config.at("ROOT_DIR") + db_name).c_str());
  auto stmt = create_statement(db.get(), select_sequences(time_begin, time_end));
  int rc;
  int count = 0;
  vector<string> evt_types = {"0", "1", "2"};
  unordered_map<tuple<string, string, string>, vector<string>, boost::hash<tuple<string, string, string>>> evt_time_sensor_data_map;
  set<string> unique_times;
  while (true){
    rc = sqlite3_step(stmt.get());
    if (rc != SQLITE_ROW){
      //LOGE("break");
      break;
    }
    dump_current_row_to_map(stmt.get(), &evt_time_sensor_data_map, &unique_times);
    count++;
  }
  cout << "there are " << count << " rows in db between " << time_begin << " to " << time_end << endl;
  if (count == 0){
    return -1;
  }

  vector<tuple<string, string, string>> keys;
  keys.reserve(evt_time_sensor_data_map.size());
  vector<vector<string>> vals;
  vals.reserve(evt_time_sensor_data_map.size());
  for (auto kv: evt_time_sensor_data_map){
    keys.push_back(kv.first);
    vals.push_back(kv.second);
  }
 
 cout << "get chunk dest path '" << dest << "'" << endl;
 if (boost::filesystem::exists(dest)){
   // do nothing
 } else boost::filesystem::create_directory(dest);
 int target_window_size = atoi(config.at("WINDOW_SIZE").c_str());
//  istringstream(config.at("WINDOW_SIZE")) >> target_window_size;
 ofstream file_list_ss;
 ofstream evt2_file_list_ss;
 file_list_ss.open(config.at("ROOT_DIR") + config.at("FILE_LIST_SQL"));
 evt2_file_list_ss.open(config.at("ROOT_DIR") + config.at("FILE_LIST_SQL_EVT2")); // although currently we don't need this for training
//  int sample_count = 0;
 for (string t: unique_times){
   for (string evt: evt_types){
     // if there is anyone of the sensors missing, skip
     try{
       tuple<string, string, string> key1(evt, t, "1"); // acc
       tuple<string, string, string> key2(evt, t, "4"); // gyro
       vector<string> val1 = evt_time_sensor_data_map.at(key1);
       vector<string> val2 = evt_time_sensor_data_map.at(key2);
       if (val1.size() != val2.size()){
	      //  cout << t + " continue" << endl;
         continue;
       } 
       int sample_rate = val1.size() / 3;
       ostringstream ss;
       ss << 0;
       int name_idx = dest.length();
       if (evt == "2"){
         dump_sensor_data_to_file(dest + evt + "_" + t + "-", val1, val2, target_window_size, 0, 1, sample_rate, evt2_file_list_ss, name_idx);
       } else {
        dump_sensor_data_to_file(dest + evt + "_" + t + "-", val1, val2, target_window_size, 0, 1, sample_rate, file_list_ss, name_idx);
       }
      //  cout << t << " ok" << endl;
     } catch (const out_of_range& oor){
      //  LOGE("Out of Range error: %s", oor.what());
       continue;
     }
   }
  //  sample_count++;
  //  if (sample_count > atoi(config.at("MAX_CURRENT_SAMPLE_AMOUNT").c_str())){
  //    break;
  //  }
 }
 file_list_ss.close();
 evt2_file_list_ss.close();
 return unique_times.size();
}

#endif