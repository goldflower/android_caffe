#define TAG "PICKUP_CREATE_DB"
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
#include "gflags/gflags.h"
#include "glog/logging.h"

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
//#include <CkGzip.h>
#include <time.h>
/*
void ChilkatSample(string path)
    {
    // This example assumes the Chilkat API to have been previously unlocked.
    // See Global Unlock Sample for sample code.

    CkGzip gzip;

    // Ungzip and untar.
    bool bNoAbsolute = true;
    const char *untarToDirectory = 0;
    untarToDirectory = "/temp/test";
    // bNoAbsolute tells the component to convert all absolute paths
    // found in the .tar to relative paths.   For example, if the .tar
    // contains a file with an absolute path such as
    // "/usr/bin/something.exe" it will
    // be extracted to "/temp/test/usr/bin/something.exe"
    bool success = gzip.UnTarGz("test.tar.gz",untarToDirectory,bNoAbsolute);
    if (success != true) {
        std::cout << gzip.lastErrorText() << "\r\n";
        return;
    }
    }
*/
//#include "../params.h"
using namespace std;
using namespace caffe;
using std::pair;
using boost::scoped_ptr;

#define axis_len 20
#define FLAGS_shuffle true

DEFINE_string(backend, "lmdb",
		"The backend {lmdb, leveldb} for storing the result");

void display_progress(float progress){
  int barWidth = 70;
  cout << "[";
  int pos = barWidth * progress;
  for (int i = 0; i < barWidth; i++){
    if (i < pos) cout << "=";
    else if (i == pos) cout << ">";
    else cout << " ";
  }
  cout << "] " << int(progress * 100.0) << " %\r";
  cout.flush();
}

float get_min(vector<float> vec, int start, int end){
  float min = INT_MAX;
  for (int i = start; i < end; i++){
    if (vec[i] < min) min = vec[i];
  }
  return min;
}

float get_max(vector<float> vec, int start, int end){
  float max = -INT_MIN;
  for (int i = start; i < end; i++){
    if (vec[i] > max) max = vec[i];
  }
  return max;
}

bool heuristic_check(vector<float> seq)
{
    int y_th = 5;
    int z_th = 5;
    // put down check
    if (seq[axis_len] - seq[axis_len * 2 - 1] < -3)
        return true;
    // head down check
    for (int i = 0; i < 3; i++)
    {
        if (seq[axis_len + i] < 0)
        {
            return true;
        }
    }
    // steady check
    float y_M = get_max(seq, axis_len, 2 * axis_len - 1);
    float y_m = get_min(seq, axis_len, 2 * axis_len - 1);
    float z_M = get_max(seq, 2 * axis_len, 3 * axis_len - 1);
    float z_m = get_min(seq, 2 * axis_len, 3 * axis_len - 1);
    if (((y_M - y_m) < y_th) && ((z_M - z_m) < z_th))
    {
        return true;
    }
    // cover check
    for (int i = 0; i < 3; i++)
    {
        if (seq[axis_len * 2 + i] < 0)
        {
            return true;
        }
    }
    return false;
}

vector<float> getSequence(const string &filename){
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


// just convert 1-d sequence to datum, not for general perpose
bool ReadSensorsToDatum(const string &filename, int label,
		        bool do_check, Datum* datum)
{
//    cout << filename << endl;
    vector<float> vec = getSequence(filename);
    if (vec.size() != axis_len * 6){
//      cout << vec.size() << endl;
//      cout << "cond1" << endl;
      return false;
    }
    if (do_check && label == 1 && heuristic_check(vec)){
      //cout << "cond2" << endl;
      return false;
    }
    //cout << "pass heuristic" << endl;
    datum->set_channels(1);
    datum->set_height(1);
    datum->set_width(axis_len * 6);
    google::protobuf::RepeatedField<float>* datumFloatData = datum->mutable_float_data();
    int count = 0;
    //cout << "vec size " << vec.size() << endl;
    for (int i = 0; i < vec.size(); i++){
      //cout << vec[i] << " ";
      datumFloatData->Add(vec[i]);
      //cout << count++ << endl;
    }
    //cout << "ok1" << endl;
    datum->set_label(label);
    //cout << "ok2" << endl;
    return true;
}


void convert_seq_to_lmdb(string target_chunk){
#ifdef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::SetUsageMessage("Convert a set of pickup sequences to the leveldb/lmdb\n"
		  "format uses as input for Caffe.\n"
		  "convert_pickup_seq [FLAGS] ROOTFOLDER LISTFILE DB_NAME");
  // reading data / label pairs,
  // => [(filename1, label1), (...), ...]
  // according to this pairs we can get the information of each data
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
  //ifstream infile(argv[2]);
  //cout << "file listed in: " << argv[2] << endl;
  ifstream infile;
  string data_folder = "";
  if (config.at("LOAD_FROM") == "V2"){
    infile.open(config.at("FILE_LIST_V2"));
    cout << "file listed in: " << config.at("FILE_LIST_V2") << endl;
  } else if (config.at("LOAD_FROM") == "SQL") {
    infile.open(config.at("FILE_LIST_SQL"));
    cout << "file listed in: " << config.at("FILE_LIST_SQL") << endl;
  } else if (config.at("LOAD_FROM") == "ALL"){ // sql + v2 negative
    cout << "file listed in: " << config.at("FILE_LIST_ALL") << endl;
    int neg_per_pos = atoi(config.at("NEG_PER_POS").c_str());
    ofstream file;
    file.open(config.at("FILE_LIST_ALL"));
    ifstream v2_file;
    string v2_neg_file_list_name = "neg_file_chunk" + target_chunk;
    data_folder = "chunk" + target_chunk;
    v2_file.open(v2_neg_file_list_name);
    vector<string> v2_lines;
    string line;
    while(std::getline(v2_file, line)){
//      cout << line.substr(line.length()-2, 1) << endl;
      if (line.substr(line.length()-2, 1) == "0"){
	//cout << "ok" << endl;
        file << line << "\n";
	v2_lines.push_back(line);
      }
    }
    v2_file.close();

    ifstream sql_file;
    sql_file.open(config.at("FILE_LIST_SQL"));
    vector<string> sql_lines;
    while(std::getline(sql_file, line)){
      file << line << "\n";
      sql_lines.push_back(line);
    }
    random_shuffle(v2_lines.begin(), v2_lines.end());
    random_shuffle(sql_lines.begin(), sql_lines.end());
    int diff = sql_lines.size() * neg_per_pos - v2_lines.size();
 //   cout << sql_lines.size() << "   " << v2_lines.size() << endl;
    if (diff > 0){
     while (diff > 0){
       file << v2_lines[diff % v2_lines.size()] << "\n";
       diff--;
     }
    } else if (diff < 0){
      diff = - diff;
      while (diff > 0){
        file << sql_lines[diff % sql_lines.size()] << "\n";
	diff--;
      }
    }

    file.close();
  } else return;

  infile.open(config.at("FILE_LIST_ALL"));
  vector<std::pair<std::string, int> > lines;
  size_t pos;
  int label;
  while (getline(infile, line)) {
    pos = line.find_last_of(' ');
    label = atoi(line.substr(pos + 1).c_str());
    lines.push_back(make_pair(line.substr(0, pos), label));
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  //LOGE(INFO) << "A total of " << lines.size() << " files on list.";
  
  // create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  //db->Open(argv[3], db::NEW);
  boost::filesystem::remove_all(config.at("LMDB_DIR"));
  db->Open(config.at("LMDB_DIR"), db::NEW);
  //cout << "db name: " << argv[3] << endl;
  cout << "db name: " << config.at("LMDB_DIR") << endl;
  scoped_ptr<db::Transaction> txn(db->NewTransaction());
  // storing to db
  // string root_folder(argv[1]);
  string command = "tar zxf " + data_folder + ".tar.gz"; 
  system(command.c_str());
  cout << "root folder: " << data_folder << endl;
  int count = 0;
  for (int line_id = 0; line_id < lines.size(); ++line_id){
    
    if (line_id % 100 == 0){
      display_progress(line_id / (lines.size()+0.0f));
    }
    
    bool status;
    Datum datum;
    
    //cout << "Handling " << lines[line_id].first << endl;
    status = ReadSensorsToDatum(data_folder + '/' + lines[line_id].first, lines[line_id].second,
		                true, &datum);
    if (!status) continue;
    // sequential
    string key_str = caffe::format_int(line_id) + "_" + lines[line_id].first;
    //cout << "key: " << key_str << endl;
    // put in db
    //cout << "ttt" << endl;
    string out; 
//    CHECK(datum.SerializeToString(&out)); // error if on mobile
    out = datum.SerializeAsString();
    txn->Put(key_str, out);

/*    
    for (int i = 0; i < datum->mutable_float_data().size(); i++){
      cout << datum->mutable_float_data()[i] << " ";
    }
*/  
    if (++count % 1000 == 0){
      // commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      //LOGE("Processed %d files", count);
    }
    //cout << "Success for " << lines[line_id].first << endl;
  }
  // write the last batch
  if (count % 1000 != 0){
    txn->Commit();
//    LOGE("Processed %d files.", count);
  }
}
