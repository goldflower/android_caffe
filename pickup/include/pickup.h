#ifndef _PICKUP_H_
#define _PICKUP_H_
#include <string>
#include "boost/filesystem.hpp"
#include <android/log.h>
#include <dlfcn.h>
using namespace std;

#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO, "PICKUP_FLOW",__VA_ARGS__)
// #define LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG, "PICKUP_FLOW", __VA_ARGS__)
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, "PICKUP_FLOW", __VA_ARGS__)

void PRINT_LOG(string log_string, int log_level, string mode){
  if (mode == "I"){
    LOGI(log_string.c_str());
  }
  switch (log_level){
    case 0: // only print simple info
      break;
    case 1: // print debug info
      if (mode == "V"){
        LOGV(log_string.c_str());
      }
      break;
  }
}

const unsigned long long MILLISECONDS_PER_DAY = 86400000;
const unsigned int MAX_NUMBER_OF_CHUNKS = 10;
const unsigned int WINDOW_SIZE = 20;
const unsigned int NUM_AXES = 6;
////// for heuristic //////
#define THREE_AXES_STEADY_TH            1.5f  // all 3 axes use the same threshold
#define THREE_AXES_STEADY_SEQUENCE_LEN  5     // only watch the latest n samples
#define PUT_DOWN_GYROX_SUM_TH          -3.0f 
#define PUT_DOWN_ACCZ_INITIAL_TH        2     // put down happened usually when the phone is facing up
#define HEAD_DOWN_ACCY_TH              -1.0f
#define HEAD_DOWN_FACING_UPWARD_TH      6.5f
#define HEAD_DOWN_FACING_DOWNWARD_TH   -6.5f
#define MAX_RANGE_ACCZ_TH               30.0f
#define FUZZY_ACCY_CROSS_COUNT_TH       4
#define GYROX_UPPER_BOUND               4.0f
#define GYROX_LOWER_BOUND              -4.0f
#define FUZZY_GYROX_OOB_COUNT_TH        3     // OOB stands for out of bounds
#define FUZZY_GYROX_IB_COUNT_TH         6     // IB stands for in bounds
#define FUZZY_GYROX_OOB_LOWER_BOUND     0
#define FUZZY_GYROX_IB_LOWER_BOUND      1
#define ANY_TWO_AXES_STEADY_TH          3.0f
#define SLIP_ACCZ_POSITIVE_TH           8.5f
#define SLIP_ACCZ_NEGATIVE_TH           -8.5f

typedef struct sns_pickup_nn_algo_s{
  float                    accel_gyro_buffer[6][20];
} sns_pickup_nn_algo_t;

static const char* alpha_property = "ro.build.alpha";
static const char* beta_property = "ro.build.beta";
static const char* mea_property = "ro.build.mea";

//////////////////////////

vector<string> get_filenames(string dir)
{
  vector<string> filenames;
  if (!boost::filesystem::exists(dir))
    return filenames;
  boost::filesystem::directory_iterator end_iter;
  for (boost::filesystem::directory_iterator iter(dir); iter != end_iter; ++iter)
  {
    if (boost::filesystem::is_regular_file(iter->status()))
    {
      filenames.push_back(iter->path().string());
    }
    else
    {
      continue;
    }
  }
  return filenames;
}

int copy_file(string src, string des)
{
  ifstream s(src, ios::binary);
  ofstream d(des, ios::binary);
  char buffer;
  if(!s || !d)
      cout << "Error" << endl;
  else
  {
      while(s.get(buffer))
      {
          d << buffer;
      }
      s.close();
      d.close();
  }
  return 0;
}


struct statistic_record
{
  int statistic_id = 0;
  unsigned long long timestamp = 0;
  int train_days = 0;
  int train_amount = 0;
  int train_pred_correct = 0;
  int last_val_amount = 0;
  int last_val_pred_correct = 0;
  int aod_amount = 0;
  int aod_pred_correct = 0;
  float neg_acc = 0;
  int nb_of_last_ok_models = 0;
  int do_not_update_wb_flag = 0;
  int version = 0;
};

struct training_status
{
  // use begin_time_for_training to end_time_for_val to train a new model
  // the last model use begin_time_for_training to end_time_for_training
  unsigned long long BEGIN_TIME_FOR_TRAINING = 0;
  unsigned long long END_TIME_FOR_TRAINING = 0; // update when there are enough data and not train yet
  unsigned long long BEGIN_TIME_FOR_VAL = 0;
  unsigned long long END_TIME_FOR_VAL = 0;
  int PROGRESS = 0; // 0: not yet training, 1: in progress,
  int NUM_DAYS = 0;
  int STOP_TRAINING_FLAG = 0;
  unsigned long long IGNORE_DB_FROM_TIME = 0;
  unsigned long long IGNORE_DB_TO_TIME = 0;
  int CURRENT_CYCLE = 0;
  int DID_NOT_IMPROVE_SIGNIFICANTLY_TIMES = 0;
  unsigned long long LAST_START_DAILY_VAL_TIMESTAMP = 0;
  unsigned long long DB_CREATION_TIME = 0;
  int NB_OF_LAST_OK_MODELS = 0;
  int CURRENT_BEST_TRAIN_AMOUNT = 0;
  float CURRENT_BEST_NEG_ACC = 0;
  int CURRENT_BEST_TRAIN_PRED_CORRECT = 0;
  int CURRENT_BEST_TRAIN_DAYS = 0;  
  int DO_NOT_UPDATE_WB_FLAG = 0;
  int VERSION = 0;
};

void write_training_status(string status_file_path, struct training_status t_status)
{
  ofstream out(status_file_path);
  out << "BEGIN_TIME_FOR_TRAINING " << t_status.BEGIN_TIME_FOR_TRAINING << "\n";
  out << "END_TIME_FOR_TRAINING " << t_status.END_TIME_FOR_TRAINING << "\n";
  out << "BEGIN_TIME_FOR_VAL " << t_status.BEGIN_TIME_FOR_VAL << "\n";
  out << "PROGRESS " << t_status.PROGRESS << "\n";
  out << "END_TIME_FOR_VAL " << t_status.END_TIME_FOR_VAL << "\n";
  out << "NUM_DAYS " << t_status.NUM_DAYS << "\n";
  out << "STOP_TRAINING_FLAG " << t_status.STOP_TRAINING_FLAG << "\n";
  out << "IGNORE_DB_FROM_TIME " << t_status.IGNORE_DB_FROM_TIME << "\n";
  out << "IGNORE_DB_TO_TIME " << t_status.IGNORE_DB_TO_TIME << "\n";
  out << "CURRENT_CYCLE " << t_status.CURRENT_CYCLE << "\n";
  out << "DID_NOT_IMPROVE_SIGNIFICANTLY_TIMES " << t_status.DID_NOT_IMPROVE_SIGNIFICANTLY_TIMES << "\n";
  out << "LAST_START_DAILY_VAL_TIMESTAMP " << t_status.LAST_START_DAILY_VAL_TIMESTAMP << "\n";
  out << "DB_CREATION_TIME " << t_status.DB_CREATION_TIME << "\n";
  out << "NB_OF_LAST_OK_MODELS " << t_status.NB_OF_LAST_OK_MODELS << "\n";
  out << "CURRENT_BEST_TRAIN_AMOUNT " << t_status.CURRENT_BEST_TRAIN_AMOUNT << "\n";
  out << "CURRENT_BEST_NEG_ACC " << t_status.CURRENT_BEST_NEG_ACC << "\n";
  out << "CURRENT_BEST_TRAIN_PRED_CORRECT " << t_status.CURRENT_BEST_TRAIN_PRED_CORRECT << "\n";
  out << "CURRENT_BEST_TRAIN_DAYS " << t_status.CURRENT_BEST_TRAIN_DAYS << "\n";
  out << "DO_NOT_UPDATE_WB_FLAG " << t_status.DO_NOT_UPDATE_WB_FLAG << "\n";
  out << "VERSION " << t_status.VERSION << "\n";
}

training_status get_training_status(string status_file_path)
{
  struct training_status t_status;
  fstream fin;
  fin.open(status_file_path, std::ios::in);
  string line;
  while (getline(fin, line))
  {
    istringstream iss(line);
    string key;
    string value;
    if (iss >> key >> value)
    {
      if (key == "BEGIN_TIME_FOR_TRAINING")
        t_status.BEGIN_TIME_FOR_TRAINING = strtoull(value.c_str(), NULL, 0);
      else if (key == "END_TIME_FOR_TRAINING")
        t_status.END_TIME_FOR_TRAINING = strtoull(value.c_str(), NULL, 0);
      else if (key == "BEGIN_TIME_FOR_VAL")
        t_status.BEGIN_TIME_FOR_VAL = strtoull(value.c_str(), NULL, 0);
      else if (key == "END_TIME_FOR_VAL")
        t_status.END_TIME_FOR_VAL = strtoull(value.c_str(), NULL, 0);
      else if (key == "NUM_DAYS")
        t_status.NUM_DAYS = atoi(value.c_str());
      else if (key == "STOP_TRAINING_FLAG")
        t_status.STOP_TRAINING_FLAG = atoi(value.c_str());
      else if (key == "IGNORE_DB_FROM_TIME")
        t_status.IGNORE_DB_FROM_TIME = strtoull(value.c_str(), NULL, 0);
      else if (key == "IGNORE_DB_TO_TIME")
        t_status.IGNORE_DB_TO_TIME = strtoull(value.c_str(), NULL, 0);
      else if (key == "CURRENT_CYCLE")
        t_status.CURRENT_CYCLE = atoi(value.c_str());
      else if (key == "PROGRESS")
        t_status.PROGRESS = atoi(value.c_str());
      else if (key == "DID_NOT_IMPROVE_SIGNIFICANTLY_TIMES")
        t_status.DID_NOT_IMPROVE_SIGNIFICANTLY_TIMES = atoi(value.c_str());
      else if (key == "START_DAILY_VAL_TIMESTAMP")
        t_status.LAST_START_DAILY_VAL_TIMESTAMP = strtoull(value.c_str(), NULL, 0);
      else if (key == "DB_CREATION_TIME")
        t_status.DB_CREATION_TIME = strtoull(value.c_str(), NULL, 0);
      else if (key == "NB_OF_LAST_OK_MODELS")
        t_status.NB_OF_LAST_OK_MODELS = atoi(value.c_str());
      else if (key == "CURRENT_BEST_TRAIN_AMOUNT")
        t_status.CURRENT_BEST_TRAIN_AMOUNT = atoi(value.c_str());
      else if (key == "CURRENT_BEST_NEG_ACC")
        t_status.CURRENT_BEST_NEG_ACC = atof(value.c_str());
      else if (key == "CURRENT_BEST_TRAIN_PRED_CORRECT")
        t_status.CURRENT_BEST_TRAIN_PRED_CORRECT = atoi(value.c_str());
      else if (key == "CURRENT_BEST_TRAIN_DAYS")
        t_status.CURRENT_BEST_TRAIN_DAYS = atoi(value.c_str());
      else if (key == "DO_NOT_UPDATE_WB_FLAG")
        t_status.DO_NOT_UPDATE_WB_FLAG = atoi(value.c_str());
      else if (key == "VERSION")
        t_status.VERSION = atoi(value.c_str());
    }
  }
  return t_status;
}

string select_sequences(unsigned long long time_start, unsigned long long time_end){
  ostringstream stm1;
  stm1 << time_start;
  ostringstream stm2;
  stm2 << time_end;
  string sql = "SELECT event_ts, event_type, sensor_type, sensor_data FROM event_table WHERE event_ts BETWEEN " + stm1.str() + " AND " + stm2.str();
  sql += " ORDER BY event_ts DESC";
  cout << "SQL: " << sql << endl;
  return sql;
}

unordered_map<string, string> get_config(){
  unordered_map<string, string> config;
  fstream fin;
  fin.open("/data/training/sarah/sarah_setting.config", std::ios::in);
  string line;
  while (getline(fin, line))
  {
    istringstream iss(line);
    string key;
    string value;
    if (iss >> key >> value)
    {
      config.insert(pair<string, string>(key, value));
    }
  }
  return config;
}
typedef int (*PFN_SYSTEM_PROP_GET)(const char *, char *);
int __system_property_get(const char* name, char* value)
{
    static PFN_SYSTEM_PROP_GET __real_system_property_get = NULL;
    if (!__real_system_property_get) {
        // libc.so should already be open, get a handle to it.
        void *handle = dlopen("libc.so", RTLD_NOLOAD);
        if (!handle) {
            LOGE("Cannot dlopen libc.so: %s.\n", dlerror());
        } else {
            __real_system_property_get = (PFN_SYSTEM_PROP_GET)dlsym(handle, "__system_property_get");
        }
        if (!__real_system_property_get) {
            LOGE("Cannot resolve __system_property_get(): %s.\n", dlerror());
        }
    }
    return (*__real_system_property_get)(name, value);
}

#endif
