//define TAG "PICKUP_TRAINING"
// caffe include
#include <caffe/common.hpp>
#include <caffe/caffe.hpp>
// stl include
#include <vector>
#include <iostream>
#include <string>
#include <unordered_map>
// linux c includes
#include <sys/types.h>
#include <dirent.h>
// lmdb related
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "boost/scoped_ptr.hpp"
#include "boost/filesystem.hpp"
// sqlite include
#include "sqlite3.h"
// for system property
#include <dlfcn.h>
// for file op
#include <boost/algorithm/string.hpp>

using std::vector;
using std::string;
using std::unordered_map;
using boost::scoped_ptr;

static const string c_strRootDir = "/data/training/sarah/";
static const string s_strChkPostfix = ".solverstate"
;

static void getSolverStates(vector<string> & vecFiles) {
    DIR* dirp = opendir(c_strRootDir.c_str());
    dirent* dp;
    while ((dp = readdir(dirp)) != NULL) {
        string file(dp->d_name);
        if (boost::algorithm::ends_with(file, s_strChkPostfix) /*&&
            boost::algorithm::starts_with(file, strChkPrefix)*/)
            vecFiles.push_back(file);
    }
    (void)closedir(dirp);
    sort(vecFiles.begin(), vecFiles.end(), std::greater<string>());
}

static void delSolverStates(int offset, vector<string> & vecFiles) {
    const string strChkPostfixModel = ".caffemodel";

    for (int i = offset; i < vecFiles.size(); ++i) {
        boost::filesystem::remove(c_strRootDir+vecFiles[i]);
        boost::algorithm::replace_all(vecFiles[i], s_strChkPostfix, strChkPostfixModel);
        boost::filesystem::remove(c_strRootDir+vecFiles[i]);
    }
}


void sarah_train(){
  vector<string> vecSolverStates;
  bool bRestore = false;
  caffe::SolverParameter solver_param;
  unordered_map<string, string> config = get_config();
  caffe::ReadSolverParamsFromTextFileOrDie(config.at("ROOT_DIR") + config.at("SOLVER_PROTOTXT_NAME"), &solver_param);
  string ckpt_path = config.at("ROOT_DIR") + config.at("CKPT_DIR");
  if (!boost::filesystem::exists(ckpt_path)){
    boost::filesystem::create_directory(ckpt_path);
  }
  getSolverStates(vecSolverStates);
  if (vecSolverStates.size() > 0){
    bRestore = true;
    // LOGI("found the latest solver state %s", vecSolverStates[0].c_str());
    if (vecSolverStates.size() > 1){
      delSolverStates(1, vecSolverStates);
    }
  }
  boost::shared_ptr<caffe::Solver<float>> solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
  if (bRestore) solver->Restore((c_strRootDir+vecSolverStates[0]).c_str());
  solver->Solve();
}
