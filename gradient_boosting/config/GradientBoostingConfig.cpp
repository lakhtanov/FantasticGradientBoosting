#include "gradient_boosting/config/GradientBoostingConfig.h"

#include <iostream>

namespace gradient_boosting {
namespace config {

using json = nlohmann::json;

using std::string;
using std::unordered_map;

using LossFunction = GradientBoostingConfig::LossFunction;
using TaskType = GradientBoostingConfig::TaskType;
using TreeType = GradientBoostingConfig::TreeType;
using Verbose = GradientBoostingConfig::Verbose;

GradientBoostingConfig::GradientBoostingConfig(const json& config)
    : to_verbose_(GetVerboseMapping())
    , to_loss_function_(GetLossFunctionMapping())
    , to_task_type_(GetTaskTypeMapping())
    , to_tree_type_(GetTreeTypeMapping())
    , loss_function_(GetLossFunction(config))
    , statistics_thresholds_(GetNumberOfStatisticsThresholds(config))
    , target_value_name_(GetTargetValueName(config))
    , task_type_(GetTaskType(config))
    , train_data_(GetTrainData(config))
    , value_thresholds_(GetNumberOfValueThresholds(config))
    , verbose_(GetVerbose(config))
    , number_of_threads_(GetNumberOfThreads(config))
    , number_of_trees_(GetNumberOfTrees(config))
    , tree_height_(GetHeight(config))
    , id_value_name_(GetIdValueName(config))
    , learning_rate_(GetLearningRate(config))
    , tree_type_(GetTreeType(config))
    , test_data_(GetTestData(config))
    , result_file_(GetResultFile(config)) {
}

// TODO(rialeksandrov) Try to avoid using this function and init map like
// 'static constexpr'.
unordered_map<string, Verbose>
GradientBoostingConfig::GetVerboseMapping() const {
  unordered_map<string, Verbose> res = {
      {"v1", Verbose::v1},
      {"v2", Verbose::v2},
      {"v3", Verbose::v3},
  };
  return res;
}

// TODO(rialeksandrov) Try to avoid using this function and init map with
// 'static constexpr'.
unordered_map<string, LossFunction>
GradientBoostingConfig::GetLossFunctionMapping() const {
  unordered_map<string, LossFunction> res = {
      {"MSE", LossFunction::MSE},
  };
  return res;
}

// TODO(rialeksandrov) Try to avoid using this function and init map with
// 'static constexpr'.
unordered_map<string, TaskType>
GradientBoostingConfig::GetTaskTypeMapping() const {
  unordered_map<string, TaskType> res = {
      {"Classification", TaskType::Classification},
  };
  return res;
}

unordered_map<string, TreeType>
GradientBoostingConfig::GetTreeTypeMapping() const {
  unordered_map<string, TreeType> res = {
      {"ObliviousTree", TreeType::ObliviousTree},
      {"Lakhtanov", TreeType::Lakhtanov},
  };
  return res;
}

Verbose GradientBoostingConfig::GetVerbose(const json& config) const {
  return to_verbose_.at(config.at("Verbose"));
}

size_t GradientBoostingConfig::GetNumberOfValueThresholds(
    const json& config) const {
  return config.at("BoostingConfig").at("NumberOfValueThresholds");
}

size_t GradientBoostingConfig::GetNumberOfStatisticsThresholds(
    const json& config) const {
  return config.at("BoostingConfig").at("NumberOfStatisticsThresholds");
}

string GradientBoostingConfig::GetTargetValueName(const json& config) const {
  return config.at("Experiment").at("TargetValueName");
}

TaskType GradientBoostingConfig::GetTaskType(const json& config) const {
  return to_task_type_.at(config.at("BoostingConfig").at("TaskType"));
}

string GradientBoostingConfig::GetTrainData(const json& config) const {
  return config.at("Experiment").at("TrainData");
}

LossFunction GradientBoostingConfig::GetLossFunction(const json& config) const {
  return to_loss_function_.at(config.at("BoostingConfig").at("LossFunction"));
}

size_t GradientBoostingConfig::GetNumberOfThreads(
    const nlohmann::json& config) const {
  return config.at("BoostingConfig").at("NumberOfThreads");
}

size_t GradientBoostingConfig::GetNumberOfTrees(const json& config) const {
  return config.at("BoostingConfig").at("NumberOfTrees");
}

size_t GradientBoostingConfig::GetHeight(const json& config) const {
  return config.at("BoostingConfig").at("TreeHeight");
}

string GradientBoostingConfig::GetIdValueName(const json& config) const {
  return config.at("Experiment").at("IdName");
}

double GradientBoostingConfig::GetLearningRate(const json& config) const {
  return config.at("BoostingConfig").at("LearningRate");
}

string GradientBoostingConfig::GetTestData(const json& config) const {
  return config.at("Experiment").at("TestData");
}

string GradientBoostingConfig::GetResultFile(const json& config) const {
  return config.at("Experiment").at("ResultFile");
}

TreeType GradientBoostingConfig::GetTreeType(const json& config) const {
  return to_tree_type_.at(config.at("BoostingConfig").at("TreeType"));
}

Verbose GradientBoostingConfig::GetVerbose() const {
  return verbose_;
}

size_t GradientBoostingConfig::GetNumberOfValueThresholds() const {
  return value_thresholds_;
}

size_t GradientBoostingConfig::GetNumberOfStatisticsThresholds() const {
  return statistics_thresholds_;
}

string GradientBoostingConfig::GetTargetValueName() const {
  return target_value_name_;
}

LossFunction GradientBoostingConfig::GetLossFunction() const {
  return loss_function_;
}

TaskType GradientBoostingConfig::GetTaskType() const {
  return task_type_;
}

string GradientBoostingConfig::GetTrainData() const {
  return train_data_;
}

size_t GradientBoostingConfig::GetNumberOfThreads() const {
  return number_of_threads_;
}

size_t GradientBoostingConfig::GetNumberOfTrees() const {
  return number_of_trees_;
}

size_t GradientBoostingConfig::GetHeight() const {
  return tree_height_;
}

string GradientBoostingConfig::GetIdValueName() const {
  return id_value_name_;
}

double GradientBoostingConfig::GetLearningRate() const {
  return learning_rate_;
}

TreeType GradientBoostingConfig::GetTreeType() const {
  return tree_type_;
}

string GradientBoostingConfig::GetTestData() const {
  return test_data_;
}

string GradientBoostingConfig::GetResultFile() const {
  return result_file_;
}

}  // namespace config
}  // namespace gradient_boosting
