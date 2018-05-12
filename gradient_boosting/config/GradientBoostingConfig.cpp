#include "gradient_boosting/config/GradientBoostingConfig.h"

#include <iostream>

namespace gradient_boosting {
namespace config {

using json = nlohmann::json;

using std::string;
using std::unordered_map;

using LossFunction = GradientBoostingConfig::LossFunction;
using Verbose = GradientBoostingConfig::Verbose;
using TaskType = GradientBoostingConfig::TaskType;

GradientBoostingConfig::GradientBoostingConfig(const json& config)
    : to_verbose_(GetVerboseMapping())
    , to_loss_function_(GetLossFunctionMapping())
    , to_task_type_(GetTaskTypeMapping())
    , verbose_(GetVerbose(config))
    , value_thresholds_(GetNumberOfValueThresholds(config))
    , statistics_thresholds_(GetNumberOfStatisticsThresholds(config))
    , loss_function_(GetLossFunction(config))
    , target_value_name_(GetTargetValueName(config))
    , task_type_(GetTaskType(config))
    , train_data_(GetTrainData(config))
{
}
// TODO(rialeksandrov) Try to avoid using this function and init map like 'static constexpr'
unordered_map<string, Verbose> GradientBoostingConfig::GetVerboseMapping() const {
  unordered_map<string, Verbose> res = {
      {"v1", Verbose::v1},
      {"v2", Verbose::v2},
      {"v3", Verbose::v3},
  };
  return res;
};
// TODO(rialeksandrov) Try to avoid using this function and init map with 'static constexpr'
unordered_map<string, LossFunction> GradientBoostingConfig::GetLossFunctionMapping() const {
  unordered_map<string, LossFunction> res = {
      {"MSE", LossFunction::MSE},
  };
  return res;
}

// TODO(rialeksandrov) Try to avoid using this function and init map with 'static constexpr'
unordered_map<string, TaskType> GradientBoostingConfig::GetTaskTypeMapping() const {
  unordered_map<string, TaskType> res = {
      {"Classification", TaskType::Classification},
  };
  return res;
}

Verbose GradientBoostingConfig::GetVerbose(const json& config) const {
  std::cout << "GradientBoostingConfig::GetVerbose" << std::endl;
  return to_verbose_.at(config.at("Verbose"));
}

size_t GradientBoostingConfig::GetNumberOfValueThresholds(const json& config) const {
  std::cout << "GradientBoostingConfig::GetNumberOfValueThresholds" << std::endl;
  return config.at("BoostingConfig").at("NumberOfValueThresholds");
}

size_t GradientBoostingConfig::GetNumberOfStatisticsThresholds(const json& config) const {
  std::cout << "GradientBoostingConfig::GetNumberOfStatisticsThresholds" << std::endl;
  return config.at("BoostingConfig").at("NumberOfStatisticsThresholds");
}

std::string GradientBoostingConfig::GetTargetValueName(const json& config) const {
  std::cout << "GradientBoostingConfig::GetTargetValueName" << std::endl;
  return config.at("Experiment").at("TargetValueName");
}

TaskType GradientBoostingConfig::GetTaskType(const json& config) const {
  std::cout << "GradientBoostingConfig::GetTaskType" << std::endl;
  return to_task_type_.at(config.at("BoostingConfig").at("TaskType"));
}

std::string GradientBoostingConfig::GetTrainData(const json& config) const {
  std::cout << "GradientBoostingConfig::GetTrainData " << config.at("Experiment").at("TrainData") << "\n";
  return config.at("Experiment").at("TrainData");
}

LossFunction GradientBoostingConfig::GetLossFunction(const json& config) const {
  std::cout << "GradientBoostingConfig::GetLossFunction" << std::endl;
  return to_loss_function_.at(config.at("BoostingConfig").at("LossFunction"));
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
  return 0;
}

size_t GradientBoostingConfig::GetNumberOfTrees() const {
  return 100;
}

size_t GradientBoostingConfig::GetHeight() const {
  return 6;
}

std::string GradientBoostingConfig::GetIdValueName () const {
  return "EventId";
}

double GradientBoostingConfig::GetLearningRate() const {
  return 0.001;
}

}  // namespace config
}  // namespace gradient_boosting
