#include "gradient_boosting/config/GradientBoostingConfig.h"


namespace gradient_boosting {
namespace config {

using json = nlohmann::json;

using std::string;
using std::unordered_map;

using LossFunction = GradientBoostingConfig::LossFunction;
using Verbose = GradientBoostingConfig::Verbose;

GradientBoostingConfig::GradientBoostingConfig(const json& config)
    : to_verbose_(GetVerboseMapping())
    , to_loss_function_(GetLossFunctionMapping())
    , verbose_(GetVerbose(config))
    , value_thresholds_(GetNumberOfValueThresholds(config))
    , statistics_thresholds_(GetNumberOfStatisticsThresholds(config))
    , loss_function_(GetLossFunction(config))
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

Verbose GradientBoostingConfig::GetVerbose(const json& config) const {
  return to_verbose_.at(config.at("Verbose"));
}

size_t GradientBoostingConfig::GetNumberOfValueThresholds(const json& config) const {
  return config.at("BoostingConfig").at("NumberOfValueThresholds");
}

size_t GradientBoostingConfig::GetNumberOfStatisticsThresholds(const json& config) const {
  return config.at("BoostingConfig").at("NumberOfStatisticsThresholds");
}

LossFunction GradientBoostingConfig::GetLossFunction(const json& config) const {
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

LossFunction GradientBoostingConfig::GetLossFunction() const {
  return loss_function_;
}

}  // namespace config
}  // namespace gradient_boosting
