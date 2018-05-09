#include "gradient_boosting/config/GradientBoostingConfig.h"

#include <cassert>
#include <string>

namespace gradient_boosting {
namespace config {

using std::string;
using json = nlohmann::json;

GradientBoostingConfig::GradientBoostingConfig(const json& config)
    : verbose_(GetVerbose(config))
    , value_thresholds_(GetNumberOfValueThresholds(config))
    , statistics_thresholds_(GetNumberOfStatisticsThresholds(config))
    , loss_function_(GetLossFunction(config))
{
}

GradientBoostingConfig::Verbose GradientBoostingConfig::GetVerbose(const json& config) {
  std::string res = config.at("Verbose");
  if (res == "v1") {
    return GradientBoostingConfig::Verbose::v1;
  }
  if (res == "v2") {
    return GradientBoostingConfig::Verbose::v2;
  }
  if (res == "v3") {
    return GradientBoostingConfig::Verbose::v3;
  }
  assert(false);
}

size_t GradientBoostingConfig::GetNumberOfValueThresholds(const json& config) {
  return config.at("BoostingConfig").at("NumberOfValueThresholds");
}

size_t GradientBoostingConfig::GetNumberOfStatisticsThresholds(const json& config) {
  return config.at("BoostingConfig").at("NumberOfStatisticsThresholds");
}

GradientBoostingConfig::LossFunction GradientBoostingConfig::GetLossFunction(const json& config) {
  std::string res = config.at("BoostingConfig").at("LossFunction");
  if (res == "MSE") {
    return GradientBoostingConfig::LossFunction::MSE;
  }
  assert(false);
}

GradientBoostingConfig::Verbose GradientBoostingConfig::GetVerbose() const {
  return verbose_;
}

size_t GradientBoostingConfig::GetNumberOfValueThresholds() const {
  return value_thresholds_;
}

size_t GradientBoostingConfig::GetNumberOfStatisticsThresholds() const {
  return statistics_thresholds_;
}

GradientBoostingConfig::LossFunction GradientBoostingConfig::GetLossFunction() const {
  return loss_function_;
}

}  // namespace config
}  // namespace gradient_boosting
