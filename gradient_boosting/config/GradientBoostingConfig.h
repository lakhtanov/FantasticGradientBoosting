#ifndef GRADIENT_BOOSTING_CONFIG_GRADIENTBOOSTINGCONFIG_H_
#define GRADIENT_BOOSTING_CONFIG_GRADIENTBOOSTINGCONFIG_H_

#include "third_party/json/single_include/nlohmann/json.hpp"

namespace gradient_boosting {
namespace config {

// TODO(rialeksandrov): class containing all the necessary configuration
// information for running GradientBoosting model.
class GradientBoostingConfig {
 public:
  explicit GradientBoostingConfig(const nlohmann::json& config);
  enum class Verbose {v1, v2, v3};
  enum class LossFunction {MSE};

  Verbose GetVerbose() const;
  size_t GetNumberOfValueThresholds() const;
  size_t GetNumberOfStatisticsThresholds() const;
  LossFunction GetLossFunction() const;
 private:
  Verbose GetVerbose(const nlohmann::json& config);
  size_t GetNumberOfValueThresholds(const nlohmann::json& config);
  size_t GetNumberOfStatisticsThresholds(const nlohmann::json& config);
  LossFunction GetLossFunction(const nlohmann::json& config);

  const Verbose verbose_;
  const size_t value_thresholds_;
  const size_t statistics_thresholds_;
  const LossFunction loss_function_;
};

}  // namespace config
}  // namespace gradient_boosting

#endif  // GRADIENT_BOOSTING_CONFIG_GRADIENTBOOSTINGCONFIG_H_
