#ifndef GRADIENT_BOOSTING_GRADIENTBOOSTING_H_
#define GRADIENT_BOOSTING_GRADIENTBOOSTING_H_

#include <unordered_map>
#include <string>

#include <gradient_boosting/config/GradientBoostingConfig.h>
#include <utils/data_containers/DataContainer.h>

#include "third_party/ctpl/ctpl_stl.h"

namespace gradient_boosting {

// TODO(rialeksandrov,lakhtanov): class for learning gradient boosting ml model.
class GradientBoosting {
 public:
  explicit GradientBoosting(const gradient_boosting::config::GradientBoostingConfig& config);
  void TestGradientBoosting(const utils::data_containers::DataContainer& data);
  void Fit(const utils::data_containers::DataContainer& data);
  std::unordered_map<std::string, double> PredictProba(const utils::data_containers::DataContainer& data) const;
  std::unordered_map<std::string, std::string> PredictClassName(const utils::data_containers::DataContainer& data) const;
 private:
  gradient_boosting::config::GradientBoostingConfig config_;
  ctpl::thread_pool thread_pool_;
};

}  // namespace gradient_boosting

#endif  // GRADIENT_BOOSTING_GRADIENTBOOSTING_H_
