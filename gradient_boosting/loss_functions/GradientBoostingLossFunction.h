#ifndef GRADIENT_BOOSTING_LOSS_FUNCTIONS_GRADIENTBOOSTINGLOSSFUNCTION_H_
#define GRADIENT_BOOSTING_LOSS_FUNCTIONS_GRADIENTBOOSTINGLOSSFUNCTION_H_

#include <vector>

#include "gradient_boosting/loss_functions/GradientBoostingSplitInfo.h"

namespace gradient_boosting {
namespace loss_functions {

class GradientBoostingLossFunction {
 public:
  explicit GradientBoostingLossFunction(
      const std::vector<std::vector<size_t>>& features_objects,
      const std::vector<std::vector<size_t>>& objects_features,
      const std::vector<double>& target_values);
  virtual void Configure(size_t feature, const std::vector<size_t>& objects);
  virtual std::vector<size_t> GetLeftSplit(size_t feature_split_value) const;
  virtual size_t GetLeftSplitSize(size_t feature_split_value) const;
  virtual GradientBoostingSplitInfo GetLoss(
      size_t feature_split_value) const = 0;
  virtual std::vector<size_t> GetRightSplit(size_t feature_split_value) const;
  virtual size_t GetRightSplitSize(size_t feature_split_value) const;
 protected:
  size_t feature_;
  const std::vector<std::vector<size_t>>& features_objects_;
  std::vector<size_t> objects_;
  const std::vector<std::vector<size_t>>& objects_features_;
  const std::vector<double>& target_values_;
};

}  // namespace loss_functions
}  // namespace gradient_boosting

#endif  // GRADIENT_BOOSTING_LOSS_FUNCTIONS_GRADIENTBOOSTINGLOSSFUNCTION_H_
