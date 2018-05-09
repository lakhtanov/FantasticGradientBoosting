#ifndef GRADIENT_BOOSTING_LOSS_FUNCTIONS_GRADIENTBOOSTINGMSELOSSFUNCTION_H_
#define GRADIENT_BOOSTING_LOSS_FUNCTIONS_GRADIENTBOOSTINGMSELOSSFUNCTION_H_

#include <vector>

#include "gradient_boosting/loss_functions/GradientBoostingLossFunction.h"

namespace gradient_boosting {
namespace loss_functions {

class GradientBoostingMSELossFunction : public GradientBoostingLossFunction {
 public:
  using GradientBoostingLossFunction::GradientBoostingLossFunction;

  void Configure(size_t feature, const std::vector<size_t>& objects) override;
  size_t GetLeftSplitSize(size_t feature_split_value) const override;
  double GetLoss(size_t feature_split_value) const override;
  size_t GetRightSplitSize(size_t feature_split_value) const override;
 private:
  double GetLossNode(
      size_t split_size,
      double split_squares_sum,
      double split_sum,
      double split_sum_avg) const;
  std::vector<size_t> objects_prefix_num_sum_;
  std::vector<double> target_values_prefix_squares_sum_;
  std::vector<double> target_values_prefix_sum_;
};

}  // namespace loss_functions
}  // namespace gradient_boosting

#endif  // GRADIENT_BOOSTING_LOSS_FUNCTIONS_GRADIENTBOOSTINGMSELOSSFUNCTION_H_
