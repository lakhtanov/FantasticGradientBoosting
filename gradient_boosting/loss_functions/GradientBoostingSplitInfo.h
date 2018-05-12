#ifndef GRADIENT_BOOSTING_LOSS_FUNCTIONS_GRADIENTBOOSTINGSPLITINFO_H_
#define GRADIENT_BOOSTING_LOSS_FUNCTIONS_GRADIENTBOOSTINGSPLITINFO_H_

#include <cstddef>

namespace gradient_boosting {
namespace loss_functions {

struct GradientBoostingSplitInfo {
  GradientBoostingSplitInfo() = default;
  GradientBoostingSplitInfo(
      const GradientBoostingSplitInfo& split_info) = default;
  explicit GradientBoostingSplitInfo(
      double loss,
      size_t left_split_size,
      double left_split_value,
      size_t right_split_size,
      double right_split_value);

  size_t left_split_size;
  double left_split_value;
  double loss;
  size_t right_split_size;
  double right_split_value;
};

}  // namespace loss_functions
}  // namespace gradient_boosting

#endif  // GRADIENT_BOOSTING_LOSS_FUNCTIONS_GRADIENTBOOSTINGSPLITINFO_H_
