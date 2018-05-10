#include "gradient_boosting/loss_functions/GradientBoostingSplitInfo.h"

namespace gradient_boosting {
namespace loss_functions {

GradientBoostingSplitInfo::GradientBoostingSplitInfo(
    double loss,
    size_t left_split_size,
    double left_split_value,
    size_t right_split_size,
    double right_split_value)
    : loss(loss)
      , left_split_size(left_split_size)
      , left_split_value(left_split_value)
      , right_split_size(right_split_size)
      , right_split_value(right_split_value) {
}

}  // namespace loss_functions
}  // namespace gradient_boosting
