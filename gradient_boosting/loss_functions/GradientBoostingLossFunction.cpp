#include "gradient_boosting/loss_functions/GradientBoostingLossFunction.h"

namespace gradient_boosting {
namespace loss_functions {

using std::vector;

GradientBoostingLossFunction::GradientBoostingLossFunction(
    const std::vector<std::vector<size_t>>& features_objects,
    const std::vector<std::vector<size_t>>& objects_features,
    const std::vector<double>& target_values)
    : features_objects_(features_objects)
      , objects_features_(objects_features)
      , target_values_(target_values) {
}

void GradientBoostingLossFunction::Configure(
    size_t feature,
    const std::vector<size_t>& objects) {
  feature_ = feature;
  objects_ = objects;
}

vector<size_t> GradientBoostingLossFunction::GetLeftSplit(
    size_t feature_split_value) const {
  std::vector<size_t> left_split;
  for (size_t object : objects_) {
    if (features_objects_[feature_][object] <= feature_split_value) {
      left_split.push_back(object);
    }
  }

  return left_split;
}

size_t GradientBoostingLossFunction::GetLeftSplitSize(
    size_t feature_split_value) const {
  size_t left_split_size = 0;
  for (size_t object : objects_) {
    if (features_objects_[feature_][object] <= feature_split_value) {
      ++left_split_size;
    }
  }

  return left_split_size;
}

vector<size_t> GradientBoostingLossFunction::GetRightSplit(
    size_t feature_split_value) const {
  std::vector<size_t> right_split;
  for (size_t object : objects_) {
    if (!(features_objects_[feature_][object] <= feature_split_value)) {
      right_split.push_back(object);
    }
  }

  return right_split;
}

size_t GradientBoostingLossFunction::GetRightSplitSize(
    size_t feature_split_value) const {
  size_t right_split_size = 0;
  for (size_t object : objects_) {
    if (!(features_objects_[feature_][object] <= feature_split_value)) {
      ++right_split_size;
    }
  }

  return right_split_size;
}

}  // namespace loss_functions
}  // namespace gradient_boosting
