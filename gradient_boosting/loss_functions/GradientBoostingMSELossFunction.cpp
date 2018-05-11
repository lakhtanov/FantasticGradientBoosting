#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <iostream>

#include "gradient_boosting/loss_functions/GradientBoostingMSELossFunction.h"
#include "gradient_boosting/loss_functions/GradientBoostingSplitInfo.h"

namespace gradient_boosting {
namespace loss_functions {

using std::max;
using std::partial_sum;
using std::vector;

std::unique_ptr<GradientBoostingLossFunction>
    GradientBoostingMSELossFunction::Clone() const {
  return
      std::make_unique<GradientBoostingMSELossFunction>(
          features_objects_,
          objects_features_,
          target_values_);
}

void GradientBoostingMSELossFunction::Configure(
    size_t feature,
    const vector<size_t>& objects) {
  GradientBoostingLossFunction::Configure(feature, objects);

  vector<double> feature_target_values(
      features_objects_[feature].size(), 0.0);
  vector<size_t> objects_num(features_objects_[feature].size(), 0);
  // std::cout << "Here we are" << std::endl;
  for (size_t object : objects) {
    // std::cout << "Here we are " <<  object << " " <<feature << " " << objects_features_.size() << " " << features_objects_.size() << std::endl;
    const size_t object_feature_value = objects_features_[object][feature];
    feature_target_values[object_feature_value] += target_values_[object];
    ++objects_num[object_feature_value];
  }
  // std::cout << "Here we are" << std::endl;
  objects_prefix_num_sum_.clear();
  objects_prefix_num_sum_.resize(objects_num.size(), 0);
  partial_sum(
      objects_num.begin(),
      objects_num.end(),
      objects_prefix_num_sum_.begin());

  target_values_prefix_squares_sum_.clear();
  target_values_prefix_squares_sum_.resize(feature_target_values.size());
  partial_sum(
      feature_target_values.begin(),
      feature_target_values.end(),
      target_values_prefix_squares_sum_.begin(),
      [](double a, double b) { return a + b * b; });

  target_values_prefix_sum_.clear();
  target_values_prefix_sum_.resize(feature_target_values.size());
  partial_sum(
      feature_target_values.begin(),
      feature_target_values.end(),
      target_values_prefix_sum_.begin());
}

size_t GradientBoostingMSELossFunction::GetLeftSplitSize(
    size_t feature_split_value) const {
  return objects_prefix_num_sum_[feature_split_value];
}

GradientBoostingSplitInfo GradientBoostingMSELossFunction::GetLoss(
    size_t feature_split_value) const {
  const double left_split_squares_sum =
      target_values_prefix_squares_sum_[feature_split_value];
  const double left_split_sum = target_values_prefix_sum_[feature_split_value];
  const size_t left_split_size = GetLeftSplitSize(feature_split_value);
  const double left_split_sum_avg =
      left_split_sum / max(left_split_size, static_cast<size_t>(1));

  const double right_split_squares_sum =
      target_values_prefix_squares_sum_.back() - left_split_squares_sum;
  const double right_split_sum =
      target_values_prefix_sum_.back() - left_split_sum;
  const size_t right_split_size = GetRightSplitSize(feature_split_value);
  const double right_split_sum_avg =
      right_split_sum / max(right_split_size, static_cast<size_t>(1));

  const double loss =
      GetLossNode(
          left_split_size,
          left_split_squares_sum,
          left_split_sum,
          left_split_sum_avg)
      + log(left_split_size + 1.0)
      + GetLossNode(
          right_split_size,
          right_split_squares_sum,
          right_split_sum,
          right_split_sum_avg)
      + log(right_split_size + 1.0);
  return GradientBoostingSplitInfo(
      loss,
      left_split_size,
      left_split_sum_avg,
      right_split_size,
      right_split_sum_avg);
}

size_t GradientBoostingMSELossFunction::GetRightSplitSize(
    size_t feature_split_value) const {
  return objects_prefix_num_sum_.back() - GetLeftSplitSize(feature_split_value);
}

double GradientBoostingMSELossFunction::GetLoss(
    double value, double target_value) const {
  const double diff = target_value - value;
  return diff * diff;
}

double GradientBoostingMSELossFunction::GetLossNode(
    size_t split_size,
    double split_squares_sum,
    double split_sum,
    double split_sum_avg) const {
  return (
      split_squares_sum
      - 2 * split_sum * split_sum_avg
      + split_size * split_sum_avg * split_sum_avg);
}

}  // namespace loss_functions
}  // namespace gradient_boosting
