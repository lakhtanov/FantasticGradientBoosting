#include <algorithm>
#include <cassert>
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
  vector<double> feature_target_square_values(
      features_objects_[feature].size(), 0.0);
  vector<size_t> objects_num(features_objects_[feature].size(), 0);
  for (size_t object : objects) {
    const size_t object_feature_value = objects_features_[object][feature];
    feature_target_values[object_feature_value] += target_values_[object];
    feature_target_square_values[object_feature_value] +=
        target_values_[object] * target_values_[object];
    ++objects_num[object_feature_value];
  }
  objects_prefix_num_sum_.clear();
  objects_prefix_num_sum_.resize(objects_num.size(), 0);
  partial_sum(
      objects_num.begin(),
      objects_num.end(),
      objects_prefix_num_sum_.begin());

  target_values_prefix_squares_sum_.clear();
  target_values_prefix_squares_sum_.resize(feature_target_values.size(), 0);
  partial_sum(
      feature_target_square_values.begin(),
      feature_target_square_values.end(),
      target_values_prefix_squares_sum_.begin());

  target_values_prefix_sum_.clear();
  target_values_prefix_sum_.resize(feature_target_values.size());
  partial_sum(
      feature_target_values.begin(),
      feature_target_values.end(),
      target_values_prefix_sum_.begin());
}

size_t GradientBoostingMSELossFunction::GetLeftSplitSize(
    size_t feature_split_value) const {
  size_t left_split_size = 0;
  for (size_t object : objects_) {
    if (objects_features_[object][feature_] <= feature_split_value) {
      ++left_split_size;
    }
  }
  return left_split_size;
  // return objects_prefix_num_sum_[feature_split_value];
}

GradientBoostingSplitInfo GradientBoostingMSELossFunction::GetLoss(
    size_t feature_split_value) const {
  double closs = 0;
  size_t cleft_split_size = 0;
  double cleft_split_sum_avg = 0;
  double cleft_split_sum_square = 0;
  size_t cright_split_size = 0;
  double cright_split_sum_avg = 0;
  double cright_split_sum_square = 0;

  for (size_t object : objects_) {
    if (objects_features_[object][feature_] <= feature_split_value) {
      ++cleft_split_size;
      cleft_split_sum_avg += target_values_[object];
      cleft_split_sum_square += target_values_[object] * target_values_[object];
    } else {
      ++cright_split_size;
      cright_split_sum_avg += target_values_[object];
      cright_split_sum_square += target_values_[object] * target_values_[object];
    }
  }

  cleft_split_sum_avg /= max(cleft_split_size, size_t(1));
  cright_split_sum_avg /= max(cright_split_size, size_t(1));

  for (size_t object : objects_) {
    if (objects_features_[object][feature_] <= feature_split_value) {
      closs += (target_values_[object] - cleft_split_sum_avg) * (target_values_[object] - cleft_split_sum_avg);
    } else {
      closs += (target_values_[object] - cright_split_sum_avg) * (target_values_[object] - cright_split_sum_avg);
    }
  }
  /*if (closs < 1e-5) {
    std::cout << "cleft_split_size " << cleft_split_size << std::endl;
    std::cout << "cright_split_size " << cright_split_size << std::endl;
    std::cout << "cleft_split_sum_avg " << cleft_split_sum_avg << std::endl;
    std::cout << "cright_split_sum_avg " << cright_split_sum_avg << std::endl;
    std::cout << "closs " << closs << std::endl;
  }*/

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
      // + log(left_split_size + 1.0)
      + GetLossNode(
          right_split_size,
          right_split_squares_sum,
          right_split_sum,
          right_split_sum_avg);
      // + log(right_split_size + 1.0);*/
  // std::cout << "left_split_sum_avg " << left_split_sum_avg << " " << cleft_split_sum_avg << std::endl;
  // std::cout << "right_split_sum_avg " << right_split_sum_avg << " " << cright_split_sum_avg << std::endl;
  // std::cout << "left_split_sum_square " << left_split_squares_sum << " " << cleft_split_sum_square << std::endl;
  // std::cout << "right_split_sum_square " << right_split_squares_sum << " " << cright_split_sum_square << std::endl;
  /* std::cout << "GradientBoostingMSELossFunction::GetLoss num_objects" << objects_.size() << std::endl;
  std::cout << "GradientBoostingMSELossFunction::GetLoss left_split_size" << cleft_split_size << " " << left_split_size << std::endl;
  std::cout << "GradientBoostingMSELossFunction::GetLoss right_split_size" << cright_split_size << " " << right_split_size << std::endl;
  std::cout << "GradientBoostingMSELossFunction::GetLoss loss" << closs << " " << loss << std::endl;
  return GradientBoostingSplitInfo(
      loss,
      left_split_size,
      left_split_sum_avg,
      right_split_size,
      right_split_sum_avg);*/
  return GradientBoostingSplitInfo(
      closs,
      cleft_split_size,
      cleft_split_sum_avg,
      cright_split_size,
      cright_split_sum_avg);
}

size_t GradientBoostingMSELossFunction::GetRightSplitSize(
    size_t feature_split_value) const {
  size_t right_split_size = 0;
  for (size_t object : objects_) {
    if (!(objects_features_[object][feature_] <= feature_split_value)) {
      ++right_split_size;
    }
  }
  return right_split_size;
  // return objects_prefix_num_sum_.back() - GetLeftSplitSize(feature_split_value);
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
