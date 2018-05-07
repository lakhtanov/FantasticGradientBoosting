#include <algorithm>
#include <set>

#include "gradient_boosting/binarization/ThresholdCreatorByStatistics.h"

namespace gradient_boosting {
namespace binarization {

using std::vector;

vector<double> ThresholdCreatorByStatistics::CreateThresholds_(
    const vector<double>& features) const {
  if (features.empty() || !num_thresholds_) {
    return vector<double>();
  }


  const double delta_statistics = 100.0 / (num_thresholds_ + 1);
  std::set<size_t> unique_threshold_feature_positions;
  for (size_t i = 0; i < num_thresholds_; ++i) {
    const double threshold_statistics = (i + 1) * delta_statistics;
    const size_t feature_position =
        static_cast<size_t>(threshold_statistics / 100 * features.size());
    unique_threshold_feature_positions.insert(feature_position);
  }

  vector<double> sorted_features = features;
  std::sort(sorted_features.begin(), sorted_features.end());

  vector<double> thresholds;
  thresholds.reserve(unique_threshold_feature_positions.size());
  for (size_t feature_position : unique_threshold_feature_positions) {
    thresholds.push_back(sorted_features[feature_position]);
  }

  return thresholds;
}

}  // namespace binarization
}  // namespace gradient_boosting
