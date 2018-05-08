#include <algorithm>
#include <vector>

#include "gradient_boosting/binarization/ThresholdCreatorByValue.h"

namespace gradient_boosting {
namespace binarization {

using std::vector;

vector<double> ThresholdCreatorByValue::CreateThresholds(
    const vector<double>& features) const {
  if (features.empty() || !num_thresholds_) {
    return vector<double>();
  }

  const double min_element =
      *std::min_element(features.begin(), features.end());
  const double max_element =
      *std::max_element(features.begin(), features.end());
  const double delta = (max_element - min_element) / (num_thresholds_ + 1);
  vector<double> thresholds(num_thresholds_, min_element);
  for (size_t i = 0; i < num_thresholds_; ++i) {
    thresholds[i] += (i + 1) * delta;
  }

  return thresholds;
}

}  // namespace binarization
}  // namespace gradient_boosting
