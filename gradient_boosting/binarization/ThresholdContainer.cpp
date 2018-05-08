#include <algorithm>

#include "gradient_boosting/binarization/ThresholdContainer.h"

namespace gradient_boosting {
namespace binarization {

using std::distance;
using std::make_move_iterator;
using std::vector;

ThresholdContainer::ThresholdContainer(
    const vector<std::unique_ptr<ThresholdCreator>>& creators,
    const vector<double>& features) {
  for (auto& creator : creators) {
    const std::vector<double> creator_thresholds =
        ThresholdCreator::CreateThresholds(*creator, features);
    thresholds_.insert(
        thresholds_.end(),
        make_move_iterator(creator_thresholds.begin()),
        make_move_iterator(creator_thresholds.end()));
  }
  std::sort(thresholds_.begin(), thresholds_.end());
  auto unique_end = std::unique(thresholds_.begin(), thresholds_.end());
  thresholds_.resize(distance(thresholds_.begin(), unique_end));
}

size_t ThresholdContainer::GetBin(double feature_value) const {
  auto position =
      std::lower_bound(thresholds_.begin(), thresholds_.end(), feature_value);
  return distance(thresholds_.begin(), position);
}

const vector<double>& ThresholdContainer::GetThresholds() const {
  return thresholds_;
}

}  // namespace binarization
}  // namespace gradient_boosting
