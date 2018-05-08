#include "gradient_boosting/binarization/ThresholdCreator.h"

namespace gradient_boosting {
namespace binarization {

using std::vector;

ThresholdCreator::ThresholdCreator(size_t num_thresholds)
    : num_thresholds_(num_thresholds) {
}

vector<double> ThresholdCreator::CreateThresholds(
    const ThresholdCreator& threshold_creator,
    const std::vector<double>& features) {
  return threshold_creator.CreateThresholds(features);
}

}  // namespace binarization
}  // namespace gradient_boosting
