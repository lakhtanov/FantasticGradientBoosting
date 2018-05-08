#ifndef GRADIENT_BOOSTING_BINARIZATION_THRESHOLDCREATORBYSTATISTICS_H_
#define GRADIENT_BOOSTING_BINARIZATION_THRESHOLDCREATORBYSTATISTICS_H_

#include <vector>

#include "gradient_boosting/binarization/ThresholdCreator.h"

namespace gradient_boosting {
namespace binarization {

class ThresholdCreatorByStatistics : public ThresholdCreator {
 public:
  using ThresholdCreator::ThresholdCreator;
 protected:
  std::vector<double> CreateThresholds(
      const std::vector<double>& features) const override;
};

}  // namespace binarization
}  // namespace gradient_boosting

#endif  // GRADIENT_BOOSTING_BINARIZATION_THRESHOLDCREATORBYSTATISTICS_H_
