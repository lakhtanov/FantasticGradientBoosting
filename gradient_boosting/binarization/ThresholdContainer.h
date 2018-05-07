#ifndef GRADIENT_BOOSTING_BINARIZATION_THRESHOLDCONTAINER_H_
#define GRADIENT_BOOSTING_BINARIZATION_THRESHOLDCONTAINER_H_

#include <memory>
#include <vector>

#include "gradient_boosting/binarization/ThresholdCreator.h"

namespace gradient_boosting {
namespace binarization {

class ThresholdContainer {
 public:
  explicit ThresholdContainer(
      const std::vector<std::unique_ptr<ThresholdCreator>>& creators,
      const std::vector<double>& features);
  size_t GetBin(double feature_value) const;
  const std::vector<double>& GetThresholds() const;
 private:
  std::vector<double> thresholds_;
};

}  // namespace binarization
}  // namespace gradient_boosting

#endif  // GRADIENT_BOOSTING_BINARIZATION_THRESHOLDCONTAINER_H_
