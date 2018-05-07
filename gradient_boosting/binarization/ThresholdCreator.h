#ifndef GRADIENT_BOOSTING_BINARIZATION_THRESHOLDCREATOR_H_
#define GRADIENT_BOOSTING_BINARIZATION_THRESHOLDCREATOR_H_

#include <memory>
#include <vector>

namespace gradient_boosting {
namespace binarization {

class ThresholdCreator {
 public:
  explicit ThresholdCreator(size_t num_thresholds);
  virtual ~ThresholdCreator() = default;
  static std::vector<double> CreateThresholds(
      const ThresholdCreator& threshold_creator,
      const std::vector<double>& features);
 protected:
  virtual std::vector<double> CreateThresholds_(
      const std::vector<double>& features) const = 0;
  size_t num_thresholds_;
};

}  // namespace binarization
}  // namespace gradient_boosting

#endif  // GRADIENT_BOOSTING_BINARIZATION_THRESHOLDCREATOR_H_
