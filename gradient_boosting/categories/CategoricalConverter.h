#ifndef GRADIENT_BOOSTING_CATEGORIES_CATEGORICALCONVERTER_H_
#define GRADIENT_BOOSTING_CATEGORIES_CATEGORICALCONVERTER_H_

#include <string>
#include <vector>

#include "gradient_boosting/categories/CategoricalContainer.h"

namespace gradient_boosting {
namespace categories {

class CategoricalConverter {
 public:
  explicit CategoricalConverter(const std::vector<std::string>& features,
                                const std::vector<size_t>& class_ids);
  std::vector<double> GetConversionResult() const;
  double Convert(const std::string& feature_value) const;
 private:
  double Convert(size_t id) const;

  double default_probability_;
  std::vector<size_t> class_sum_;
  const CategoricalContainer container_;
  std::vector<double> conversion_result_;
  std::vector<size_t> features_count_;
  const double lambda_;
};

}  // namespace categories
}  // namespace gradient_boosting

#endif  // GRADIENT_BOOSTING_CATEGORIES_CATEGORICALCONVERTER_H_
