#include <algorithm>
#include <cassert>
#include <cmath>

#include "gradient_boosting/categories/CategoricalConverter.h"

namespace gradient_boosting {
namespace categories {

CategoricalConverter::CategoricalConverter(const std::vector<std::string>& features,
                                           const std::vector<size_t>& class_ids)
    : container_(features)
    , lambda_(log(features.size() + 2)) {
  if (!class_ids.empty()) {
    size_t number_of_classes = *std::max_element(class_ids.begin(), class_ids.end());
    number_of_classes += 1;
    assert(number_of_classes <= 2 && features.size() == class_ids.size());
    features_count_.resize(container_.Size());
    conversion_result_.resize(container_.Size());
    for (size_t index = 0; index < features.size(); ++index) {
      const auto &feature_value = features[index];
      size_t id = container_.GetId(feature_value);
      features_count_[id]++;
      class_sum_[id] += class_ids[index];
      default_probability_+= class_ids[index];
      conversion_result_[index] = Convert(id);
    }
    default_probability_ /= class_ids.size();
  }
}

std::vector<double> CategoricalConverter::GetConversionResult() const {
  return conversion_result_;
}

double CategoricalConverter::Convert(const std::string& feature_value) {
  size_t id = container_.GetId(feature_value);
  if (id < class_sum_.size()) {
    return Convert(id);
  } else {
    return default_probability_;
  }
}

double CategoricalConverter::Convert(size_t id) const {
  return (class_sum_[id] + lambda_ / 2.0) / (features_count_[id] + lambda_);
}

}  // namespace categories
}  // namespace gradient_boosting
