#include <algorithm>
#include <cassert>
#include <cmath>

#include "gradient_boosting/categories/CategoricalConverter.h"

namespace gradient_boosting {
namespace categories {

using std::string;
using std::vector;

CategoricalConverter::CategoricalConverter(const vector<string>& features,
                                           const vector<double>& class_ids)
    : container_(features)
    , lambda_(log(features.size() + 2)) {
  if (class_ids.empty()) {
    default_probability_ = 0.5;
  } else {
    const double number_of_classes = (*std::max_element(class_ids.begin(), class_ids.end())) + 1.0;
    assert(number_of_classes < 3 && features.size() == class_ids.size());
    features_count_.resize(container_.Size());
    conversion_result_.resize(container_.Size());
    for (size_t index = 0; index < features.size(); ++index) {
      const auto result = container_.GetId(features[index]);
      const size_t id = result.second;
      ++features_count_[id];
      class_sum_[id] += class_ids[index];
      default_probability_+= class_ids[index];
      conversion_result_[index] = Convert(id);
    }
    default_probability_ = (default_probability_ + lambda_ / 2.0) / (class_ids.size() + lambda_);
  }
}

vector<double> CategoricalConverter::GetConversionResult() const {
  return conversion_result_;
}

double CategoricalConverter::Convert(const string& feature_value) const {
  const auto result = container_.GetId(feature_value);
  if (result.first) {
    return Convert(result.second);
  } else {
    return default_probability_;
  }
}

double CategoricalConverter::Convert(size_t id) const {
  return (class_sum_[id] + lambda_ / 2.0) / (features_count_[id] + lambda_);
}

}  // namespace categories
}  // namespace gradient_boosting
