#include "gradient_boosting/categories/CategoricalContainer.h"

namespace gradient_boosting {
namespace categories {

CategoricalContainer::CategoricalContainer(const std::vector<std::string>& features) {
  for (const auto& feature_value : features) {
    GetId(feature_value);
  }
}

size_t CategoricalContainer::GetId(const std::string& feature_value) {
  if (!table_.count(feature_value)) {
    size_t next_id = table_.size();
    table_[feature_value] = next_id;
  }
  return table_[feature_value];
}

size_t CategoricalContainer::Size() const {
  return table_.size();
}

}  // namespace categories
}  // namespace gradient_boosting
