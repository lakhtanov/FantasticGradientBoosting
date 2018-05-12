#include <utility>

#include "gradient_boosting/categories/CategoricalContainer.h"

namespace gradient_boosting {
namespace categories {

using std::string;

CategoricalContainer::CategoricalContainer(
    const std::vector<string>& features) {
  for (const auto& feature_value : features) {
    if (!table_.count(feature_value)) {
      const size_t next_id = table_.size();
      table_[feature_value] = next_id;
    }
  }
}

std::pair<bool, size_t> CategoricalContainer::GetId(
    const string& feature_value) const {
  if (table_.count(feature_value)) {
    return {true, table_.at(feature_value)};
  } else {
    return {false, 0};
  }
}

size_t CategoricalContainer::Size() const {
  return table_.size();
}

}  // namespace categories
}  // namespace gradient_boosting
