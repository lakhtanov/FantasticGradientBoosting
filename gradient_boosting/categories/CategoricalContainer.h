#ifndef GRADIENT_BOOSTING_CATEGORIES_CATEGORICALCONTAINER_H_
#define GRADIENT_BOOSTING_CATEGORIES_CATEGORICALCONTAINER_H_

#include <string>
#include <vector>
#include <unordered_map>

namespace gradient_boosting {
namespace categories {

class CategoricalContainer {
 public:
  explicit CategoricalContainer(const std::vector<std::string>& features);
  size_t GetId(const std::string& feature_value);
  size_t Size() const;
 private:
  std::unordered_map<std::string, size_t> table_;
};

}  // namespace categories
}  // namespace gradient_boosting

#endif  // GRADIENT_BOOSTING_CATEGORIES_CATEGORICALCONTAINER_H_
