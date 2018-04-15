//
// Created by Roman Aleksandrov on 11.04.2018.
//

#ifndef UTILS_DATA_CONTAINERS_DATACONTAINER_H_
#define UTILS_DATA_CONTAINERS_DATACONTAINER_H_

#include <string>
#include <vector>

#include "utils/data_containers/ElementContainer.h"

namespace utils {
namespace data_containers {

class DataContainer {
 public:
  DataContainer(const std::vector<std::string> &names,
                const std::vector<std::vector<std::string>> &table);

 private:
  void Validation();

  const std::vector<std::string> names_;
  std::vector<std::vector<ElementContainer>> data_;
  size_t rows_, columns_;
  bool validated_;
};

class DataValidator {
 private:
  friend DataContainer;
  explicit DataValidator(const ElementContainer &el);
  void Merge(const ElementContainer &el);
  void Apply(ElementContainer* el) const;
  ElementContainer::DataType data_type_;
};

}  // namespace data_containers
}  // namespace utils


#endif  // UTILS_DATA_CONTAINERS_DATACONTAINER_H_
