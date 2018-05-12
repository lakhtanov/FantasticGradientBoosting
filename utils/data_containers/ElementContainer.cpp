#include "utils/data_containers/ElementContainer.h"

#include <cassert>
#include <string>

namespace utils {

namespace data_containers {

ElementContainer::ElementContainer(const std::string &str)
    : raw_data_(str), type_(DataType::String) {
  try {
    std::stod(raw_data_);
    type_ = DataType::Double;
  } catch (...) {
    // TODO(rialeksandrov) Debug Output here.
  }
}

bool ElementContainer::IsDouble() const {
  return type_ == DataType::Double;
}

bool ElementContainer::IsString() const {
  return type_ == DataType::String;
}

double ElementContainer::GetDouble() const {
  assert(IsDouble());
  return std::stod(raw_data_);
}

std::string ElementContainer::GetString() const {
  return raw_data_;
}

ElementContainer::DataType ElementContainer::GetDataType() const {
  return type_;
}

void ElementContainer::ChangeDataType(const ElementContainer::DataType& type) {
  type_ = type;
}

}  // namespace data_containers
}  // namespace utils
