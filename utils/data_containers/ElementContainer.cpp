#include "utils/data_containers/ElementContainer.h"

#include <cassert>
#include <string>

namespace utils {

namespace data_containers {

ElementContainer::ElementContainer(const std::string &str)
    : raw_data_(str), type_(DataType::String) {
  try {
    double el = std::stod(raw_data_);
    type_ = DataType::Double;
  } catch (...) {
    // TODO(rialeksandrov) Debug Output here.
  }
}

bool ElementContainer::IsDouble() {
  return type_ == DataType::Double;
}

bool ElementContainer::IsString() {
  return type_ == DataType::String;
}

double ElementContainer::GetDouble() {
  assert(IsDouble());
  return std::stod(raw_data_);
}

std::string ElementContainer::GetString() {
  return raw_data_;
}

}  // namespace data_containers
}  // namespace utils
