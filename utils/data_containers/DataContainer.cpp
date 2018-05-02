#include "utils/data_containers/DataContainer.h"

#include <cassert>

namespace utils {
namespace data_containers {


DataContainer::DataContainer(const std::vector<std::vector<std::string>> &table)
    : DataContainer(std::vector<std::string>(table.front().size()), table)
{
}

DataContainer::DataContainer(const std::vector<std::string> &names,
                             const std::vector<std::vector<std::string>> &table)
    : names_(names)
    , rows_(table.size())
    , columns_(names.size())
    , validated_(false) {
  data_.resize(rows_);
  for (size_t row_idx = 0; row_idx < table.size(); ++row_idx) {
    const auto& row = table[row_idx];
    assert(row.size() == columns_);
    for (const auto& el : row) {
      data_[row_idx].emplace_back(el);
    }
  }
  Validation();
}

void DataContainer::Validation() {
  for (size_t column = 0; column < columns_; ++column) {
    if (rows_ <= 1) {
      continue;
    }
    DataValidator validator(data_[0][column]);
    for (size_t row = 0; row < rows_; ++row) {
      validator.Merge(data_[row][column]);
    }
    for (size_t row = 0; row < rows_; ++row) {
      validator.Apply(&data_[row][column]);
    }
  }
  validated_ = true;
}


DataValidator::DataValidator(const ElementContainer& el)
    : data_type_(el.GetDataType()) {
}

void DataValidator::Merge(const ElementContainer& el) {
  if (el.IsString()) {
    data_type_ = el.GetDataType();
  }
}

void DataValidator::Apply(ElementContainer* el) {
  el->ChangeDataType(data_type_);
}

}  // namespace data_containers
}  // namespace utils
