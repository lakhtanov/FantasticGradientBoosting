#ifndef UTILS_DATA_CONTAINERS_ELEMENTCONTAINER_H_
#define UTILS_DATA_CONTAINERS_ELEMENTCONTAINER_H_

#include <string>

namespace utils {
namespace data_containers {

class ElementContainer {
 public:
  enum class DataType {Double, String};
  explicit ElementContainer(const std::string& str);
  bool IsDouble() const;
  bool IsString() const;
  double GetDouble() const;
  std::string GetString() const;
 private:
  DataType GetDataType() const;
  void ChangeDataType(const DataType& type);
  friend class DataValidator;
  std::string raw_data_;
  DataType type_;
};

}  // namespace data_containers
}  // namespace utils


#endif  // UTILS_DATA_CONTAINERS_ELEMENTCONTAINER_H_
