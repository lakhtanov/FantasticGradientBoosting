#ifndef UTILS_DATA_CONTAINERS_ELEMENTCONTAINER_H_
#define UTILS_DATA_CONTAINERS_ELEMENTCONTAINER_H_

#include <string>

namespace utils {

namespace  data_containers {

class ElementContainer {
 public:
  enum class DataType {Double, String, LongInteger};
  explicit ElementContainer(const std::string& str);
  bool IsDouble();
  bool IsString();
  double GetDouble();
  std::string GetString();

 private:
  std::string raw_data_;
  DataType type_;
};

}  // namespace data_containers
}  // namespace utils


#endif  // UTILS_DATA_CONTAINERS_ELEMENTCONTAINER_H_
