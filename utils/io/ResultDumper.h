#ifndef UTILS_IO_RESULTDUMPER_H_
#define UTILS_IO_RESULTDUMPER_H_

#include <fstream>
#include <string>
#include <vector>

namespace utils {
namespace io {

class ResultDumper {
 public:
  explicit ResultDumper(
      const std::string& file_name,
      const std::string& id_name,
      const std::string& target_value_name);
  void AddResult(std::string id, double result);
  void AddResults(
      const std::vector<std::string>& ids,
      const std::vector<double>& results);

 private:
  std::ofstream out_;
  const std::string file_name_;
  const std::string id_name_;
  const std::string target_value_name_;
};

}  // namespace io
}  // namespace utils

#endif  // UTILS_IO_RESULTDUMPER_H_
