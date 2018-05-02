#ifndef UTILS_IO_SIMPLECSVREADER_H_
#define UTILS_IO_SIMPLECSVREADER_H_

#include <string>

#include "utils/io/CSVReaderBase.h"

namespace utils {
namespace io {

class SimpleCSVReader final : public CSVReaderBase {
 public:
  explicit SimpleCSVReader(const std::string& file_name);
  SimpleCSVReader(const std::string& file_name, bool header);
  data_containers::DataContainer ReadFile() const final;
};

}  // namespace io
}  // namespace utils


#endif  // UTILS_IO_SIMPLECSVREADER_H_
