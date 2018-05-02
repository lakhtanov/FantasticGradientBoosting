#ifndef UTILS_IO_CSVREADERBASE_H_
#define UTILS_IO_CSVREADERBASE_H_

#include <string>

#include "utils/data_containers/DataContainer.h"

namespace utils {
namespace io {

class CSVReaderBase {
 public:
  explicit CSVReaderBase(const std::string& file_name, bool header);
  virtual data_containers::DataContainer ReadFile() const = 0;
 protected:
  const std::string file_name_;
  const bool header_;
};

}  // namespace io
}  // namespace utils

#endif  // UTILS_IO_CSVREADERBASE_H_
