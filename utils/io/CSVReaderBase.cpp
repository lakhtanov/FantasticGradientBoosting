#include "utils/io/CSVReaderBase.h"

namespace utils {
namespace io {

CSVReaderBase::CSVReaderBase(const std::string &file_name)
    : file_name_(file_name) {
}

}  // namespace io
}  // namespace utils
